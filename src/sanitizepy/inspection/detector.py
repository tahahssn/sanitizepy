"""
Intelligent Issue Detector and Dataset Quality Inspector.

Inspects a pandas DataFrame across multiple dimensions to generate structured
DatasetIssues, composite quality scores, and actionable Recommendations.
"""

from __future__ import annotations

import re
from typing import Final

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

from sanitizepy.exceptions import DependencyError
from sanitizepy.inspection.anomalies import AnomalyInspector, AnomalyMethod
from sanitizepy.inspection.health import DatasetHealthReport, DatasetIssue
from sanitizepy.inspection.near_duplicates import NearDuplicateDetector
from sanitizepy.models.base import ColumnReference
from sanitizepy.models.recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationCategory,
    RecommendationImpact,
    RecommendationPriority,
    RecommendationReason,
)

# ---------------------------------------------------------------------------
# Encoding-artifact detection helpers (Requirement 5.1)
# ---------------------------------------------------------------------------

# Mojibake often appears as sequences of Latin-1 Supplement characters
# (U+00C0–U+00FF) where Unicode letters would not normally cluster together.
# The pattern below flags the most reliable mojibake indicators that arise from
# UTF-8 bytes mis-decoded as Latin-1: Ã followed by a character in the
# Latin-1 supplement block, or common two-char prefix sequences such as Â·.
_MOJIBAKE_RE: re.Pattern[str] = re.compile(
    r"[\xc0-\xc3][\x80-\xbf\xa0-\xff]"  # UTF-8 2-byte encoded as Latin-1
    r"|Ã[^\s]"  # Classic Ã-prefix mojibake
    r"|â\x80[\x93\x94\x98\x99\x9c\x9d\xa2\xa6\xa0]"  # Common 3-byte sequences
)

# Unicode REPLACEMENT CHARACTER – produced by lossy decoding.
_REPLACEMENT_CHAR: str = "\ufffd"

# ASCII control characters excluding common printable-range whitespace
# (HT=\x09, LF=\x0a, CR=\x0d).  U+007F DELETE and U+0000 NULL are included.
_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _count_encoding_artifact_values(series: pd.Series) -> dict[str, int]:
    """
    Return counts of encoding-artifact cells in a string/object *series*.

    Keys
    ----
    ``"mojibake"``
        Values where the text matches common UTF-8-decoded-as-Latin-1 patterns.
    ``"replacement_char"``
        Values containing the Unicode replacement character U+FFFD.
    ``"control_chars"``
        Values containing non-printable ASCII control characters.

    Non-string items (including ``None`` / ``NaN``) are skipped.
    """
    mojibake_count = 0
    replacement_count = 0
    control_count = 0

    for value in series:
        if not isinstance(value, str):
            continue
        if _MOJIBAKE_RE.search(value):
            mojibake_count += 1
        if _REPLACEMENT_CHAR in value:
            replacement_count += 1
        if _CONTROL_CHAR_RE.search(value):
            control_count += 1

    return {
        "mojibake": mojibake_count,
        "replacement_char": replacement_count,
        "control_chars": control_count,
    }


# ---------------------------------------------------------------------------
# Anomaly detection wiring (Requirements 12.3, 12.5)
# ---------------------------------------------------------------------------

# Deterministic method used when reporting anomalies as DatasetIssues.
# Both "iqr" and "zscore" are Core_Stack-only, so no optional extra is
# required for the default method.
_ANOMALY_METHOD: Final[AnomalyMethod] = "iqr"

# Maps any anomaly method that requires an optional dependency to the extra
# that provides it. Core_Stack-only methods ("iqr", "zscore") are absent, so
# they never trigger a DependencyError. The structure is kept ready so that a
# future method requiring an extra raises DependencyError naming that extra.
_ANOMALY_METHOD_EXTRAS: Final[dict[str, str]] = {}


class IssueDetector:
    """
    Automated dataset inspector that identifies data quality defects,
    calculates composite health scores, and constructs explainable recommendations.
    """

    def inspect(self, dataframe: pd.DataFrame) -> DatasetHealthReport:
        """
        Inspect input DataFrame and return a DatasetHealthReport.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        rows, cols = dataframe.shape
        if rows == 0 or cols == 0:
            return DatasetHealthReport(
                health_score=0,
                completeness_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                validity_score=0.0,
                integrity_score=0.0,
                rows=rows,
                columns=cols,
                memory_mb=0.0,
                issues=[
                    DatasetIssue(
                        title="Empty DataFrame",
                        description="DataFrame contains 0 rows or 0 columns.",
                        severity="CRITICAL",
                        category="general",
                        evidence=f"Shape: ({rows}, {cols})",
                        recommendation_text=(
                            "Provide a non-empty DataFrame for analysis."
                        ),
                    )
                ],
                recommendations=[],
            )

        memory_mb = float(dataframe.memory_usage(deep=True).sum() / (1024 * 1024))
        issues: list[DatasetIssue] = []
        recommendations: list[Recommendation] = []

        total_cells = rows * cols
        missing_cells = int(dataframe.isna().sum().sum())
        dup_rows = int(dataframe.duplicated().sum())

        empty_cols: list[str] = []
        constant_cols: list[str] = []
        unparsed_dtype_cols: list[str] = []
        casing_issue_cols: list[str] = []
        outlier_cols: list[str] = []

        # 1. Inspect Duplicates
        if dup_rows > 0:
            dup_pct = (dup_rows / rows) * 100.0
            severity = "CRITICAL" if dup_pct > 10.0 else "WARNING"
            issues.append(
                DatasetIssue(
                    title="Duplicate Rows Detected",
                    description=(
                        f"Dataset contains {dup_rows:,} ({dup_pct:.1f}%) duplicate "
                        "rows."
                    ),
                    severity=severity,
                    category="duplicates",
                    evidence=f"{dup_rows} exact row duplicates found across dataset",
                    recommendation_text="Remove exact duplicate rows.",
                )
            )
            recommendations.append(
                Recommendation(
                    title="Remove Duplicate Rows",
                    description=f"Drop {dup_rows} identical rows from dataset.",
                    category=RecommendationCategory.DUPLICATES,
                    priority=(
                        RecommendationPriority.CRITICAL
                        if dup_pct > 10
                        else RecommendationPriority.HIGH
                    ),
                    action=RecommendationAction.REMOVE_DUPLICATES,
                    reason=RecommendationReason(
                        title="Duplicate Row Check",
                        explanation=(
                            f"{dup_rows} rows ({dup_pct:.1f}%) share identical values "
                            "across all columns."
                        ),
                    ),
                    impact=RecommendationImpact(
                        row_change=-dup_rows,
                        quality_score_delta=min(20.0, dup_pct * 0.8),
                    ),
                    confidence=1.0,
                    automatic=True,
                )
            )

        # 2. Inspect Column by Column
        for col in dataframe.columns:
            series = dataframe[col]
            n_missing = int(series.isna().sum())
            missing_pct = (n_missing / rows) * 100.0
            n_unique = int(series.nunique(dropna=True))

            # Missingness checks
            if missing_pct == 100.0:
                empty_cols.append(col)
                issues.append(
                    DatasetIssue(
                        title=f"Column '{col}' is 100% Missing",
                        description=f"Column '{col}' contains no non-null values.",
                        severity="CRITICAL",
                        category="missing_values",
                        column=col,
                        evidence="100% missing values (0 non-null values)",
                        recommendation_text=f"Drop empty column '{col}'.",
                    )
                )
                recommendations.append(
                    Recommendation(
                        title=f"Drop Empty Column '{col}'",
                        description=(
                            f"Remove column '{col}' because it contains only null "
                            "values."
                        ),
                        category=RecommendationCategory.MISSING_VALUES,
                        priority=RecommendationPriority.CRITICAL,
                        action=RecommendationAction.DROP_COLUMN,
                        column=ColumnReference(name=col, dtype=str(series.dtype)),
                        reason=RecommendationReason(
                            title="100% Missingness",
                            explanation=(
                                f"Column '{col}' provides zero informational "
                                "content."
                            ),
                        ),
                        impact=RecommendationImpact(
                            column_change=-1, quality_score_delta=5.0
                        ),
                        confidence=1.0,
                        automatic=True,
                    )
                )
            elif missing_pct > 0.0:
                severity = "WARNING" if missing_pct >= 15.0 else "INFO"
                rec_action = (
                    RecommendationAction.FILL_MEDIAN
                    if is_numeric_dtype(series)
                    else RecommendationAction.FILL_MODE
                )
                strategy = "median" if is_numeric_dtype(series) else "mode"
                issues.append(
                    DatasetIssue(
                        title=f"Column '{col}' Has {missing_pct:.1f}% Missing Values",
                        description=(
                            f"{n_missing:,} missing cells detected in column "
                            f"'{col}'."
                        ),
                        severity=severity,
                        category="missing_values",
                        column=col,
                        evidence=f"{n_missing}/{rows} values are missing",
                        recommendation_text=(
                            f"Fill missing values in '{col}' using {strategy}."
                        ),
                    )
                )
                recommendations.append(
                    Recommendation(
                        title=f"Impute Missing Values in '{col}'",
                        description=(
                            f"Fill {n_missing} missing values in '{col}' using "
                            f"{strategy}."
                        ),
                        category=RecommendationCategory.MISSING_VALUES,
                        priority=(
                            RecommendationPriority.HIGH
                            if missing_pct >= 15
                            else RecommendationPriority.MEDIUM
                        ),
                        action=rec_action,
                        column=ColumnReference(name=col, dtype=str(series.dtype)),
                        reason=RecommendationReason(
                            title=f"Missing Value Imputation ({strategy})",
                            explanation=(
                                f"Column '{col}' has {missing_pct:.1f}% missingness."
                            ),
                        ),
                        impact=RecommendationImpact(
                            quality_score_delta=min(10.0, missing_pct * 0.3)
                        ),
                        confidence=0.85,
                        automatic=missing_pct <= 50,
                    )
                )

            # Constant / Zero Variance checks
            if n_unique == 1 and missing_pct < 100.0:
                constant_cols.append(col)
                issues.append(
                    DatasetIssue(
                        title=f"Constant Column '{col}'",
                        description=(
                            f"Column '{col}' contains only 1 unique non-null " "value."
                        ),
                        severity="WARNING",
                        category="integrity",
                        column=col,
                        evidence="Single unique value across all non-null entries",
                        recommendation_text=(
                            f"Consider removing constant column '{col}'."
                        ),
                    )
                )
                recommendations.append(
                    Recommendation(
                        title=f"Drop Constant Column '{col}'",
                        description=f"Remove column '{col}' as it has zero variance.",
                        category=RecommendationCategory.FEATURE_SELECTION,
                        priority=RecommendationPriority.MEDIUM,
                        action=RecommendationAction.DROP_COLUMN,
                        column=ColumnReference(name=col, dtype=str(series.dtype)),
                        reason=RecommendationReason(
                            title="Zero Variance",
                            explanation=(
                                f"Column '{col}' holds the same constant value "
                                "across all rows."
                            ),
                        ),
                        impact=RecommendationImpact(
                            column_change=-1, quality_score_delta=2.0
                        ),
                        confidence=0.9,
                        automatic=True,
                    )
                )

            # Datatype inspection for Object / String columns
            if is_object_dtype(series) or is_string_dtype(series):
                non_null = series.dropna().astype(str)
                if len(non_null) > 0:
                    # Casing inconsistencies
                    cleaned_str = non_null.str.strip().str.lower()
                    if cleaned_str.nunique() < non_null.nunique():
                        casing_issue_cols.append(col)
                        issues.append(
                            DatasetIssue(
                                title=f"Inconsistent String Casing in '{col}'",
                                description=(
                                    f"Column '{col}' has casing or trailing whitespace "
                                    "variants."
                                ),
                                severity="WARNING",
                                category="consistency",
                                column=col,
                                evidence=(
                                    f"{non_null.nunique()} raw unique values -> "
                                    f"{cleaned_str.nunique()} normalized unique values"
                                ),
                                recommendation_text=(
                                    f"Normalize string casing and trim whitespace "
                                    f"in '{col}'."
                                ),
                            )
                        )

                    # Unparsed datetime string detection
                    sample_vals = non_null.iloc[: min(100, len(non_null))]
                    date_match_count = 0
                    for val in sample_vals:
                        if re.search(
                            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
                            val,
                        ):
                            date_match_count += 1
                    if date_match_count / len(sample_vals) > 0.7:
                        unparsed_dtype_cols.append(col)
                        match_pct = int(date_match_count / len(sample_vals) * 100)
                        issues.append(
                            DatasetIssue(
                                title=f"Unparsed Datetime String in '{col}'",
                                description=(
                                    f"Column '{col}' is stored as string/object but "
                                    "resembles datetime values."
                                ),
                                severity="WARNING",
                                category="data_types",
                                column=col,
                                evidence=(
                                    f">{match_pct}% of sample values match date "
                                    "formats"
                                ),
                                recommendation_text=(
                                    f"Convert '{col}' to datetime dtype."
                                ),
                            )
                        )
                        recommendations.append(
                            Recommendation(
                                title=f"Convert '{col}' to Datetime",
                                description=(
                                    f"Parse string column '{col}' into datetime "
                                    "format."
                                ),
                                category=RecommendationCategory.DATA_TYPES,
                                priority=RecommendationPriority.HIGH,
                                action=RecommendationAction.CONVERT_TYPE,
                                column=ColumnReference(
                                    name=col, dtype=str(series.dtype)
                                ),
                                reason=RecommendationReason(
                                    title="Datetime Pattern Match",
                                    explanation=(
                                        f"Column '{col}' strings follow standard date "
                                        "patterns."
                                    ),
                                ),
                                impact=RecommendationImpact(quality_score_delta=5.0),
                                confidence=0.9,
                                automatic=True,
                            )
                        )

                    # Encoding-artifact detection (Requirement 5.1)
                    # Additively flag mojibake, replacement characters, and
                    # control characters as a separate "encoding_artifacts"
                    # category without altering or replacing any existing issue.
                    artifact_counts = _count_encoding_artifact_values(series)
                    total_artifacts = sum(artifact_counts.values())

                    if total_artifacts > 0:
                        artifact_pct = (total_artifacts / len(non_null)) * 100.0
                        severity = "CRITICAL" if artifact_pct > 10.0 else "WARNING"
                        evidence_parts: list[str] = []
                        if artifact_counts["mojibake"]:
                            evidence_parts.append(
                                f"{artifact_counts['mojibake']} mojibake pattern(s)"
                            )
                        if artifact_counts["replacement_char"]:
                            evidence_parts.append(
                                f"{artifact_counts['replacement_char']} "
                                "replacement character(s) (U+FFFD)"
                            )
                        if artifact_counts["control_chars"]:
                            evidence_parts.append(
                                f"{artifact_counts['control_chars']} "
                                "control character(s)"
                            )
                        issues.append(
                            DatasetIssue(
                                title=(f"Encoding Artifacts Detected in '{col}'"),
                                description=(
                                    f"Column '{col}' contains {total_artifacts:,} "
                                    f"cell(s) ({artifact_pct:.1f}%) with encoding "
                                    "artifacts (mojibake, replacement characters, "
                                    "or control characters)."
                                ),
                                severity=severity,
                                category="encoding_artifacts",
                                column=col,
                                evidence="; ".join(evidence_parts),
                                recommendation_text=(
                                    f"Inspect and repair encoding artifacts "
                                    f"in column '{col}'."
                                ),
                            )
                        )

            # Outlier checks for numeric columns
            if is_numeric_dtype(series) and n_unique > 5:
                s_clean = series.dropna()
                if len(s_clean) > 10:
                    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = (
                            (s_clean < (q1 - 1.5 * iqr)) | (s_clean > (q3 + 1.5 * iqr))
                        ).sum()
                        outlier_pct = (outliers / len(s_clean)) * 100.0
                        if outlier_pct > 3.0:
                            outlier_cols.append(col)
                            issues.append(
                                DatasetIssue(
                                    title=f"Outliers Detected in '{col}'",
                                    description=(
                                        f"{outliers:,} ({outlier_pct:.1f}%) "
                                        "statistical "
                                        f"outliers in '{col}'."
                                    ),
                                    severity="INFO" if outlier_pct < 10 else "WARNING",
                                    category="outliers",
                                    column=col,
                                    evidence=(
                                        f"{outliers} points outside 1.5*IQR " "bounds"
                                    ),
                                    recommendation_text=(
                                        "Investigate numerical distribution and "
                                        "extreme "
                                        f"values in '{col}'."
                                    ),
                                )
                            )

        # 2b. Anomaly detection (Requirements 12.3, 12.5)
        # Additively report anomaly findings as a distinct "anomalies"
        # category without altering any existing issue. The method used and
        # the flagged count are included in each issue.
        self._append_anomaly_issues(dataframe, issues)

        # 2c. Near-duplicate detection (Requirement 10.4)
        # Additively report near-duplicates as a distinct "near_duplicates"
        # category, separate from the exact-duplicate "duplicates" finding
        # emitted above. Only reported when there are near-duplicates beyond
        # the exact row duplicates already accounted for.
        self._append_near_duplicate_issues(dataframe, issues, dup_rows)

        # 3. Composite Health Score Calculation
        completeness_score = max(
            0.0, 100.0 * (1.0 - (missing_cells / max(1, total_cells)))
        )
        uniqueness_score = max(0.0, 100.0 * (1.0 - (dup_rows / max(1, rows))))
        integrity_score = max(
            0.0, 100.0 - (100.0 * (len(constant_cols) + len(empty_cols)) / max(1, cols))
        )
        consistency_score = max(
            0.0,
            100.0 - (15.0 * len(unparsed_dtype_cols) + 10.0 * len(casing_issue_cols)),
        )
        validity_score = max(0.0, 100.0 - (8.0 * len(outlier_cols)))

        composite_score = int(
            round(
                0.35 * completeness_score
                + 0.25 * uniqueness_score
                + 0.15 * integrity_score
                + 0.15 * consistency_score
                + 0.10 * validity_score
            )
        )
        health_score = max(0, min(100, composite_score))

        return DatasetHealthReport(
            health_score=health_score,
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            integrity_score=integrity_score,
            rows=rows,
            columns=cols,
            memory_mb=memory_mb,
            issues=issues,
            recommendations=recommendations,
        )

    def _append_anomaly_issues(
        self, dataframe: pd.DataFrame, issues: list[DatasetIssue]
    ) -> None:
        """
        Additively append anomaly findings as ``"anomalies"`` DatasetIssues.

        Uses the deterministic, Core_Stack-only default method. Each emitted
        issue records the anomaly method used and the flagged count. If the
        configured method required an optional dependency, a
        :class:`DependencyError` naming the extra would be raised; the current
        Core_Stack-only methods never trigger this.
        """
        extra = _ANOMALY_METHOD_EXTRAS.get(_ANOMALY_METHOD)
        if extra is not None:
            raise DependencyError(
                f"Anomaly method '{_ANOMALY_METHOD}' requires the optional "
                f"dependency provided by '{extra}'. Install it with "
                f"'pip install {extra}'."
            )

        result = AnomalyInspector().inspect(dataframe, method=_ANOMALY_METHOD)

        for report in result.reports:
            if report.anomaly_count <= 0:
                continue

            severity = "WARNING" if report.anomaly_percentage >= 10.0 else "INFO"
            issues.append(
                DatasetIssue(
                    title=f"Anomalies Detected in '{report.column}'",
                    description=(
                        f"{report.anomaly_count:,} "
                        f"({report.anomaly_percentage:.1f}%) anomalous value(s) "
                        f"flagged in column '{report.column}' using the "
                        f"'{report.method}' method."
                    ),
                    severity=severity,
                    category="anomalies",
                    column=report.column,
                    evidence=(
                        f"method={report.method}; "
                        f"flagged={report.anomaly_count}/{report.analyzed_count}"
                    ),
                    recommendation_text=(
                        f"Investigate the {report.anomaly_count} anomalous "
                        f"value(s) in '{report.column}'."
                    ),
                )
            )

    def _append_near_duplicate_issues(
        self,
        dataframe: pd.DataFrame,
        issues: list[DatasetIssue],
        exact_duplicate_rows: int,
    ) -> None:
        """
        Additively append near-duplicate findings as a distinct DatasetIssue.

        Runs :class:`NearDuplicateDetector` in the deterministic,
        Core_Stack-only ``"exact_normalized"`` mode (the similarity mode, which
        needs the optional ``sanitizepy[fuzzy]`` extra, is never used here).
        Rows that match only after normalization (case, whitespace) but are not
        byte-for-byte identical are the "near" duplicates.

        The finding uses the ``"near_duplicates"`` category, which is distinct
        from the exact-duplicate ``"duplicates"`` category emitted elsewhere in
        :meth:`inspect`. To keep the two findings non-overlapping, this issue is
        only emitted when the normalized-duplicate count exceeds the count of
        exact row duplicates already reported (i.e. there are genuine near
        duplicates beyond the exact ones). Existing findings are never altered.
        """
        result = NearDuplicateDetector().detect(dataframe, method="exact_normalized")

        # Redundant records found only through normalization, over and above
        # the exact row duplicates already surfaced as the "duplicates" issue.
        near_only_count = result.duplicate_count - exact_duplicate_rows

        if near_only_count <= 0:
            return

        group_count = len(result.groups)
        rows = dataframe.shape[0]
        near_pct = (near_only_count / rows) * 100.0 if rows else 0.0
        severity = "WARNING" if near_pct > 5.0 else "INFO"

        issues.append(
            DatasetIssue(
                title="Near-Duplicate Rows Detected",
                description=(
                    f"Dataset contains {near_only_count:,} near-duplicate "
                    f"record(s) across {group_count:,} group(s) that match "
                    "only after normalization (case and whitespace) beyond the "
                    "exact duplicate rows already reported."
                ),
                severity=severity,
                category="near_duplicates",
                evidence=(
                    f"method={result.method}; "
                    f"normalized_duplicate_count={result.duplicate_count}; "
                    f"exact_duplicate_rows={exact_duplicate_rows}; "
                    f"near_duplicate_records={near_only_count}; "
                    f"groups={group_count}"
                ),
                recommendation_text=(
                    "Review and remove near-duplicate records that differ only "
                    "by casing or whitespace."
                ),
            )
        )
