"""
Intelligent Issue Detector and Dataset Quality Inspector.

Inspects a pandas DataFrame across multiple dimensions to generate structured
DatasetIssues, composite quality scores, and actionable Recommendations.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

from cleaner.inspection.health import DatasetHealthReport, DatasetIssue
from cleaner.models.base import ColumnReference
from cleaner.models.recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationCategory,
    RecommendationImpact,
    RecommendationPriority,
    RecommendationReason,
)


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
                        recommendation_text="Provide a non-empty DataFrame for analysis.",
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
                    description=f"Dataset contains {dup_rows:,} ({dup_pct:.1f}%) duplicate rows.",
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
                    priority=RecommendationPriority.CRITICAL if dup_pct > 10 else RecommendationPriority.HIGH,
                    action=RecommendationAction.REMOVE_DUPLICATES,
                    reason=RecommendationReason(
                        title="Duplicate Row Check",
                        explanation=f"{dup_rows} rows ({dup_pct:.1f}%) share identical values across all columns.",
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
                        description=f"Remove column '{col}' because it contains only null values.",
                        category=RecommendationCategory.MISSING_VALUES,
                        priority=RecommendationPriority.CRITICAL,
                        action=RecommendationAction.DROP_COLUMN,
                        column=ColumnReference(name=col, dtype=str(series.dtype)),
                        reason=RecommendationReason(
                            title="100% Missingness",
                            explanation=f"Column '{col}' provides zero informational content.",
                        ),
                        impact=RecommendationImpact(column_change=-1, quality_score_delta=5.0),
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
                        description=f"{n_missing:,} missing cells detected in column '{col}'.",
                        severity=severity,
                        category="missing_values",
                        column=col,
                        evidence=f"{n_missing}/{rows} values are missing",
                        recommendation_text=f"Fill missing values in '{col}' using {strategy}.",
                    )
                )
                recommendations.append(
                    Recommendation(
                        title=f"Impute Missing Values in '{col}'",
                        description=f"Fill {n_missing} missing values in '{col}' using {strategy}.",
                        category=RecommendationCategory.MISSING_VALUES,
                        priority=RecommendationPriority.HIGH if missing_pct >= 15 else RecommendationPriority.MEDIUM,
                        action=rec_action,
                        column=ColumnReference(name=col, dtype=str(series.dtype)),
                        reason=RecommendationReason(
                            title=f"Missing Value Imputation ({strategy})",
                            explanation=f"Column '{col}' has {missing_pct:.1f}% missingness.",
                        ),
                        impact=RecommendationImpact(quality_score_delta=min(10.0, missing_pct * 0.3)),
                        confidence=0.85,
                        automatic=False if missing_pct > 50 else True,
                    )
                )

            # Constant / Zero Variance checks
            if n_unique == 1 and missing_pct < 100.0:
                constant_cols.append(col)
                issues.append(
                    DatasetIssue(
                        title=f"Constant Column '{col}'",
                        description=f"Column '{col}' contains only 1 unique non-null value.",
                        severity="WARNING",
                        category="integrity",
                        column=col,
                        evidence="Single unique value across all non-null entries",
                        recommendation_text=f"Consider removing constant column '{col}'.",
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
                            explanation=f"Column '{col}' holds the same constant value across all rows.",
                        ),
                        impact=RecommendationImpact(column_change=-1, quality_score_delta=2.0),
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
                                description=f"Column '{col}' has casing or trailing whitespace variants.",
                                severity="WARNING",
                                category="consistency",
                                column=col,
                                evidence=f"{non_null.nunique()} raw unique values -> {cleaned_str.nunique()} normalized unique values",
                                recommendation_text=f"Normalize string casing and trim whitespace in '{col}'.",
                            )
                        )

                    # Unparsed datetime string detection
                    sample_vals = non_null.iloc[: min(100, len(non_null))]
                    date_match_count = 0
                    for val in sample_vals:
                        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", val):
                            date_match_count += 1
                    if date_match_count / len(sample_vals) > 0.7:
                        unparsed_dtype_cols.append(col)
                        issues.append(
                            DatasetIssue(
                                title=f"Unparsed Datetime String in '{col}'",
                                description=f"Column '{col}' is stored as string/object but resembles datetime values.",
                                severity="WARNING",
                                category="data_types",
                                column=col,
                                evidence=f">{int(date_match_count/len(sample_vals)*100)}% of sample values match date formats",
                                recommendation_text=f"Convert '{col}' to datetime dtype.",
                            )
                        )
                        recommendations.append(
                            Recommendation(
                                title=f"Convert '{col}' to Datetime",
                                description=f"Parse string column '{col}' into datetime format.",
                                category=RecommendationCategory.DATA_TYPES,
                                priority=RecommendationPriority.HIGH,
                                action=RecommendationAction.CONVERT_TYPE,
                                column=ColumnReference(name=col, dtype=str(series.dtype)),
                                reason=RecommendationReason(
                                    title="Datetime Pattern Match",
                                    explanation=f"Column '{col}' strings follow standard date patterns.",
                                ),
                                impact=RecommendationImpact(quality_score_delta=5.0),
                                confidence=0.9,
                                automatic=True,
                            )
                        )

            # Outlier checks for numeric columns
            if is_numeric_dtype(series) and n_unique > 5:
                s_clean = series.dropna()
                if len(s_clean) > 10:
                    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = ((s_clean < (q1 - 1.5 * iqr)) | (s_clean > (q3 + 1.5 * iqr))).sum()
                        outlier_pct = (outliers / len(s_clean)) * 100.0
                        if outlier_pct > 3.0:
                            outlier_cols.append(col)
                            issues.append(
                                DatasetIssue(
                                    title=f"Outliers Detected in '{col}'",
                                    description=f"{outliers:,} ({outlier_pct:.1f}%) statistical outliers in '{col}'.",
                                    severity="INFO" if outlier_pct < 10 else "WARNING",
                                    category="outliers",
                                    column=col,
                                    evidence=f"{outliers} points outside 1.5*IQR bounds",
                                    recommendation_text=f"Investigate numerical distribution and extreme values in '{col}'.",
                                )
                            )

        # 3. Composite Health Score Calculation
        completeness_score = max(0.0, 100.0 * (1.0 - (missing_cells / max(1, total_cells))))
        uniqueness_score = max(0.0, 100.0 * (1.0 - (dup_rows / max(1, rows))))
        integrity_score = max(0.0, 100.0 - (100.0 * (len(constant_cols) + len(empty_cols)) / max(1, cols)))
        consistency_score = max(0.0, 100.0 - (15.0 * len(unparsed_dtype_cols) + 10.0 * len(casing_issue_cols)))
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
