"""
Tests for sanitizepy.inspection.detector.IssueDetector

Task 6.3 – extend/create tests for the additive encoding-artifact detector
Requirements: 5.1

Covers:
  - Encoding-artifact detection produces DatasetIssue(category="encoding_artifacts")
  - Mojibake patterns detected per column
  - Replacement character (U+FFFD) detected per column
  - Control characters detected per column
  - Issues are additive (existing categories unaffected)
  - Correct severity thresholds (>10% → CRITICAL, else WARNING)
  - Evidence string reflects artifact type counts
  - Non-object columns are not flagged for encoding artifacts
  - Already-clean columns produce no encoding_artifacts issue
  - Multiple artifact types in the same column produce one combined issue
  - Edge cases: empty, single-row, all-null, mixed-type, wide, tall
  - Internal helper _count_encoding_artifact_values tested directly
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanitizepy.inspection.detector import (
    IssueDetector,
    _count_encoding_artifact_values,
)
from sanitizepy.inspection.health import DatasetIssue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Classic UTF-8-as-Latin-1 mojibake
MOJIBAKE_STRING: str = "CafÃ©"

# Unicode replacement character
REPLACEMENT_CHAR_STRING: str = "hello\ufffdworld"

# Pure replacement character (unrecoverable)
PURE_REPLACEMENT_STRING: str = "\ufffd\ufffd\ufffd"

# ASCII control character
CONTROL_CHAR_STRING: str = "hello\x01world"

# Clean string with no artifacts
CLEAN_STRING: str = "This is normal text."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


def _get_encoding_issues(report_issues: list[DatasetIssue]) -> list[DatasetIssue]:
    """Filter issues to only encoding_artifacts category."""
    return [i for i in report_issues if i.category == "encoding_artifacts"]


# ---------------------------------------------------------------------------
# 1. Tests for _count_encoding_artifact_values helper
# ---------------------------------------------------------------------------


class TestCountEncodingArtifactValues:
    """Direct unit tests for the internal counting helper."""

    def test_mojibake_counted(self) -> None:
        series = pd.Series([MOJIBAKE_STRING, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["mojibake"] == 1
        assert counts["replacement_char"] == 0
        assert counts["control_chars"] == 0

    def test_replacement_char_counted(self) -> None:
        series = pd.Series([REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["replacement_char"] == 1
        assert counts["mojibake"] == 0
        assert counts["control_chars"] == 0

    def test_control_chars_counted(self) -> None:
        series = pd.Series([CONTROL_CHAR_STRING, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["control_chars"] == 1
        assert counts["mojibake"] == 0
        assert counts["replacement_char"] == 0

    def test_multiple_artifact_types_counted_independently(self) -> None:
        series = pd.Series(
            [
                MOJIBAKE_STRING,
                REPLACEMENT_CHAR_STRING,
                CONTROL_CHAR_STRING,
            ]
        )
        counts = _count_encoding_artifact_values(series)
        assert counts["mojibake"] >= 1
        assert counts["replacement_char"] >= 1
        assert counts["control_chars"] >= 1

    def test_non_string_values_skipped(self) -> None:
        series = pd.Series([1, 2.0, None, True, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["mojibake"] == 0
        assert counts["replacement_char"] == 0
        assert counts["control_chars"] == 0

    def test_nan_values_skipped(self) -> None:
        series = pd.Series([np.nan, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert sum(counts.values()) == 0

    def test_empty_series_all_zero(self) -> None:
        series = pd.Series([], dtype="object")
        counts = _count_encoding_artifact_values(series)
        assert counts["mojibake"] == 0
        assert counts["replacement_char"] == 0
        assert counts["control_chars"] == 0

    def test_all_clean_strings_all_zero(self) -> None:
        series = pd.Series([CLEAN_STRING, "Another clean value", "hello world"])
        counts = _count_encoding_artifact_values(series)
        assert sum(counts.values()) == 0

    def test_pure_replacement_char_counted(self) -> None:
        series = pd.Series([PURE_REPLACEMENT_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["replacement_char"] >= 1

    def test_multiple_rows_with_same_artifact_counted(self) -> None:
        series = pd.Series([CONTROL_CHAR_STRING, CONTROL_CHAR_STRING, CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert counts["control_chars"] == 2

    def test_keys_always_present(self) -> None:
        """The returned dict must always have all three keys."""
        series = pd.Series([CLEAN_STRING])
        counts = _count_encoding_artifact_values(series)
        assert "mojibake" in counts
        assert "replacement_char" in counts
        assert "control_chars" in counts


# ---------------------------------------------------------------------------
# 2. IssueDetector produces encoding_artifacts issues additively (Req 5.1)
# ---------------------------------------------------------------------------


class TestEncodingArtifactDetection:
    """IssueDetector additively emits encoding_artifacts DatasetIssues."""

    def test_mojibake_produces_encoding_artifacts_issue(self) -> None:
        df = _make_df(text=[MOJIBAKE_STRING, CLEAN_STRING, CLEAN_STRING])
        detector = IssueDetector()
        report = detector.inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1

    def test_replacement_char_produces_encoding_artifacts_issue(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        detector = IssueDetector()
        report = detector.inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1

    def test_control_chars_produce_encoding_artifacts_issue(self) -> None:
        df = _make_df(text=[CONTROL_CHAR_STRING, CLEAN_STRING, CLEAN_STRING])
        detector = IssueDetector()
        report = detector.inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1

    def test_issue_category_is_encoding_artifacts(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        for issue in enc_issues:
            assert issue.category == "encoding_artifacts"

    def test_issue_references_correct_column(self) -> None:
        df = _make_df(
            clean_col=[CLEAN_STRING, CLEAN_STRING],
            dirty_col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING],
        )
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        affected_columns = {i.column for i in enc_issues if i.column is not None}
        assert "dirty_col" in affected_columns
        assert "clean_col" not in affected_columns

    def test_clean_column_no_encoding_artifacts_issue(self) -> None:
        df = _make_df(text=[CLEAN_STRING, "Another clean value", "More text"])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0

    def test_encoding_artifacts_issues_additive(self) -> None:
        """Existing issue categories must not be removed or replaced."""
        # Create a DataFrame with both duplicates AND encoding artifacts
        df = pd.DataFrame(
            {
                "text": [REPLACEMENT_CHAR_STRING, CLEAN_STRING, CLEAN_STRING],
                "dup": [1, 1, 1],  # will trigger duplicate issue
            }
        )
        # Make a duplicate row to trigger the duplicate issue
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = IssueDetector().inspect(df)
        categories = {i.category for i in report.issues}
        # Both encoding artifacts and duplicates should be present
        assert "encoding_artifacts" in categories
        assert "duplicates" in categories

    def test_encoding_artifacts_issue_per_column(self) -> None:
        """Each column with artifacts gets its own issue."""
        df = _make_df(
            a=[REPLACEMENT_CHAR_STRING, CLEAN_STRING],
            b=[CONTROL_CHAR_STRING, CLEAN_STRING],
        )
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        affected_columns = {i.column for i in enc_issues if i.column is not None}
        assert "a" in affected_columns
        assert "b" in affected_columns


# ---------------------------------------------------------------------------
# 3. Severity thresholds
# ---------------------------------------------------------------------------


class TestSeverityThresholds:
    """Issues with >10% affected cells → CRITICAL; otherwise → WARNING."""

    def test_low_artifact_rate_severity_warning(self) -> None:
        """Less than 10% artifact cells → WARNING."""
        # 1 artifact out of 20 rows = 5%
        col_values = [REPLACEMENT_CHAR_STRING] + [CLEAN_STRING] * 19
        df = _make_df(text=col_values)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert enc_issues[0].severity == "WARNING"

    def test_high_artifact_rate_severity_critical(self) -> None:
        """More than 10% artifact cells → CRITICAL."""
        # 3 artifacts out of 10 rows = 30%
        col_values = [REPLACEMENT_CHAR_STRING] * 3 + [CLEAN_STRING] * 7
        df = _make_df(text=col_values)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert enc_issues[0].severity == "CRITICAL"

    def test_boundary_10_percent_warning(self) -> None:
        """Exactly 10% artifacts → WARNING (boundary is strictly > 10% for CRITICAL)."""
        # 1 artifact out of 10 rows = exactly 10%
        col_values = [REPLACEMENT_CHAR_STRING] + [CLEAN_STRING] * 9
        df = _make_df(text=col_values)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert enc_issues[0].severity == "WARNING"


# ---------------------------------------------------------------------------
# 4. Issue content (title, description, evidence)
# ---------------------------------------------------------------------------


class TestIssueContent:
    def test_issue_title_mentions_column(self) -> None:
        df = _make_df(my_column=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert "my_column" in enc_issues[0].title

    def test_issue_description_mentions_artifacts(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        desc = enc_issues[0].description.lower()
        assert "encoding" in desc or "artifact" in desc or "replacement" in desc

    def test_issue_evidence_reflects_artifact_types(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        evidence = enc_issues[0].evidence.lower()
        # Evidence should mention replacement character
        assert (
            "replacement" in evidence or "fffd" in evidence or "character" in evidence
        )

    def test_issue_recommendation_text_present(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert len(enc_issues[0].recommendation_text) > 0


# ---------------------------------------------------------------------------
# 5. Non-object columns not flagged
# ---------------------------------------------------------------------------


class TestNonObjectColumnsNotFlagged:
    def test_integer_column_no_encoding_issue(self) -> None:
        df = _make_df(num=[1, 2, 3, 4, 5])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0

    def test_float_column_no_encoding_issue(self) -> None:
        df = _make_df(vals=[1.0, 2.0, 3.0])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0

    def test_datetime_column_no_encoding_issue(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-06-15"])
        df = pd.DataFrame({"d": dates})
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0


# ---------------------------------------------------------------------------
# 6. Multiple artifact types in same column → single combined issue
# ---------------------------------------------------------------------------


class TestMultipleArtifactTypesSameColumn:
    def test_mojibake_and_control_chars_same_column_one_issue(self) -> None:
        """A column with both mojibake and control chars yields exactly one
        encoding_artifacts issue for that column."""
        df = _make_df(
            text=[
                MOJIBAKE_STRING,
                CONTROL_CHAR_STRING,
                CLEAN_STRING,
                CLEAN_STRING,
            ]
        )
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        # Filter for column "text"
        text_enc_issues = [i for i in enc_issues if i.column == "text"]
        # Should be exactly 1 issue for the column
        assert len(text_enc_issues) == 1


# ---------------------------------------------------------------------------
# 7. Edge case: Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_empty_rows_no_encoding_issue(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0

    def test_empty_dataframe_returns_report(self) -> None:
        df = pd.DataFrame()
        report = IssueDetector().inspect(df)
        # Must return a report (not crash)
        assert report is not None


# ---------------------------------------------------------------------------
# 8. Edge case: Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_with_artifact_detected(self) -> None:
        df = _make_df(text=[REPLACEMENT_CHAR_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1

    def test_single_row_clean_no_encoding_issue(self) -> None:
        df = _make_df(text=[CLEAN_STRING])
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0


# ---------------------------------------------------------------------------
# 9. Edge case: All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_column_no_encoding_issue(self) -> None:
        """NaN values are not strings and should not trigger encoding issues."""
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0


# ---------------------------------------------------------------------------
# 10. Edge case: Mixed-type column (object dtype)
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_mixed_type_column_integer_objects_not_flagged(self) -> None:
        """Non-string items in an object column are skipped."""
        df = pd.DataFrame(
            {"col": pd.array([1, 2.0, None, CLEAN_STRING], dtype="object")}
        )
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0

    def test_mixed_type_column_with_artifact_flagged(self) -> None:
        """String artifact in an otherwise mixed column is still detected."""
        df = pd.DataFrame(
            {"col": pd.array([1, REPLACEMENT_CHAR_STRING, 3.0], dtype="object")}
        )
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1


# ---------------------------------------------------------------------------
# 11. Edge case: Wide DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_only_artifact_columns_flagged(self) -> None:
        n_clean = 15
        data: dict[str, list[str]] = {
            f"clean{i}": [CLEAN_STRING, CLEAN_STRING] for i in range(n_clean)
        }
        data["dirty"] = [REPLACEMENT_CHAR_STRING, CLEAN_STRING]
        df = pd.DataFrame(data)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        # Only one column has artifacts
        assert len(enc_issues) == 1
        assert enc_issues[0].column == "dirty"

    def test_wide_multiple_artifact_columns_all_flagged(self) -> None:
        data: dict[str, list[str]] = {
            f"dirty{i}": [REPLACEMENT_CHAR_STRING, CLEAN_STRING] for i in range(5)
        }
        df = pd.DataFrame(data)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 5


# ---------------------------------------------------------------------------
# 12. Edge case: Tall DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_low_artifact_rate_warning(self) -> None:
        """3% artifact rate on tall DataFrame → WARNING."""
        # 3% → 30 artifacts out of 1000 rows
        col_values = [REPLACEMENT_CHAR_STRING] * 30 + [CLEAN_STRING] * 970
        df = _make_df(text=col_values)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert enc_issues[0].severity == "WARNING"

    def test_tall_high_artifact_rate_critical(self) -> None:
        """50% artifact rate on tall DataFrame → CRITICAL."""
        n_rows = 500
        half = n_rows // 2
        col_values = [REPLACEMENT_CHAR_STRING] * half + [CLEAN_STRING] * half
        df = _make_df(text=col_values)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) >= 1
        assert enc_issues[0].severity == "CRITICAL"

    def test_tall_all_clean_no_encoding_issue(self) -> None:
        n_rows = 2_000
        df = _make_df(text=[CLEAN_STRING] * n_rows)
        report = IssueDetector().inspect(df)
        enc_issues = _get_encoding_issues(report.issues)
        assert len(enc_issues) == 0
