"""
Edge-case tests for data quality inspection and cleaning operations.

Covers: empty DataFrames, single row/column, all-null columns,
mixed types, NaN/None/inf, duplicate column names, high cardinality,
wide/tall DataFrames, and the cleaning engine dry-run / audit trail
contract under adversarial inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sanitizepy import (
    AnomalyInspector,
    Cleaner,
    DatasetProfiler,
    EncodingRepairOperation,
    MissingTokenOperation,
    NearDuplicateDetector,
    NearDuplicateRemovalOperation,
    TextNormalizationOperation,
    TextQualityAnalyzer,
    TypeCoercionOperation,
)
from sanitizepy.cleaning import (
    CleaningEngine,
    CleaningResult,
    DropColumns,
    DropDuplicates,
    DropMissingRows,
    FillMissing,
    OperationResult,
)
from sanitizepy.exceptions import DataTypeConversionError
from sanitizepy.inspection.detector import IssueDetector
from sanitizepy.inspection.health import DatasetHealthReport

# ---------------------------------------------------------------------------
# IssueDetector edge cases
# ---------------------------------------------------------------------------


class TestIssueDetectorEdgeCases:

    def test_empty_dataframe(self) -> None:
        report = IssueDetector().inspect(pd.DataFrame())
        assert report.health_score == 0
        assert report.rows == 0
        assert report.columns == 0
        assert len(report.critical_issues) >= 1

    def test_single_row_single_column(self) -> None:
        df = pd.DataFrame({"a": [42]})
        report = IssueDetector().inspect(df)
        assert report.rows == 1
        assert report.columns == 1
        assert report.health_score > 0

    def test_all_null_column(self) -> None:
        df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
        report = IssueDetector().inspect(df)
        critical_cols = [i.column for i in report.critical_issues]
        assert "x" in critical_cols

    def test_all_null_dataframe(self) -> None:
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        report = IssueDetector().inspect(df)
        assert report.completeness_score == 0.0
        assert report.health_score < 50

    def test_no_issues_clean_data(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10.1, 20.2, 30.3, 40.4, 50.5],
            }
        )
        report = IssueDetector().inspect(df)
        assert report.health_score >= 80
        assert len(report.critical_issues) == 0

    def test_mixed_types_column(self) -> None:
        df = pd.DataFrame({"mixed": [1, "two", 3.0, None, True]})
        report = IssueDetector().inspect(df)
        assert report.rows == 5
        assert report.columns == 1

    def test_inf_values_in_numeric(self) -> None:
        df = pd.DataFrame({"vals": [1.0, 2.0, np.inf, -np.inf, 5.0]})
        report = IssueDetector().inspect(df)
        assert report.rows == 5

    def test_duplicate_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [10, 10, 10, 20, 20]})
        report = IssueDetector().inspect(df)
        dup_issues = [i for i in report.issues if "Duplicate" in i.title]
        assert len(dup_issues) >= 1

    def test_constant_column_detected(self) -> None:
        df = pd.DataFrame({"const": [99, 99, 99], "var": [1, 2, 3]})
        report = IssueDetector().inspect(df)
        const_issues = [i for i in report.issues if "Constant" in i.title]
        assert len(const_issues) >= 1

    def test_casing_inconsistency_detection(self) -> None:
        df = pd.DataFrame({"country": ["USA", "usa", "Usa", "UK", "uk"]})
        report = IssueDetector().inspect(df)
        casing_issues = [i for i in report.issues if "Casing" in i.title]
        assert len(casing_issues) >= 1

    def test_datetime_string_detection(self) -> None:
        df = pd.DataFrame(
            {
                "date": [
                    "2023-01-01",
                    "2023-02-15",
                    "2023-03-20",
                    "2023-04-10",
                    "2023-05-05",
                    "2023-06-30",
                ]
            }
        )
        report = IssueDetector().inspect(df)
        dt_issues = [i for i in report.issues if "Datetime" in i.title]
        assert len(dt_issues) >= 1

    def test_wide_dataframe(self) -> None:
        data = {f"col_{i}": [i] for i in range(200)}
        df = pd.DataFrame(data)
        report = IssueDetector().inspect(df)
        assert report.columns == 200
        assert report.health_score > 0

    def test_tall_dataframe(self) -> None:
        df = pd.DataFrame({"x": list(range(10_000))})
        report = IssueDetector().inspect(df)
        assert report.rows == 10_000

    def test_type_error_on_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            IssueDetector().inspect("not a dataframe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CleaningEngine edge cases
# ---------------------------------------------------------------------------


class TestCleaningEngineEdgeCases:

    def test_run_empty_operations(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        engine = CleaningEngine()
        result = engine.run_with_result(df)
        assert result.data.shape == df.shape
        assert len(result.operations) == 0
        assert len(result.audit_log) == 0

    def test_run_with_result_type(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        engine = CleaningEngine([DropDuplicates()])
        result = engine.run_with_result(df)
        assert isinstance(result, CleaningResult)
        assert isinstance(result.operations[0], OperationResult)

    def test_dry_run_returns_original_shape(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, 5, 6]})
        engine = CleaningEngine(
            [
                DropDuplicates(),
                DropMissingRows(),
            ]
        )
        result = engine.run_with_result(df, dry_run=True)
        assert result.dry_run is True
        assert result.data.shape == df.shape

    def test_real_run_changes_shape(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, 5, 6]})
        engine = CleaningEngine(
            [
                DropDuplicates(),
                DropMissingRows(),
            ]
        )
        result = engine.run_with_result(df, dry_run=False)
        assert result.dry_run is False
        assert result.data.shape[0] < df.shape[0]

    def test_audit_log_populated(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2]})
        engine = CleaningEngine([DropDuplicates()])
        result = engine.run_with_result(df)
        assert len(result.audit_log) == 1
        log_entry = result.audit_log[0]
        assert "timestamp" in log_entry
        assert "operation" in log_entry
        assert log_entry["operation"] == "drop_duplicates"

    def test_summary_string(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2]})
        engine = CleaningEngine([DropDuplicates()])
        result = engine.run_with_result(df)
        summary = result.summary()
        assert "drop_duplicates" in summary
        assert "1 operation" in summary

    def test_fill_missing_on_empty_column(self) -> None:
        df = pd.DataFrame({"x": [None, None, None]})
        engine = CleaningEngine([FillMissing(value=0.0, subset=["x"])])
        result = engine.run_with_result(df)
        assert result.data["x"].isna().sum() == 0

    def test_drop_columns_removes_column(self) -> None:
        df = pd.DataFrame({"keep": [1, 2], "drop_me": [3, 4]})
        engine = CleaningEngine([DropColumns(columns=["drop_me"])])
        result = engine.run_with_result(df)
        assert "drop_me" not in result.data.columns
        assert "keep" in result.data.columns

    def test_type_error_on_non_dataframe(self) -> None:
        engine = CleaningEngine()
        with pytest.raises(TypeError):
            engine.run_with_result([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# High-level Cleaner edge cases
# ---------------------------------------------------------------------------


class TestCleanerEdgeCases:

    def test_inspect_returns_health_report(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = Cleaner().inspect(df)
        assert isinstance(report, DatasetHealthReport)

    def test_plan_from_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "empty": [None, None, None]})
        plan = Cleaner().plan(df)
        assert len(plan.steps) > 0

    def test_clean_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "empty": [None, None, None]})
        result = Cleaner().clean(df, dry_run=True)
        assert result.dry_run is True
        assert result.data.shape == df.shape

    def test_clean_real_removes_empty_col(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
        result = Cleaner().clean(df, dry_run=False)
        assert "empty" not in result.data.columns


# ===========================================================================
# Edge-case regression coverage for production-grade-evolution capabilities
# ===========================================================================
#
# Task 22.2: exercise empty / single-row / all-null / mixed-type / infinite /
# wide / tall inputs across the new operations and inspectors, asserting the
# real contracts each capability established rather than incidental behaviour:
#
# * Operations return a DataFrame (possibly empty) for empty input, never
#   mutate the caller's frame, and record impact in an OperationResult.
# * TypeCoercionOperation raises KeyError when a target column is absent.
# * The new inspectors/analyzers raise ValueError on an empty DataFrame.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared edge-case fixtures
# ---------------------------------------------------------------------------


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"text": ["hello"], "num": [1.0]})


def _all_null_df() -> pd.DataFrame:
    return pd.DataFrame({"text": [None, None, None], "num": [np.nan, np.nan, np.nan]})


def _mixed_type_df() -> pd.DataFrame:
    return pd.DataFrame({"mixed": [1, "two", 3.0, None, True]})


def _infinite_df() -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, np.inf, -np.inf, 5.0]})


def _wide_df(n_cols: int = 200) -> pd.DataFrame:
    return pd.DataFrame({f"col_{i}": ["a", "b", "c"] for i in range(n_cols)})


def _tall_df(n_rows: int = 10_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"value {i % 7}" for i in range(n_rows)],
            "num": list(range(n_rows)),
        }
    )


# ---------------------------------------------------------------------------
# MissingTokenOperation edge cases
# ---------------------------------------------------------------------------


class TestMissingTokenOperationEdgeCases:

    def test_empty_dataframe_returns_empty(self) -> None:
        df = _empty_df()
        out, result = MissingTokenOperation().apply_with_result(df)
        assert out.empty
        assert out.shape == df.shape
        assert result.operation_name == "missing_token_normalization"

    def test_single_row(self) -> None:
        df = pd.DataFrame({"c": ["N/A"]})
        out, result = MissingTokenOperation().apply_with_result(df)
        assert out["c"].isna().all()
        assert result.rows_affected == 1

    def test_all_null_column_unchanged(self) -> None:
        df = _all_null_df()
        out, result = MissingTokenOperation().apply_with_result(df)
        assert out["text"].isna().all()
        assert result.rows_affected == 0

    def test_mixed_type_column(self) -> None:
        df = _mixed_type_df()
        out, _ = MissingTokenOperation().apply_with_result(df)
        # No token strings present; values preserved (non-str untouched).
        assert out.shape == df.shape

    def test_infinite_values_numeric_untouched(self) -> None:
        df = _infinite_df()
        out, result = MissingTokenOperation().apply_with_result(df)
        assert np.isinf(out["num"]).sum() == 2
        assert result.rows_affected == 0

    def test_wide_dataframe(self) -> None:
        df = _wide_df()
        out, _ = MissingTokenOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_tall_dataframe(self) -> None:
        df = _tall_df()
        out, result = MissingTokenOperation().apply_with_result(df)
        assert out.shape == df.shape
        assert result.rows_affected == 0

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"c": ["null", "keep"]})
        original = df.copy()
        MissingTokenOperation().apply_with_result(df)
        pd.testing.assert_frame_equal(df, original)

    def test_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"c": ["null", "keep"]})
        out, result = MissingTokenOperation().apply_with_result(df, dry_run=True)
        assert result.dry_run is True
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# TypeCoercionOperation edge cases
# ---------------------------------------------------------------------------


class TestTypeCoercionOperationEdgeCases:

    def test_empty_dataframe_missing_column_raises_key_error(self) -> None:
        # An empty frame has no columns, so a named target is absent.
        df = _empty_df()
        op = TypeCoercionOperation({"num": "int64"})
        with pytest.raises(KeyError):
            op.apply_with_result(df)

    def test_empty_frame_with_existing_column(self) -> None:
        df = pd.DataFrame({"num": pd.Series([], dtype="object")})
        op = TypeCoercionOperation({"num": "float64"})
        out, result = op.apply_with_result(df)
        assert out.empty
        assert result.rows_affected == 0

    def test_single_row(self) -> None:
        df = pd.DataFrame({"num": ["42"]})
        out, _ = TypeCoercionOperation({"num": "int64"}).apply_with_result(df)
        assert out["num"].iloc[0] == 42

    def test_all_null_column_coerce(self) -> None:
        df = pd.DataFrame({"num": [None, None, None]})
        out, result = TypeCoercionOperation(
            {"num": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert out["num"].isna().all()
        assert result.rows_affected == 0

    def test_mixed_type_coerce_records_affected(self) -> None:
        df = pd.DataFrame({"num": [1, "two", 3.0, None]})
        out, result = TypeCoercionOperation(
            {"num": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        # "two" is non-convertible -> becomes missing and is counted.
        assert result.rows_affected == 1
        assert out["num"].isna().sum() == 2

    def test_mixed_type_raise_policy(self) -> None:
        df = pd.DataFrame({"num": [1, "two", 3.0]})
        with pytest.raises(DataTypeConversionError):
            TypeCoercionOperation({"num": "int64"}).apply_with_result(df)

    def test_infinite_values_preserved(self) -> None:
        df = _infinite_df()
        out, _ = TypeCoercionOperation(
            {"num": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert np.isinf(out["num"]).sum() == 2

    def test_missing_column_raises_key_error(self) -> None:
        df = _single_row_df()
        with pytest.raises(KeyError):
            TypeCoercionOperation({"absent": "int64"}).apply_with_result(df)

    def test_wide_dataframe(self) -> None:
        df = pd.DataFrame({f"c{i}": ["1", "2", "3"] for i in range(50)})
        targets = {f"c{i}": "int64" for i in range(50)}
        out, result = TypeCoercionOperation(targets).apply_with_result(df)
        assert result.columns_affected == 50
        assert (out.dtypes == "int64").all()

    def test_tall_dataframe(self) -> None:
        df = pd.DataFrame({"num": [str(i) for i in range(10_000)]})
        out, _ = TypeCoercionOperation({"num": "int64"}).apply_with_result(df)
        assert out["num"].dtype == "int64"

    def test_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        out, result = TypeCoercionOperation({"num": "int64"}).apply_with_result(
            df, dry_run=True
        )
        assert result.dry_run is True
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# TextNormalizationOperation edge cases
# ---------------------------------------------------------------------------


class TestTextNormalizationOperationEdgeCases:

    def test_empty_dataframe_returns_empty(self) -> None:
        df = _empty_df()
        out, result = TextNormalizationOperation().apply_with_result(df)
        assert out.empty
        assert result.operation_name == "text_normalization"

    def test_single_row(self) -> None:
        df = pd.DataFrame({"c": ["  Hello  "]})
        out, result = TextNormalizationOperation(case="lower").apply_with_result(df)
        assert out["c"].iloc[0] == "hello"
        assert result.rows_affected == 1

    def test_all_null_column_unchanged(self) -> None:
        df = _all_null_df()
        out, result = TextNormalizationOperation().apply_with_result(df)
        assert out["text"].isna().all()
        assert result.rows_affected == 0

    def test_mixed_type_column(self) -> None:
        df = _mixed_type_df()
        out, _ = TextNormalizationOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_infinite_values_numeric_untouched(self) -> None:
        df = _infinite_df()
        out, result = TextNormalizationOperation().apply_with_result(df)
        assert np.isinf(out["num"]).sum() == 2
        assert result.rows_affected == 0

    def test_empty_after_normalization_becomes_missing(self) -> None:
        df = pd.DataFrame({"c": ["   ", "keep"]})
        out, _ = TextNormalizationOperation().apply_with_result(df)
        assert pd.isna(out["c"].iloc[0])
        assert out["c"].iloc[1] == "keep"

    def test_wide_dataframe(self) -> None:
        df = _wide_df()
        out, _ = TextNormalizationOperation(case="upper").apply_with_result(df)
        assert out.shape == df.shape
        assert out["col_0"].iloc[0] == "A"

    def test_tall_dataframe(self) -> None:
        df = _tall_df()
        out, _ = TextNormalizationOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"c": ["  x  ", "  y  "]})
        out, result = TextNormalizationOperation().apply_with_result(df, dry_run=True)
        assert result.dry_run is True
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# EncodingRepairOperation edge cases
# ---------------------------------------------------------------------------


class TestEncodingRepairOperationEdgeCases:

    def test_empty_dataframe_returns_empty(self) -> None:
        df = _empty_df()
        out, result = EncodingRepairOperation().apply_with_result(df)
        assert out.empty
        assert result.operation_name == "encoding_repair"

    def test_single_row_clean(self) -> None:
        df = pd.DataFrame({"c": ["clean text"]})
        out, result = EncodingRepairOperation().apply_with_result(df)
        assert out["c"].iloc[0] == "clean text"
        assert result.rows_affected == 0

    def test_all_null_column_unchanged(self) -> None:
        df = _all_null_df()
        out, result = EncodingRepairOperation().apply_with_result(df)
        assert out["text"].isna().all()
        assert result.rows_affected == 0

    def test_mixed_type_column(self) -> None:
        df = _mixed_type_df()
        out, _ = EncodingRepairOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_infinite_values_numeric_untouched(self) -> None:
        df = _infinite_df()
        out, result = EncodingRepairOperation().apply_with_result(df)
        assert np.isinf(out["num"]).sum() == 2
        assert result.rows_affected == 0

    def test_control_characters_repaired(self) -> None:
        df = pd.DataFrame({"c": ["ab\x00cd"]})
        out, result = EncodingRepairOperation().apply_with_result(df)
        assert out["c"].iloc[0] == "abcd"
        assert result.rows_affected == 1

    def test_wide_dataframe(self) -> None:
        df = _wide_df()
        out, _ = EncodingRepairOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_tall_dataframe(self) -> None:
        df = _tall_df()
        out, _ = EncodingRepairOperation().apply_with_result(df)
        assert out.shape == df.shape

    def test_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"c": ["ab\x00cd", "ok"]})
        out, result = EncodingRepairOperation().apply_with_result(df, dry_run=True)
        assert result.dry_run is True
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# NearDuplicateRemovalOperation edge cases
# ---------------------------------------------------------------------------


class TestNearDuplicateRemovalOperationEdgeCases:

    def test_empty_dataframe_returns_empty(self) -> None:
        df = _empty_df()
        out, result = NearDuplicateRemovalOperation().apply_with_result(df)
        assert out.empty
        assert result.rows_affected == 0

    def test_single_row(self) -> None:
        df = _single_row_df()
        out, result = NearDuplicateRemovalOperation().apply_with_result(df)
        assert out.shape == df.shape
        assert result.rows_affected == 0

    def test_all_null_rows_collapse(self) -> None:
        df = _all_null_df()
        out, result = NearDuplicateRemovalOperation().apply_with_result(df)
        # All-null rows normalize to identical keys -> one representative kept.
        assert out.shape[0] == 1
        assert result.rows_affected == 2

    def test_mixed_type_column(self) -> None:
        df = _mixed_type_df()
        out, _ = NearDuplicateRemovalOperation().apply_with_result(df)
        assert out.shape[0] <= df.shape[0]

    def test_infinite_values(self) -> None:
        df = _infinite_df()
        out, result = NearDuplicateRemovalOperation().apply_with_result(df)
        # All distinct numeric values -> nothing removed.
        assert out.shape == df.shape
        assert result.rows_affected == 0

    def test_keep_last_policy(self) -> None:
        df = pd.DataFrame({"c": ["a", "A", "b"]})
        out, result = NearDuplicateRemovalOperation(keep="last").apply_with_result(df)
        assert result.rows_affected == 1
        assert out.shape[0] == 2

    def test_wide_dataframe(self) -> None:
        df = _wide_df()
        out, result = NearDuplicateRemovalOperation().apply_with_result(df)
        # Each row differs across the wide columns -> nothing collapses.
        assert out.shape == df.shape
        assert result.rows_affected == 0

    def test_tall_dataframe(self) -> None:
        df = _tall_df()
        out, result = NearDuplicateRemovalOperation(subset=["text"]).apply_with_result(
            df
        )
        # Only 7 distinct normalized text values remain.
        assert out.shape[0] == 7
        assert result.rows_affected == 10_000 - 7

    def test_dry_run_preserves_original(self) -> None:
        df = pd.DataFrame({"c": ["a", "A", "b"]})
        out, result = NearDuplicateRemovalOperation().apply_with_result(
            df, dry_run=True
        )
        assert result.dry_run is True
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# DatasetProfiler edge cases
# ---------------------------------------------------------------------------


class TestDatasetProfilerEdgeCases:

    def test_empty_dataframe_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            DatasetProfiler().profile(_empty_df())

    def test_single_row(self) -> None:
        profile = DatasetProfiler().profile(_single_row_df())
        assert profile.row_count == 1
        assert profile.column_count == 2

    def test_all_null_dataframe(self) -> None:
        profile = DatasetProfiler().profile(_all_null_df())
        assert profile.row_count == 3
        assert profile.missing_values is not None

    def test_mixed_type_column(self) -> None:
        profile = DatasetProfiler().profile(_mixed_type_df())
        assert profile.row_count == 5
        assert profile.column_count == 1

    def test_infinite_values(self) -> None:
        profile = DatasetProfiler().profile(_infinite_df())
        assert profile.row_count == 5
        assert profile.statistics is not None

    def test_wide_dataframe(self) -> None:
        profile = DatasetProfiler().profile(_wide_df())
        assert profile.column_count == 200

    def test_tall_dataframe(self) -> None:
        profile = DatasetProfiler().profile(_tall_df())
        assert profile.row_count == 10_000


# ---------------------------------------------------------------------------
# AnomalyInspector edge cases
# ---------------------------------------------------------------------------


class TestAnomalyInspectorEdgeCases:

    def test_empty_dataframe_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AnomalyInspector().inspect(_empty_df())

    def test_single_row(self) -> None:
        result = AnomalyInspector().inspect(_single_row_df(), method="zscore")
        assert result.total_anomalies == 0

    def test_all_null_numeric_column_skipped(self) -> None:
        df = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
        result = AnomalyInspector().inspect(df, method="zscore")
        assert result.reports == ()

    def test_mixed_type_column(self) -> None:
        df = pd.DataFrame(
            {"num": [1.0, 2.0, 3.0, 1000.0], "text": ["a", "b", "c", "d"]}
        )
        result = AnomalyInspector().inspect(df, method="iqr")
        assert "num" in result.analyzed_columns
        assert "text" not in result.analyzed_columns

    def test_infinite_values_ignored(self) -> None:
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0, np.inf, -np.inf, 4.0]})
        result = AnomalyInspector().inspect(df, method="zscore")
        assert result.analyzed_columns == ("num",)

    def test_wide_dataframe(self) -> None:
        df = pd.DataFrame(
            {f"c{i}": [float(i), float(i + 1), float(i + 2)] for i in range(50)}
        )
        result = AnomalyInspector().inspect(df, method="iqr")
        assert len(result.analyzed_columns) == 50

    def test_tall_dataframe(self) -> None:
        df = pd.DataFrame({"num": [1.0] * 9_999 + [1_000.0]})
        result = AnomalyInspector().inspect(df, method="iqr")
        assert result.total_anomalies >= 1

    def test_iqr_and_zscore_deterministic(self) -> None:
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 100.0]})
        first = AnomalyInspector().inspect(df, method="zscore", seed=7)
        second = AnomalyInspector().inspect(df, method="zscore", seed=7)
        assert first == second


# ---------------------------------------------------------------------------
# NearDuplicateDetector edge cases
# ---------------------------------------------------------------------------


class TestNearDuplicateDetectorEdgeCases:

    def test_empty_dataframe_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            NearDuplicateDetector().detect(_empty_df())

    def test_single_row(self) -> None:
        result = NearDuplicateDetector().detect(_single_row_df())
        assert result.groups == ()
        assert result.duplicate_count == 0

    def test_all_null_rows_group(self) -> None:
        result = NearDuplicateDetector().detect(_all_null_df())
        assert result.duplicate_count == 2

    def test_mixed_type_column(self) -> None:
        result = NearDuplicateDetector().detect(_mixed_type_df())
        assert result.duplicate_count >= 0

    def test_infinite_values(self) -> None:
        result = NearDuplicateDetector().detect(_infinite_df())
        assert result.duplicate_count == 0

    def test_case_and_whitespace_normalized(self) -> None:
        df = pd.DataFrame({"c": ["Hello", "  hello  ", "world"]})
        result = NearDuplicateDetector().detect(df)
        assert result.duplicate_count == 1

    def test_wide_dataframe(self) -> None:
        result = NearDuplicateDetector().detect(_wide_df())
        # Each of the three rows is distinct across the wide columns.
        assert result.duplicate_count == 0

    def test_tall_dataframe(self) -> None:
        result = NearDuplicateDetector().detect(_tall_df(), subset=["text"])
        assert result.duplicate_count == 10_000 - 7

    def test_deterministic(self) -> None:
        df = pd.DataFrame({"c": ["a", "A", "b", "B"]})
        assert NearDuplicateDetector().detect(df) == NearDuplicateDetector().detect(df)


# ---------------------------------------------------------------------------
# TextQualityAnalyzer edge cases
# ---------------------------------------------------------------------------


class TestTextQualityAnalyzerEdgeCases:

    def test_empty_dataframe_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            TextQualityAnalyzer().analyze(_empty_df())

    def test_single_row(self) -> None:
        results = TextQualityAnalyzer().analyze(pd.DataFrame({"c": ["hello world"]}))
        assert len(results) == 1
        assert results[0].non_null_count == 1
        assert results[0].token_count_max == 2

    def test_all_null_text_column(self) -> None:
        results = TextQualityAnalyzer().analyze(pd.DataFrame({"c": [None, None, None]}))
        assert len(results) == 1
        assert results[0].non_null_count == 0

    def test_mixed_type_column_analyzed_as_object(self) -> None:
        results = TextQualityAnalyzer().analyze(_mixed_type_df())
        # The object-dtype "mixed" column is analyzed after str-coercion.
        assert len(results) == 1
        assert results[0].column == "mixed"

    def test_numeric_only_frame_yields_no_results(self) -> None:
        results = TextQualityAnalyzer().analyze(_infinite_df())
        assert results == ()

    def test_empty_after_strip_counted(self) -> None:
        results = TextQualityAnalyzer().analyze(pd.DataFrame({"c": ["   ", "real"]}))
        assert results[0].empty_after_strip_count == 1

    def test_wide_dataframe(self) -> None:
        results = TextQualityAnalyzer().analyze(_wide_df())
        assert len(results) == 200

    def test_tall_dataframe(self) -> None:
        results = TextQualityAnalyzer().analyze(_tall_df(), subset=["text"])
        assert len(results) == 1
        assert results[0].non_null_count == 10_000

    def test_deterministic(self) -> None:
        df = pd.DataFrame({"c": ["alpha", "beta", "gamma"]})
        assert TextQualityAnalyzer().analyze(df) == TextQualityAnalyzer().analyze(df)
