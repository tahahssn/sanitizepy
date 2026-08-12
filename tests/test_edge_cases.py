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

from cleaner import Cleaner
from cleaner.cleaning import (
    CleaningEngine,
    CleaningResult,
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
    OperationResult,
)
from cleaner.inspection.detector import IssueDetector
from cleaner.inspection.health import DatasetHealthReport


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
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10.1, 20.2, 30.3, 40.4, 50.5],
        })
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
        df = pd.DataFrame({"date": [
            "2023-01-01", "2023-02-15", "2023-03-20",
            "2023-04-10", "2023-05-05", "2023-06-30",
        ]})
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
        engine = CleaningEngine([
            DropDuplicates(),
            DropMissingRows(),
        ])
        result = engine.run_with_result(df, dry_run=True)
        assert result.dry_run is True
        assert result.data.shape == df.shape

    def test_real_run_changes_shape(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, 5, 6]})
        engine = CleaningEngine([
            DropDuplicates(),
            DropMissingRows(),
        ])
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
