"""
Tests for cleaner.inspection.missing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaner.inspection.missing import (
    MissingColumnReport,
    MissingInspectionResult,
    MissingSummary,
    MissingValueInspector,
)


def _make_df(**kwargs):
    """Helper: build a DataFrame from keyword args (column=list)."""
    return pd.DataFrame(kwargs)


class TestMissingValueInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = MissingValueInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())


class TestMissingValueInspectorNoMissing:
    def setup_method(self):
        self.df = _make_df(a=[1, 2, 3], b=["x", "y", "z"])
        self.inspector = MissingValueInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_missing_inspection_result(self):
        assert isinstance(self.result, MissingInspectionResult)

    def test_summary_total_rows(self):
        assert self.result.summary.total_rows == 3

    def test_summary_total_columns(self):
        assert self.result.summary.total_columns == 2

    def test_summary_total_cells(self):
        assert self.result.summary.total_cells == 6

    def test_summary_missing_cells_zero(self):
        assert self.result.summary.missing_cells == 0

    def test_summary_missing_percentage_zero(self):
        assert self.result.summary.missing_percentage == 0.0

    def test_summary_complete_cells(self):
        assert self.result.summary.complete_cells == 6

    def test_summary_complete_percentage_100(self):
        assert self.result.summary.complete_percentage == 100.0

    def test_summary_columns_with_missing_empty(self):
        assert len(self.result.summary.columns_with_missing) == 0

    def test_summary_complete_columns(self):
        assert set(self.result.summary.complete_columns) == {"a", "b"}

    def test_summary_complete_rows(self):
        assert self.result.summary.complete_rows == 3

    def test_summary_rows_with_missing_zero(self):
        assert self.result.summary.rows_with_missing == 0

    def test_severity_low(self):
        assert self.result.summary.severity == "LOW"

    def test_column_reports_tuple(self):
        assert isinstance(self.result.column_reports, tuple)

    def test_missing_mask_is_dataframe(self):
        assert isinstance(self.result.missing_mask, pd.DataFrame)

    def test_missing_mask_all_false(self):
        assert not self.result.missing_mask.any().any()


class TestMissingValueInspectorWithMissing:
    def setup_method(self):
        self.df = _make_df(
            a=[1, None, 3, None, 5],
            b=["x", "y", None, None, "z"],
            c=[10, 20, 30, 40, 50],
        )
        self.inspector = MissingValueInspector()
        self.result = self.inspector.inspect(self.df)

    def test_summary_missing_cells(self):
        # a has 2 missing, b has 2 missing, c has 0 = 4 total
        assert self.result.summary.missing_cells == 4

    def test_summary_columns_with_missing(self):
        assert set(self.result.summary.columns_with_missing) == {"a", "b"}

    def test_summary_complete_columns(self):
        assert self.result.summary.complete_columns == ("c",)

    def test_column_reports_sorted_descending(self):
        percentages = [r.missing_percentage for r in self.result.column_reports]
        assert percentages == sorted(percentages, reverse=True)

    def test_column_report_has_missing(self):
        report_a = next(r for r in self.result.column_reports if r.column == "a")
        assert report_a.has_missing == True  # noqa: E712 (np.True_ compatible)

    def test_column_report_no_missing(self):
        reports = {r.column: r for r in self.result.column_reports}
        assert reports["c"].has_missing == False  # noqa: E712 (np.False_ compatible)

    def test_column_report_counts(self):
        reports = {r.column: r for r in self.result.column_reports}
        assert reports["a"].missing_count == 2

    def test_column_report_complete_count(self):
        reports = {r.column: r for r in self.result.column_reports}
        assert reports["a"].complete_count == 3

    def test_missing_percentages_series(self):
        assert isinstance(self.result.missing_percentages, pd.Series)


class TestMissingValueInspectorThreshold:
    def setup_method(self):
        self.df = _make_df(
            low=[None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            high=[None, None, None, None, None, None, 60, 70, 80, 90, 100],
        )
        self.inspector = MissingValueInspector()

    def test_threshold_filters_columns(self):
        result = self.inspector.inspect(self.df, threshold=40.0)
        # 'low' has ~9% missing, 'high' has ~54% missing
        # threshold=40 → only 'high' should remain in column_reports
        column_names = [r.column for r in result.column_reports]
        assert "low" not in column_names
        assert "high" in column_names

    def test_no_threshold_returns_all_columns(self):
        result = self.inspector.inspect(self.df)
        column_names = [r.column for r in result.column_reports]
        assert "low" in column_names
        assert "high" in column_names


class TestSeverityBoundaries:
    def _make_pct_df(self, pct: float) -> pd.DataFrame:
        n = 100
        missing = int(n * pct / 100)
        vals = [None] * missing + list(range(n - missing))
        return pd.DataFrame({"x": vals})

    def test_severity_low(self):
        inspector = MissingValueInspector()
        result = inspector.inspect(self._make_pct_df(0.0))
        assert result.summary.severity == "LOW"

    def test_severity_medium(self):
        inspector = MissingValueInspector()
        result = inspector.inspect(self._make_pct_df(10.0))
        assert result.summary.severity == "MEDIUM"

    def test_severity_high(self):
        inspector = MissingValueInspector()
        result = inspector.inspect(self._make_pct_df(25.0))
        assert result.summary.severity == "HIGH"

    def test_severity_critical(self):
        inspector = MissingValueInspector()
        result = inspector.inspect(self._make_pct_df(50.0))
        assert result.summary.severity == "CRITICAL"
