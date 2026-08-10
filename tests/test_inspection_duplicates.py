"""
Tests for cleaner.inspection.duplicates
"""

from __future__ import annotations

import pandas as pd
import pytest

from cleaner.inspection.duplicates import (
    DuplicateInspectionResult,
    DuplicateInspector,
    DuplicateSummary,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


class TestDuplicateInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = DuplicateInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())


class TestDuplicateInspectorNoDuplicates:
    def setup_method(self):
        self.df = _make_df(a=[1, 2, 3], b=["x", "y", "z"])
        self.inspector = DuplicateInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_duplicate_inspection_result(self):
        assert isinstance(self.result, DuplicateInspectionResult)

    def test_summary_total_rows(self):
        assert self.result.summary.total_rows == 3

    def test_summary_duplicate_rows_zero(self):
        assert self.result.summary.duplicate_rows == 0

    def test_summary_unique_rows(self):
        assert self.result.summary.unique_rows == 3

    def test_summary_duplicate_percentage_zero(self):
        assert self.result.summary.duplicate_percentage == 0.0

    def test_summary_unique_percentage_100(self):
        assert self.result.summary.unique_percentage == 100.0

    def test_severity_low(self):
        assert self.result.summary.severity == "LOW"

    def test_duplicate_count_zero(self):
        assert self.result.duplicate_count == 0

    def test_duplicate_indices_empty(self):
        assert len(self.result.duplicate_indices) == 0

    def test_duplicate_dataframe_empty(self):
        assert len(self.result.duplicate_dataframe) == 0


class TestDuplicateInspectorWithDuplicates:
    def setup_method(self):
        self.df = _make_df(
            a=[1, 2, 2, 3, 3, 3],
            b=["x", "y", "y", "z", "z", "z"],
        )
        self.inspector = DuplicateInspector()
        self.result = self.inspector.inspect(self.df)

    def test_duplicate_count(self):
        # 'first' strategy: row 2 (dup of row 1), rows 4,5 (dups of row 3) = 3 dups
        assert self.result.duplicate_count == 3

    def test_summary_duplicate_rows(self):
        assert self.result.summary.duplicate_rows == 3

    def test_summary_unique_rows(self):
        assert self.result.summary.unique_rows == 3

    def test_duplicate_indices_is_tuple(self):
        assert isinstance(self.result.duplicate_indices, tuple)

    def test_duplicate_dataframe_len(self):
        assert len(self.result.duplicate_dataframe) == 3


class TestDuplicateInspectorKeepStrategies:
    def setup_method(self):
        self.df = _make_df(a=[1, 1, 2], b=["x", "x", "y"])
        self.inspector = DuplicateInspector()

    def test_keep_first(self):
        result = self.inspector.inspect(self.df, keep="first")
        assert result.duplicate_count == 1
        assert 1 in result.duplicate_indices

    def test_keep_last(self):
        result = self.inspector.inspect(self.df, keep="last")
        assert result.duplicate_count == 1
        assert 0 in result.duplicate_indices

    def test_keep_false(self):
        result = self.inspector.inspect(self.df, keep=False)
        # Both rows 0 and 1 are duplicates
        assert result.duplicate_count == 2


class TestDuplicateInspectorSubset:
    def setup_method(self):
        self.df = _make_df(
            a=[1, 1, 2],
            b=["x", "y", "z"],
        )
        self.inspector = DuplicateInspector()

    def test_subset_reduces_duplicate_detection(self):
        # on subset=["a"] rows 0 and 1 are duplicates
        result = self.inspector.inspect(self.df, subset=["a"])
        assert result.duplicate_count == 1

    def test_no_subset_no_duplicates(self):
        result = self.inspector.inspect(self.df)
        assert result.duplicate_count == 0


class TestHasDuplicates:
    def setup_method(self):
        self.inspector = DuplicateInspector()

    def test_returns_true_when_duplicates_exist(self):
        df = _make_df(a=[1, 1, 2])
        assert self.inspector.has_duplicates(df) is True

    def test_returns_false_when_no_duplicates(self):
        df = _make_df(a=[1, 2, 3])
        assert self.inspector.has_duplicates(df) is False


class TestDuplicateColumns:
    def setup_method(self):
        self.inspector = DuplicateInspector()

    def test_no_duplicate_columns(self):
        df = _make_df(a=[1], b=[2])
        assert self.inspector.duplicate_columns(df) == ()

    def test_duplicate_column_count_zero(self):
        df = _make_df(a=[1], b=[2])
        assert self.inspector.duplicate_column_count(df) == 0


class TestSeverityBoundaries:
    def _make_df_with_pct(self, pct: float) -> pd.DataFrame:
        n = 100
        num_dups = int(n * pct / 100)
        unique_rows = n - num_dups
        a_vals = list(range(unique_rows)) + [0] * num_dups
        return pd.DataFrame({"a": a_vals})

    def test_severity_low(self):
        inspector = DuplicateInspector()
        result = inspector.inspect(self._make_df_with_pct(0.0))
        assert result.summary.severity == "LOW"

    def test_severity_medium(self):
        inspector = DuplicateInspector()
        result = inspector.inspect(self._make_df_with_pct(3.0))
        assert result.summary.severity == "MEDIUM"

    def test_severity_high(self):
        inspector = DuplicateInspector()
        result = inspector.inspect(self._make_df_with_pct(10.0))
        assert result.summary.severity == "HIGH"

    def test_severity_critical(self):
        inspector = DuplicateInspector()
        result = inspector.inspect(self._make_df_with_pct(20.0))
        assert result.summary.severity == "CRITICAL"
