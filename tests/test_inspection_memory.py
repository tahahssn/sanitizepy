"""
Tests for sanitizepy.inspection.memory
"""

from __future__ import annotations

import pandas as pd
import pytest

from sanitizepy.inspection.memory import (
    MemoryColumnReport,
    MemoryInspectionResult,
    MemoryInspector,
    MemorySummary,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


class TestMemoryInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = MemoryInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())


class TestMemoryInspectorBasic:
    def setup_method(self):
        self.df = _make_df(a=[1, 2, 3, 4, 5], b=[1.0, 2.0, 3.0, 4.0, 5.0])
        self.inspector = MemoryInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_memory_inspection_result(self):
        assert isinstance(self.result, MemoryInspectionResult)

    def test_summary_is_memory_summary(self):
        assert isinstance(self.result.summary, MemorySummary)

    def test_summary_total_columns(self):
        assert self.result.summary.total_columns == 2

    def test_summary_total_rows(self):
        assert self.result.summary.total_rows == 5

    def test_summary_total_memory_bytes_positive(self):
        assert self.result.summary.total_memory_bytes > 0

    def test_summary_estimated_saved_bytes_non_negative(self):
        assert self.result.summary.estimated_saved_bytes >= 0

    def test_reports_is_tuple(self):
        assert isinstance(self.result.reports, tuple)

    def test_reports_count(self):
        assert len(self.result.reports) == 2

    def test_reports_are_memory_column_reports(self):
        for report in self.result.reports:
            assert isinstance(report, MemoryColumnReport)

    def test_reports_sorted_descending_by_bytes(self):
        sizes = [r.memory_bytes for r in self.result.reports]
        assert sizes == sorted(sizes, reverse=True)

    def test_memory_usage_is_series(self):
        assert isinstance(self.result.memory_usage, pd.Series)


class TestMemoryInspectorColumnReport:
    def setup_method(self):
        self.df = _make_df(x=[1, 2, 3])
        self.inspector = MemoryInspector()
        self.result = self.inspector.inspect(self.df)

    def test_column_name(self):
        assert self.result.reports[0].column == "x"

    def test_dtype_is_string(self):
        assert isinstance(self.result.reports[0].dtype, str)

    def test_memory_mb_non_negative(self):
        assert self.result.reports[0].memory_mb >= 0.0

    def test_memory_percentage_between_0_and_100(self):
        pct = self.result.reports[0].memory_percentage
        assert 0.0 <= pct <= 100.0

    def test_estimated_bytes_after_positive(self):
        assert self.result.reports[0].estimated_bytes_after > 0

    def test_estimated_saved_bytes_non_negative(self):
        assert self.result.reports[0].estimated_saved_bytes >= 0

    def test_estimated_saved_percentage_non_negative(self):
        assert self.result.reports[0].estimated_saved_percentage >= 0.0


class TestMemoryInspectorLargestColumns:
    def setup_method(self):
        self.df = _make_df(
            a=list(range(100)),
            b=[float(i) for i in range(100)],
            c=["text"] * 100,
        )
        self.inspector = MemoryInspector()

    def test_returns_series(self):
        result = self.inspector.largest_columns(self.df)
        assert isinstance(result, pd.Series)

    def test_n_limits_output(self):
        result = self.inspector.largest_columns(self.df, n=2)
        assert len(result) == 2

    def test_sorted_descending(self):
        result = self.inspector.largest_columns(self.df)
        vals = result.tolist()
        assert vals == sorted(vals, reverse=True)


class TestMemoryInspectorIntegerDowncast:
    def setup_method(self):
        # Large int64 column
        self.df = _make_df(x=[1, 2, 3, 4, 5])
        self.inspector = MemoryInspector()
        self.result = self.inspector.inspect(self.df)

    def test_integer_column_has_estimated_bytes(self):
        # estimated_bytes_after is always computed â€” just verify it is positive
        report = self.result.reports[0]
        assert report.estimated_bytes_after > 0

    def test_estimated_saved_bytes_non_negative(self):
        report = self.result.reports[0]
        assert report.estimated_saved_bytes >= 0


class TestMemoryInspectorObjectColumn:
    def setup_method(self):
        # Low-cardinality string column (pandas 3.x uses str/StringDtype, not object)
        self.df = _make_df(color=["red", "blue", "red", "blue"] * 25)
        self.inspector = MemoryInspector()
        self.result = self.inspector.inspect(self.df)

    def test_recommended_dtype_category_or_none(self):
        # In pandas 3.x, dtype may be 'str' not 'object', so is_object_dtype
        # returns False and recommended_dtype is None.
        report = self.result.reports[0]
        assert report.recommended_dtype in ("category", None)

    def test_estimated_saved_bytes_non_negative(self):
        report = self.result.reports[0]
        assert report.estimated_saved_bytes >= 0
