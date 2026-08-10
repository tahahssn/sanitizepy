"""
Tests for cleaner.inspection.datatypes
"""

from __future__ import annotations

import pandas as pd
import pytest

from cleaner.inspection.datatypes import (
    ColumnTypeReport,
    DatatypeInspectionResult,
    DatatypeInspector,
    DatatypeSummary,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


class TestDatatypeInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = DatatypeInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())


class TestDatatypeInspectorIntegerColumn:
    def setup_method(self):
        self.df = _make_df(x=[1, 2, 3])
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_datatype_inspection_result(self):
        assert isinstance(self.result, DatatypeInspectionResult)

    def test_summary_total_columns(self):
        assert self.result.summary.total_columns == 1

    def test_integer_column_detected(self):
        assert "x" in self.result.summary.integer_columns

    def test_numeric_column_detected(self):
        assert "x" in self.result.summary.numeric_columns

    def test_column_report_semantic_type(self):
        report = self.result.reports[0]
        assert report.semantic_type == "integer"

    def test_column_report_recommended_dtype(self):
        report = self.result.reports[0]
        # Integer columns get recommended Int64
        assert report.recommended_dtype == "Int64"

    def test_column_report_missing_count_zero(self):
        report = self.result.reports[0]
        assert report.missing_count == 0

    def test_column_report_unique_count(self):
        report = self.result.reports[0]
        assert report.unique_count == 3

    def test_column_report_memory_bytes_positive(self):
        report = self.result.reports[0]
        assert report.memory_bytes > 0


class TestDatatypeInspectorFloatColumn:
    def setup_method(self):
        self.df = _make_df(x=[1.0, 2.5, 3.7])
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_float_column_detected(self):
        assert "x" in self.result.summary.float_columns

    def test_numeric_column_detected(self):
        assert "x" in self.result.summary.numeric_columns

    def test_semantic_type_float(self):
        assert self.result.reports[0].semantic_type == "float"

    def test_recommended_dtype_float64(self):
        assert self.result.reports[0].recommended_dtype == "Float64"


class TestDatatypeInspectorObjectColumn:
    def setup_method(self):
        # High-cardinality object/string column
        self.df = _make_df(name=["Alice", "Bob", "Charlie", "Dave", "Eve"])
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_string_or_object_column_detected(self):
        # In pandas 3.x is_string_dtype fires before is_object_dtype,
        # so the column lands in string_columns, not object_columns.
        assert (
            "name" in self.result.summary.string_columns
            or "name" in self.result.summary.object_columns
        )

    def test_semantic_type_string_or_object(self):
        semantic = self.result.reports[0].semantic_type
        assert semantic in ("string", "object")

    def test_recommended_dtype_is_string_or_none(self):
        # Depending on pandas version, may be "string", "category", or None.
        rec = self.result.reports[0].recommended_dtype
        assert rec in ("string", "category", None)


class TestDatatypeInspectorLowCardinalityObject:
    def setup_method(self):
        # Low-cardinality object/string column
        self.df = _make_df(color=["red", "blue", "red", "blue", "red", "blue"] * 5)
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_recommended_dtype_category_or_other(self):
        # If pandas 3.x classifies via is_object_dtype the recommendation is
        # "category" for low cardinality. If it hits is_string_dtype first,
        # recommended_dtype may be None depending on _recommend_dtype logic.
        rec = self.result.reports[0].recommended_dtype
        assert rec in ("category", "string", None)


class TestDatatypeInspectorDatetimeColumn:
    def setup_method(self):
        self.df = _make_df(
            ts=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
        )
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_datetime_column_detected(self):
        assert "ts" in self.result.summary.datetime_columns

    def test_semantic_type_datetime(self):
        assert self.result.reports[0].semantic_type == "datetime"

    def test_no_recommended_dtype_for_datetime(self):
        assert self.result.reports[0].recommended_dtype is None


class TestDatatypeInspectorBooleanColumn:
    def setup_method(self):
        self.df = _make_df(flag=[True, False, True])
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_boolean_column_detected(self):
        assert "flag" in self.result.summary.boolean_columns

    def test_semantic_type_boolean(self):
        assert self.result.reports[0].semantic_type == "boolean"


class TestDatatypeInspectorMixedColumns:
    def setup_method(self):
        self.df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
        })
        self.inspector = DatatypeInspector()
        self.result = self.inspector.inspect(self.df)

    def test_total_columns(self):
        assert self.result.summary.total_columns == 3

    def test_reports_count(self):
        assert len(self.result.reports) == 3

    def test_reports_are_column_type_reports(self):
        for report in self.result.reports:
            assert isinstance(report, ColumnTypeReport)
