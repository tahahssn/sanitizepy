"""
Tests for cleaner.inspection.statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaner.inspection.statistics import (
    StatisticsInspectionResult,
    StatisticsInspector,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


class TestStatisticsInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = StatisticsInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())


class TestStatisticsInspectorNoNumeric:
    def setup_method(self):
        self.df = _make_df(name=["alice", "bob", "charlie"])
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_result(self):
        assert isinstance(self.result, StatisticsInspectionResult)

    def test_no_reports_for_non_numeric(self):
        assert len(self.result.reports) == 0

    def test_summary_numeric_columns_zero(self):
        assert self.result.summary.numeric_columns == 0


class TestStatisticsInspectorBasic:
    def setup_method(self):
        self.df = _make_df(x=[1.0, 2.0, 3.0, 4.0, 5.0])
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_returns_result(self):
        assert isinstance(self.result, StatisticsInspectionResult)

    def test_one_report(self):
        assert len(self.result.reports) == 1

    def test_report_column_name(self):
        assert self.result.reports[0].column == "x"

    def test_count(self):
        assert self.result.reports[0].count == 5

    def test_minimum(self):
        assert self.result.reports[0].minimum == 1.0

    def test_maximum(self):
        assert self.result.reports[0].maximum == 5.0

    def test_mean(self):
        assert self.result.reports[0].mean == pytest.approx(3.0)

    def test_median(self):
        assert self.result.reports[0].median == pytest.approx(3.0)

    def test_sum(self):
        assert self.result.reports[0].sum == pytest.approx(15.0)

    def test_zero_count(self):
        assert self.result.reports[0].zero_count == 0

    def test_negative_count_zero(self):
        assert self.result.reports[0].negative_count == 0

    def test_not_constant(self):
        assert self.result.reports[0].constant is False

    def test_missing_zero(self):
        assert self.result.reports[0].missing == 0

    def test_q1_q2_q3_order(self):
        r = self.result.reports[0]
        assert r.q1 <= r.q2 <= r.q3

    def test_iqr_positive(self):
        r = self.result.reports[0]
        assert r.iqr >= 0.0

    def test_unique(self):
        assert self.result.reports[0].unique == 5

    def test_summary_analyzed_columns(self):
        assert "x" in self.result.summary.analyzed_columns


class TestStatisticsInspectorConstantColumn:
    def setup_method(self):
        self.df = _make_df(c=[5.0, 5.0, 5.0, 5.0])
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_constant_flag(self):
        assert self.result.reports[0].constant is True

    def test_constant_column_in_summary(self):
        assert "c" in self.result.summary.constant_columns


class TestStatisticsInspectorNegativeValues:
    def setup_method(self):
        self.df = _make_df(x=[-3, -2, -1, 0, 1, 2, 3])
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_negative_count(self):
        assert self.result.reports[0].negative_count == 3

    def test_zero_count(self):
        assert self.result.reports[0].zero_count == 1


class TestStatisticsInspectorInfiniteValues:
    def setup_method(self):
        self.df = _make_df(x=[1.0, 2.0, np.inf, -np.inf, 5.0])
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_infinite_count(self):
        assert self.result.reports[0].infinite_count == 2

    def test_finite_values_used_for_stats(self):
        # Only finite values 1,2,5 should be in clean series
        assert self.result.reports[0].count == 3


class TestStatisticsInspectorMixedColumns:
    def setup_method(self):
        self.df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "text": ["a", "b", "c"],
            }
        )
        self.inspector = StatisticsInspector()
        self.result = self.inspector.inspect(self.df)

    def test_only_numeric_analyzed(self):
        assert len(self.result.reports) == 1
        assert self.result.reports[0].column == "num"


class TestDescribe:
    def test_returns_dataframe(self):
        inspector = StatisticsInspector()
        df = _make_df(x=[1, 2, 3])
        result = inspector.describe(df)
        assert isinstance(result, pd.DataFrame)


class TestCorrelation:
    def test_returns_dataframe(self):
        inspector = StatisticsInspector()
        df = _make_df(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
        result = inspector.correlation(df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self):
        inspector = StatisticsInspector()
        df = _make_df(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
        result = inspector.correlation(df)
        assert result.loc["x", "x"] == pytest.approx(1.0)


class TestCovariance:
    def test_returns_dataframe(self):
        inspector = StatisticsInspector()
        df = _make_df(x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
        result = inspector.covariance(df)
        assert isinstance(result, pd.DataFrame)
