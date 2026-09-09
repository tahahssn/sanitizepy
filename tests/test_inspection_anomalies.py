"""
Tests for sanitizepy.inspection.anomalies

Covers the AnomalyInspector (z-score method, seed reproducibility, preserved
IQR behavior) and the additive IssueDetector wiring that reports anomalies as
DatasetIssues including the method used and flagged count.

Requirements: 12.1, 12.3, 12.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sanitizepy.inspection.anomalies import (
    AnomalyInspector,
    AnomalyResult,
    ColumnAnomalyReport,
)
from sanitizepy.inspection.detector import IssueDetector


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestAnomalyInspectorEmpty:
    def test_raises_on_empty_dataframe(self):
        inspector = AnomalyInspector()
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(pd.DataFrame())

    def test_raises_on_dataframe_with_columns_but_no_rows(self):
        inspector = AnomalyInspector()
        empty = pd.DataFrame({"x": pd.Series([], dtype="float64")})
        with pytest.raises(ValueError, match="empty"):
            inspector.inspect(empty)


def _cluster_with_outlier() -> tuple[list[float], int]:
    """
    Build a large tight cluster plus one outlier whose z-score exceeds 3.

    With enough in-cluster points, a single distant value is not masked and
    its standard score comfortably clears the default threshold of 3.0. The
    outlier is placed last, so its positional index equals ``len - 1``.
    """
    cluster = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8] * 5
    outlier_index = len(cluster)
    return [*cluster, 60.0], outlier_index


class TestAnomalyInspectorZScore:
    def setup_method(self):
        # A tight cluster of values plus one clear outlier beyond 3 sigma.
        self.values, self.outlier_index = _cluster_with_outlier()
        self.df = _make_df(x=self.values)
        self.inspector = AnomalyInspector()
        self.result = self.inspector.inspect(self.df, method="zscore")

    def test_returns_result(self):
        assert isinstance(self.result, AnomalyResult)

    def test_records_method(self):
        assert self.result.method == "zscore"

    def test_analyzes_numeric_column(self):
        assert self.result.analyzed_columns == ("x",)

    def test_flags_the_outlier(self):
        assert self.result.total_anomalies == 1

    def test_report_shape(self):
        report = self.result.reports[0]
        assert isinstance(report, ColumnAnomalyReport)
        assert report.column == "x"
        assert report.method == "zscore"
        assert report.anomaly_count == 1
        # The last positional index holds the lone outlier.
        assert report.anomaly_indices == (self.outlier_index,)

    def test_report_bounds_populated(self):
        report = self.result.reports[0]
        assert report.lower_bound is not None
        assert report.upper_bound is not None
        assert report.lower_bound < report.upper_bound

    def test_percentage_computed(self):
        report = self.result.reports[0]
        expected = round((1 / report.analyzed_count) * 100.0, 4)
        assert report.anomaly_percentage == expected

    def test_threshold_controls_sensitivity(self):
        # A very high threshold flags nothing; a very low one flags at least
        # as many as the default threshold.
        strict = self.inspector.inspect(
            self.df, method="zscore", zscore_threshold=100.0
        )
        loose = self.inspector.inspect(self.df, method="zscore", zscore_threshold=0.5)
        assert strict.total_anomalies == 0
        assert loose.total_anomalies >= self.result.total_anomalies

    def test_constant_column_yields_no_anomalies(self):
        df = _make_df(c=[5.0, 5.0, 5.0, 5.0])
        result = self.inspector.inspect(df, method="zscore")
        assert result.total_anomalies == 0
        report = result.reports[0]
        assert report.anomaly_count == 0
        assert report.lower_bound is None
        assert report.upper_bound is None


class TestAnomalyInspectorSeedReproducibility:
    def setup_method(self):
        values, _ = _cluster_with_outlier()
        self.df = _make_df(x=values)
        self.inspector = AnomalyInspector()

    def test_seed_recorded_in_result(self):
        result = self.inspector.inspect(self.df, method="zscore", seed=42)
        assert result.seed == 42

    def test_same_seed_reproduces_identical_results(self):
        first = self.inspector.inspect(self.df, method="zscore", seed=7)
        second = self.inspector.inspect(self.df, method="zscore", seed=7)
        assert first == second

    def test_different_seeds_produce_identical_deterministic_results(self):
        # The current methods are deterministic, so the seed must not change
        # the detected anomalies.
        seed_a = self.inspector.inspect(self.df, method="zscore", seed=1)
        seed_b = self.inspector.inspect(self.df, method="zscore", seed=999)
        assert seed_a.reports == seed_b.reports
        assert seed_a.total_anomalies == seed_b.total_anomalies

    def test_none_seed_matches_seeded_detection(self):
        no_seed = self.inspector.inspect(self.df, method="zscore", seed=None)
        seeded = self.inspector.inspect(self.df, method="zscore", seed=123)
        assert no_seed.reports == seeded.reports


class TestAnomalyInspectorIQRPreserved:
    def setup_method(self):
        self.df = _make_df(x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
        self.inspector = AnomalyInspector()

    def test_default_method_is_iqr(self):
        result = self.inspector.inspect(self.df)
        assert result.method == "iqr"

    def test_iqr_flags_outlier(self):
        result = self.inspector.inspect(self.df, method="iqr")
        assert result.total_anomalies == 1
        assert result.reports[0].anomaly_indices == (9,)

    def test_iqr_bounds_match_tukey_fences(self):
        result = self.inspector.inspect(self.df, method="iqr", iqr_multiplier=1.5)
        report = result.reports[0]
        values = self.df["x"]
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        assert report.lower_bound == pytest.approx(q1 - 1.5 * iqr)
        assert report.upper_bound == pytest.approx(q3 + 1.5 * iqr)

    def test_iqr_multiplier_widens_fences(self):
        wide = self.inspector.inspect(self.df, method="iqr", iqr_multiplier=100.0)
        assert wide.total_anomalies == 0

    def test_iqr_deterministic_across_runs(self):
        first = self.inspector.inspect(self.df, method="iqr")
        second = self.inspector.inspect(self.df, method="iqr")
        assert first == second


class TestAnomalyInspectorColumnHandling:
    def setup_method(self):
        self.inspector = AnomalyInspector()

    def test_skips_non_numeric_columns(self):
        df = _make_df(
            name=["alice", "bob", "charlie"],
            value=[1.0, 2.0, 1000.0],
        )
        result = self.inspector.inspect(df, method="zscore")
        assert result.analyzed_columns == ("value",)

    def test_all_non_numeric_yields_no_reports(self):
        df = _make_df(name=["alice", "bob", "charlie"])
        result = self.inspector.inspect(df, method="zscore")
        assert result.reports == ()
        assert result.analyzed_columns == ()
        assert result.total_anomalies == 0

    def test_infinite_values_are_ignored(self):
        df = _make_df(x=[1.0, 2.0, 3.0, np.inf, -np.inf, 4.0])
        result = self.inspector.inspect(df, method="zscore")
        report = result.reports[0]
        # inf / -inf are dropped, leaving the four finite values analyzed.
        assert report.analyzed_count == 4

    def test_does_not_mutate_input(self):
        df = _make_df(x=[1.0, 2.0, np.inf, 1000.0])
        before = df.copy(deep=True)
        self.inspector.inspect(df, method="zscore")
        pd.testing.assert_frame_equal(df, before)


class TestAnomalyInspectorEdgeCases:
    def setup_method(self):
        self.inspector = AnomalyInspector()

    def test_single_row(self):
        df = _make_df(x=[42.0])
        result = self.inspector.inspect(df, method="zscore")
        # std is 0 for a single value, so nothing is flagged.
        assert result.total_anomalies == 0

    def test_all_null_numeric_column_skipped(self):
        df = _make_df(x=[np.nan, np.nan, np.nan])
        result = self.inspector.inspect(df, method="zscore")
        assert result.analyzed_columns == ()
        assert result.reports == ()

    def test_mixed_type_columns(self):
        df = _make_df(
            num=[1.0, 2.0, 3.0, 1000.0],
            text=["a", "b", "c", "d"],
            flag=[True, False, True, False],
        )
        result = self.inspector.inspect(df, method="zscore")
        # Numeric and boolean columns qualify as numeric dtype; text does not.
        assert "text" not in result.analyzed_columns
        assert "num" in result.analyzed_columns

    def test_wide_dataframe(self):
        data = {f"c{i}": [float(i), float(i + 1), float(i + 2)] for i in range(50)}
        df = pd.DataFrame(data)
        result = self.inspector.inspect(df, method="zscore")
        assert len(result.analyzed_columns) == 50

    def test_tall_dataframe(self):
        values = [1.0] * 999 + [1000.0]
        df = _make_df(x=values)
        result = self.inspector.inspect(df, method="zscore")
        assert result.total_anomalies == 1
        assert result.reports[0].analyzed_count == 1000


class TestIssueDetectorAnomalyWiring:
    def setup_method(self):
        self.detector = IssueDetector()

    def _anomaly_issues(self, report):
        return [i for i in report.issues if i.category == "anomalies"]

    def test_reports_anomaly_issue_for_clear_outliers(self):
        # Enough spread and rows to trigger the IQR-based anomaly detector.
        values = [float(v) for v in range(1, 30)] + [100000.0]
        df = _make_df(measurement=values)
        report = self.detector.inspect(df)
        anomaly_issues = self._anomaly_issues(report)
        assert len(anomaly_issues) >= 1

    def test_anomaly_issue_includes_method(self):
        values = [float(v) for v in range(1, 30)] + [100000.0]
        df = _make_df(measurement=values)
        report = self.detector.inspect(df)
        issue = self._anomaly_issues(report)[0]
        assert "method=" in issue.evidence
        assert "iqr" in issue.evidence

    def test_anomaly_issue_includes_flagged_count(self):
        values = [float(v) for v in range(1, 30)] + [100000.0]
        df = _make_df(measurement=values)
        report = self.detector.inspect(df)
        issue = self._anomaly_issues(report)[0]
        # The evidence encodes flagged=<count>/<analyzed>.
        assert "flagged=" in issue.evidence
        assert issue.column == "measurement"

    def test_no_anomaly_issue_when_no_outliers(self):
        df = _make_df(measurement=[float(v) for v in range(1, 40)])
        report = self.detector.inspect(df)
        assert self._anomaly_issues(report) == []


# ===========================================================================
# Property tests (task 15.3)
# **Property 14: Anomaly detection is deterministic**
# **Validates: Requirements 12.2**
# ===========================================================================

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@st.composite
def _numeric_cell(draw: st.DrawFn) -> object:
    """
    Draw a single numeric-column cell.

    The pool mixes ordinary finite floats, a few extreme outlier magnitudes,
    infinities, and ``NaN`` so the generated columns exercise the analyzer's
    finite-filtering and anomaly-flagging branches deterministically.
    """
    ordinary = st.floats(
        min_value=-1000.0,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    outlier = st.sampled_from([-1_000_000.0, 1_000_000.0, 500_000.0, -500_000.0])
    special = st.sampled_from([float("inf"), float("-inf"), float("nan")])
    return draw(st.one_of(ordinary, outlier, special))


@st.composite
def _numeric_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """
    Draw a non-empty DataFrame with one or more numeric columns.

    Each column mixes ordinary finite values, extreme outliers, infinities,
    and ``NaN`` (via :func:`_numeric_cell`). All columns share the same length
    so pandas accepts the mapping, and the dtype is float64 so the analyzer
    treats every column as numeric.
    """
    n_cols = draw(st.integers(min_value=1, max_value=4))
    n_rows = draw(st.integers(min_value=1, max_value=40))
    data = {
        f"col_{i}": pd.array(
            [draw(_numeric_cell()) for _ in range(n_rows)], dtype="float64"
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


class TestAnomalyDeterminismProperty:
    """
    **Property 14: Anomaly detection is deterministic**
    **Validates: Requirements 12.2**

    Inspecting the same DataFrame twice with the same method and seed must
    yield an identical ``AnomalyResult`` (reports, total_anomalies, and
    analyzed_columns all equal). The property holds for both the ``iqr`` and
    ``zscore`` methods and with a fixed seed.
    """

    @given(_numeric_dataframe(), st.sampled_from(["iqr", "zscore"]))
    @settings(max_examples=150, deadline=None)
    def test_repeated_inspection_yields_identical_result(
        self, df: pd.DataFrame, method: str
    ) -> None:
        inspector = AnomalyInspector()
        first = inspector.inspect(df, method=method)  # type: ignore[arg-type]
        second = inspector.inspect(df, method=method)  # type: ignore[arg-type]
        assert first == second

    @given(
        _numeric_dataframe(),
        st.sampled_from(["iqr", "zscore"]),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=150, deadline=None)
    def test_same_seed_reproduces_identical_result(
        self, df: pd.DataFrame, method: str, seed: int
    ) -> None:
        inspector = AnomalyInspector()
        first = inspector.inspect(df, method=method, seed=seed)  # type: ignore[arg-type]
        second = inspector.inspect(df, method=method, seed=seed)  # type: ignore[arg-type]
        assert first == second

    @given(
        _numeric_dataframe(),
        st.sampled_from(["iqr", "zscore"]),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=150, deadline=None)
    def test_inspection_of_copy_matches_original(
        self, df: pd.DataFrame, method: str, seed: int
    ) -> None:
        inspector = AnomalyInspector()
        first = inspector.inspect(df, method=method, seed=seed)  # type: ignore[arg-type]
        second = inspector.inspect(
            df.copy(deep=True),
            method=method,  # type: ignore[arg-type]
            seed=seed,
        )
        assert first == second

    @given(
        _numeric_dataframe(),
        st.sampled_from(["iqr", "zscore"]),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=100, deadline=None)
    def test_every_field_is_identical_across_runs(
        self, df: pd.DataFrame, method: str, seed: int
    ) -> None:
        inspector = AnomalyInspector()
        first = inspector.inspect(df, method=method, seed=seed)  # type: ignore[arg-type]
        second = inspector.inspect(df, method=method, seed=seed)  # type: ignore[arg-type]

        assert first.method == second.method
        assert first.seed == second.seed
        assert first.analyzed_columns == second.analyzed_columns
        assert first.total_anomalies == second.total_anomalies
        assert first.reports == second.reports
