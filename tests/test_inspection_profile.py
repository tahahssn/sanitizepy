"""
Unit tests for sanitizepy.inspection.profile

Covers the ``DatasetProfiler`` builder (per-dimension exposure, empty-frame
ValueError, non-mutation) and ``profile_to_report`` rendering through the
existing reports subsystem, plus the required edge cases (empty, single-row,
all-null, mixed-type, infinite, wide, tall).

NOTE: task 11.4 adds its property test(s) to this same module. To avoid any
collision, every class and helper defined here is prefixed with ``Unit`` /
``_unit_`` and kept in clearly separated, distinctly named test classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sanitizepy.inspection.datatypes import DatatypeInspectionResult
from sanitizepy.inspection.duplicates import DuplicateInspectionResult
from sanitizepy.inspection.memory import MemoryInspectionResult
from sanitizepy.inspection.missing import MissingInspectionResult
from sanitizepy.inspection.profile import DatasetProfiler, profile_to_report
from sanitizepy.inspection.statistics import StatisticsInspectionResult
from sanitizepy.models.profile import DatasetProfile
from sanitizepy.reports.report import Report


def _unit_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 2.0, 4.0],
            "count": [10, 20, 20, 40],
            "text": ["alpha", "beta", "beta", None],
        }
    )


class TestDatasetProfilerUnitBasic:
    def setup_method(self):
        self.df = _unit_sample_df()
        self.profiler = DatasetProfiler()
        self.profile = self.profiler.profile(self.df)

    def test_returns_dataset_profile(self):
        assert isinstance(self.profile, DatasetProfile)

    def test_row_count_matches(self):
        assert self.profile.row_count == 4

    def test_column_count_matches(self):
        assert self.profile.column_count == 3

    def test_exposes_datatype_result(self):
        assert isinstance(self.profile.datatypes, DatatypeInspectionResult)
        columns = {r.column for r in self.profile.datatypes.reports}
        assert columns == {"num", "count", "text"}

    def test_exposes_missing_result_per_column(self):
        assert isinstance(self.profile.missing_values, MissingInspectionResult)
        by_column = {
            r.column: r.missing_count
            for r in self.profile.missing_values.column_reports
        }
        assert by_column["text"] == 1
        assert by_column["num"] == 0

    def test_exposes_duplicate_result(self):
        assert isinstance(self.profile.duplicates, DuplicateInspectionResult)
        # Rows 1 and 2 are identical -> exactly one duplicate.
        assert self.profile.duplicates.duplicate_count == 1

    def test_exposes_memory_result_per_column(self):
        assert isinstance(self.profile.memory, MemoryInspectionResult)
        columns = {r.column for r in self.profile.memory.reports}
        assert columns == {"num", "count", "text"}

    def test_exposes_statistics_result(self):
        assert isinstance(self.profile.statistics, StatisticsInspectionResult)
        analyzed = {r.column for r in self.profile.statistics.reports}
        assert "num" in analyzed

    def test_optional_slots_none_for_core_profile(self):
        assert self.profile.text_quality is None
        assert self.profile.anomalies is None
        assert self.profile.near_duplicate is None


class TestDatasetProfilerUnitNonMutation:
    def test_does_not_mutate_input(self):
        df = _unit_sample_df()
        before = df.copy(deep=True)
        DatasetProfiler().profile(df)
        pd.testing.assert_frame_equal(df, before)

    def test_reuses_inspector_results_matching_standalone(self):
        # The profile must expose the same values the standalone inspectors
        # produce, i.e. it aggregates rather than recomputing differently.
        df = _unit_sample_df()
        profile = DatasetProfiler().profile(df)

        from sanitizepy.inspection.duplicates import DuplicateInspector
        from sanitizepy.inspection.missing import MissingValueInspector

        standalone_missing = MissingValueInspector().inspect(df)
        standalone_dupes = DuplicateInspector().inspect(df)

        assert (
            profile.missing_values.summary.missing_cells
            == standalone_missing.summary.missing_cells
        )
        assert profile.duplicates.duplicate_count == standalone_dupes.duplicate_count


class TestDatasetProfilerUnitEmpty:
    def test_raises_value_error_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            DatasetProfiler().profile(pd.DataFrame())

    def test_raises_value_error_on_columns_but_no_rows(self):
        # A frame with declared columns but zero rows is still empty.
        empty_rows = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        with pytest.raises(ValueError, match="empty"):
            DatasetProfiler().profile(empty_rows)


class TestProfileToReportUnit:
    def setup_method(self):
        self.df = _unit_sample_df()
        self.profile = DatasetProfiler().profile(self.df)
        self.report = profile_to_report(self.profile)

    def test_returns_report(self):
        assert isinstance(self.report, Report)

    def test_default_title(self):
        assert self.report.title == "Dataset Profile"

    def test_custom_title(self):
        report = profile_to_report(self.profile, title="My Profile")
        assert report.title == "My Profile"

    def test_core_sections_present(self):
        for name in (
            "overview",
            "datatypes",
            "missing_values",
            "duplicates",
            "memory",
            "statistics",
        ):
            assert self.report.has_section(name), name

    def test_overview_section_content(self):
        overview = self.report.get_section("overview")
        assert overview is not None
        assert overview.content["row_count"] == 4
        assert overview.content["column_count"] == 3

    def test_optional_sections_absent_for_core_profile(self):
        assert not self.report.has_section("text_quality")
        assert not self.report.has_section("anomalies")
        assert not self.report.has_section("near_duplicate")

    def test_section_titles_are_humanized(self):
        section = self.report.get_section("missing_values")
        assert section is not None
        assert section.title == "Missing Values"


class TestDatasetProfilerUnitEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        profile = DatasetProfiler().profile(df)
        assert profile.row_count == 1
        assert profile.column_count == 2
        assert profile.duplicates.duplicate_count == 0

    def test_all_null_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        profile = DatasetProfiler().profile(df)
        by_column = {
            r.column: r.missing_count for r in profile.missing_values.column_reports
        }
        assert by_column["a"] == 3

    def test_mixed_type_object_column(self):
        df = pd.DataFrame({"mixed": [1, "two", 3.0, None]})
        profile = DatasetProfiler().profile(df)
        assert profile.row_count == 4
        mixed_cols = profile.datatypes.summary.mixed_object_columns
        assert "mixed" in mixed_cols

    def test_infinite_values(self):
        df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 4.0]})
        profile = DatasetProfiler().profile(df)
        stats = {r.column: r for r in profile.statistics.reports}
        assert "x" in stats
        assert stats["x"].infinite_count == 2

    def test_wide_dataframe(self):
        data = {f"c{i}": [i, i + 1] for i in range(60)}
        df = pd.DataFrame(data)
        profile = DatasetProfiler().profile(df)
        assert profile.column_count == 60
        assert profile.row_count == 2
        report = profile_to_report(profile)
        assert isinstance(report, Report)

    def test_tall_dataframe(self):
        df = pd.DataFrame({"x": list(range(5000))})
        profile = DatasetProfiler().profile(df)
        assert profile.row_count == 5000
        assert profile.column_count == 1
        assert profile.duplicates.duplicate_count == 0


# ===========================================================================
# Property test — Task 11.4
# **Property 8: DatasetProfile reuses inspector results**
# **Validates: Requirements 8.1, 8.2**
#
# The profile's slots must equal what running each standalone inspector
# directly on the same DataFrame produces. This proves ``DatasetProfiler``
# aggregates (reuses) the standalone inspectors rather than recomputing any
# dimension differently.
#
# Equality note: several inspector results are frozen dataclasses that hold
# pandas objects (Series / DataFrame). Direct dataclass ``==`` on those raises
# on the ambiguous truth value of a pandas object, so those results are
# compared field-by-field: scalar/tuple fields with ``==`` and pandas fields
# with ``pd.testing``. Results made only of scalars/tuples
# (``datatypes``, ``statistics``) are compared with dataclass ``==`` directly.
# ===========================================================================

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from sanitizepy.inspection.datatypes import DatatypeInspector  # noqa: E402
from sanitizepy.inspection.duplicates import DuplicateInspector  # noqa: E402
from sanitizepy.inspection.memory import MemoryInspector  # noqa: E402
from sanitizepy.inspection.missing import MissingValueInspector  # noqa: E402
from sanitizepy.inspection.statistics import StatisticsInspector  # noqa: E402


@st.composite
def _property_profile_frame(draw: st.DrawFn) -> pd.DataFrame:
    """
    Draw a non-empty DataFrame with mixed columns, missing values, and rows
    that may repeat (so duplicates, missing, memory, datatype and statistics
    dimensions all have something to report).
    """

    size = draw(st.integers(min_value=1, max_value=30))

    floats = draw(
        st.lists(
            st.one_of(
                st.floats(
                    min_value=-1_000.0,
                    max_value=1_000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.none(),
            ),
            min_size=size,
            max_size=size,
        )
    )

    ints = draw(
        st.lists(
            st.integers(min_value=-1_000, max_value=1_000),
            min_size=size,
            max_size=size,
        )
    )

    texts = draw(
        st.lists(
            st.one_of(
                st.sampled_from(["alpha", "beta", "gamma", "", "  spaced  "]),
                st.none(),
            ),
            min_size=size,
            max_size=size,
        )
    )

    return pd.DataFrame({"num": floats, "count": ints, "text": texts})


def _scalar_equal(a: object, b: object) -> bool:
    """
    NaN-aware scalar equality. Two ``NaN`` floats compare equal here so that
    results carrying NaN statistics (e.g. variance of a single value) are
    still recognised as identical when produced from the same input.
    """

    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


def _assert_result_equal(actual: object, expected: object) -> None:
    """
    Compare two frozen-dataclass inspector results field by field.

    Handles the three field kinds these results hold:
      * pandas ``DataFrame`` / ``Series`` -> ``pd.testing`` comparators;
      * nested frozen dataclasses (summaries, per-column reports, and tuples
        of them) -> recursive comparison;
      * plain scalars/tuples -> NaN-aware scalar equality.
    """

    from dataclasses import fields, is_dataclass

    for f in fields(actual):  # type: ignore[arg-type]
        av = getattr(actual, f.name)
        ev = getattr(expected, f.name)

        if isinstance(av, pd.DataFrame):
            pd.testing.assert_frame_equal(av, ev)
        elif isinstance(av, pd.Series):
            pd.testing.assert_series_equal(av, ev)
        elif is_dataclass(av) and not isinstance(av, type):
            _assert_result_equal(av, ev)
        elif isinstance(av, tuple):
            assert len(av) == len(ev), f"field {f.name!r} length differs"
            for a_item, e_item in zip(av, ev, strict=True):
                if is_dataclass(a_item) and not isinstance(a_item, type):
                    _assert_result_equal(a_item, e_item)
                else:
                    assert _scalar_equal(
                        a_item, e_item
                    ), f"field {f.name!r} item differs: {a_item!r} != {e_item!r}"
        else:
            assert _scalar_equal(av, ev), f"field {f.name!r} differs: {av!r} != {ev!r}"


class TestDatasetProfileReusesInspectorResultsProperty:
    """
    **Property 8: DatasetProfile reuses inspector results**
    **Validates: Requirements 8.1, 8.2**
    """

    @given(df=_property_profile_frame())
    @settings(max_examples=100, deadline=None)
    def test_profile_slots_match_standalone_inspectors(self, df: pd.DataFrame) -> None:
        profile = DatasetProfiler().profile(df)

        # Row/column counts mirror the frame shape (Requirement 8.1).
        assert profile.row_count == df.shape[0]
        assert profile.column_count == df.shape[1]

        # Every aggregated slot must equal the standalone inspector's result
        # on the same frame -> the profiler reuses, not recomputes
        # (Requirement 8.2). Comparison is NaN-aware and pandas-aware because
        # these frozen results carry NaN statistics and pandas objects.
        _assert_result_equal(profile.datatypes, DatatypeInspector().inspect(df))
        _assert_result_equal(profile.statistics, StatisticsInspector().inspect(df))
        _assert_result_equal(
            profile.missing_values, MissingValueInspector().inspect(df)
        )
        _assert_result_equal(profile.duplicates, DuplicateInspector().inspect(df))
        _assert_result_equal(profile.memory, MemoryInspector().inspect(df))
