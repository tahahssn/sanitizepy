"""
Unit and integration tests for the high-level sanitizepy API:
inspect(), plan(), clean(), Cleaner(), dry-run mode, and health scoring.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import sanitizepy
from sanitizepy import (
    Cleaner,
    CleaningPlan,
    CleaningResult,
    ColumnContract,
    DataContract,
    DatasetHealthReport,
    DatasetProfile,
)
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingRows,
    FillMissing,
)
from sanitizepy.rules.rule import RuleResult


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Sample raw dataframe containing various data quality issues."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 3, 5, 6, 7, 8, 9, 10],
            "name": [
                "Alice",
                "Bob",
                "Charlie",
                "Charlie",
                "Eve",
                "Frank",
                "Grace",
                "Heidi",
                "Ivan",
                "Judy",
            ],
            "age": [25.0, 30.0, None, None, 45.0, 50.0, 35.0, 40.0, 28.0, 33.0],
            "signup_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-03",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
                "2023-01-09",
                "2023-01-10",
            ],
            "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "empty_col": [None, None, None, None, None, None, None, None, None, None],
        }
    )


def test_top_level_inspect(sample_raw_dataframe: pd.DataFrame) -> None:
    report = sanitizepy.inspect(sample_raw_dataframe)
    assert isinstance(report, DatasetHealthReport)
    assert 0 <= report.health_score <= 100
    assert report.rows == 10
    assert report.columns == 6
    assert len(report.issues) > 0
    assert len(report.recommendations) > 0

    # Critical issues should include empty_col
    critical_cols = [i.column for i in report.critical_issues]
    assert "empty_col" in critical_cols


def test_top_level_plan(sample_raw_dataframe: pd.DataFrame) -> None:
    report = sanitizepy.inspect(sample_raw_dataframe)
    plan_from_report = sanitizepy.plan(report)
    plan_from_df = sanitizepy.plan(sample_raw_dataframe)

    assert isinstance(plan_from_report, CleaningPlan)
    assert isinstance(plan_from_df, CleaningPlan)
    assert len(plan_from_report.steps) > 0


def test_plan_enable_disable(sample_raw_dataframe: pd.DataFrame) -> None:
    cleaning_plan = sanitizepy.plan(sample_raw_dataframe)
    first_step_idx = cleaning_plan.steps[0].index

    cleaning_plan.disable(first_step_idx)
    assert cleaning_plan.steps[0].enabled is False

    cleaning_plan.enable(first_step_idx)
    assert cleaning_plan.steps[0].enabled is True


def test_dry_run_mode(sample_raw_dataframe: pd.DataFrame) -> None:
    c = Cleaner()
    cleaning_plan = c.plan(sample_raw_dataframe)

    # Dry run execution should not change underlying shape or values permanently
    result = c.clean(sample_raw_dataframe, plan=cleaning_plan, dry_run=True)

    assert isinstance(result, CleaningResult)
    assert result.dry_run is True
    assert result.data.shape == sample_raw_dataframe.shape
    assert len(result.operations) > 0
    assert len(result.audit_log) == len(result.operations)
    assert result.audit_log[0]["dry_run"] is True


def test_actual_clean_execution(sample_raw_dataframe: pd.DataFrame) -> None:
    c = Cleaner()
    result = c.clean(sample_raw_dataframe, dry_run=False)

    assert isinstance(result, CleaningResult)
    assert result.dry_run is False
    # Shape after cleaning should be transformed (e.g. empty_col dropped,
    # duplicates removed)
    assert result.data.shape != sample_raw_dataframe.shape
    assert "empty_col" not in result.data.columns
    assert result.data.duplicated().sum() == 0


def test_edge_case_empty_dataframe() -> None:
    empty_df = pd.DataFrame()
    report = sanitizepy.inspect(empty_df)
    assert report.health_score == 0
    assert len(report.critical_issues) > 0


def test_edge_case_single_cell() -> None:
    single_df = pd.DataFrame({"col": [1]})
    report = sanitizepy.inspect(single_df)
    assert report.health_score > 50
    assert report.rows == 1
    assert report.columns == 1


# ---------------------------------------------------------------------------
# Task 19.3 - Property 20: Optional dependencies do not contaminate core imports
# Validates: Requirements 16.1, 16.3
# ---------------------------------------------------------------------------


class TestCoreImportPurity:
    """
    Property 20: Optional dependencies (rapidfuzz, ftfy) must never be pulled
    into sys.modules by a plain ``import sanitizepy``. They are only imported
    lazily inside the capabilities that need them.
    """

    _PROBE = (
        "import sys, sanitizepy; "
        "print('rapidfuzz' in sys.modules, 'ftfy' in sys.modules)"
    )

    def test_import_sanitizepy_does_not_import_optional_deps(self) -> None:
        # Run in a fresh subprocess so the result is not polluted by whatever
        # the test process itself has already imported.
        completed = subprocess.run(
            [sys.executable, "-c", self._PROBE],
            capture_output=True,
            text=True,
            check=True,
        )

        stdout = completed.stdout.strip()
        assert stdout == "False False", (
            "import sanitizepy must not pull optional dependencies into "
            f"sys.modules; probe output was: {stdout!r} "
            f"(stderr: {completed.stderr!r})"
        )

    @pytest.mark.parametrize("optional_module", ["rapidfuzz", "ftfy"])
    def test_specific_optional_dep_absent_after_core_import(
        self, optional_module: str
    ) -> None:
        probe = "import sys, sanitizepy; " f"print({optional_module!r} in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == "False", (
            f"{optional_module} was imported by core `import sanitizepy` "
            f"(stderr: {completed.stderr!r})"
        )


# ---------------------------------------------------------------------------
# Task 20.2 - Property 19: Existing APIs retain behavior
# Validates: Requirements 15.1, 15.4
# ---------------------------------------------------------------------------


@st.composite
def _dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate varied DataFrames with 1-3 numeric columns of equal length."""
    n_cols = draw(st.integers(min_value=1, max_value=3))
    length = draw(st.integers(min_value=1, max_value=25))
    data = {}
    for i in range(n_cols):
        column = draw(
            st.lists(
                st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
                min_size=length,
                max_size=length,
            )
        )
        data[f"col_{i}"] = column
    return pd.DataFrame(data)


class TestExistingApiBehaviorPreserved:
    """
    Property 19: Existing public APIs retain their documented behavior across
    a wide range of generated DataFrames.
    """

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_inspect_contract(self, df: pd.DataFrame) -> None:
        report = sanitizepy.inspect(df)
        assert isinstance(report, DatasetHealthReport)
        assert 0 <= report.health_score <= 100
        assert report.rows == df.shape[0]
        assert report.columns == df.shape[1]

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_dry_run_never_mutates_input(self, df: pd.DataFrame) -> None:
        original = df.copy(deep=True)
        c = Cleaner()
        result = c.clean(df, dry_run=True)
        assert isinstance(result, CleaningResult)
        assert result.dry_run is True
        # Input DataFrame must be untouched by a dry run.
        pd.testing.assert_frame_equal(df, original)

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_fill_missing_constant_preserves_behavior(self, df: pd.DataFrame) -> None:
        original = df.copy(deep=True)
        op = FillMissing(value=0)
        transformed = op.apply(df)
        # Input is not mutated.
        pd.testing.assert_frame_equal(df, original)
        # Same shape, no remaining missing values in the filled columns.
        assert transformed.shape == df.shape
        assert int(transformed.isna().to_numpy().sum()) == 0

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_drop_duplicates_preserves_behavior(self, df: pd.DataFrame) -> None:
        original = df.copy(deep=True)
        op = DropDuplicates()
        transformed = op.apply(df)
        pd.testing.assert_frame_equal(df, original)
        assert transformed.duplicated().sum() == 0
        assert transformed.shape[0] <= df.shape[0]
        assert list(transformed.columns) == list(df.columns)

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_drop_columns_preserves_behavior(self, df: pd.DataFrame) -> None:
        original = df.copy(deep=True)
        target = list(df.columns)[0]
        op = DropColumns(columns=[target])
        transformed = op.apply(df)
        pd.testing.assert_frame_equal(df, original)
        assert target not in transformed.columns
        assert transformed.shape[1] == df.shape[1] - 1

    @given(df=_dataframes())
    @settings(max_examples=75, deadline=None)
    def test_drop_missing_rows_preserves_behavior(self, df: pd.DataFrame) -> None:
        original = df.copy(deep=True)
        op = DropMissingRows()
        transformed = op.apply(df)
        pd.testing.assert_frame_equal(df, original)
        # No missing values remain and no rows were added.
        assert int(transformed.isna().to_numpy().sum()) == 0
        assert transformed.shape[0] <= df.shape[0]
        assert list(transformed.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Task 20.3 (facade portion) - public API facade methods work
# Validates: Requirements 15.1, 15.3
# ---------------------------------------------------------------------------


class TestFacadeMethods:
    """The additive Cleaner facade methods work without altering existing API."""

    def test_profile_facade(self, sample_raw_dataframe: pd.DataFrame) -> None:
        c = Cleaner()
        profile = c.profile(sample_raw_dataframe)
        assert isinstance(profile, DatasetProfile)

    def test_validate_facade(self, sample_raw_dataframe: pd.DataFrame) -> None:
        c = Cleaner()
        contract = DataContract(columns={"id": ColumnContract(dtype="int64")})
        results = c.validate(sample_raw_dataframe, contract)
        assert isinstance(results, tuple)
        assert len(results) >= 1
        assert all(isinstance(r, RuleResult) for r in results)
