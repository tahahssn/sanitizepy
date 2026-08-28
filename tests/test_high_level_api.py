"""
Unit and integration tests for the high-level sanitizepy API:
inspect(), plan(), clean(), Cleaner(), dry-run mode, and health scoring.
"""

from __future__ import annotations

import pandas as pd
import pytest

import cleaner
from cleaner import Cleaner, CleaningPlan, CleaningResult, DatasetHealthReport


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
    report = cleaner.inspect(sample_raw_dataframe)
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
    report = cleaner.inspect(sample_raw_dataframe)
    plan_from_report = cleaner.plan(report)
    plan_from_df = cleaner.plan(sample_raw_dataframe)

    assert isinstance(plan_from_report, CleaningPlan)
    assert isinstance(plan_from_df, CleaningPlan)
    assert len(plan_from_report.steps) > 0


def test_plan_enable_disable(sample_raw_dataframe: pd.DataFrame) -> None:
    cleaning_plan = cleaner.plan(sample_raw_dataframe)
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
    report = cleaner.inspect(empty_df)
    assert report.health_score == 0
    assert len(report.critical_issues) > 0


def test_edge_case_single_cell() -> None:
    single_df = pd.DataFrame({"col": [1]})
    report = cleaner.inspect(single_df)
    assert report.health_score > 50
    assert report.rows == 1
    assert report.columns == 1
