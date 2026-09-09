"""
Unit tests for ``CleaningPlan.from_report`` mappings — task 9.2.

Covers the Phase-1 recommendation-to-operation wiring added in task 9.1:

- Each ``FILL_*`` action maps to a ``FillMissing`` step with the correct
  strategy and target column subset.
- ``CONVERT_TYPE`` maps to a ``TypeCoercionOperation`` only when a resolvable
  ``target_dtype`` is present; otherwise no coercion step is created.
- Disabled recommendations and recommendations that cannot be turned into a
  constructible operation produce no orphaned steps.

Edge cases exercised: recommendation with a missing column, and unresolvable
``target_dtype`` (``None`` / empty string).

_Requirements: 1.6, 2.2_
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.operations import DropColumns, DropDuplicates, FillMissing
from sanitizepy.cleaning.plan import CleaningPlan, PlanStep
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation
from sanitizepy.exceptions import DeserializationError
from sanitizepy.inspection.health import DatasetHealthReport
from sanitizepy.models.base import ColumnReference
from sanitizepy.models.recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationCategory,
    RecommendationImpact,
    RecommendationPriority,
    RecommendationReason,
)
from sanitizepy.models.replay import ReplayablePlan, ReplayOperation
from sanitizepy.version import VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recommendation(
    action: RecommendationAction,
    *,
    column: str | None = "age",
    target_dtype: str | None = None,
    enabled: bool = True,
    title: str = "Recommendation",
) -> Recommendation:
    """Build a minimal valid ``Recommendation`` for plan construction."""
    return Recommendation(
        title=title,
        description="Generated for from_report mapping tests.",
        category=RecommendationCategory.CLEANING,
        priority=RecommendationPriority.MEDIUM,
        action=action,
        column=ColumnReference(name=column) if column is not None else None,
        target_dtype=target_dtype,
        reason=RecommendationReason(
            title="Reason",
            explanation="Explanation.",
        ),
        impact=RecommendationImpact(),
        confidence=0.8,
        enabled=enabled,
    )


def _make_report(recommendations: list[Recommendation]) -> DatasetHealthReport:
    """Wrap recommendations in a minimal health report."""
    return DatasetHealthReport(
        health_score=90,
        completeness_score=100.0,
        uniqueness_score=100.0,
        consistency_score=100.0,
        validity_score=100.0,
        integrity_score=100.0,
        rows=10,
        columns=3,
        memory_mb=0.1,
        recommendations=recommendations,
    )


def _plan_from(recommendations: list[Recommendation]) -> CleaningPlan:
    return CleaningPlan.from_report(_make_report(recommendations))


# ---------------------------------------------------------------------------
# FILL_* mappings (Requirement 1.6)
# ---------------------------------------------------------------------------


class TestFillMappings:
    @pytest.mark.parametrize(
        ("action", "expected_strategy"),
        [
            (RecommendationAction.FILL_MEDIAN, "median"),
            (RecommendationAction.FILL_MEAN, "mean"),
            (RecommendationAction.FILL_MODE, "mode"),
            (RecommendationAction.FILL_CONSTANT, "constant"),
        ],
    )
    def test_fill_action_maps_to_strategy_and_column(
        self, action: RecommendationAction, expected_strategy: str
    ) -> None:
        plan = _plan_from([_make_recommendation(action, column="age")])

        assert len(plan.steps) == 1
        op = plan.steps[0].operation
        assert isinstance(op, FillMissing)
        assert op.strategy == expected_strategy
        assert op.subset == ["age"]

    def test_fill_step_preserves_recommendation_and_title(self) -> None:
        rec = _make_recommendation(
            RecommendationAction.FILL_MEAN, column="salary", title="Fill salary"
        )
        plan = _plan_from([rec])

        step = plan.steps[0]
        assert step.title == "Fill salary"
        assert step.recommendation is rec
        assert step.enabled is True
        assert step.index == 1

    def test_multiple_fill_recommendations_indexed_sequentially(self) -> None:
        plan = _plan_from(
            [
                _make_recommendation(RecommendationAction.FILL_MEDIAN, column="a"),
                _make_recommendation(RecommendationAction.FILL_MODE, column="b"),
            ]
        )

        assert [s.index for s in plan.steps] == [1, 2]
        assert [s.operation.subset for s in plan.steps] == [["a"], ["b"]]

    def test_fill_without_column_produces_no_step(self) -> None:
        """A FILL_* recommendation missing a column cannot be constructed."""
        plan = _plan_from(
            [_make_recommendation(RecommendationAction.FILL_MEDIAN, column=None)]
        )

        assert plan.steps == []


# ---------------------------------------------------------------------------
# CONVERT_TYPE mappings (Requirement 2.2)
# ---------------------------------------------------------------------------


class TestConvertTypeMappings:
    def test_convert_type_with_target_dtype_maps_to_coercion(self) -> None:
        plan = _plan_from(
            [
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE,
                    column="age",
                    target_dtype="int64",
                )
            ]
        )

        assert len(plan.steps) == 1
        op = plan.steps[0].operation
        assert isinstance(op, TypeCoercionOperation)
        assert op.target_dtypes == {"age": "int64"}
        assert op.columns == ["age"]

    def test_convert_type_without_target_dtype_produces_no_step(self) -> None:
        """Unresolvable dtype (None) must create no coercion step."""
        plan = _plan_from(
            [
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE,
                    column="age",
                    target_dtype=None,
                )
            ]
        )

        assert plan.steps == []

    def test_convert_type_with_empty_target_dtype_produces_no_step(self) -> None:
        """An empty-string dtype is falsy and unresolvable -> no step."""
        plan = _plan_from(
            [
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE,
                    column="age",
                    target_dtype="",
                )
            ]
        )

        assert plan.steps == []

    def test_convert_type_without_column_produces_no_step(self) -> None:
        plan = _plan_from(
            [
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE,
                    column=None,
                    target_dtype="int64",
                )
            ]
        )

        assert plan.steps == []


# ---------------------------------------------------------------------------
# No orphaned steps / mixed recommendations
# ---------------------------------------------------------------------------


class TestNoOrphanedSteps:
    def test_disabled_recommendation_is_skipped(self) -> None:
        plan = _plan_from(
            [
                _make_recommendation(
                    RecommendationAction.FILL_MEDIAN, column="age", enabled=False
                )
            ]
        )

        assert plan.steps == []

    def test_unconstructible_recommendations_produce_no_steps(self) -> None:
        """Actions that need a column/dtype but lack them are dropped."""
        plan = _plan_from(
            [
                _make_recommendation(RecommendationAction.DROP_COLUMN, column=None),
                _make_recommendation(RecommendationAction.DROP_ROWS, column=None),
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE, column="age", target_dtype=None
                ),
            ]
        )

        assert plan.steps == []

    def test_mixed_recommendations_only_constructible_become_steps(self) -> None:
        plan = _plan_from(
            [
                # constructible
                _make_recommendation(RecommendationAction.FILL_MEAN, column="a"),
                # not constructible (no dtype)
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE, column="b", target_dtype=None
                ),
                # constructible
                _make_recommendation(
                    RecommendationAction.CONVERT_TYPE, column="c", target_dtype="int64"
                ),
                # constructible (no column needed)
                _make_recommendation(
                    RecommendationAction.REMOVE_DUPLICATES, column=None
                ),
                # constructible
                _make_recommendation(RecommendationAction.DROP_COLUMN, column="d"),
            ]
        )

        ops = [step.operation for step in plan.steps]
        assert len(ops) == 4
        assert isinstance(ops[0], FillMissing)
        assert isinstance(ops[1], TypeCoercionOperation)
        assert isinstance(ops[2], DropDuplicates)
        assert isinstance(ops[3], DropColumns)
        # indices are contiguous with no gaps for the dropped recommendation
        assert [step.index for step in plan.steps] == [1, 2, 3, 4]

    def test_empty_recommendations_produces_empty_plan(self) -> None:
        plan = _plan_from([])
        assert plan.steps == []


# ===========================================================================
# Serialization / replay tests — tasks 17.4 and 17.5
# ===========================================================================
#
# Task 17.4 (property): **Property 16: Replay produces equivalent results**
#   **Validates: Requirements 13.2**
#
# Task 17.5 (unit) — Requirements: 13.1, 13.3, 13.4, 13.5
#   serialize/deserialize round-trips for empty, single-step, and multi-step
#   plans (only ENABLED steps are serialized), plus preservation of audit-log
#   ordering across an original-vs-replayed run.


def _plan_from_operations(operations: list[object]) -> CleaningPlan:
    """Build a CleaningPlan of enabled steps from live operations."""
    steps = [
        PlanStep(index=i, title=op.name, operation=op, enabled=True)
        for i, op in enumerate(operations, start=1)
    ]
    return CleaningPlan(steps=steps)


@st.composite
def _replayable_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """Draw a small DataFrame with numeric and string columns and some nulls."""
    n_rows = draw(st.integers(min_value=1, max_value=8))
    numeric = draw(
        st.lists(
            st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    text = draw(
        st.lists(
            st.sampled_from(["alpha", "beta", "gamma", "alpha", "n/a"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    # Force a float dtype for the numeric column so statistical fill strategies
    # apply even when every value is null. An all-None object column would be
    # rejected by FillMissing's own preconditions, which is unrelated to the
    # replay-equivalence property under test here.
    return pd.DataFrame({"a": pd.Series(numeric, dtype="float64"), "b": text})


@st.composite
def _equivalence_plans(draw: st.DrawFn) -> CleaningPlan:
    """Draw a plan of deterministic operations applicable to the test frame."""
    operations: list[object] = []
    count = draw(st.integers(min_value=1, max_value=3))
    for _ in range(count):
        kind = draw(
            st.sampled_from(
                ["fill_a", "fill_b_mode", "drop_duplicates", "missing_token"]
            )
        )
        if kind == "fill_a":
            operations.append(
                FillMissing(
                    strategy=draw(st.sampled_from(["median", "mean"])), subset=["a"]
                )
            )
        elif kind == "fill_b_mode":
            operations.append(FillMissing(strategy="mode", subset=["b"]))
        elif kind == "drop_duplicates":
            operations.append(
                DropDuplicates(keep=draw(st.sampled_from(["first", "last"])))
            )
        else:  # missing_token
            from sanitizepy.cleaning.missing_tokens import MissingTokenOperation

            operations.append(MissingTokenOperation(subset=["b"]))
    return _plan_from_operations(operations)


class TestReplayEquivalenceProperty:
    """**Property 16: Replay produces equivalent results**

    **Validates: Requirements 13.2**

    Applying a plan, then serializing + deserializing that plan and applying
    the reconstruction to the same DataFrame yields an equivalent DataFrame.
    """

    @settings(max_examples=75, deadline=None)
    @given(plan=_equivalence_plans(), dataframe=_replayable_dataframe())
    def test_replayed_plan_produces_equivalent_frame(
        self, plan: CleaningPlan, dataframe: pd.DataFrame
    ) -> None:
        original_result = plan.apply(dataframe.copy())

        reconstructed = CleaningPlan.deserialize(
            ReplayablePlan.from_json(plan.serialize().to_json())
        )
        replayed_result = reconstructed.apply(dataframe.copy())

        pd.testing.assert_frame_equal(original_result.data, replayed_result.data)


class TestPlanSerializeDeserialize:
    """Unit coverage for ``CleaningPlan.serialize`` / ``deserialize`` (13.1, 13.5)."""

    def test_empty_plan_serializes_to_empty_replayable_plan(self) -> None:
        plan = CleaningPlan(steps=[])
        replayable = plan.serialize()

        assert replayable.operations == ()
        assert replayable.version == VERSION
        assert replayable.seeds == {}

        reconstructed = CleaningPlan.deserialize(replayable)
        assert reconstructed.steps == []

    def test_single_step_round_trip(self) -> None:
        op = FillMissing(strategy="median", subset=["age"])
        plan = _plan_from_operations([op])

        reconstructed = CleaningPlan.deserialize(plan.serialize())

        assert len(reconstructed.steps) == 1
        rebuilt = reconstructed.steps[0].operation
        assert isinstance(rebuilt, FillMissing)
        assert rebuilt.strategy == "median"
        assert rebuilt.subset == ["age"]

    def test_multi_step_round_trip_preserves_order_and_params(self) -> None:
        plan = _plan_from_operations(
            [
                FillMissing(strategy="mean", subset=["a"]),
                TypeCoercionOperation(
                    target_dtypes={"a": "int64"}, error_policy="coerce"
                ),
                DropDuplicates(keep="last"),
            ]
        )

        replayable = plan.serialize()
        assert [op.name for op in replayable.operations] == [
            "fill_missing",
            "type_coercion",
            "drop_duplicates",
        ]

        reconstructed = CleaningPlan.deserialize(
            ReplayablePlan.from_json(replayable.to_json())
        )
        ops = [step.operation for step in reconstructed.steps]
        assert isinstance(ops[0], FillMissing)
        assert ops[0].strategy == "mean"
        assert isinstance(ops[1], TypeCoercionOperation)
        assert ops[1].target_dtypes == {"a": "int64"}
        assert ops[1].error_policy == "coerce"
        assert isinstance(ops[2], DropDuplicates)
        assert ops[2].keep == "last"

    def test_only_enabled_steps_are_serialized(self) -> None:
        enabled = FillMissing(strategy="mean", subset=["a"])
        disabled = DropColumns(columns=["b"])
        plan = CleaningPlan(
            steps=[
                PlanStep(index=1, title="keep", operation=enabled, enabled=True),
                PlanStep(index=2, title="skip", operation=disabled, enabled=False),
            ]
        )

        replayable = plan.serialize()

        assert [op.name for op in replayable.operations] == ["fill_missing"]

    def test_unknown_operation_name_raises_deserialization_error(self) -> None:
        plan = ReplayablePlan(
            operations=(ReplayOperation(name="not_registered", parameters={}),)
        )
        with pytest.raises(DeserializationError, match="Unknown operation"):
            CleaningPlan.deserialize(plan)

    def test_bad_parameters_raise_deserialization_error(self) -> None:
        # DropColumns requires a non-empty ``columns`` list; an empty list is
        # rejected by its constructor and surfaces as DeserializationError.
        plan = ReplayablePlan(
            operations=(
                ReplayOperation(name="drop_columns", parameters={"columns": []}),
            )
        )
        with pytest.raises(DeserializationError):
            CleaningPlan.deserialize(plan)


class TestReplayAuditOrdering:
    """Audit ordering is preserved across original and replayed runs (13.2)."""

    def test_audit_log_order_matches_between_original_and_replay(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, None, 3, 3],
                "b": ["x", "y", "z", "z"],
            }
        )
        plan = _plan_from_operations(
            [
                FillMissing(strategy="median", subset=["a"]),
                DropDuplicates(keep="first"),
            ]
        )

        original = plan.apply(df.copy())
        reconstructed = CleaningPlan.deserialize(plan.serialize())
        replayed = reconstructed.apply(df.copy())

        original_sequence = [
            (entry["order"], entry["operation"]) for entry in original.audit_log
        ]
        replayed_sequence = [
            (entry["order"], entry["operation"]) for entry in replayed.audit_log
        ]

        assert original_sequence == [(1, "fill_missing"), (2, "drop_duplicates")]
        assert replayed_sequence == original_sequence
