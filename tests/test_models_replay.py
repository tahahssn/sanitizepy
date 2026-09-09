"""
Tests for the replayable-plan model and plan serialization round-trip.

This module contains:

1. Property test (task 17.3)
   **Property 15: Plan serialization round-trips**
   **Validates: Requirements 13.1, 13.5**

   A ``CleaningPlan`` built from built-in operations serializes to a
   ``ReplayablePlan``, survives a ``to_json()`` / ``from_json()`` JSON round
   trip, and deserializes back to a plan whose operations match the original
   (same names and same ``describe()`` reconstruction parameters).

2. Unit tests (task 17.5) — Requirements: 13.1, 13.3, 13.4, 13.5
   Covers the ``ReplayablePlan`` / ``ReplayOperation`` model directly: JSON
   round-trip, ``SerializationError`` on a non-serializable parameter,
   ``DeserializationError`` on malformed JSON and on an unknown operation name
   (via ``CleaningPlan.deserialize``), and seed persistence across the JSON
   boundary.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.encoding import EncodingRepairOperation
from sanitizepy.cleaning.missing_tokens import MissingTokenOperation
from sanitizepy.cleaning.near_duplicates import NearDuplicateRemovalOperation
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    FillMissing,
)
from sanitizepy.cleaning.plan import CleaningPlan, PlanStep
from sanitizepy.cleaning.text_normalization import TextNormalizationOperation
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation
from sanitizepy.exceptions import DeserializationError, SerializationError
from sanitizepy.models.replay import ReplayablePlan, ReplayOperation
from sanitizepy.version import VERSION

# ===========================================================================
# Shared builders / strategies
# ===========================================================================

# A pool of column names used by generated operations.
_COLUMNS = ["a", "b", "c", "age", "name"]


@st.composite
def _built_in_operations(draw: st.DrawFn) -> object:
    """
    Draw a single built-in :class:`CleaningOperation` with varied parameters.

    Only deterministic, reconstructable operations are drawn so that
    ``serialize()`` -> JSON -> ``from_json()`` -> ``deserialize()`` produces an
    operation whose ``describe()`` matches the original.
    """
    kind = draw(
        st.sampled_from(
            [
                "fill",
                "drop_duplicates",
                "drop_columns",
                "type_coercion",
                "missing_token",
                "text_normalization",
                "encoding",
                "near_duplicate",
            ]
        )
    )
    columns = draw(
        st.lists(st.sampled_from(_COLUMNS), min_size=1, max_size=3, unique=True)
    )

    if kind == "fill":
        strategy = draw(st.sampled_from(["median", "mean", "mode", "constant"]))
        return FillMissing(strategy=strategy, subset=columns)
    if kind == "drop_duplicates":
        keep = draw(st.sampled_from(["first", "last", False]))
        return DropDuplicates(keep=keep, subset=columns)
    if kind == "drop_columns":
        return DropColumns(columns=columns)
    if kind == "type_coercion":
        dtype = draw(st.sampled_from(["int64", "float64", "string"]))
        policy = draw(st.sampled_from(["raise", "coerce"]))
        return TypeCoercionOperation(
            target_dtypes=dict.fromkeys(columns, dtype),
            error_policy=policy,
        )
    if kind == "missing_token":
        extra = draw(
            st.lists(st.sampled_from(["x", "y", "??"]), max_size=2, unique=True)
        )
        return MissingTokenOperation(extra_tokens=set(extra) or None, subset=columns)
    if kind == "text_normalization":
        return TextNormalizationOperation(
            subset=columns,
            unicode_form=draw(st.sampled_from(["NFC", "NFKC", "none"])),
            normalize_whitespace=draw(st.booleans()),
            case=draw(st.sampled_from(["lower", "upper", "title", "none"])),
        )
    if kind == "encoding":
        return EncodingRepairOperation(
            subset=columns,
            mode=draw(st.sampled_from(["core", "advanced"])),
            error_on_unrepaired=draw(st.booleans()),
        )
    # near_duplicate
    return NearDuplicateRemovalOperation(
        subset=columns,
        keep=draw(st.sampled_from(["first", "last"])),
    )


@st.composite
def _cleaning_plans(draw: st.DrawFn) -> CleaningPlan:
    """Draw a :class:`CleaningPlan` composed of enabled built-in operations."""
    operations = draw(st.lists(_built_in_operations(), min_size=1, max_size=5))
    steps = [
        PlanStep(
            index=index,
            title=operation.name,
            operation=operation,
            enabled=True,
        )
        for index, operation in enumerate(operations, start=1)
    ]
    return CleaningPlan(steps=steps)


def _operation_signature(plan: CleaningPlan) -> list[tuple[str, dict[str, object]]]:
    """Return (name, reconstruction-parameters) pairs for a plan's steps."""
    return [
        (
            step.operation.name,
            CleaningPlan._reconstruction_parameters(step.operation),
        )
        for step in plan.steps
    ]


# ===========================================================================
# Property test (task 17.3) — Property 15: Plan serialization round-trips
# ===========================================================================


class TestPlanSerializationRoundTripProperty:
    """**Property 15: Plan serialization round-trips**

    **Validates: Requirements 13.1, 13.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(plan=_cleaning_plans())
    def test_serialize_json_deserialize_preserves_operations(
        self, plan: CleaningPlan
    ) -> None:
        original_signature = _operation_signature(plan)

        # serialize() -> ReplayablePlan
        replayable = plan.serialize()
        assert isinstance(replayable, ReplayablePlan)

        # to_json() / from_json() round-trip
        restored_plan = ReplayablePlan.from_json(replayable.to_json())
        assert restored_plan == replayable

        # deserialize() back to a CleaningPlan
        reconstructed = CleaningPlan.deserialize(restored_plan)

        # Same operation names, in the same order.
        assert [name for name, _ in _operation_signature(reconstructed)] == [
            name for name, _ in original_signature
        ]

        # Same describe()-derived reconstruction parameters per step.
        assert _operation_signature(reconstructed) == original_signature


# ===========================================================================
# Unit tests (task 17.5) — ReplayablePlan / ReplayOperation model
# ===========================================================================


class TestReplayOperationModel:
    def test_defaults_to_empty_parameters(self) -> None:
        op = ReplayOperation(name="fill_missing")
        assert op.name == "fill_missing"
        assert op.parameters == {}

    def test_stores_parameters(self) -> None:
        op = ReplayOperation(name="drop_columns", parameters={"columns": ["a", "b"]})
        assert op.parameters == {"columns": ["a", "b"]}


class TestReplayablePlanJsonRoundTrip:
    def test_defaults(self) -> None:
        plan = ReplayablePlan()
        assert plan.version == VERSION
        assert plan.operations == ()
        assert plan.configuration == {}
        assert plan.seeds == {}

    def test_json_round_trip_preserves_all_fields(self) -> None:
        plan = ReplayablePlan(
            version="1.2.3",
            operations=(
                ReplayOperation(name="fill_missing", parameters={"strategy": "mean"}),
                ReplayOperation(name="drop_columns", parameters={"columns": ["a"]}),
            ),
            configuration={"chunk_size": 100},
            seeds={"near_duplicate_removal": 42},
        )

        restored = ReplayablePlan.from_json(plan.to_json())

        assert restored == plan
        assert restored.version == "1.2.3"
        assert restored.operations[0].name == "fill_missing"
        assert restored.operations[0].parameters == {"strategy": "mean"}
        assert restored.configuration == {"chunk_size": 100}

    def test_empty_plan_json_round_trip(self) -> None:
        plan = ReplayablePlan()
        assert ReplayablePlan.from_json(plan.to_json()) == plan


class TestReplayablePlanSeedPersistence:
    """Requirement 13.4: seeds survive the JSON boundary."""

    def test_seeds_persist_through_json(self) -> None:
        plan = ReplayablePlan(
            operations=(ReplayOperation(name="near_duplicate_removal"),),
            seeds={"near_duplicate_removal": 7},
        )

        restored = ReplayablePlan.from_json(plan.to_json())

        assert restored.seeds == {"near_duplicate_removal": 7}

    def test_multiple_seeds_persist(self) -> None:
        plan = ReplayablePlan(seeds={"op_a": 1, "op_b": 2, "op_c": 3})
        restored = ReplayablePlan.from_json(plan.to_json())
        assert restored.seeds == {"op_a": 1, "op_b": 2, "op_c": 3}


class TestReplayablePlanSerializationErrors:
    def test_non_serializable_parameter_raises_serialization_error(self) -> None:
        plan = ReplayablePlan(
            operations=(ReplayOperation(name="custom", parameters={"bad": {1, 2, 3}}),),
        )
        with pytest.raises(SerializationError):
            plan.to_json()

    def test_non_serializable_configuration_raises_serialization_error(self) -> None:
        plan = ReplayablePlan(configuration={"bad": object()})
        with pytest.raises(SerializationError):
            plan.to_json()


class TestReplayablePlanDeserializationErrors:
    def test_malformed_json_raises_deserialization_error(self) -> None:
        with pytest.raises(DeserializationError):
            ReplayablePlan.from_json("{not valid json")

    def test_schema_mismatch_raises_deserialization_error(self) -> None:
        # ``operations`` must be a list of objects, not a bare string.
        with pytest.raises(DeserializationError):
            ReplayablePlan.from_json('{"operations": "nope"}')

    def test_unknown_operation_name_raises_via_plan_deserialize(self) -> None:
        """Requirement 13.3: unknown operation names are rejected."""
        plan = ReplayablePlan(
            operations=(ReplayOperation(name="does_not_exist", parameters={}),),
        )
        with pytest.raises(DeserializationError, match="Unknown operation"):
            CleaningPlan.deserialize(plan)
