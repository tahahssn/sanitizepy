"""
Unit tests for sanitizepy.models.recommendations — task 3.2.

Covers:
- ``target_dtype`` defaults to ``None`` (additive field from task 3.1)
- Existing ``Recommendation`` construction is unaffected
- JSON / dict serialization round-trips preserve ``target_dtype``
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sanitizepy.models.recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationCategory,
    RecommendationGroup,
    RecommendationImpact,
    RecommendationPriority,
    RecommendationReason,
    RecommendationResult,
    RecommendationSummary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reason(**kwargs: object) -> RecommendationReason:
    defaults: dict[str, object] = {
        "title": "High missing rate",
        "explanation": "More than 50 % of values are missing.",
    }
    defaults.update(kwargs)
    return RecommendationReason(**defaults)  # type: ignore[arg-type]


def _make_impact(**kwargs: object) -> RecommendationImpact:
    defaults: dict[str, object] = {
        "memory_change_mb": 0.0,
        "row_change": 0,
        "column_change": 0,
        "quality_score_delta": 0.0,
    }
    defaults.update(kwargs)
    return RecommendationImpact(**defaults)  # type: ignore[arg-type]


def _make_recommendation(**kwargs: object) -> Recommendation:
    """Return a minimal valid ``Recommendation``, overridable by *kwargs*."""
    defaults: dict[str, object] = {
        "title": "Fill missing values",
        "description": "Column 'age' has missing values.",
        "category": RecommendationCategory.MISSING_VALUES,
        "priority": RecommendationPriority.HIGH,
        "action": RecommendationAction.FILL_MEDIAN,
        "reason": _make_reason(),
        "impact": _make_impact(),
        "confidence": 0.9,
    }
    defaults.update(kwargs)
    return Recommendation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# target_dtype field — default is None (Requirement 2.2)
# ---------------------------------------------------------------------------


class TestTargetDtypeDefault:
    def test_default_is_none(self):
        rec = _make_recommendation()
        assert rec.target_dtype is None

    def test_default_is_none_for_fill_action(self):
        """Ensure the default holds for every FILL_* action."""
        for action in (
            RecommendationAction.FILL_MEAN,
            RecommendationAction.FILL_MEDIAN,
            RecommendationAction.FILL_MODE,
            RecommendationAction.FILL_CONSTANT,
        ):
            rec = _make_recommendation(action=action)
            assert rec.target_dtype is None

    def test_default_is_none_for_convert_type_action(self):
        """CONVERT_TYPE with no explicit target_dtype still defaults to None."""
        rec = _make_recommendation(action=RecommendationAction.CONVERT_TYPE)
        assert rec.target_dtype is None

    def test_explicit_none_accepted(self):
        rec = _make_recommendation(target_dtype=None)
        assert rec.target_dtype is None

    def test_string_dtype_accepted(self):
        rec = _make_recommendation(
            action=RecommendationAction.CONVERT_TYPE,
            target_dtype="int64",
        )
        assert rec.target_dtype == "int64"

    def test_various_dtype_strings_accepted(self):
        for dtype in ("float32", "datetime64[ns]", "category", "bool", "object"):
            rec = _make_recommendation(target_dtype=dtype)
            assert rec.target_dtype == dtype


# ---------------------------------------------------------------------------
# Existing construction is unaffected (Requirement 15.1)
# ---------------------------------------------------------------------------


class TestExistingConstructionUnaffected:
    """Pre-3.1 Recommendation construction must still work unchanged."""

    def test_minimal_construction_without_target_dtype(self):
        """Build without specifying target_dtype at all — must succeed."""
        rec = _make_recommendation()
        assert rec.title == "Fill missing values"
        assert rec.confidence == pytest.approx(0.9)

    def test_all_existing_fields_present(self):
        rec = _make_recommendation()
        # Fields that existed before task 3.1
        assert hasattr(rec, "title")
        assert hasattr(rec, "description")
        assert hasattr(rec, "category")
        assert hasattr(rec, "priority")
        assert hasattr(rec, "action")
        assert hasattr(rec, "column")
        assert hasattr(rec, "reason")
        assert hasattr(rec, "impact")
        assert hasattr(rec, "confidence")
        assert hasattr(rec, "automatic")
        assert hasattr(rec, "enabled")

    def test_column_defaults_to_none(self):
        rec = _make_recommendation()
        assert rec.column is None

    def test_automatic_defaults_to_false(self):
        rec = _make_recommendation()
        assert rec.automatic is False

    def test_enabled_defaults_to_true(self):
        rec = _make_recommendation()
        assert rec.enabled is True

    def test_model_is_frozen(self):
        rec = _make_recommendation()
        with pytest.raises(ValidationError):
            rec.title = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            _make_recommendation(nonexistent_field="value")

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            _make_recommendation(confidence=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            _make_recommendation(confidence=1.01)

    def test_confidence_boundary_zero_accepted(self):
        rec = _make_recommendation(confidence=0.0)
        assert rec.confidence == pytest.approx(0.0)

    def test_confidence_boundary_one_accepted(self):
        rec = _make_recommendation(confidence=1.0)
        assert rec.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Serialization round-trip (Requirement 2.2, 15.1)
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_model_dump_includes_target_dtype_none(self):
        rec = _make_recommendation()
        data = rec.model_dump()
        assert "target_dtype" in data
        assert data["target_dtype"] is None

    def test_model_dump_includes_target_dtype_value(self):
        rec = _make_recommendation(
            action=RecommendationAction.CONVERT_TYPE,
            target_dtype="float64",
        )
        data = rec.model_dump()
        assert data["target_dtype"] == "float64"

    def test_json_round_trip_none(self):
        rec = _make_recommendation()
        json_str = rec.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["target_dtype"] is None
        # model_validate_json handles strict-mode StrEnum deserialization correctly
        restored = Recommendation.model_validate_json(json_str)
        assert restored == rec

    def test_json_round_trip_with_value(self):
        rec = _make_recommendation(
            action=RecommendationAction.CONVERT_TYPE,
            target_dtype="int32",
        )
        json_str = rec.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["target_dtype"] == "int32"
        restored = Recommendation.model_validate_json(json_str)
        assert restored == rec

    def test_dict_round_trip_none(self):
        rec = _make_recommendation()
        data = rec.model_dump()
        # strict=False is required because model_dump serialises StrEnum fields to
        # plain strings; model_validate needs to coerce them back.
        restored = Recommendation.model_validate(data, strict=False)
        assert restored == rec

    def test_dict_round_trip_with_value(self):
        rec = _make_recommendation(
            action=RecommendationAction.CONVERT_TYPE,
            target_dtype="category",
        )
        data = rec.model_dump()
        restored = Recommendation.model_validate(data, strict=False)
        assert restored == rec

    def test_omitted_target_dtype_in_payload_treated_as_none(self):
        """Deserializing legacy payloads that have no target_dtype key."""
        rec = _make_recommendation()
        data = rec.model_dump()
        del data["target_dtype"]
        restored = Recommendation.model_validate(data, strict=False)
        assert restored.target_dtype is None

    def test_json_serializable_without_target_dtype(self):
        """Confirm model_dump_json produces valid JSON for the default case."""
        rec = _make_recommendation()
        json_str = rec.model_dump_json()
        # Must not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_json_serializable_with_target_dtype(self):
        rec = _make_recommendation(target_dtype="uint8")
        json_str = rec.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["target_dtype"] == "uint8"


# ---------------------------------------------------------------------------
# Sibling models are unaffected by the additive change (Requirement 15.1)
# ---------------------------------------------------------------------------


class TestSiblingModelsUnaffected:
    """
    RecommendationGroup, RecommendationSummary, and RecommendationResult
    must continue to construct and serialise correctly.
    """

    def test_recommendation_group_construction(self):
        group = RecommendationGroup(
            category=RecommendationCategory.DATA_TYPES,
            recommendations=[_make_recommendation()],
        )
        assert group.category == RecommendationCategory.DATA_TYPES
        assert len(group.recommendations) == 1

    def test_recommendation_summary_construction(self):
        summary = RecommendationSummary(
            total=3,
            critical=1,
            high=1,
            medium=1,
            low=0,
        )
        assert summary.total == 3

    def test_recommendation_result_round_trip(self):
        rec = _make_recommendation(
            action=RecommendationAction.CONVERT_TYPE,
            target_dtype="int64",
        )
        group = RecommendationGroup(
            category=RecommendationCategory.DATA_TYPES,
            recommendations=[rec],
        )
        summary = RecommendationSummary(
            total=1,
            critical=0,
            high=1,
            medium=0,
            low=0,
        )
        result = RecommendationResult(
            summary=summary,
            groups=[group],
            recommendations=[rec],
        )
        data = result.model_dump()
        restored = RecommendationResult.model_validate(data, strict=False)
        assert restored == result
        assert restored.recommendations[0].target_dtype == "int64"
