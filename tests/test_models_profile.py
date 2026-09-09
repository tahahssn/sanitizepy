"""
Unit tests for sanitizepy.models.profile

Covers the immutable ``DatasetProfile`` aggregate and the ``TextQualityResult``
model: construction from real inspector results, frozen/immutable semantics,
and the forward-compatible optional slots.

These tests live in a distinct module (``test_models_profile.py``) and use
distinctly named test classes so they never collide with the property tests
that task 11.4 adds to ``test_inspection_profile.py``.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from sanitizepy.inspection.datatypes import (
    DatatypeInspectionResult,
    DatatypeInspector,
)
from sanitizepy.inspection.duplicates import (
    DuplicateInspectionResult,
    DuplicateInspector,
)
from sanitizepy.inspection.memory import (
    MemoryInspectionResult,
    MemoryInspector,
)
from sanitizepy.inspection.missing import (
    MissingInspectionResult,
    MissingValueInspector,
)
from sanitizepy.inspection.statistics import (
    StatisticsInspectionResult,
    StatisticsInspector,
)
from sanitizepy.models.profile import DatasetProfile, TextQualityResult


def _build_profile(dataframe: pd.DataFrame) -> DatasetProfile:
    """Assemble a profile directly from the standalone inspectors."""

    return DatasetProfile(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        datatypes=DatatypeInspector().inspect(dataframe),
        missing_values=MissingValueInspector().inspect(dataframe),
        duplicates=DuplicateInspector().inspect(dataframe),
        memory=MemoryInspector().inspect(dataframe),
        statistics=StatisticsInspector().inspect(dataframe),
    )


def _sample_text_quality() -> TextQualityResult:
    return TextQualityResult(
        column="notes",
        non_null_count=3,
        empty_after_strip_count=1,
        character_length_min=0,
        character_length_max=10,
        character_length_mean=5.0,
        character_length_median=5.0,
        token_count_min=0,
        token_count_max=2,
        token_count_mean=1.0,
        token_count_median=1.0,
        boilerplate_count=0,
        encoding_garbage_count=0,
    )


class TestDatasetProfileModelConstruction:
    def setup_method(self):
        self.df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "text": ["a", "b", "c"],
            }
        )
        self.profile = _build_profile(self.df)

    def test_is_dataset_profile_instance(self):
        assert isinstance(self.profile, DatasetProfile)

    def test_row_and_column_counts(self):
        assert self.profile.row_count == 3
        assert self.profile.column_count == 2

    def test_composes_datatype_result(self):
        assert isinstance(self.profile.datatypes, DatatypeInspectionResult)

    def test_composes_missing_result(self):
        assert isinstance(self.profile.missing_values, MissingInspectionResult)

    def test_composes_duplicate_result(self):
        assert isinstance(self.profile.duplicates, DuplicateInspectionResult)

    def test_composes_memory_result(self):
        assert isinstance(self.profile.memory, MemoryInspectionResult)

    def test_composes_statistics_result(self):
        assert isinstance(self.profile.statistics, StatisticsInspectionResult)

    def test_optional_slots_default_to_none(self):
        assert self.profile.text_quality is None
        assert self.profile.anomalies is None
        assert self.profile.near_duplicate is None


class TestDatasetProfileModelImmutability:
    def setup_method(self):
        self.profile = _build_profile(pd.DataFrame({"num": [1, 2, 3]}))

    def test_cannot_reassign_row_count(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.profile.row_count = 99  # type: ignore[misc]

    def test_cannot_reassign_datatypes(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.profile.datatypes = None  # type: ignore[assignment,misc]

    def test_cannot_reassign_optional_slot(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.profile.anomalies = ()  # type: ignore[misc]

    def test_has_no_dict_due_to_slots(self):
        assert not hasattr(self.profile, "__dict__")


class TestDatasetProfileModelOptionalSlots:
    def setup_method(self):
        self.df = pd.DataFrame({"text": ["a", "b", "c"]})

    def test_text_quality_slot_holds_tuple(self):
        tq = _sample_text_quality()
        profile = DatasetProfile(
            row_count=3,
            column_count=1,
            datatypes=DatatypeInspector().inspect(self.df),
            missing_values=MissingValueInspector().inspect(self.df),
            duplicates=DuplicateInspector().inspect(self.df),
            memory=MemoryInspector().inspect(self.df),
            statistics=StatisticsInspector().inspect(self.df),
            text_quality=(tq,),
        )
        assert profile.text_quality == (tq,)
        assert profile.text_quality[0].column == "notes"

    def test_anomalies_and_near_duplicate_accept_values(self):
        profile = DatasetProfile(
            row_count=3,
            column_count=1,
            datatypes=DatatypeInspector().inspect(self.df),
            missing_values=MissingValueInspector().inspect(self.df),
            duplicates=DuplicateInspector().inspect(self.df),
            memory=MemoryInspector().inspect(self.df),
            statistics=StatisticsInspector().inspect(self.df),
            anomalies=("anomaly-placeholder",),
            near_duplicate="near-dup-placeholder",
        )
        assert profile.anomalies == ("anomaly-placeholder",)
        assert profile.near_duplicate == "near-dup-placeholder"


class TestTextQualityResultModel:
    def setup_method(self):
        self.result = _sample_text_quality()

    def test_is_text_quality_result_instance(self):
        assert isinstance(self.result, TextQualityResult)

    def test_fields_are_preserved(self):
        assert self.result.column == "notes"
        assert self.result.non_null_count == 3
        assert self.result.empty_after_strip_count == 1
        assert self.result.character_length_max == 10
        assert self.result.token_count_mean == 1.0
        assert self.result.boilerplate_count == 0
        assert self.result.encoding_garbage_count == 0

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.result.column = "other"  # type: ignore[misc]

    def test_has_no_dict_due_to_slots(self):
        assert not hasattr(self.result, "__dict__")

    def test_is_convertible_via_asdict(self):
        data = dataclasses.asdict(self.result)
        assert data["column"] == "notes"
        assert data["character_length_mean"] == 5.0
