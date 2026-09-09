"""
Unit tests for sanitizepy.models.contracts

Covers the declarative ``ColumnContract`` and ``DataContract`` pydantic
models: default (unset) expectations, explicit expectations, frozen /
immutable semantics, forbidden extra fields, and the column mapping.

These models are purely declarative -- they carry no validation logic --
so the tests here assert construction, defaults, and immutability only.
The behavioural validation of contracts against DataFrames lives in
``test_rules.py``.

Validates: Requirements 9.1
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sanitizepy.models.contracts import ColumnContract, DataContract

# ---------------------------------------------------------------------------
# ColumnContract
# ---------------------------------------------------------------------------


class TestColumnContractDefaults:
    def test_all_fields_default_to_none(self):
        contract = ColumnContract()
        assert contract.dtype is None
        assert contract.nullable is None
        assert contract.allowed_values is None
        assert contract.min_value is None
        assert contract.max_value is None
        assert contract.regex is None
        assert contract.unique is None

    def test_partial_declaration_leaves_others_none(self):
        contract = ColumnContract(dtype="int64")
        assert contract.dtype == "int64"
        assert contract.nullable is None
        assert contract.allowed_values is None


class TestColumnContractExplicitValues:
    def test_dtype_set(self):
        assert ColumnContract(dtype="string").dtype == "string"

    def test_nullable_set(self):
        assert ColumnContract(nullable=False).nullable is False

    def test_allowed_values_set(self):
        contract = ColumnContract(allowed_values=("a", "b", "c"))
        assert contract.allowed_values == ("a", "b", "c")

    def test_range_set(self):
        contract = ColumnContract(min_value=0.0, max_value=10.0)
        assert contract.min_value == 0.0
        assert contract.max_value == 10.0

    def test_regex_set(self):
        assert ColumnContract(regex=r"^\d+$").regex == r"^\d+$"

    def test_unique_set(self):
        assert ColumnContract(unique=True).unique is True


class TestColumnContractImmutability:
    def test_is_frozen(self):
        contract = ColumnContract(dtype="int64")
        with pytest.raises(ValidationError):
            contract.dtype = "float64"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ColumnContract(unknown_field="value")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DataContract
# ---------------------------------------------------------------------------


class TestDataContract:
    def test_default_columns_empty(self):
        contract = DataContract()
        assert contract.columns == {}

    def test_columns_mapping(self):
        contract = DataContract(
            columns={
                "age": ColumnContract(dtype="int64", min_value=0.0),
                "name": ColumnContract(nullable=False),
            }
        )
        assert set(contract.columns) == {"age", "name"}
        assert contract.columns["age"].dtype == "int64"
        assert contract.columns["name"].nullable is False

    def test_is_frozen(self):
        contract = DataContract(columns={"a": ColumnContract()})
        with pytest.raises(ValidationError):
            contract.columns = {}  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            DataContract(unknown_field="value")  # type: ignore[call-arg]
