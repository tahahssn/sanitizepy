"""
Tests for sanitizepy.cleaning.base (CleaningOperation)
"""

from __future__ import annotations

import pandas as pd
import pytest

from sanitizepy.cleaning.base import CleaningOperation
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)


class _ConcreteOp(CleaningOperation):
    name = "concrete_op"

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        return dataframe.copy()


class TestCleaningOperationBase:
    def test_apply_accepts_dataframe(self):
        op = _ConcreteOp()
        df = pd.DataFrame({"a": [1, 2]})
        result = op.apply(df)
        assert isinstance(result, pd.DataFrame)

    def test_apply_raises_on_non_dataframe(self):
        op = _ConcreteOp()
        with pytest.raises(TypeError, match="DataFrame"):
            op.apply([1, 2, 3])  # type: ignore[arg-type]

    def test_apply_raises_on_dict(self):
        op = _ConcreteOp()
        with pytest.raises(TypeError):
            op.apply({"a": [1, 2]})  # type: ignore[arg-type]

    def test_describe_returns_dict_with_name(self):
        op = _ConcreteOp()
        desc = op.describe()
        assert isinstance(desc, dict)
        assert desc["name"] == "concrete_op"


class TestClassificationAttributeDefaults:
    """
    Classification attributes (``is_chunk_safe`` / ``is_inplace_safe``) must
    default to ``False`` so operations behave conservatively unless a subclass
    explicitly opts in (Requirements 6.1, 15.2).
    """

    def test_abc_defaults_are_false(self):
        assert CleaningOperation.is_chunk_safe is False
        assert CleaningOperation.is_inplace_safe is False

    def test_concrete_subclass_inherits_false_defaults(self):
        # Class-level access.
        assert _ConcreteOp.is_chunk_safe is False
        assert _ConcreteOp.is_inplace_safe is False
        # Instance-level access.
        op = _ConcreteOp()
        assert op.is_chunk_safe is False
        assert op.is_inplace_safe is False

    @pytest.mark.parametrize(
        "operation",
        [
            DropMissingRows(),
            DropMissingColumns(),
            FillMissing(value=0),
            DropDuplicates(),
            DropColumns(columns=["a"]),
        ],
    )
    def test_existing_operations_default_to_false(
        self, operation: CleaningOperation
    ) -> None:
        assert operation.is_chunk_safe is False
        assert operation.is_inplace_safe is False

    @pytest.mark.parametrize(
        "operation_cls",
        [
            DropMissingRows,
            DropMissingColumns,
            FillMissing,
            DropDuplicates,
            DropColumns,
        ],
    )
    def test_existing_operation_classes_default_to_false(
        self, operation_cls: type[CleaningOperation]
    ) -> None:
        # Verify defaults at the class level too, independent of construction.
        assert operation_cls.is_chunk_safe is False
        assert operation_cls.is_inplace_safe is False

    def test_attributes_are_booleans(self):
        assert isinstance(CleaningOperation.is_chunk_safe, bool)
        assert isinstance(CleaningOperation.is_inplace_safe, bool)


class TestClassificationAttributeOverrides:
    """A subclass may override the flags independently of one another."""

    def test_subclass_overriding_only_chunk_safe(self):
        class _ChunkOnly(CleaningOperation):
            name = "chunk_only"
            is_chunk_safe = True

            def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
                return dataframe.copy()

        assert _ChunkOnly.is_chunk_safe is True
        # The other flag remains at the inherited default.
        assert _ChunkOnly.is_inplace_safe is False

    def test_subclass_overriding_only_inplace_safe(self):
        class _InplaceOnly(CleaningOperation):
            name = "inplace_only"
            is_inplace_safe = True

            def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
                return dataframe.copy()

        assert _InplaceOnly.is_inplace_safe is True
        # The other flag remains at the inherited default.
        assert _InplaceOnly.is_chunk_safe is False

    def test_override_does_not_leak_to_sibling_or_base(self):
        class _BothTrue(CleaningOperation):
            name = "both_true"
            is_chunk_safe = True
            is_inplace_safe = True

            def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
                return dataframe.copy()

        assert _BothTrue.is_chunk_safe is True
        assert _BothTrue.is_inplace_safe is True
        # Base defaults are untouched by the override.
        assert CleaningOperation.is_chunk_safe is False
        assert CleaningOperation.is_inplace_safe is False
        # A separate conservative subclass is unaffected.
        assert _ConcreteOp.is_chunk_safe is False
        assert _ConcreteOp.is_inplace_safe is False


class TestAbstractInstantiationGuard:
    """The ABC cannot be instantiated directly; ``apply`` is abstract."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            CleaningOperation()  # type: ignore[abstract]

    def test_subclass_without_apply_cannot_instantiate(self):
        class _Incomplete(CleaningOperation):
            name = "incomplete"

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]
