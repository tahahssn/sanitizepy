"""
Tests for cleaner.cleaning.base (CleaningOperation)
"""

from __future__ import annotations

import pandas as pd
import pytest

from cleaner.cleaning.base import CleaningOperation


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
