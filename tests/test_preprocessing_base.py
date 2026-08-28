"""
Tests for sanitizepy.preprocessing.base (FeatureOperation)
"""

from __future__ import annotations

import pandas as pd
import pytest

from sanitizepy.preprocessing.base import FeatureOperation


class _ConcreteFeatureOp(FeatureOperation):
    name = "concrete_feature_op"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return data.copy()


class TestFeatureOperationBase:
    def test_initial_is_fitted_false(self):
        op = _ConcreteFeatureOp()
        assert op.is_fitted is False

    def test_fit_sets_is_fitted_true(self):
        op = _ConcreteFeatureOp()
        df = pd.DataFrame({"a": [1, 2]})
        op.fit(df)
        assert op.is_fitted is True

    def test_fit_returns_self(self):
        op = _ConcreteFeatureOp()
        df = pd.DataFrame({"a": [1, 2]})
        result = op.fit(df)
        assert result is op

    def test_transform_requires_fitting(self):
        op = _ConcreteFeatureOp()
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(RuntimeError, match="fitted"):
            op.transform(df)

    def test_transform_after_fit_succeeds(self):
        op = _ConcreteFeatureOp()
        df = pd.DataFrame({"a": [1, 2]})
        op.fit(df)
        result = op.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_returns_dataframe(self):
        op = _ConcreteFeatureOp()
        df = pd.DataFrame({"a": [1, 2]})
        result = op.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_validate_input_rejects_non_dataframe(self):
        op = _ConcreteFeatureOp()
        with pytest.raises(TypeError):
            op.fit([1, 2, 3])  # type: ignore[arg-type]

    def test_get_params_returns_empty_dict_by_default(self):
        op = _ConcreteFeatureOp()
        assert op.get_params() == {}

    def test_repr_contains_name(self):
        op = _ConcreteFeatureOp()
        r = repr(op)
        assert "concrete_feature_op" in r

    def test_repr_contains_fitted_state(self):
        op = _ConcreteFeatureOp()
        assert "False" in repr(op)
        op.fit(pd.DataFrame({"a": [1]}))
        assert "True" in repr(op)
