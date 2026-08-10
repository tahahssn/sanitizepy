"""
Base abstractions for feature engineering operations.

This module defines the contract that every feature engineering operation
must follow. Operations are intentionally stateful so that fitting can be
performed on training data and the learned state can later be reused on
validation/test data without data leakage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class FeatureOperation(ABC):
    """
    Abstract base class for feature engineering operations.

    A feature operation follows the standard fit/transform contract:

        operation.fit(X_train)
        X_train_transformed = operation.transform(X_train)
        X_test_transformed = operation.transform(X_test)

    This allows operations that learn information from data to preserve
    that state and apply the exact same transformation to future datasets.
    """

    name: str = "feature_operation"

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the operation has been fitted.
        """
        return self._is_fitted

    def fit(self, data: pd.DataFrame) -> FeatureOperation:
        """
        Fit the operation using the supplied dataframe.

        Subclasses should override this method when they need to learn
        parameters from the input data.

        Parameters
        ----------
        data:
            Input dataframe used to learn transformation state.

        Returns
        -------
        FeatureOperation
            The fitted operation.
        """
        self._validate_input(data)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataframe using the fitted operation.

        Parameters
        ----------
        data:
            Input dataframe to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed dataframe.
        """
        raise NotImplementedError

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the operation and transform the same dataframe.

        Parameters
        ----------
        data:
            Input dataframe used for fitting and transformation.

        Returns
        -------
        pandas.DataFrame
            Transformed dataframe.
        """
        self.fit(data)
        return self.transform(data)

    def get_params(self) -> dict[str, Any]:
        """
        Return configuration parameters for the operation.

        Subclasses can override this method when they expose additional
        configuration that should be serializable or inspectable.
        """
        return {}

    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate the common input contract.

        Feature engineering operates on pandas DataFrames. Input validation
        is centralized here so individual operations do not need to repeat
        the same checks.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} expects a pandas DataFrame, "
                f"got {type(data).__name__}."
            )

    def _require_fitted(self) -> None:
        """
        Ensure the operation has been fitted before transformation.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform()."
            )

    def __repr__(self) -> str:
        """
        Return a concise representation of the operation.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"fitted={self._is_fitted}"
            f")"
        )