from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .base import FeatureOperation


class FeatureEngineeringEngine:
    """
    Orchestrates a sequence of feature engineering operations.

    Operations are executed in the order in which they are registered.
    Each operation is fitted once and then transformed against the same
    dataframe during an execution.

    The engine does not implement feature-generation logic itself.
    Concrete feature behavior belongs to FeatureOperation subclasses.
    """

    def __init__(
        self,
        operations: Iterable[FeatureOperation] | None = None,
    ) -> None:
        self._operations: list[FeatureOperation] = []

        if operations is not None:
            self.add_operations(operations)

    @property
    def operations(self) -> tuple[FeatureOperation, ...]:
        """Return the registered operations as an immutable view."""
        return tuple(self._operations)

    def add_operation(self, operation: FeatureOperation) -> FeatureEngineeringEngine:
        """
        Register a feature engineering operation.

        Parameters
        ----------
        operation:
            A FeatureOperation instance.

        Returns
        -------
        FeatureEngineeringEngine
            The current engine instance.
        """
        if not isinstance(operation, FeatureOperation):
            raise TypeError(
                "operation must be an instance of FeatureOperation, "
                f"got {type(operation).__name__}"
            )

        self._operations.append(operation)
        return self

    def add_operations(
        self,
        operations: Iterable[FeatureOperation],
    ) -> FeatureEngineeringEngine:
        """
        Register multiple feature engineering operations.

        Parameters
        ----------
        operations:
            Iterable containing FeatureOperation instances.

        Returns
        -------
        FeatureEngineeringEngine
            The current engine instance.
        """
        for operation in operations:
            self.add_operation(operation)

        return self

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()

    def fit(self, data: pd.DataFrame) -> FeatureEngineeringEngine:
        """
        Fit all registered operations against the supplied dataframe.

        Parameters
        ----------
        data:
            Input dataframe used to learn operation-specific state.

        Returns
        -------
        FeatureEngineeringEngine
            The fitted engine.
        """
        self._validate_input(data)

        current = data

        for operation in self._operations:
            operation.fit(current)
            current = operation.transform(current)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered operations sequentially.

        The engine does not mutate the caller's dataframe. Each operation
        receives the dataframe produced by the preceding operation.

        Parameters
        ----------
        data:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            Transformed dataframe.
        """
        self._validate_input(data)

        current = data.copy()

        for operation in self._operations:
            current = operation.transform(current)

        return current

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and apply all registered operations sequentially.

        Parameters
        ----------
        data:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            Transformed dataframe.
        """
        self._validate_input(data)

        current = data.copy()

        for operation in self._operations:
            operation.fit(current)
            current = operation.transform(current)

        return current

    def get_params(self) -> dict[str, Any]:
        """
        Return the engine configuration.

        Returns
        -------
        dict[str, Any]
            Registered operation configurations.
        """
        return {
            "operations": [
                operation.get_params()
                for operation in self._operations
            ]
        }

    def __len__(self) -> int:
        """Return the number of registered operations."""
        return len(self._operations)

    def __repr__(self) -> str:
        operation_names = ", ".join(
            type(operation).__name__
            for operation in self._operations
        )

        return (
            f"{type(self).__name__}("
            f"operations=[{operation_names}]"
            f")"
        )

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> None:
        """Validate the dataframe supplied to the engine."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas.DataFrame, "
                f"got {type(data).__name__}"
            )