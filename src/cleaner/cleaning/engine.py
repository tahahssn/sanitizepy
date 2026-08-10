from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .base import CleaningOperation


class CleaningEngine:
    """
    Execute an ordered sequence of cleaning operations.

    The engine is intentionally orchestration-only. Individual cleaning
    operations are responsible for their own transformation logic.
    """

    def __init__(
        self,
        operations: Iterable[CleaningOperation] | None = None,
    ) -> None:
        self._operations: list[CleaningOperation] = []

        if operations is not None:
            for operation in operations:
                self.add(operation)

    @property
    def operations(self) -> tuple[CleaningOperation, ...]:
        """Return the configured operations as an immutable sequence."""
        return tuple(self._operations)

    def add(self, operation: CleaningOperation) -> None:
        """
        Add a cleaning operation to the execution sequence.

        Operations execute in the exact order in which they are added.
        """
        if not isinstance(operation, CleaningOperation):
            raise TypeError(
                "operation must be an instance of CleaningOperation"
            )

        self._operations.append(operation)

    def clear(self) -> None:
        """Remove all configured cleaning operations."""
        self._operations.clear()

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all configured cleaning operations.

        Parameters
        ----------
        dataframe:
            Input dataframe to clean.

        Returns
        -------
        pandas.DataFrame
            The cleaned dataframe.

        Raises
        ------
        TypeError
            If the supplied object is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        result = dataframe.copy()

        for operation in self._operations:
            result = operation.apply(result)

        return result

    def describe(self) -> list[dict[str, object]]:
        """
        Return descriptions of the configured operations in execution order.
        """
        return [operation.describe() for operation in self._operations]