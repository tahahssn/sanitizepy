from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OperationResult:
    """
    Structured, explainable outcome of applying a single cleaning operation.

    Attributes
    ----------
    operation_name:
        Canonical name of the operation.
    affected_columns:
        List of column names modified or evaluated.
    rows_affected:
        Number of rows removed or filled.
    columns_affected:
        Number of columns removed or modified.
    before_shape:
        (rows, cols) tuple prior to operation execution.
    after_shape:
        (rows, cols) tuple following operation execution.
    strategy_description:
        Human-readable summary of what was performed.
    dry_run:
        Whether the operation was evaluated in dry-run mode.
    details:
        Additional operation-specific metadata.
    """

    operation_name: str
    affected_columns: list[str] = field(default_factory=list)
    rows_affected: int = 0
    columns_affected: int = 0
    before_shape: tuple[int, int] = (0, 0)
    after_shape: tuple[int, int] = (0, 0)
    strategy_description: str = ""
    dry_run: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class CleaningOperation(ABC):
    """
    Base contract for all dataframe cleaning operations.

    A cleaning operation receives a pandas DataFrame and returns a cleaned
    DataFrame. Implementations must not mutate the input DataFrame unless
    explicitly documented as part of their contract.

    Class Attributes
    ----------------
    is_chunk_safe:
        Whether the operation produces identical results when applied
        independently to row-wise chunks of the dataset. Defaults to
        ``False`` so operations execute over the whole dataset unless a
        subclass explicitly opts in.
    is_inplace_safe:
        Whether the operation may be applied without a defensive copy of
        the input DataFrame when not running in dry-run mode. Defaults to
        ``False`` so the engine keeps its conservative copy-based behavior
        unless a subclass explicitly opts in.
    """

    name: str
    is_chunk_safe: bool = False
    is_inplace_safe: bool = False

    @abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the cleaning operation to a dataframe.

        Parameters
        ----------
        dataframe:
            The dataframe to transform.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe.

        Raises
        ------
        TypeError
            If the supplied object is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )
        return dataframe

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        """
        Apply the cleaning operation and return both the resultant dataframe
        and an OperationResult tracking exact metrics and dry-run state.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        before_shape = dataframe.shape
        transformed = self.apply(dataframe)
        after_shape = transformed.shape

        rows_affected = abs(before_shape[0] - after_shape[0])
        cols_affected = abs(before_shape[1] - after_shape[1])

        desc = self.describe()
        affected_cols = desc.get("subset") or desc.get("columns") or []

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(affected_cols),
            rows_affected=rows_affected,
            columns_affected=cols_affected,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=f"{self.name} on {before_shape} -> {after_shape}",
            dry_run=dry_run,
            details=desc,
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        """
        Return a serializable description of the operation.

        The description is intentionally lightweight so that the cleaning
        engine and reporting layer can record which operations were applied
        without coupling themselves to concrete operation implementations.
        """
        return {
            "name": self.name,
        }
