from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..exceptions import DataValidationError
from .base import CleaningOperation, OperationResult

FillStrategy = Literal["median", "mean", "mode", "constant"]


class DropMissingRows(CleaningOperation):
    """Remove rows containing missing values."""

    name = "drop_missing_rows"

    def __init__(self, subset: list[str] | None = None) -> None:
        self.subset = subset

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        if self.subset is not None:
            missing_columns = [
                column for column in self.subset if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        return dataframe.dropna(subset=self.subset).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
        }


class DropMissingColumns(CleaningOperation):
    """Remove columns containing missing values."""

    name = "drop_missing_columns"

    def __init__(self, subset: list[str] | None = None) -> None:
        self.subset = subset

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        columns = self.subset

        if columns is None:
            columns = dataframe.columns[dataframe.isna().any()].tolist()
        else:
            missing_columns = [
                column for column in columns if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        return dataframe.drop(columns=columns).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
        }


class FillMissing(CleaningOperation):
    """
    Fill missing values using an explicit constant or a statistical strategy.

    Four fill strategies are supported:

    * ``"constant"`` (default) replaces missing values with the caller-supplied
      ``value``, preserving the original constant-fill behavior.
    * ``"median"`` replaces missing values in each targeted numeric column with
      the column median computed over its non-missing values.
    * ``"mean"`` replaces missing values in each targeted numeric column with the
      column mean computed over its non-missing values.
    * ``"mode"`` replaces missing values in each targeted column with the most
      frequent non-missing value, selecting the first in sorted order on ties.

    The ``"median"`` and ``"mean"`` strategies require numeric columns; applying
    them to a non-numeric column raises :class:`DataValidationError`.
    """

    name = "fill_missing"

    def __init__(
        self,
        value: Any = None,
        subset: list[str] | None = None,
        strategy: FillStrategy = "constant",
    ) -> None:
        if strategy not in {"median", "mean", "mode", "constant"}:
            raise ValueError(
                "strategy must be one of: 'median', 'mean', 'mode', 'constant'"
            )

        self.value = value
        self.subset = subset
        self.strategy = strategy

    def _target_columns(self, dataframe: pd.DataFrame) -> list[str]:
        if self.subset is not None:
            missing_columns = [
                column for column in self.subset if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")

            return list(self.subset)

        return list(dataframe.columns)

    def _fill_value_for_column(self, series: pd.Series) -> Any:
        """
        Compute the replacement value for a single column under the current
        statistical strategy. ``constant`` is handled by the caller.
        """
        if self.strategy in {"median", "mean"} and not is_numeric_dtype(series):
            raise DataValidationError(
                f"Cannot apply '{self.strategy}' fill strategy to non-numeric "
                f"column '{series.name}' with dtype '{series.dtype}'."
            )

        non_missing = series.dropna()

        if self.strategy == "median":
            return non_missing.median()

        if self.strategy == "mean":
            return non_missing.mean()

        # mode: most frequent non-missing value, first in sorted order on ties.
        if non_missing.empty:
            return None

        counts = non_missing.value_counts()
        top_count = counts.iloc[0]
        tied = [value for value, count in counts.items() if count == top_count]
        return sorted(tied)[0]  # type: ignore[type-var]

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        columns = self._target_columns(dataframe)

        if self.strategy == "constant":
            result = dataframe.copy()
            for column in columns:
                mask = result[column].isna()
                if mask.any():
                    result.loc[mask, column] = self.value
            return result

        result = dataframe.copy()
        for column in columns:
            fill_value = self._fill_value_for_column(result[column])
            if fill_value is not None:
                mask = result[column].isna()
                if mask.any():
                    result.loc[mask, column] = fill_value
        return result

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        before_shape = dataframe.shape
        columns = self._target_columns(dataframe)
        missing_before = int(dataframe[columns].isna().to_numpy().sum())

        transformed = self.apply(dataframe)
        after_shape = transformed.shape
        missing_after = int(transformed[columns].isna().to_numpy().sum())
        values_filled = missing_before - missing_after

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(columns),
            rows_affected=values_filled,
            columns_affected=0,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(
                f"{self.name} ({self.strategy}) filled {values_filled} value(s)"
            ),
            dry_run=dry_run,
            details={**self.describe(), "values_filled": values_filled},
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "value": self.value,
            "subset": list(self.subset) if self.subset is not None else None,
        }


class DropDuplicates(CleaningOperation):
    """Remove duplicate rows."""

    name = "drop_duplicates"

    def __init__(
        self,
        subset: list[str] | None = None,
        keep: Literal["first", "last"] | Literal[False] = "first",
    ) -> None:
        if keep not in {"first", "last", False}:
            raise ValueError("keep must be one of: 'first', 'last', or False")

        self.subset = subset
        self.keep = keep

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        if self.subset is not None:
            missing_columns = [
                column for column in self.subset if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        return dataframe.drop_duplicates(
            subset=self.subset,
            keep=self.keep,
        ).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
            "keep": self.keep,
        }


class DropColumns(CleaningOperation):
    """Remove explicitly specified columns."""

    name = "drop_columns"

    def __init__(self, columns: list[str]) -> None:
        if not columns:
            raise ValueError("columns must contain at least one column name")

        self.columns = list(columns)

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        missing_columns = [
            column for column in self.columns if column not in dataframe.columns
        ]

        if missing_columns:
            raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        return dataframe.drop(columns=self.columns).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
        }
