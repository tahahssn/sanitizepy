from __future__ import annotations

from typing import Any

import pandas as pd

from .base import CleaningOperation


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
                raise KeyError(
                    f"Columns not found in dataframe: {missing_columns}"
                )

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
                raise KeyError(
                    f"Columns not found in dataframe: {missing_columns}"
                )

        return dataframe.drop(columns=columns).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
        }


class FillMissing(CleaningOperation):
    """
    Fill missing values using an explicitly supplied value.

    The operation intentionally requires the caller to provide the value.
    No automatic statistical assumptions are made here.
    """

    name = "fill_missing"

    def __init__(
        self,
        value: Any,
        subset: list[str] | None = None,
    ) -> None:
        self.value = value
        self.subset = subset

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        if self.subset is not None:
            missing_columns = [
                column for column in self.subset if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(
                    f"Columns not found in dataframe: {missing_columns}"
                )

            result = dataframe.copy()
            result.loc[:, self.subset] = result.loc[:, self.subset].fillna(
                self.value
            )
            return result

        return dataframe.fillna(self.value).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "subset": list(self.subset) if self.subset is not None else None,
        }


class DropDuplicates(CleaningOperation):
    """Remove duplicate rows."""

    name = "drop_duplicates"

    def __init__(
        self,
        subset: list[str] | None = None,
        keep: str | bool = "first",
    ) -> None:
        if keep not in {"first", "last", False}:
            raise ValueError(
                "keep must be one of: 'first', 'last', or False"
            )

        self.subset = subset
        self.keep = keep

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        if self.subset is not None:
            missing_columns = [
                column for column in self.subset if column not in dataframe.columns
            ]

            if missing_columns:
                raise KeyError(
                    f"Columns not found in dataframe: {missing_columns}"
                )

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
            raise KeyError(
                f"Columns not found in dataframe: {missing_columns}"
            )

        return dataframe.drop(columns=self.columns).copy()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
        }