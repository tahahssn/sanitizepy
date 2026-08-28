from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

KeepStrategy = Literal["first", "last", False]


@dataclass(frozen=True, slots=True)
class DuplicateSummary:
    """
    Dataset-level duplicate summary.
    """

    total_rows: int
    duplicate_rows: int
    unique_rows: int
    duplicate_percentage: float
    unique_percentage: float
    severity: str


@dataclass(frozen=True, slots=True)
class DuplicateInspectionResult:
    """
    Duplicate inspection result.
    """

    summary: DuplicateSummary

    duplicate_mask: pd.Series

    duplicate_indices: tuple[int, ...]

    duplicate_dataframe: pd.DataFrame

    duplicate_count: int


class DuplicateInspector:
    """
    High-performance duplicate inspection.

    This class never mutates the dataframe.
    """

    __slots__: Final = ()

    _LOW = 2.0
    _MEDIUM = 5.0
    _HIGH = 15.0

    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
        keep: KeepStrategy = "first",
    ) -> DuplicateInspectionResult:
        """
        Inspect duplicate rows.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        subset:
            Columns used for duplicate comparison.

        keep:
            Same behavior as pandas.DataFrame.duplicated()

        Returns
        -------
        DuplicateInspectionResult
        """

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        duplicate_mask = dataframe.duplicated(
            subset=subset,
            keep=keep,
        )

        duplicate_dataframe = dataframe.loc[duplicate_mask]

        duplicate_indices = tuple(duplicate_dataframe.index.tolist())

        total_rows = len(dataframe)

        duplicate_count = int(duplicate_mask.sum())

        unique_rows = total_rows - duplicate_count

        duplicate_percentage = round(
            (duplicate_count / total_rows) * 100.0,
            4,
        )

        unique_percentage = round(
            100.0 - duplicate_percentage,
            4,
        )

        severity = self._severity(duplicate_percentage)

        summary = DuplicateSummary(
            total_rows=total_rows,
            duplicate_rows=duplicate_count,
            unique_rows=unique_rows,
            duplicate_percentage=duplicate_percentage,
            unique_percentage=unique_percentage,
            severity=severity,
        )

        return DuplicateInspectionResult(
            summary=summary,
            duplicate_mask=duplicate_mask,
            duplicate_indices=duplicate_indices,
            duplicate_dataframe=duplicate_dataframe,
            duplicate_count=duplicate_count,
        )

    def has_duplicates(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """
        Return True if duplicate rows exist.
        """

        return bool(dataframe.duplicated(subset=subset).any())

    def duplicate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[str, ...]:
        """
        Return duplicated column names.
        """

        mask = dataframe.columns.duplicated()

        return tuple(dataframe.columns[mask])

    def duplicate_column_count(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Count duplicated column names.
        """

        return int(dataframe.columns.duplicated().sum())

    def _severity(self, percentage: float) -> str:
        """
        Compute duplicate severity.
        """

        if percentage < self._LOW:
            return "LOW"

        if percentage < self._MEDIUM:
            return "MEDIUM"

        if percentage < self._HIGH:
            return "HIGH"

        return "CRITICAL"
