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

    def __repr__(self) -> str:
        return (
            f"DuplicateInspectionResult(duplicates={self.duplicate_count}, "
            f"rate={self.summary.duplicate_percentage:.1f}%)"
        )

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "summary": asdict(self.summary),
            "duplicate_count": self.duplicate_count,
            "duplicate_indices": list(self.duplicate_indices),
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: object, options: object) -> object:
        from sanitizepy.ui import (
            SYMBOL_OK,
            SYMBOL_WARN,
            Text,
            render_footer,
            render_header,
            render_metric,
            render_status_row,
        )

        yield render_header("duplicates")
        yield Text("")

        yield render_metric("Rows", f"{self.summary.total_rows:,}")
        yield render_metric("Duplicate rows", f"{self.summary.duplicate_rows:,}")
        yield render_metric("Unique rows", f"{self.summary.unique_rows:,}")
        dup_rate = f"{self.summary.duplicate_percentage:.1f}%"
        yield render_metric("Duplicate rate", dup_rate)
        yield Text("")

        if self.duplicate_count > 0:
            yield render_status_row(
                SYMBOL_WARN,
                "duplicate rows detected",
                count=self.duplicate_count,
            )
        else:
            yield render_status_row(SYMBOL_OK, "No duplicate rows detected")

        yield Text("")
        has_dup_df = (
            hasattr(self, "duplicate_dataframe")
            and self.duplicate_dataframe is not None
        )
        total_cols = len(self.duplicate_dataframe.columns) if has_dup_df else 0
        yield render_footer((self.summary.total_rows, total_cols))


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
