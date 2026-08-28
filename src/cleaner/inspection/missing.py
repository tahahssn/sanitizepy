from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True, slots=True)
class MissingColumnReport:
    """
    Missing value statistics for a single column.
    """

    column: str
    missing_count: int
    missing_percentage: float
    complete_count: int
    complete_percentage: float
    has_missing: bool


@dataclass(frozen=True, slots=True)
class MissingSummary:
    """
    Dataset-level missing value summary.
    """

    total_rows: int
    total_columns: int
    total_cells: int

    missing_cells: int
    missing_percentage: float

    complete_cells: int
    complete_percentage: float

    columns_with_missing: tuple[str, ...]
    complete_columns: tuple[str, ...]

    complete_rows: int
    rows_with_missing: int

    severity: str


@dataclass(frozen=True, slots=True)
class MissingInspectionResult:
    """
    Complete missing value inspection result.
    """

    summary: MissingSummary

    column_reports: tuple[MissingColumnReport, ...]

    missing_mask: pd.DataFrame

    missing_counts: pd.Series

    missing_percentages: pd.Series


class MissingValueInspector:
    """
    High-performance missing value inspection.

    This class never mutates the dataframe.

    It only performs analysis.
    """

    __slots__: Final = ()

    _LOW = 5.0
    _MEDIUM = 15.0
    _HIGH = 35.0

    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> MissingInspectionResult:
        """
        Inspect missing values.

        Parameters
        ----------
        dataframe:
            Input dataframe.

        threshold:
            Optional minimum missing percentage required
            for a column to appear in the report.

        Returns
        -------
        MissingInspectionResult
        """

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        mask = dataframe.isna()

        total_rows = len(dataframe)
        total_columns = len(dataframe.columns)
        total_cells = total_rows * total_columns

        missing_counts = mask.sum(axis=0).astype("int64")

        missing_percentages = missing_counts.div(total_rows).mul(100.0).round(4)

        if threshold is not None:
            keep = missing_percentages >= threshold
            missing_counts = missing_counts.loc[keep]
            missing_percentages = missing_percentages.loc[keep]

        reports = tuple(
            MissingColumnReport(
                column=column,
                missing_count=int(count),
                missing_percentage=float(percent),
                complete_count=total_rows - int(count),
                complete_percentage=round(100.0 - float(percent), 4),
                has_missing=count > 0,
            )
            for column, count, percent in zip(
                missing_counts.index,
                missing_counts.values,
                missing_percentages.values,
                strict=True,
            )
        )

        reports = tuple(
            sorted(
                reports,
                key=lambda report: report.missing_percentage,
                reverse=True,
            )
        )

        total_missing = int(mask.to_numpy().sum())

        missing_percentage = (
            round((total_missing / total_cells) * 100.0, 4) if total_cells else 0.0
        )

        complete_cells = total_cells - total_missing

        complete_percentage = round(
            100.0 - missing_percentage,
            4,
        )

        columns_with_missing = tuple(dataframe.columns[mask.any(axis=0)])

        complete_columns = tuple(dataframe.columns[~mask.any(axis=0)])

        rows_with_missing = int(mask.any(axis=1).sum())

        complete_rows = total_rows - rows_with_missing

        severity = self._severity(missing_percentage)

        summary = MissingSummary(
            total_rows=total_rows,
            total_columns=total_columns,
            total_cells=total_cells,
            missing_cells=total_missing,
            missing_percentage=missing_percentage,
            complete_cells=complete_cells,
            complete_percentage=complete_percentage,
            columns_with_missing=columns_with_missing,
            complete_columns=complete_columns,
            complete_rows=complete_rows,
            rows_with_missing=rows_with_missing,
            severity=severity,
        )

        return MissingInspectionResult(
            summary=summary,
            column_reports=reports,
            missing_mask=mask,
            missing_counts=missing_counts.sort_values(ascending=False),
            missing_percentages=missing_percentages.sort_values(ascending=False),
        )

    def _severity(self, percentage: float) -> str:
        """
        Compute dataset missing severity.
        """

        if percentage < self._LOW:
            return "LOW"

        if percentage < self._MEDIUM:
            return "MEDIUM"

        if percentage < self._HIGH:
            return "HIGH"

        return "CRITICAL"
