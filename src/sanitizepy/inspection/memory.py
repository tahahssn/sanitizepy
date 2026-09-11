from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)


@dataclass(frozen=True, slots=True)
class MemoryColumnReport:
    """
    Memory statistics for a single column.
    """

    column: str
    dtype: str

    memory_bytes: int
    memory_mb: float
    memory_percentage: float

    recommended_dtype: str | None

    estimated_bytes_after: int
    estimated_saved_bytes: int
    estimated_saved_percentage: float


@dataclass(frozen=True, slots=True)
class MemorySummary:
    """
    Dataset memory summary.
    """

    total_columns: int
    total_rows: int

    total_memory_bytes: int
    total_memory_mb: float

    estimated_memory_bytes: int
    estimated_memory_mb: float

    estimated_saved_bytes: int
    estimated_saved_mb: float
    estimated_saved_percentage: float


@dataclass(frozen=True, slots=True)
class MemoryInspectionResult:
    """
    Complete memory inspection.
    """

    summary: MemorySummary

    reports: tuple[MemoryColumnReport, ...]

    memory_usage: pd.Series

    def __repr__(self) -> str:
        return (
            f"MemoryInspectionResult(total_mb={self.summary.total_memory_mb:.2f}, "
            f"potential_saved_mb={self.summary.estimated_saved_mb:.2f})"
        )

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "summary": asdict(self.summary),
            "reports": [asdict(r) for r in self.reports],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: object, options: object) -> object:
        from sanitizepy.ui import (
            COLOR_META,
            Text,
            render_footer,
            render_header,
            render_metric,
        )

        yield render_header("memory")
        yield Text("")

        yield render_metric("Current memory", f"{self.summary.total_memory_mb:.1f} MB")
        yield Text("")

        yield Text("  Largest columns", style=f"bold {COLOR_META}")
        yield Text("  " + "─" * 32, style="dim")

        sorted_cols = sorted(self.reports, key=lambda r: r.memory_bytes, reverse=True)
        for r in sorted_cols[:3]:
            yield render_metric(r.column, f"{r.memory_mb:.1f} MB")

        yield Text("")
        saved_str = f"{self.summary.estimated_saved_mb:.1f} MB"
        yield render_metric("Potential savings", saved_str)

        yield Text("")
        yield render_footer(
            (
                self.summary.total_rows,
                self.summary.total_columns,
                self.summary.total_memory_mb,
            )
        )


class MemoryInspector:
    """
    Production-grade memory inspection.

    Performs analysis only.

    Never mutates the dataframe.
    """

    __slots__: Final = ()

    CATEGORY_THRESHOLD = 0.50

    def inspect(
        self,
        dataframe: pd.DataFrame,
    ) -> MemoryInspectionResult:

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        usage = dataframe.memory_usage(index=True, deep=True)

        column_usage = dataframe.memory_usage(index=False, deep=True)

        total_memory = int(usage.sum())

        reports: list[MemoryColumnReport] = []

        estimated_total = 0

        for column in dataframe.columns:

            series = dataframe[column]

            before = int(column_usage[column])

            after = self._estimate_memory(series)

            saved = max(before - after, 0)

            reports.append(
                MemoryColumnReport(
                    column=column,
                    dtype=str(series.dtype),
                    memory_bytes=before,
                    memory_mb=round(before / (1024**2), 4),
                    memory_percentage=round(
                        before / total_memory * 100,
                        4,
                    ),
                    recommended_dtype=self._recommend_dtype(series),
                    estimated_bytes_after=after,
                    estimated_saved_bytes=saved,
                    estimated_saved_percentage=(
                        round(
                            saved / before * 100,
                            4,
                        )
                        if before
                        else 0.0
                    ),
                )
            )

            estimated_total += after

        sorted_reports = tuple(
            sorted(
                reports,
                key=lambda x: x.memory_bytes,
                reverse=True,
            )
        )

        saved = max(total_memory - estimated_total, 0)

        summary = MemorySummary(
            total_columns=len(dataframe.columns),
            total_rows=len(dataframe),
            total_memory_bytes=total_memory,
            total_memory_mb=round(
                total_memory / (1024**2),
                4,
            ),
            estimated_memory_bytes=estimated_total,
            estimated_memory_mb=round(
                estimated_total / (1024**2),
                4,
            ),
            estimated_saved_bytes=saved,
            estimated_saved_mb=round(
                saved / (1024**2),
                4,
            ),
            estimated_saved_percentage=(
                round(
                    saved / total_memory * 100,
                    4,
                )
                if total_memory
                else 0.0
            ),
        )

        return MemoryInspectionResult(
            summary=summary,
            reports=sorted_reports,
            memory_usage=usage.sort_values(ascending=False),
        )

    def largest_columns(
        self,
        dataframe: pd.DataFrame,
        n: int = 10,
    ) -> pd.Series:
        """
        Return the largest memory-consuming columns.
        """

        return (
            dataframe.memory_usage(
                deep=True,
                index=False,
            )
            .sort_values(ascending=False)
            .head(n)
        )

    def _estimate_memory(
        self,
        series: pd.Series,
    ) -> int:
        """
        Estimate optimized memory usage.
        """

        current = int(series.memory_usage(deep=True))

        if is_integer_dtype(series):

            optimized = pd.to_numeric(
                series,
                downcast="integer",
            )

            return int(optimized.memory_usage(deep=True))

        if is_float_dtype(series):

            optimized = pd.to_numeric(
                series,
                downcast="float",
            )

            return int(optimized.memory_usage(deep=True))

        if is_object_dtype(series):

            ratio = series.nunique(dropna=True) / max(len(series), 1)

            if ratio < self.CATEGORY_THRESHOLD:

                optimized = series.astype("category")

                return int(optimized.memory_usage(deep=True))

        return current

    def _recommend_dtype(
        self,
        series: pd.Series,
    ) -> str | None:
        """
        Recommend optimized dtype.
        """

        if is_integer_dtype(series):

            return str(
                pd.to_numeric(
                    series,
                    downcast="integer",
                ).dtype
            )

        if is_float_dtype(series):

            return str(
                pd.to_numeric(
                    series,
                    downcast="float",
                ).dtype
            )

        if is_object_dtype(series):

            ratio = series.nunique(dropna=True) / max(len(series), 1)

            if ratio < self.CATEGORY_THRESHOLD:
                return "category"

            return "string"

        return None
