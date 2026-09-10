from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CorrelationMethod = Literal["pearson", "kendall", "spearman"]


@dataclass(frozen=True, slots=True)
class NumericColumnStatistics:
    """
    Statistical profile of a numeric column.
    """

    column: str

    count: int
    missing: int
    unique: int

    minimum: float
    maximum: float

    mean: float
    median: float
    mode: float | None

    variance: float
    standard_deviation: float

    q1: float
    q2: float
    q3: float

    iqr: float

    skewness: float
    kurtosis: float

    sum: float

    zero_count: int
    negative_count: int
    infinite_count: int

    outlier_count: int

    constant: bool


@dataclass(frozen=True, slots=True)
class StatisticsSummary:
    """
    Dataset statistical summary.
    """

    numeric_columns: int

    constant_columns: tuple[str, ...]

    analyzed_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StatisticsInspectionResult:
    """
    Complete statistics inspection.
    """

    summary: StatisticsSummary

    reports: tuple[NumericColumnStatistics, ...]

    def __repr__(self) -> str:
        return (
            f"StatisticsInspectionResult(analyzed_columns="
            f"{len(self.summary.analyzed_columns)})"
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "summary": asdict(self.summary),
            "reports": [asdict(r) for r in self.reports],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from sanitizepy.ui import (
            Text,
            render_footer,
            render_header,
            render_table,
        )

        yield render_header("statistics")
        yield Text("")

        headers = ["column", "count", "mean", "median", "std", "min", "max"]
        rows = []
        for r in self.reports:
            min_str = (
                f"{r.minimum:,.1f}"
                if not float(r.minimum).is_integer()
                else f"{int(r.minimum):,}"
            )
            max_str = (
                f"{r.maximum:,.1f}"
                if not float(r.maximum).is_integer()
                else f"{int(r.maximum):,}"
            )
            rows.append(
                [
                    r.column,
                    f"{r.count:,}",
                    f"{r.mean:,.1f}",
                    f"{r.median:,.1f}",
                    f"{r.standard_deviation:,.1f}",
                    min_str,
                    max_str,
                ]
            )

        yield render_table(headers, rows)
        yield Text("")
        yield render_footer(f"{len(self.summary.analyzed_columns)} numeric columns")


class StatisticsInspector:
    """
    Production-grade statistical inspection.

    Read-only.

    Never mutates the dataframe.
    """

    __slots__: Final = ()

    def inspect(
        self,
        dataframe: pd.DataFrame,
    ) -> StatisticsInspectionResult:

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        reports: list[NumericColumnStatistics] = []

        constant_columns: list[str] = []

        analyzed_columns: list[str] = []

        for column in dataframe.columns:

            series = dataframe[column]

            if not is_numeric_dtype(series):
                continue

            clean = series.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            values = clean.dropna()

            if values.empty:
                continue

            # Only count this column as analyzed once we know it has data.
            analyzed_columns.append(column)

            q1 = float(values.quantile(0.25))
            q2 = float(values.quantile(0.50))
            q3 = float(values.quantile(0.75))

            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            outliers = ((values < lower) | (values > upper)).sum()

            mode = values.mode()

            constant = values.nunique() == 1

            if constant:
                constant_columns.append(column)

            reports.append(
                NumericColumnStatistics(
                    column=column,
                    count=int(values.count()),
                    missing=int(series.isna().sum()),
                    unique=int(values.nunique()),
                    minimum=float(cast(Any, values.min())),
                    maximum=float(cast(Any, values.max())),
                    mean=float(cast(Any, values.mean())),
                    median=float(cast(Any, values.median())),
                    mode=float(mode.iloc[0]) if not mode.empty else None,
                    # ddof=1: NaN when only one non-NaN value exists; coerce to 0.0.
                    variance=(
                        float(cast(Any, values.var())) if len(values) > 1 else 0.0
                    ),
                    standard_deviation=(
                        float(cast(Any, values.std())) if len(values) > 1 else 0.0
                    ),
                    q1=q1,
                    q2=q2,
                    q3=q3,
                    iqr=float(iqr),
                    skewness=float(cast(Any, values.skew())),
                    kurtosis=float(cast(Any, values.kurt())),
                    sum=float(values.sum()),
                    zero_count=int((values == 0).sum()),
                    negative_count=int((values < 0).sum()),
                    infinite_count=int(
                        (~np.isfinite(series)).sum() - int(series.isna().sum())
                    ),
                    outlier_count=outliers,
                    constant=constant,
                )
            )

        summary = StatisticsSummary(
            numeric_columns=len(reports),
            constant_columns=tuple(constant_columns),
            analyzed_columns=tuple(analyzed_columns),
        )

        return StatisticsInspectionResult(
            summary=summary,
            reports=tuple(reports),
        )

    def describe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Wrapper around pandas describe().
        """

        return dataframe.describe(
            include="all",
        )

    def correlation(
        self,
        dataframe: pd.DataFrame,
        *,
        method: CorrelationMethod = "pearson",
    ) -> pd.DataFrame:
        """
        Correlation matrix.
        """

        return dataframe.corr(
            numeric_only=True,
            method=method,
        )

    def covariance(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Covariance matrix.
        """

        return dataframe.cov(
            numeric_only=True,
        )
