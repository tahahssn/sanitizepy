"""
Deterministic anomaly detection.

This module provides read-only, deterministic anomaly-detection methods
over numeric columns of a pandas DataFrame. It never mutates the input.

Two methods are provided:

- ``iqr``: single-column Tukey fences at 1.5 * IQR, preserving the exact
  behavior already used by :class:`StatisticsInspector`.
- ``zscore``: single-column standard-score thresholding where a value is
  flagged when ``|x - mean| / std`` exceeds a caller-supplied threshold.

Both methods are fully deterministic. A fixed ``seed`` is accepted and
applied to any method that involves randomness so that repeated runs with
the same seed produce identical results. The current methods are purely
deterministic and do not draw random numbers, but the seed is honored to
keep the contract stable as future randomized methods are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

AnomalyMethod = Literal["iqr", "zscore"]

DEFAULT_IQR_MULTIPLIER: Final = 1.5
DEFAULT_ZSCORE_THRESHOLD: Final = 3.0


@dataclass(frozen=True, slots=True)
class ColumnAnomalyReport:
    """
    Per-column anomaly detection report.
    """

    column: str

    method: AnomalyMethod

    analyzed_count: int

    anomaly_count: int

    anomaly_percentage: float

    lower_bound: float | None

    upper_bound: float | None

    anomaly_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """
    Complete anomaly inspection result.

    Immutable; never references the caller's DataFrame.
    """

    method: AnomalyMethod

    seed: int | None

    analyzed_columns: tuple[str, ...]

    total_anomalies: int

    reports: tuple[ColumnAnomalyReport, ...]

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(total_anomalies={self.total_anomalies}, "
            f"method='{self.method}')"
        )

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "method": self.method,
            "seed": self.seed,
            "analyzed_columns": list(self.analyzed_columns),
            "total_anomalies": self.total_anomalies,
            "reports": [asdict(r) for r in self.reports],
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
            render_status_row,
            render_table,
        )

        yield render_header("anomalies")
        yield Text("")

        headers = ["column", "method", "flagged"]
        flagged_reports = [r for r in self.reports if r.anomaly_count > 0]
        display_reports = flagged_reports if flagged_reports else self.reports

        rows = []
        for r in display_reports:
            method_str = "IQR" if r.method == "iqr" else "z-score"
            if r.method not in ("iqr", "zscore"):
                method_str = str(r.method)
            rows.append([r.column, method_str, f"{r.anomaly_count:,}"])

        if rows:
            yield render_table(headers, rows)
            yield Text("")

        if self.total_anomalies > 0:
            msg = (
                f"{self.total_anomalies:,} observations flagged "
                "— review before removing"
            )
            yield render_status_row(SYMBOL_WARN, msg)
        else:
            yield render_status_row(SYMBOL_OK, "No anomalies detected")

        yield Text("")
        yield render_footer(f"{len(self.analyzed_columns)} columns analyzed")


class AnomalyInspector:
    """
    Deterministic anomaly inspection.

    Read-only. Never mutates the dataframe.
    """

    __slots__: Final = ()

    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        method: AnomalyMethod = "iqr",
        iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
        zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
        seed: int | None = None,
    ) -> AnomalyResult:
        """
        Detect anomalies across numeric columns.

        Parameters
        ----------
        dataframe:
            Input dataframe (never mutated).

        method:
            ``"iqr"`` (Tukey fences, preserving existing behavior) or
            ``"zscore"`` (standard-score thresholding).

        iqr_multiplier:
            Multiplier applied to the IQR when computing Tukey fences.

        zscore_threshold:
            Absolute z-score above which a value is flagged.

        seed:
            Fixed seed applied to any randomized step. The current methods
            are deterministic and draw no random numbers, but the seed is
            honored so results stay reproducible if randomness is added.

        Returns
        -------
        AnomalyResult
        """

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        # Apply the fixed seed so any randomized step is reproducible.
        # Current methods are deterministic and draw no random numbers.
        if seed is not None:
            np.random.seed(seed)

        reports: list[ColumnAnomalyReport] = []
        analyzed_columns: list[str] = []
        total_anomalies = 0

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

            analyzed_columns.append(str(column))

            if method == "iqr":
                mask, lower, upper = self._iqr_mask(values, iqr_multiplier)
            else:
                mask, lower, upper = self._zscore_mask(values, zscore_threshold)

            anomaly_indices = tuple(int(i) for i in values.index[mask].tolist())
            anomaly_count = len(anomaly_indices)
            analyzed_count = int(values.count())

            anomaly_percentage = round(
                (anomaly_count / analyzed_count) * 100.0,
                4,
            )

            total_anomalies += anomaly_count

            reports.append(
                ColumnAnomalyReport(
                    column=str(column),
                    method=method,
                    analyzed_count=analyzed_count,
                    anomaly_count=anomaly_count,
                    anomaly_percentage=anomaly_percentage,
                    lower_bound=lower,
                    upper_bound=upper,
                    anomaly_indices=anomaly_indices,
                )
            )

        return AnomalyResult(
            method=method,
            seed=seed,
            analyzed_columns=tuple(analyzed_columns),
            total_anomalies=total_anomalies,
            reports=tuple(reports),
        )

    def _iqr_mask(
        self,
        values: pd.Series,
        multiplier: float,
    ) -> tuple[pd.Series, float | None, float | None]:
        """
        Tukey-fence mask, preserving existing 1.5 * IQR behavior.
        """

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))

        iqr = q3 - q1

        lower = q1 - (multiplier * iqr)
        upper = q3 + (multiplier * iqr)

        mask = (values < lower) | (values > upper)

        return mask, lower, upper

    def _zscore_mask(
        self,
        values: pd.Series,
        threshold: float,
    ) -> tuple[pd.Series, float | None, float | None]:
        """
        Standard-score mask: ``|x - mean| / std > threshold``.

        A zero standard deviation (constant or single-value column) yields
        no anomalies, matching the intuition that identical values are not
        anomalous relative to one another.
        """

        mean = float(values.mean())

        # ddof=1 matches StatisticsInspector; guard the single-value case.
        std = float(values.std()) if len(values) > 1 else 0.0

        if std == 0.0:
            empty_mask = pd.Series(False, index=values.index)
            return empty_mask, None, None

        z = (values - mean) / std
        mask = z.abs() > threshold

        lower = mean - (threshold * std)
        upper = mean + (threshold * std)

        return mask, lower, upper
