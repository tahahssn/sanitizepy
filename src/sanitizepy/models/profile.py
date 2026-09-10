"""
Dataset profile model.

``DatasetProfile`` is an immutable aggregate of the results produced by the
standalone inspectors (``MissingValueInspector``, ``DuplicateInspector``,
``DatatypeInspector``, ``MemoryInspector``, ``StatisticsInspector``).

It follows the same immutable convention used by those inspector result
types (frozen dataclasses that may hold pandas objects), rather than the
pydantic ``BaseCleanerModel`` convention, because it composes those pandas
carrying dataclass results directly.

The ``text_quality``, ``anomalies`` and ``near_duplicate`` slots are
forward-compatible and default to ``None``. The analyzers that populate them
are introduced by later tasks:

- ``TextQualityResult`` (defined here; populated by task 14.2 /
  ``inspection/text_quality.py``)
- ``AnomalyResult`` (task 15.x / ``inspection/anomalies.py``)
- ``NearDuplicateResult`` (task 13.x / ``inspection/near_duplicates.py``)

The ``anomalies`` and ``near_duplicate`` slots stay typed loosely until
those types exist, so the profile can be assembled today and tightened
without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sanitizepy.inspection.datatypes import DatatypeInspectionResult
from sanitizepy.inspection.duplicates import DuplicateInspectionResult
from sanitizepy.inspection.memory import MemoryInspectionResult
from sanitizepy.inspection.missing import MissingInspectionResult
from sanitizepy.inspection.statistics import StatisticsInspectionResult


@dataclass(frozen=True, slots=True)
class TextQualityResult:
    """
    Deterministic per-column text-quality result.

    Immutable, column-oriented result produced by the text-quality analyzer
    (``inspection/text_quality.py``, task 14.2). One instance describes a
    single free-form text column; the analyzer returns one per analyzed
    column and the profile aggregates them as a tuple.

    Follows the Standalone_Inspector convention (frozen ``slots=True``
    dataclass) so it composes into ``DatasetProfile`` alongside the other
    immutable inspector results.

    All statistics are computed over non-null values only. Character-length
    and token-count aggregates describe the same non-null population counted
    by ``non_null_count``.

    Attributes
    ----------
    column:
        Name of the analyzed text column.

    non_null_count:
        Number of non-null values analyzed in the column.

    empty_after_strip_count:
        Number of non-null values that are empty once leading/trailing
        whitespace is stripped.

    character_length_min:
        Minimum character length across non-null values.

    character_length_max:
        Maximum character length across non-null values.

    character_length_mean:
        Mean character length across non-null values.

    character_length_median:
        Median character length across non-null values.

    token_count_min:
        Minimum whitespace-delimited token count across non-null values.

    token_count_max:
        Maximum whitespace-delimited token count across non-null values.

    token_count_mean:
        Mean whitespace-delimited token count across non-null values.

    token_count_median:
        Median whitespace-delimited token count across non-null values.

    boilerplate_count:
        Number of non-null values matching deterministic boilerplate
        patterns.

    encoding_garbage_count:
        Number of non-null values matching deterministic encoding-garbage
        (mojibake / replacement / control-character) patterns.
    """

    column: str

    non_null_count: int

    empty_after_strip_count: int

    character_length_min: int
    character_length_max: int
    character_length_mean: float
    character_length_median: float

    token_count_min: int
    token_count_max: int
    token_count_mean: float
    token_count_median: float

    boilerplate_count: int
    encoding_garbage_count: int

    def __repr__(self) -> str:
        return f"TextQualityResult(column={self.column!r}, non_null={self.non_null_count})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "non_null_count": self.non_null_count,
            "empty_after_strip_count": self.empty_after_strip_count,
            "character_length_min": self.character_length_min,
            "character_length_max": self.character_length_max,
            "character_length_mean": self.character_length_mean,
            "character_length_median": self.character_length_median,
            "token_count_min": self.token_count_min,
            "token_count_max": self.token_count_max,
            "token_count_mean": self.token_count_mean,
            "token_count_median": self.token_count_median,
            "boilerplate_count": self.boilerplate_count,
            "encoding_garbage_count": self.encoding_garbage_count,
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from sanitizepy.ui import render_header, render_table, Text

        yield render_header("text quality")
        yield Text("")
        headers = ["column", "avg chars", "avg tokens", "empty", "encoding"]
        pct_empty = (self.empty_after_strip_count / max(1, self.non_null_count)) * 100
        rows = [
            [
                self.column,
                f"{self.character_length_mean:.1f}",
                f"{self.token_count_mean:.1f}",
                f"{pct_empty:.1f}%",
                f"{self.encoding_garbage_count:,}",
            ]
        ]
        yield render_table(headers, rows)


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """
    Immutable aggregate profile of a dataset.

    Combines the immutable results of the standalone inspectors without
    recomputing any analysis they already provide.
    """

    row_count: int
    column_count: int

    datatypes: DatatypeInspectionResult
    missing_values: MissingInspectionResult
    duplicates: DuplicateInspectionResult
    memory: MemoryInspectionResult
    statistics: StatisticsInspectionResult

    # Per-column text-quality results (task 14.1). The remaining slots stay
    # typed loosely (``Any``) until ``AnomalyResult`` (15.x) and
    # ``NearDuplicateResult`` (13.x) exist.
    text_quality: tuple[TextQualityResult, ...] | None = field(default=None)
    anomalies: tuple[Any, ...] | None = field(default=None)
    near_duplicate: Any | None = field(default=None)

    def __repr__(self) -> str:
        return f"DatasetProfile(rows={self.row_count}, columns={self.column_count})"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        result: dict[str, Any] = {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "datatypes": (
                self.datatypes.to_dict()
                if hasattr(self.datatypes, "to_dict")
                else asdict(self.datatypes)
            ),
            "missing_values": (
                self.missing_values.to_dict()
                if hasattr(self.missing_values, "to_dict")
                else asdict(self.missing_values)
            ),
            "duplicates": (
                self.duplicates.to_dict()
                if hasattr(self.duplicates, "to_dict")
                else asdict(self.duplicates)
            ),
            "memory": (
                self.memory.to_dict()
                if hasattr(self.memory, "to_dict")
                else asdict(self.memory)
            ),
            "statistics": (
                self.statistics.to_dict()
                if hasattr(self.statistics, "to_dict")
                else asdict(self.statistics)
            ),
        }
        if self.text_quality is not None:
            result["text_quality"] = [
                t.to_dict() if hasattr(t, "to_dict") else asdict(t)
                for t in self.text_quality
            ]
        if self.anomalies is not None:
            result["anomalies"] = [
                a.to_dict() if hasattr(a, "to_dict") else asdict(a)
                for a in self.anomalies
            ]
        if self.near_duplicate is not None:
            result["near_duplicate"] = (
                self.near_duplicate.to_dict()
                if hasattr(self.near_duplicate, "to_dict")
                else asdict(self.near_duplicate)
            )
        return result

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from sanitizepy.ui import (
            COLOR_META,
            Text,
            render_footer,
            render_header,
            render_health_score,
            render_metric,
            render_table,
        )

        yield render_header("profile")
        yield Text("")

        # DATASET
        yield Text("DATASET", style=f"bold {COLOR_META}")
        yield render_metric("Rows", f"{self.row_count:,}")
        yield render_metric("Columns", f"{self.column_count:,}")
        numeric_count = len(self.datatypes.summary.numeric_columns)
        text_count = len(self.datatypes.summary.string_columns) + len(
            self.datatypes.summary.object_columns
        )
        datetime_count = len(self.datatypes.summary.datetime_columns)
        yield render_metric("Numeric", f"{numeric_count:,}")
        yield render_metric("Text", f"{text_count:,}")
        yield render_metric("Datetime", f"{datetime_count:,}")
        yield render_metric("Memory", f"{self.memory.summary.total_memory_mb:.1f} MB")
        yield Text("")

        # QUALITY
        yield Text("QUALITY", style=f"bold {COLOR_META}")
        yield render_metric(
            "Missing", f"{self.missing_values.summary.missing_percentage:.1f}%"
        )
        yield render_metric(
            "Duplicates", f"{self.duplicates.summary.duplicate_percentage:.1f}%"
        )

        missing_cells = self.missing_values.summary.missing_cells
        total_cells = max(1, self.missing_values.summary.total_cells)
        dup_rows = self.duplicates.summary.duplicate_rows
        comp_score = max(0.0, 100.0 * (1.0 - (missing_cells / total_cells)))
        uniq_score = max(0.0, 100.0 * (1.0 - (dup_rows / max(1, self.row_count))))
        const_cols = len(self.statistics.summary.constant_columns)
        integ_score = max(0.0, 100.0 - (100.0 * const_cols / max(1, self.column_count)))
        rec_diff_count = sum(
            1
            for r in self.datatypes.reports
            if r.recommended_dtype and r.recommended_dtype != r.dtype
        )
        consist_score = max(0.0, 100.0 - 15.0 * rec_diff_count)
        outlier_cols = sum(1 for r in self.statistics.reports if r.outlier_count > 0)
        valid_score = max(0.0, 100.0 - 8.0 * outlier_cols)
        health = max(
            0,
            min(
                100,
                int(
                    round(
                        0.35 * comp_score
                        + 0.25 * uniq_score
                        + 0.15 * integ_score
                        + 0.15 * consist_score
                        + 0.10 * valid_score
                    )
                ),
            ),
        )
        yield render_health_score(health)
        yield Text("")

        # COLUMN PROFILE
        yield Text("COLUMN PROFILE", style=f"bold {COLOR_META}")
        headers = ["column", "dtype", "nulls", "unique", "min", "max"]
        missing_map = {
            r.column: r.missing_percentage for r in self.missing_values.column_reports
        }
        stats_map = {r.column: r for r in self.statistics.reports}

        rows = []
        for col_report in self.datatypes.reports:
            c = col_report.column
            null_pct = missing_map.get(c, 0.0)
            null_str = f"{null_pct:.1f}%"
            uniq_str = f"{col_report.unique_count:,}"

            if c in stats_map:
                st = stats_map[c]
                min_str = (
                    f"{st.minimum:,.1f}"
                    if not float(st.minimum).is_integer()
                    else f"{int(st.minimum):,}"
                )
                max_str = (
                    f"{st.maximum:,.1f}"
                    if not float(st.maximum).is_integer()
                    else f"{int(st.maximum):,}"
                )
            else:
                min_str = "—"
                max_str = "—"

            rows.append([c, col_report.dtype, null_str, uniq_str, min_str, max_str])

        yield render_table(headers, rows)
        yield Text("")
        yield render_footer(self)


__all__ = [
    "DatasetProfile",
    "TextQualityResult",
]
