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


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """
    Immutable aggregate profile of a dataset.

    Combines the immutable results of the standalone inspectors without
    recomputing any analysis they already provide.

    Attributes
    ----------
    row_count:
        Number of rows in the profiled DataFrame.

    column_count:
        Number of columns in the profiled DataFrame.

    datatypes:
        Result from ``DatatypeInspector``.

    missing_values:
        Result from ``MissingValueInspector``.

    duplicates:
        Result from ``DuplicateInspector``.

    memory:
        Result from ``MemoryInspector``.

    statistics:
        Result from ``StatisticsInspector``.

    text_quality:
        Per-column ``TextQualityResult`` values. Populated by the
        text-quality analyzer added in a later task (14.2); ``None`` when
        text-quality analysis has not been run.

    anomalies:
        Forward-compatible slot for per-column ``AnomalyResult`` values.
        Populated by the anomaly analyzer added in a later task; ``None``
        when anomaly detection has not been run.

    near_duplicate:
        Forward-compatible slot for a ``NearDuplicateResult``. Populated by
        the near-duplicate detector added in a later task; ``None`` when
        near-duplicate detection has not been run.
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


__all__ = [
    "DatasetProfile",
    "TextQualityResult",
]
