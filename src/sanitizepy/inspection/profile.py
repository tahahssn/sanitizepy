"""
Dataset profile builder.

``DatasetProfiler`` assembles a :class:`~sanitizepy.models.profile.DatasetProfile`
by delegating to the standalone inspectors and aggregating their immutable
results. It does not reimplement any analysis: each inspector remains the
single source of truth for its own dimension.

The core profile is Core_Stack-only (pandas + stdlib); it never touches any
optional dependency. The forward-compatible ``text_quality``, ``anomalies``
and ``near_duplicate`` slots on ``DatasetProfile`` are left at their ``None``
defaults here and are populated by the dedicated analyzers in later tasks.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Final

import pandas as pd

from sanitizepy.inspection.datatypes import DatatypeInspector
from sanitizepy.inspection.duplicates import DuplicateInspector
from sanitizepy.inspection.memory import MemoryInspector
from sanitizepy.inspection.missing import MissingValueInspector
from sanitizepy.inspection.statistics import StatisticsInspector
from sanitizepy.models.profile import DatasetProfile
from sanitizepy.reports.engine import ReportEngine
from sanitizepy.reports.report import Report


class DatasetProfiler:
    """
    Build an immutable :class:`DatasetProfile` from a DataFrame.

    The profiler reuses the standalone inspectors verbatim, aggregating their
    immutable results rather than recomputing any analysis. It never mutates
    the input DataFrame.
    """

    __slots__: Final = (
        "_missing",
        "_duplicates",
        "_datatypes",
        "_memory",
        "_statistics",
    )

    def __init__(self) -> None:
        self._missing = MissingValueInspector()
        self._duplicates = DuplicateInspector()
        self._datatypes = DatatypeInspector()
        self._memory = MemoryInspector()
        self._statistics = StatisticsInspector()

    def profile(
        self,
        dataframe: pd.DataFrame,
    ) -> DatasetProfile:
        """
        Profile a DataFrame by aggregating the standalone inspector results.

        Parameters
        ----------
        dataframe:
            Input DataFrame. Never mutated.

        Returns
        -------
        DatasetProfile
            Immutable aggregate of the five core inspector results.

        Raises
        ------
        ValueError
            If ``dataframe`` is empty, consistent with the standalone
            inspectors.
        """

        if dataframe.empty:
            raise ValueError("Cannot profile an empty DataFrame.")

        datatypes = self._datatypes.inspect(dataframe)
        missing_values = self._missing.inspect(dataframe)
        duplicates = self._duplicates.inspect(dataframe)
        memory = self._memory.inspect(dataframe)
        statistics = self._statistics.inspect(dataframe)

        return DatasetProfile(
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            datatypes=datatypes,
            missing_values=missing_values,
            duplicates=duplicates,
            memory=memory,
            statistics=statistics,
        )


def profile_to_report(
    profile: DatasetProfile,
    *,
    title: str = "Dataset Profile",
) -> Report:
    """
    Transform a :class:`DatasetProfile` into a canonical :class:`Report`.

    This reuses the existing reports subsystem: it assembles ordered,
    JSON-serializable section content and hands it to the existing
    :class:`ReportEngine`, which produces the canonical :class:`Report`.
    Rendering and exporting are then handled by the existing
    renderers/exporters (``TextRenderer``, ``JSONRenderer``,
    ``StringExporter``, ``FileExporter``). No parallel reporting mechanism is
    introduced.

    Only populated slots are turned into sections. The core inspector
    dimensions (datatypes, missing values, duplicates, memory, statistics) are
    always present; the ``text_quality``, ``anomalies`` and ``near_duplicate``
    slots contribute a section only when populated.

    Section content is built from the immutable inspector result summaries and
    per-column reports (scalar fields only). The pandas-carrying fields on the
    inspector results (masks, usage series, duplicate frames) are intentionally
    excluded so the report stays cleanly renderable and serializable.

    Parameters
    ----------
    profile:
        The profile to render. Not mutated.

    title:
        Human-readable report title.

    Returns
    -------
    Report
        The canonical report produced by :class:`ReportEngine`.
    """

    sections: dict[str, Any] = {
        "overview": {
            "row_count": profile.row_count,
            "column_count": profile.column_count,
        },
        "datatypes": {
            "summary": asdict(profile.datatypes.summary),
            "columns": [asdict(report) for report in profile.datatypes.reports],
        },
        "missing_values": {
            "summary": asdict(profile.missing_values.summary),
            "columns": [
                asdict(report) for report in profile.missing_values.column_reports
            ],
        },
        "duplicates": {
            "summary": asdict(profile.duplicates.summary),
            "duplicate_count": profile.duplicates.duplicate_count,
            "duplicate_indices": list(profile.duplicates.duplicate_indices),
        },
        "memory": {
            "summary": asdict(profile.memory.summary),
            "columns": [asdict(report) for report in profile.memory.reports],
        },
        "statistics": {
            "summary": asdict(profile.statistics.summary),
            "columns": [asdict(report) for report in profile.statistics.reports],
        },
    }

    if profile.text_quality is not None:
        sections["text_quality"] = [asdict(result) for result in profile.text_quality]

    if profile.anomalies is not None:
        sections["anomalies"] = [asdict(result) for result in profile.anomalies]

    if profile.near_duplicate is not None:
        sections["near_duplicate"] = asdict(profile.near_duplicate)

    return ReportEngine().run(sections, title=title)


__all__ = [
    "DatasetProfiler",
    "profile_to_report",
]
