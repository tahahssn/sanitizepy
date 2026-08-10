"""
cleaner.reports.engine
~~~~~~~~~~~~~~~~~~~~~~

Report generation engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..engine.base import BaseEngine
from .report import Report, ReportSection


class ReportEngine(BaseEngine):
    """
    Engine responsible for constructing structured reports.

    The ReportEngine does not render or export reports. It converts
    supplied processing results into the canonical :class:`Report`
    representation used by the rest of Cleaner.
    """

    def run(
        self,
        results: Mapping[str, Any] | Iterable[ReportSection],
        *,
        title: str = "Cleaner Report",
        metadata: Mapping[str, Any] | None = None,
    ) -> Report:
        """
        Generate a structured report from processing results.

        Parameters
        ----------
        results:
            Structured processing results. A mapping is converted into
            report sections, while an iterable of ``ReportSection``
            instances is preserved as supplied.

        title:
            Human-readable report title.

        metadata:
            Optional report metadata.

        Returns
        -------
        Report
            A structured immutable report.

        Raises
        ------
        TypeError
            If results contain unsupported section values.
        """
        sections = self._build_sections(results)

        return Report(
            title=title,
            sections=sections,
            metadata={} if metadata is None else dict(metadata),
        )

    def _build_sections(
        self,
        results: Mapping[str, Any] | Iterable[ReportSection],
    ) -> tuple[ReportSection, ...]:
        """
        Convert processing results into report sections.
        """
        if isinstance(results, Mapping):
            return tuple(
                ReportSection(
                    name=str(name),
                    title=self._format_title(str(name)),
                    content=content,
                )
                for name, content in results.items()
            )

        sections = tuple(results)

        if not all(isinstance(section, ReportSection) for section in sections):
            raise TypeError(
                "Iterable report results must contain only ReportSection "
                "instances."
            )

        return sections

    @staticmethod
    def _format_title(name: str) -> str:
        """
        Convert a section identifier into a human-readable title.
        """
        return name.replace("_", " ").strip().title()


__all__ = [
    "ReportEngine",
]