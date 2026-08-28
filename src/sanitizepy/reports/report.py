"""
sanitizepy.reports.report
~~~~~~~~~~~~~~~~~~~~~~

Report data structures produced by the report engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ReportSection:
    """
    A single section of a generated report.

    A section contains a stable identifier, a human-readable title,
    and arbitrary structured content produced by an engine.
    """

    name: str
    title: str
    content: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Report section name cannot be empty.")

        if not self.title:
            raise ValueError("Report section title cannot be empty.")


@dataclass(frozen=True, slots=True)
class Report:
    """
    Immutable report produced by the report engine.

    A Report is a structured representation of processing results.
    Rendering and exporting are intentionally handled by separate
    components.
    """

    title: str
    sections: tuple[ReportSection, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("Report title cannot be empty.")

        if self.created_at.tzinfo is None:
            raise ValueError("Report created_at must be timezone-aware.")

        if not isinstance(self.sections, tuple):
            raise TypeError("Report sections must be a tuple.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("Report metadata must implement Mapping.")

    @property
    def section_count(self) -> int:
        """
        Return the number of sections contained in the report.
        """
        return len(self.sections)

    def get_section(self, name: str) -> ReportSection | None:
        """
        Return a section by its identifier.

        Parameters
        ----------
        name:
            Section identifier.

        Returns
        -------
        ReportSection | None
            The matching section, or None when it does not exist.
        """
        for section in self.sections:
            if section.name == name:
                return section

        return None

    def has_section(self, name: str) -> bool:
        """
        Return whether a section with the given identifier exists.
        """
        return self.get_section(name) is not None


__all__ = [
    "Report",
    "ReportSection",
]
