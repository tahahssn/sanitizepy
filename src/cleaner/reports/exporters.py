"""
cleaner.reports.exporters
~~~~~~~~~~~~~~~~~~~~~~~~~

Exporters for writing rendered reports to external destinations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .renderers import BaseRenderer
from .report import Report


class BaseExporter(ABC):
    """
    Base class for all report exporters.
    """

    @abstractmethod
    def export(
        self,
        report: Report,
        renderer: BaseRenderer,
    ) -> str | Path:
        """
        Export a report using the supplied renderer.
        """
        raise NotImplementedError


class StringExporter(BaseExporter):
    """
    Export a report into an in-memory string.

    This exporter does not perform any filesystem or network I/O.
    """

    def export(
        self,
        report: Report,
        renderer: BaseRenderer,
    ) -> str:
        """
        Render and return the report as a string.
        """
        return renderer.render(report)


class FileExporter(BaseExporter):
    """
    Export a rendered report to a filesystem path.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """
        Return the configured output path.
        """
        return self._path

    def export(
        self,
        report: Report,
        renderer: BaseRenderer,
    ) -> Path:
        """
        Render the report and write it to the configured path.

        Parent directories are created when they do not already exist.
        """
        rendered = renderer.render(report)

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._path.write_text(
            rendered,
            encoding="utf-8",
        )

        return self._path


__all__ = [
    "BaseExporter",
    "FileExporter",
    "StringExporter",
]
