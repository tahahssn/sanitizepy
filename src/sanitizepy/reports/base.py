"""
sanitizepy.reports.base
~~~~~~~~~~~~~~~~~~~~

Base abstraction for all report implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReport(ABC):
    """
    Abstract base class for every report.

    Reports expose multiple representations of the same
    information without modifying the underlying data.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Return the report as a Python dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def to_text(self) -> str:
        """
        Return the report as plain text.
        """
        raise NotImplementedError

    @abstractmethod
    def to_markdown(self) -> str:
        """
        Return the report as Markdown.
        """
        raise NotImplementedError

    @abstractmethod
    def to_html(self) -> str:
        """
        Return the report as HTML.
        """
        raise NotImplementedError


__all__ = [
    "BaseReport",
]
