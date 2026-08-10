"""
cleaner.reports.renderers
~~~~~~~~~~~~~~~~~~~~~~~~~

Renderers for converting structured reports into presentation formats.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .report import Report


class BaseRenderer(ABC):
    """
    Base class for all report renderers.
    """

    @abstractmethod
    def render(self, report: Report) -> str:
        """
        Render a report into a string representation.
        """
        raise NotImplementedError


class TextRenderer(BaseRenderer):
    """
    Render a report as human-readable plain text.
    """

    def render(self, report: Report) -> str:
        """
        Render the supplied report as plain text.
        """
        lines: list[str] = [
            report.title,
            "=" * len(report.title),
        ]

        for section in report.sections:
            lines.append("")
            lines.append(section.title)
            lines.append("-" * len(section.title))
            lines.append(self._format_content(section.content))

        return "\n".join(lines)

    @staticmethod
    def _format_content(content: Any) -> str:
        """
        Convert section content into a readable text representation.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, (dict, list, tuple, set)):
            return json.dumps(
                content,
                indent=2,
                default=str,
                ensure_ascii=False,
            )

        return str(content)


class JSONRenderer(BaseRenderer):
    """
    Render a report as JSON.
    """

    def render(self, report: Report) -> str:
        """
        Render the supplied report as a JSON document.
        """
        payload = {
            "title": report.title,
            "created_at": report.created_at.isoformat(),
            "metadata": dict(report.metadata),
            "sections": [
                {
                    "name": section.name,
                    "title": section.title,
                    "content": section.content,
                }
                for section in report.sections
            ],
        }

        return json.dumps(
            payload,
            indent=2,
            default=str,
            ensure_ascii=False,
        )


__all__ = [
    "BaseRenderer",
    "JSONRenderer",
    "TextRenderer",
]