"""
cleaner.rules.rule
~~~~~~~~~~~~~~~~~~

Core data models used by the Rule Engine.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuleSeverity(StrEnum):
    """
    Severity level of a rule.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(StrEnum):
    """
    Logical category for a rule.
    """

    DATA_QUALITY = "data_quality"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    STRUCTURE = "structure"
    CUSTOM = "custom"


class RuleResult(BaseModel):
    """
    Result returned after evaluating a rule.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    rule: str = Field(
        ...,
        description="Unique rule name.",
    )

    passed: bool = Field(
        ...,
        description="Whether the rule passed.",
    )

    severity: RuleSeverity = Field(
        ...,
        description="Severity assigned to the rule.",
    )

    category: RuleCategory = Field(
        ...,
        description="Rule category.",
    )

    message: str = Field(
        ...,
        description="Human-readable outcome.",
    )

    affected_columns: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Columns affected by the rule.",
    )

    affected_rows: int = Field(
        default=0,
        ge=0,
        description="Number of affected rows.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule-specific information.",
    )


__all__ = [
    "RuleSeverity",
    "RuleCategory",
    "RuleResult",
]
