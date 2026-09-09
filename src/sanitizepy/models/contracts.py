"""
Data contract models.

These models express declarative, column-level expectations that a
DataFrame can be validated against deterministically.

A ``ColumnContract`` describes the expectations for a single column
(dtype, nullability, allowed values, numeric range, regex pattern, and
uniqueness). A ``DataContract`` groups per-column contracts into a single
declarative schema keyed by column name.

The contracts are purely declarative: they carry no validation logic.
Deterministic validation and ``RuleResult`` production are handled by the
rules subsystem.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sanitizepy.models.base import BaseCleanerModel


class ColumnContract(BaseCleanerModel):
    """
    Declarative expectations for a single column.

    Every field is optional. An unset (``None``) field declares no
    expectation for that concern, so a contract can constrain only the
    aspects a caller cares about.
    """

    dtype: str | None = Field(
        default=None,
        description="Expected column dtype, e.g. 'int64' or 'string'.",
    )

    nullable: bool | None = Field(
        default=None,
        description="Whether missing values are permitted in the column.",
    )

    allowed_values: tuple[Any, ...] | None = Field(
        default=None,
        description="Exhaustive set of permitted values for the column.",
    )

    min_value: float | None = Field(
        default=None,
        description="Inclusive lower bound for numeric values.",
    )

    max_value: float | None = Field(
        default=None,
        description="Inclusive upper bound for numeric values.",
    )

    regex: str | None = Field(
        default=None,
        description="Regex pattern every non-missing value must match.",
    )

    unique: bool | None = Field(
        default=None,
        description="Whether the column must contain only unique values.",
    )


class DataContract(BaseCleanerModel):
    """
    A declarative schema mapping column names to their expectations.
    """

    columns: dict[str, ColumnContract] = Field(
        default_factory=dict,
        description="Mapping of column name to its declared expectations.",
    )


__all__ = [
    "ColumnContract",
    "DataContract",
]
