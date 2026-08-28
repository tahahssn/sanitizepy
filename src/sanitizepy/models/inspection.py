"""
Inspection models.

These models define the standardized output returned by all
inspection modules.

Every inspector should return immutable, validated models instead
of raw dictionaries.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from sanitizepy.models.base import (
    BaseCleanerModel,
    ColumnReference,
    DatasetInfo,
    ExecutionTime,
)


class Severity(StrEnum):
    """
    Severity level for detected issues.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InspectionStatus(StrEnum):
    """
    Inspection execution status.
    """

    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


class ColumnIssue(BaseCleanerModel):
    """
    Represents an issue detected in a single column.
    """

    column: ColumnReference

    issue: str = Field(
        min_length=1,
        description="Issue description.",
    )

    severity: Severity

    value: float | int | str | None = None

    recommendation: str | None = None


class MissingValueResult(BaseCleanerModel):
    """
    Missing value statistics for one column.
    """

    column: ColumnReference

    missing_count: int = Field(
        ge=0,
    )

    missing_percentage: float = Field(
        ge=0,
        le=100,
    )


class DuplicateResult(BaseCleanerModel):
    """
    Duplicate statistics.
    """

    duplicate_rows: int = Field(
        ge=0,
    )

    duplicate_percentage: float = Field(
        ge=0,
        le=100,
    )


class DataTypeResult(BaseCleanerModel):
    """
    Data type inspection result.
    """

    column: ColumnReference

    detected_dtype: str

    nullable: bool

    unique_values: int = Field(
        ge=0,
    )


class NumericStatistics(BaseCleanerModel):
    """
    Statistics for numeric columns.
    """

    minimum: float

    maximum: float

    mean: float

    median: float

    std: float

    variance: float

    skewness: float

    kurtosis: float

    zeros: int = Field(
        ge=0,
    )


class CategoricalStatistics(BaseCleanerModel):
    """
    Statistics for categorical columns.
    """

    unique_values: int = Field(
        ge=0,
    )

    most_frequent: str | None = None

    frequency: int = Field(
        ge=0,
    )


class ColumnInspection(BaseCleanerModel):
    """
    Complete inspection result for one column.
    """

    column: ColumnReference

    missing: MissingValueResult | None = None

    datatype: DataTypeResult | None = None

    numeric: NumericStatistics | None = None

    categorical: CategoricalStatistics | None = None

    issues: list[ColumnIssue] = Field(
        default_factory=list,
    )


class InspectionSummary(BaseCleanerModel):
    """
    High-level inspection summary.
    """

    total_columns: int = Field(
        ge=0,
    )

    total_rows: int = Field(
        ge=0,
    )

    missing_columns: int = Field(
        ge=0,
    )

    duplicate_rows: int = Field(
        ge=0,
    )

    issues_found: int = Field(
        ge=0,
    )


class InspectionResult(BaseCleanerModel):
    """
    Final object returned by every inspection pipeline.
    """

    dataset: DatasetInfo

    status: InspectionStatus

    execution: ExecutionTime

    summary: InspectionSummary

    duplicates: DuplicateResult

    columns: list[ColumnInspection] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    errors: list[str] = Field(
        default_factory=list,
    )


__all__ = [
    "Severity",
    "InspectionStatus",
    "ColumnIssue",
    "MissingValueResult",
    "DuplicateResult",
    "DataTypeResult",
    "NumericStatistics",
    "CategoricalStatistics",
    "ColumnInspection",
    "InspectionSummary",
    "InspectionResult",
]
