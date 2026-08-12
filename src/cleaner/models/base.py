"""
Base models used throughout the Cleaner library.

All public models inherit from BaseCleanerModel to provide:

- Strict validation
- Immutable models
- JSON serialization
- Timestamp support
- Consistent configuration
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """
    return datetime.now(timezone.utc)


class BaseCleanerModel(BaseModel):
    """
    Root model inherited by every public model.

    Features
    --------
    - Frozen (immutable)
    - Strict validation
    - Assignment validation
    - Extra fields forbidden
    - Automatic serialization support
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
        extra="forbid",
        populate_by_name=True,
        use_enum_values=True,
    )


class IdentifiedModel(BaseCleanerModel):
    """
    Adds a globally unique identifier.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique object identifier.",
    )


class TimestampedModel(BaseCleanerModel):
    """
    Adds creation timestamp.
    """

    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC creation timestamp.",
    )


class MetadataModel(BaseCleanerModel):
    """
    Generic metadata container.

    Used when inspectors or engines need
    arbitrary structured metadata.
    """

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )


class ResourceUsage(BaseCleanerModel):
    """
    Memory usage statistics.
    """

    memory_bytes: int = Field(
        ge=0,
        description="Memory usage in bytes.",
    )

    memory_mb: float = Field(
        ge=0,
        description="Memory usage in megabytes.",
    )

    memory_gb: float = Field(
        ge=0,
        description="Memory usage in gigabytes.",
    )


class ExecutionTime(BaseCleanerModel):
    """
    Execution timing statistics.
    """

    seconds: float = Field(
        ge=0,
        description="Execution time in seconds.",
    )


class DataShape(BaseCleanerModel):
    """
    Represents dataframe dimensions.
    """

    rows: int = Field(
        ge=0,
        description="Number of rows.",
    )

    columns: int = Field(
        ge=0,
        description="Number of columns.",
    )

    @property
    def size(self) -> int:
        """
        Total number of cells.
        """
        return self.rows * self.columns


class ColumnReference(BaseCleanerModel):
    """
    Reference to a dataframe column.
    """

    name: str = Field(
        min_length=1,
        description="Column name.",
    )

    dtype: str = Field(
        default="unknown",
        min_length=1,
        description="Detected dtype.",
    )


class DatasetInfo(
    IdentifiedModel,
    TimestampedModel,
):
    """
    Basic dataset information shared by reports.
    """

    name: str = Field(
        min_length=1,
        description="Dataset name.",
    )

    shape: DataShape

    memory: ResourceUsage


class VersionInfo(BaseCleanerModel):
    """
    Library version information.
    """

    library: str

    version: str

    python: str


__all__ = [
    "BaseCleanerModel",
    "IdentifiedModel",
    "TimestampedModel",
    "MetadataModel",
    "ResourceUsage",
    "ExecutionTime",
    "DataShape",
    "ColumnReference",
    "DatasetInfo",
    "VersionInfo",
]