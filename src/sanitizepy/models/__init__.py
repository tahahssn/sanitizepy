"""
Cleaner Data Models

Public exports for all data models used throughout the Cleaner
library.

Users can simply import models directly:

    from sanitizepy.models import CleanerReport
    from sanitizepy.models import Recommendation
    from sanitizepy.models import InspectionResult
"""

from .base import (
    BaseCleanerModel,
    ColumnReference,
    DatasetInfo,
    DataShape,
    ExecutionTime,
    IdentifiedModel,
    MetadataModel,
    ResourceUsage,
    TimestampedModel,
    VersionInfo,
)
from .contracts import (
    ColumnContract,
    DataContract,
)
from .inspection import (
    CategoricalStatistics,
    ColumnInspection,
    ColumnIssue,
    DataTypeResult,
    DuplicateResult,
    InspectionResult,
    InspectionStatus,
    InspectionSummary,
    MissingValueResult,
    NumericStatistics,
    Severity,
)
from .recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationCategory,
    RecommendationGroup,
    RecommendationImpact,
    RecommendationPriority,
    RecommendationReason,
    RecommendationResult,
    RecommendationSummary,
)
from .replay import (
    ReplayablePlan,
    ReplayOperation,
)
from .report import (
    CleanerReport,
    ExecutionMetadata,
    ReportSummary,
)

__all__ = [
    # Base
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
    # Contracts
    "ColumnContract",
    "DataContract",
    # Inspection
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
    # Recommendations
    "RecommendationCategory",
    "RecommendationPriority",
    "RecommendationAction",
    "RecommendationReason",
    "RecommendationImpact",
    "Recommendation",
    "RecommendationGroup",
    "RecommendationSummary",
    "RecommendationResult",
    # Replay
    "ReplayOperation",
    "ReplayablePlan",
    # Reports
    "CleanerReport",
    "ExecutionMetadata",
    "ReportSummary",
]
