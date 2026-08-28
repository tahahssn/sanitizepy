"""
Cleaner Data Models

Public exports for all data models used throughout the Cleaner
library.

Users can simply import models directly:

    from cleaner.models import CleanerReport
    from cleaner.models import Recommendation
    from cleaner.models import InspectionResult
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
    # Reports
    "CleanerReport",
    "ExecutionMetadata",
    "ReportSummary",
]
