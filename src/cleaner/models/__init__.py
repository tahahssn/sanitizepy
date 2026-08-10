"""
Cleaner Data Models

Public exports for all data models used throughout the Cleaner
library.

Users can simply import models directly:

    from cleaner.models import CleanerReport
    from cleaner.models import Recommendation
    from cleaner.models import InspectionResult
"""

from .base import CleanerBaseModel

from .inspection import (
    ColumnInspection,
    InspectionResult,
)

from .recommendations import (
    Recommendation,
    RecommendationAction,
    RecommendationSeverity,
)

from .report import (
    CleanerReport,
    ExecutionMetadata,
    ReportSummary,
)

__all__ = [
    # Base
    "CleanerBaseModel",

    # Inspection
    "ColumnInspection",
    "InspectionResult",

    # Recommendations
    "Recommendation",
    "RecommendationAction",
    "RecommendationSeverity",

    # Reports
    "CleanerReport",
    "ExecutionMetadata",
    "ReportSummary",
]