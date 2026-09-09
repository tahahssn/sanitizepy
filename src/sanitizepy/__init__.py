"""
sanitizepy

An open-source Python engine for automated tabular data quality inspection,
explainable cleaning, and preprocessing.
"""

from __future__ import annotations

from .api import (
    DEFAULT_CONFIG,
    VERSION,
    VERSION_INFO,
    AnomalyInspector,
    Cleaner,
    CleanerConfig,
    CleanerError,
    CleaningEngine,
    CleaningPlan,
    CleaningResult,
    ColumnContract,
    ConfigurationError,
    DataContract,
    DatasetHealthReport,
    DatasetIssue,
    DatasetProfile,
    DatasetProfiler,
    DataValidationError,
    EncodingRepairOperation,
    EngineError,
    MissingTokenOperation,
    NearDuplicateDetector,
    NearDuplicateRemovalOperation,
    OperationRegistry,
    OperationResult,
    ReplayablePlan,
    ReplayOperation,
    TextNormalizationOperation,
    TextQualityAnalyzer,
    TextQualityResult,
    TypeCoercionOperation,
    clean,
    get_version,
    inspect,
    plan,
    profile_to_report,
    registry,
)

__all__ = [
    "Cleaner",
    "inspect",
    "plan",
    "clean",
    "DatasetHealthReport",
    "DatasetIssue",
    "CleaningPlan",
    "CleaningResult",
    "OperationResult",
    "CleaningEngine",
    "CleanerConfig",
    "DEFAULT_CONFIG",
    "CleanerError",
    "ConfigurationError",
    "DataValidationError",
    "EngineError",
    "VERSION",
    "VERSION_INFO",
    "get_version",
    # --- Additive exports (production-grade-evolution) ---
    # Operations
    "MissingTokenOperation",
    "TypeCoercionOperation",
    "TextNormalizationOperation",
    "EncodingRepairOperation",
    "NearDuplicateRemovalOperation",
    "OperationRegistry",
    "registry",
    # Inspectors
    "DatasetProfiler",
    "profile_to_report",
    "AnomalyInspector",
    "NearDuplicateDetector",
    "TextQualityAnalyzer",
    # Models
    "DatasetProfile",
    "TextQualityResult",
    "ColumnContract",
    "DataContract",
    "ReplayablePlan",
    "ReplayOperation",
]

__version__ = VERSION
