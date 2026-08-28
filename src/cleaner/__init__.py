"""
sanitizepy

An open-source Python engine for automated tabular data quality inspection,
explainable cleaning, and preprocessing.
"""

from __future__ import annotations

from .api import (
    Cleaner,
    CleanerConfig,
    CleanerError,
    CleaningEngine,
    CleaningPlan,
    CleaningResult,
    ConfigurationError,
    DEFAULT_CONFIG,
    DataValidationError,
    DatasetHealthReport,
    DatasetIssue,
    EngineError,
    OperationResult,
    VERSION,
    VERSION_INFO,
    clean,
    get_version,
    inspect,
    plan,
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
]

__version__ = VERSION