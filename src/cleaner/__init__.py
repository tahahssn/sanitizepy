"""
Cleaner

Production-ready Python library for automated data inspection,
cleaning, preprocessing, and feature engineering.
"""

from __future__ import annotations

from .api import (
    Cleaner,
    CleanerConfig,
    DEFAULT_CONFIG,
    CleanerError,
    ConfigurationError,
    DataValidationError,
    EngineError,
    VERSION,
    VERSION_INFO,
    get_version,
)

__all__ = [
    "Cleaner",
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