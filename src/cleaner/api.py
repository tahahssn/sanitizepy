"""
cleaner.api

Stable public API for the Cleaner library.
"""

from __future__ import annotations

from .config import CleanerConfig, DEFAULT_CONFIG
from .core import Cleaner
from .exceptions import (
    CleanerError,
    ConfigurationError,
    DataValidationError,
    EngineError,
)
from .version import VERSION, VERSION_INFO, get_version

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