"""
cleaner.core
~~~~~~~~~~~~

Public entry point for the Cleaner library.
"""

from __future__ import annotations

from .config import CleanerConfig, DEFAULT_CONFIG
from .logger import get_logger


class Cleaner:
    """
    Main entry point of the Cleaner library.

    A Cleaner instance stores the global configuration
    that will be shared across all processing engines.
    """

    def __init__(
        self,
        config: CleanerConfig | None = None,
    ) -> None:
        self._config: CleanerConfig = config or DEFAULT_CONFIG
        self._logger = get_logger(self.__class__.__name__)

    @property
    def config(self) -> CleanerConfig:
        """
        Return the active configuration.
        """
        return self._config

    @property
    def logger(self):
        """
        Return the package logger.
        """
        return self._logger


__all__ = [
    "Cleaner",
]