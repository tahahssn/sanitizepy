"""
sanitizepy.engine.base
~~~~~~~~~~~~~~~~~~~

Base engine shared by all processing components.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import DEFAULT_CONFIG, CleanerConfig
from ..logger import get_logger


class BaseEngine(ABC):
    """
    Base class for every engine inside Cleaner.

    All inspection, cleaning, preprocessing and reporting
    engines inherit from this class.
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
        Return the engine configuration.
        """
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """
        Return the engine logger.
        """
        return self._logger

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the engine.

        Every concrete engine must implement this method.
        """
        raise NotImplementedError


__all__ = [
    "BaseEngine",
]
