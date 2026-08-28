"""
sanitizepy.logger
~~~~~~~~~~~~~~

Centralized logging for the Cleaner package.
"""

from __future__ import annotations

import logging
from typing import Final

from .constants import PACKAGE_NAME

# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

_DEFAULT_LOG_LEVEL: Final[int] = logging.INFO

_DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# LOGGER FACTORY
# ============================================================================


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str | None, default=None
        Logger name. If None, the package logger is returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger_name = PACKAGE_NAME if name is None else f"{PACKAGE_NAME}.{name}"

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt=_DEFAULT_LOG_FORMAT,
            datefmt=_DEFAULT_DATE_FORMAT,
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    logger.setLevel(_DEFAULT_LOG_LEVEL)
    logger.propagate = False

    return logger


# ============================================================================
# ROOT LOGGER
# ============================================================================

logger: Final[logging.Logger] = get_logger()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "logger",
    "get_logger",
]
