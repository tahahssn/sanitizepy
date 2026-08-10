"""
cleaner.version
~~~~~~~~~~~~~~~~

Single source of truth for package version information.
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# VERSION
# ============================================================================

VERSION: Final[str] = "0.1.0"

VERSION_INFO: Final[tuple[int, int, int]] = (
    0,
    1,
    0,
)

# ============================================================================
# HELPERS
# ============================================================================

def get_version() -> str:
    """
    Return the installed package version.

    Returns
    -------
    str
        Package version.
    """
    return VERSION


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "VERSION",
    "VERSION_INFO",
    "get_version",
]