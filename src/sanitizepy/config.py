"""
sanitizepy.config
~~~~~~~~~~~~~~

Central configuration system for the Cleaner library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_ENCODING,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_MISSING_VALUE_TOKENS,
    DEFAULT_PREVIEW_ROWS,
    DEFAULT_TOP_VALUES,
)


@dataclass(slots=True, kw_only=True)
class CleanerConfig:
    """
    Global configuration for the Cleaner library.

    This configuration is shared across inspection,
    cleaning, preprocessing and reporting modules.
    """

    # =========================================================================
    # GENERAL
    # =========================================================================

    encoding: str = DEFAULT_ENCODING

    float_precision: int = DEFAULT_FLOAT_PRECISION

    # =========================================================================
    # INSPECTION
    # =========================================================================

    preview_rows: int = DEFAULT_PREVIEW_ROWS

    top_values: int = DEFAULT_TOP_VALUES

    missing_value_tokens: frozenset[str] = field(default_factory=frozenset)
    """
    Additional sentinel strings to treat as missing.

    Configured tokens *extend* (not replace) ``DEFAULT_MISSING_VALUE_TOKENS``.
    The default is an empty extension, so the built-in defaults always apply.
    After initialization this holds the effective token set
    (defaults + configured extension), casefolded for case-insensitive matching.
    """

    # =========================================================================
    # REPORTING
    # =========================================================================

    report_directory: Path | None = None

    # =========================================================================
    # LOGGING
    # =========================================================================

    enable_logging: bool = True

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def __post_init__(self) -> None:
        """
        Validate configuration values.
        """

        if self.preview_rows <= 0:
            raise ValueError("'preview_rows' must be greater than zero.")

        if self.top_values <= 0:
            raise ValueError("'top_values' must be greater than zero.")

        if self.float_precision < 0:
            raise ValueError("'float_precision' cannot be negative.")

        if self.report_directory is not None:
            self.report_directory = Path(self.report_directory).expanduser().resolve()

        self.missing_value_tokens = frozenset(
            token.casefold()
            for token in (*DEFAULT_MISSING_VALUE_TOKENS, *self.missing_value_tokens)
        )


DEFAULT_CONFIG = CleanerConfig()

__all__ = [
    "CleanerConfig",
    "DEFAULT_CONFIG",
]
