"""
cleaner.constants
~~~~~~~~~~~~~~~~~

Library-wide immutable constants.

Only values that are intended to remain stable across the entire package
should exist here. Runtime configuration belongs in config.py.
"""

from __future__ import annotations

from pathlib import Path

# ============================================================================
# PACKAGE INFORMATION
# ============================================================================

PACKAGE_NAME: str = "cleaner"

# ============================================================================
# FILESYSTEM
# ============================================================================

DEFAULT_ENCODING: str = "utf-8"

HOME_DIRECTORY: Path = Path.home()

# ============================================================================
# DATA TYPES
# ============================================================================

NUMERIC_DTYPES: frozenset[str] = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    }
)

BOOLEAN_DTYPES: frozenset[str] = frozenset(
    {
        "bool",
        "boolean",
    }
)

STRING_DTYPES: frozenset[str] = frozenset(
    {
        "object",
        "string",
    }
)

DATETIME_DTYPES: frozenset[str] = frozenset(
    {
        "datetime64[ns]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[s]",
        "timedelta64[ns]",
    }
)

CATEGORICAL_DTYPES: frozenset[str] = frozenset(
    {
        "category",
    }
)

SUPPORTED_TABULAR_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".csv",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xls",
        ".xlsm",
        ".parquet",
        ".feather",
        ".json",
        ".jsonl",
        ".pickle",
        ".pkl",
    }
)

# ============================================================================
# MISSING VALUE TOKENS
# ============================================================================

DEFAULT_MISSING_VALUE_TOKENS: frozenset[str] = frozenset(
    {
        "",
        " ",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "nil",
        "?",
        "-",
    }
)

# ============================================================================
# INSPECTION LIMITS
# ============================================================================

DEFAULT_PREVIEW_ROWS: int = 10

DEFAULT_TOP_VALUES: int = 10

DEFAULT_FLOAT_PRECISION: int = 6

# ============================================================================
# MEMORY
# ============================================================================

BYTES_IN_KB: int = 1024
BYTES_IN_MB: int = BYTES_IN_KB * 1024
BYTES_IN_GB: int = BYTES_IN_MB * 1024

# ============================================================================
# REPORTING
# ============================================================================

DEFAULT_REPORT_TITLE: str = "Cleaner Inspection Report"

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PACKAGE_NAME",
    "DEFAULT_ENCODING",
    "HOME_DIRECTORY",
    "NUMERIC_DTYPES",
    "BOOLEAN_DTYPES",
    "STRING_DTYPES",
    "DATETIME_DTYPES",
    "CATEGORICAL_DTYPES",
    "SUPPORTED_TABULAR_EXTENSIONS",
    "DEFAULT_MISSING_VALUE_TOKENS",
    "DEFAULT_PREVIEW_ROWS",
    "DEFAULT_TOP_VALUES",
    "DEFAULT_FLOAT_PRECISION",
    "BYTES_IN_KB",
    "BYTES_IN_MB",
    "BYTES_IN_GB",
    "DEFAULT_REPORT_TITLE",
]
