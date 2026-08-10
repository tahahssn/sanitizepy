"""
Tests for cleaner.constants
"""

from __future__ import annotations

from pathlib import Path

from cleaner.constants import (
    BOOLEAN_DTYPES,
    BYTES_IN_GB,
    BYTES_IN_KB,
    BYTES_IN_MB,
    CATEGORICAL_DTYPES,
    DATETIME_DTYPES,
    DEFAULT_ENCODING,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_MISSING_VALUE_TOKENS,
    DEFAULT_PREVIEW_ROWS,
    DEFAULT_REPORT_TITLE,
    DEFAULT_TOP_VALUES,
    HOME_DIRECTORY,
    NUMERIC_DTYPES,
    PACKAGE_NAME,
    STRING_DTYPES,
    SUPPORTED_TABULAR_EXTENSIONS,
)


class TestPackageConstants:
    def test_package_name(self):
        assert PACKAGE_NAME == "cleaner"

    def test_default_encoding(self):
        assert DEFAULT_ENCODING == "utf-8"

    def test_home_directory_is_path(self):
        assert isinstance(HOME_DIRECTORY, Path)

    def test_home_directory_exists(self):
        assert HOME_DIRECTORY.exists()


class TestDtypeConstants:
    def test_numeric_dtypes_is_frozenset(self):
        assert isinstance(NUMERIC_DTYPES, frozenset)

    def test_numeric_dtypes_contains_int64(self):
        assert "int64" in NUMERIC_DTYPES

    def test_numeric_dtypes_contains_float64(self):
        assert "float64" in NUMERIC_DTYPES

    def test_boolean_dtypes_is_frozenset(self):
        assert isinstance(BOOLEAN_DTYPES, frozenset)

    def test_boolean_dtypes_contains_bool(self):
        assert "bool" in BOOLEAN_DTYPES

    def test_string_dtypes_is_frozenset(self):
        assert isinstance(STRING_DTYPES, frozenset)

    def test_string_dtypes_contains_object(self):
        assert "object" in STRING_DTYPES

    def test_datetime_dtypes_is_frozenset(self):
        assert isinstance(DATETIME_DTYPES, frozenset)

    def test_categorical_dtypes_is_frozenset(self):
        assert isinstance(CATEGORICAL_DTYPES, frozenset)

    def test_categorical_dtypes_contains_category(self):
        assert "category" in CATEGORICAL_DTYPES


class TestFileExtensions:
    def test_supported_extensions_is_frozenset(self):
        assert isinstance(SUPPORTED_TABULAR_EXTENSIONS, frozenset)

    def test_csv_is_supported(self):
        assert ".csv" in SUPPORTED_TABULAR_EXTENSIONS

    def test_parquet_is_supported(self):
        assert ".parquet" in SUPPORTED_TABULAR_EXTENSIONS

    def test_json_is_supported(self):
        assert ".json" in SUPPORTED_TABULAR_EXTENSIONS


class TestMissingValueTokens:
    def test_tokens_is_frozenset(self):
        assert isinstance(DEFAULT_MISSING_VALUE_TOKENS, frozenset)

    def test_tokens_contains_na(self):
        assert "na" in DEFAULT_MISSING_VALUE_TOKENS

    def test_tokens_contains_nan(self):
        assert "nan" in DEFAULT_MISSING_VALUE_TOKENS

    def test_tokens_contains_null(self):
        assert "null" in DEFAULT_MISSING_VALUE_TOKENS

    def test_tokens_non_empty(self):
        assert len(DEFAULT_MISSING_VALUE_TOKENS) > 0


class TestInspectionLimits:
    def test_preview_rows_positive(self):
        assert DEFAULT_PREVIEW_ROWS > 0

    def test_top_values_positive(self):
        assert DEFAULT_TOP_VALUES > 0

    def test_float_precision_non_negative(self):
        assert DEFAULT_FLOAT_PRECISION >= 0


class TestMemoryConstants:
    def test_bytes_in_kb(self):
        assert BYTES_IN_KB == 1024

    def test_bytes_in_mb(self):
        assert BYTES_IN_MB == 1024 * 1024

    def test_bytes_in_gb(self):
        assert BYTES_IN_GB == 1024 * 1024 * 1024

    def test_hierarchy(self):
        assert BYTES_IN_KB < BYTES_IN_MB < BYTES_IN_GB


class TestReportingConstants:
    def test_default_report_title_is_string(self):
        assert isinstance(DEFAULT_REPORT_TITLE, str)

    def test_default_report_title_non_empty(self):
        assert len(DEFAULT_REPORT_TITLE) > 0
