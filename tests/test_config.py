"""
Tests for cleaner.config
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleaner.config import DEFAULT_CONFIG, CleanerConfig
from cleaner.constants import (
    DEFAULT_ENCODING,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_PREVIEW_ROWS,
    DEFAULT_TOP_VALUES,
)


class TestCleanerConfigDefaults:
    def test_encoding_default(self):
        config = CleanerConfig()
        assert config.encoding == DEFAULT_ENCODING

    def test_float_precision_default(self):
        config = CleanerConfig()
        assert config.float_precision == DEFAULT_FLOAT_PRECISION

    def test_preview_rows_default(self):
        config = CleanerConfig()
        assert config.preview_rows == DEFAULT_PREVIEW_ROWS

    def test_top_values_default(self):
        config = CleanerConfig()
        assert config.top_values == DEFAULT_TOP_VALUES

    def test_report_directory_default_none(self):
        config = CleanerConfig()
        assert config.report_directory is None

    def test_enable_logging_default_true(self):
        config = CleanerConfig()
        assert config.enable_logging is True

    def test_missing_value_tokens_is_frozenset(self):
        config = CleanerConfig()
        assert isinstance(config.missing_value_tokens, frozenset)

    def test_missing_value_tokens_normalized_to_lowercase(self):
        """All tokens must be casefolded per __post_init__."""
        config = CleanerConfig()
        for token in config.missing_value_tokens:
            assert token == token.casefold()

    def test_default_missing_tokens_include_na(self):
        config = CleanerConfig()
        assert "na" in config.missing_value_tokens


class TestCleanerConfigValidation:
    def test_preview_rows_zero_raises(self):
        with pytest.raises(ValueError, match="preview_rows"):
            CleanerConfig(preview_rows=0)

    def test_preview_rows_negative_raises(self):
        with pytest.raises(ValueError, match="preview_rows"):
            CleanerConfig(preview_rows=-1)

    def test_top_values_zero_raises(self):
        with pytest.raises(ValueError, match="top_values"):
            CleanerConfig(top_values=0)

    def test_top_values_negative_raises(self):
        with pytest.raises(ValueError, match="top_values"):
            CleanerConfig(top_values=-1)

    def test_float_precision_negative_raises(self):
        with pytest.raises(ValueError, match="float_precision"):
            CleanerConfig(float_precision=-1)

    def test_float_precision_zero_is_valid(self):
        config = CleanerConfig(float_precision=0)
        assert config.float_precision == 0


class TestCleanerConfigReportDirectory:
    def test_report_directory_accepts_string(self, tmp_path):
        config = CleanerConfig(report_directory=tmp_path)
        assert isinstance(config.report_directory, Path)

    def test_report_directory_resolves_path(self, tmp_path):
        config = CleanerConfig(report_directory=tmp_path)
        assert config.report_directory == tmp_path.resolve()

    def test_report_directory_none_remains_none(self):
        config = CleanerConfig(report_directory=None)
        assert config.report_directory is None


class TestCleanerConfigCustomTokens:
    def test_custom_tokens_stored_as_frozenset(self):
        config = CleanerConfig(missing_value_tokens=frozenset({"N/A", "MISSING"}))
        assert isinstance(config.missing_value_tokens, frozenset)

    def test_custom_tokens_casefolded(self):
        config = CleanerConfig(missing_value_tokens=frozenset({"N/A", "MISSING"}))
        assert "n/a" in config.missing_value_tokens
        assert "missing" in config.missing_value_tokens


class TestDefaultConfig:
    def test_default_config_is_cleaner_config(self):
        assert isinstance(DEFAULT_CONFIG, CleanerConfig)

    def test_default_config_uses_default_encoding(self):
        assert DEFAULT_CONFIG.encoding == DEFAULT_ENCODING

    def test_default_config_uses_default_preview_rows(self):
        assert DEFAULT_CONFIG.preview_rows == DEFAULT_PREVIEW_ROWS
