"""
Tests for sanitizepy.config
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanitizepy.config import DEFAULT_CONFIG, CleanerConfig
from sanitizepy.constants import (
    DEFAULT_ENCODING,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_MISSING_VALUE_TOKENS,
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

    # --- task 4.2 additions ---

    def test_configured_tokens_extend_not_replace_defaults(self):
        """Defaults must still be present when custom tokens are supplied."""
        config = CleanerConfig(missing_value_tokens=frozenset({"MISSING", "TBD"}))
        # Original defaults survive
        for default in DEFAULT_MISSING_VALUE_TOKENS:
            assert (
                default.casefold() in config.missing_value_tokens
            ), f"default token {default!r} missing after custom tokens configured"

    def test_configured_tokens_added_on_top_of_defaults(self):
        """Custom tokens appear in the effective set alongside defaults."""
        config = CleanerConfig(missing_value_tokens=frozenset({"custom_sentinel"}))
        assert "custom_sentinel" in config.missing_value_tokens
        assert "na" in config.missing_value_tokens  # default still present

    def test_no_custom_tokens_yields_only_defaults(self):
        """Empty extension produces effective set equal to casefolded defaults."""
        config = CleanerConfig()
        expected = frozenset(t.casefold() for t in DEFAULT_MISSING_VALUE_TOKENS)
        assert config.missing_value_tokens == expected

    def test_effective_set_is_superset_of_defaults(self):
        """Any configured extension must be a strict superset of defaults."""
        extra = frozenset({"bespoke_token"})
        config = CleanerConfig(missing_value_tokens=extra)
        defaults_casefolded = frozenset(
            t.casefold() for t in DEFAULT_MISSING_VALUE_TOKENS
        )
        assert defaults_casefolded.issubset(config.missing_value_tokens)

    def test_duplicate_custom_token_matching_default_is_deduplicated(self):
        """A custom token that matches a default (possibly different case) does
        not inflate the set — frozenset semantics ensure uniqueness."""
        config_with_dup = CleanerConfig(missing_value_tokens=frozenset({"NA", "NaN"}))
        config_baseline = CleanerConfig()
        # All defaults still present and no double-counting surprises
        for token in config_baseline.missing_value_tokens:
            assert token in config_with_dup.missing_value_tokens

    def test_effective_tokens_all_casefolded(self):
        """Every token in the final set must be fully casefolded."""
        config = CleanerConfig(missing_value_tokens=frozenset({"MixedCase", "UPPER"}))
        for token in config.missing_value_tokens:
            assert token == token.casefold(), f"token {token!r} is not casefolded"

    def test_empty_frozenset_extension_leaves_defaults_intact(self):
        """Passing an explicit empty frozenset is the same as passing nothing."""
        config_default = CleanerConfig()
        config_empty = CleanerConfig(missing_value_tokens=frozenset())
        assert config_default.missing_value_tokens == config_empty.missing_value_tokens


class TestDefaultConfig:
    def test_default_config_is_sanitizepy_config(self):
        assert isinstance(DEFAULT_CONFIG, CleanerConfig)

    def test_default_config_uses_default_encoding(self):
        assert DEFAULT_CONFIG.encoding == DEFAULT_ENCODING

    def test_default_config_uses_default_preview_rows(self):
        assert DEFAULT_CONFIG.preview_rows == DEFAULT_PREVIEW_ROWS
