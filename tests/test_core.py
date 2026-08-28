"""
Tests for cleaner.core (Cleaner) and cleaner public API
"""

from __future__ import annotations

import cleaner
from cleaner import (
    DEFAULT_CONFIG,
    VERSION,
    VERSION_INFO,
    Cleaner,
    CleanerConfig,
    CleanerError,
    ConfigurationError,
    DataValidationError,
    EngineError,
    get_version,
)
from cleaner.core import Cleaner as CoreCleaner


class TestCleanerInit:
    def test_default_construction(self):
        c = Cleaner()
        assert c is not None

    def test_default_config_is_used_when_none_given(self):
        c = Cleaner()
        assert c.config is DEFAULT_CONFIG

    def test_custom_config_is_stored(self):
        custom = CleanerConfig(preview_rows=5)
        c = Cleaner(config=custom)
        assert c.config is custom

    def test_config_property_returns_cleaner_config(self):
        c = Cleaner()
        assert isinstance(c.config, CleanerConfig)

    def test_logger_property_returns_logger(self):
        import logging

        c = Cleaner()
        assert isinstance(c.logger, logging.Logger)


class TestCleanerPublicAPI:
    def test_cleaner_importable_from_package(self):
        assert Cleaner is cleaner.Cleaner

    def test_cleaner_config_importable_from_package(self):
        assert CleanerConfig is cleaner.CleanerConfig

    def test_default_config_importable_from_package(self):
        assert DEFAULT_CONFIG is cleaner.DEFAULT_CONFIG

    def test_version_importable_from_package(self):
        assert VERSION is cleaner.VERSION

    def test_version_info_importable_from_package(self):
        assert VERSION_INFO is cleaner.VERSION_INFO

    def test_get_version_importable_from_package(self):
        assert get_version is cleaner.get_version

    def test_cleaner_error_importable_from_package(self):
        assert CleanerError is cleaner.CleanerError

    def test_configuration_error_importable_from_package(self):
        assert ConfigurationError is cleaner.ConfigurationError

    def test_data_validation_error_importable_from_package(self):
        assert DataValidationError is cleaner.DataValidationError

    def test_engine_error_importable_from_package(self):
        assert EngineError is cleaner.EngineError

    def test_core_cleaner_is_same_as_api_cleaner(self):
        assert CoreCleaner is Cleaner
