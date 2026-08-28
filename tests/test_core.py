"""
Tests for sanitizepy.core (Cleaner) and sanitizepy public API
"""

from __future__ import annotations

import sanitizepy
from sanitizepy import (
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
from sanitizepy.core import Cleaner as CoreCleaner


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

    def test_config_property_returns_sanitizepy_config(self):
        c = Cleaner()
        assert isinstance(c.config, CleanerConfig)

    def test_logger_property_returns_logger(self):
        import logging

        c = Cleaner()
        assert isinstance(c.logger, logging.Logger)


class TestCleanerPublicAPI:
    def test_sanitizepy_importable_from_package(self):
        assert Cleaner is sanitizepy.Cleaner

    def test_sanitizepy_config_importable_from_package(self):
        assert CleanerConfig is sanitizepy.CleanerConfig

    def test_default_config_importable_from_package(self):
        assert DEFAULT_CONFIG is sanitizepy.DEFAULT_CONFIG

    def test_version_importable_from_package(self):
        assert VERSION is sanitizepy.VERSION

    def test_version_info_importable_from_package(self):
        assert VERSION_INFO is sanitizepy.VERSION_INFO

    def test_get_version_importable_from_package(self):
        assert get_version is sanitizepy.get_version

    def test_sanitizepy_error_importable_from_package(self):
        assert CleanerError is sanitizepy.CleanerError

    def test_configuration_error_importable_from_package(self):
        assert ConfigurationError is sanitizepy.ConfigurationError

    def test_data_validation_error_importable_from_package(self):
        assert DataValidationError is sanitizepy.DataValidationError

    def test_engine_error_importable_from_package(self):
        assert EngineError is sanitizepy.EngineError

    def test_core_sanitizepy_is_same_as_api_sanitizepy(self):
        assert CoreCleaner is Cleaner
