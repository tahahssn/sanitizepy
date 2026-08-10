"""
Tests for cleaner.logger
"""

from __future__ import annotations

import logging

from cleaner.logger import get_logger, logger


class TestGetLogger:
    def test_returns_logger_instance(self):
        result = get_logger()
        assert isinstance(result, logging.Logger)

    def test_root_logger_name(self):
        result = get_logger()
        assert result.name == "cleaner"

    def test_named_logger_includes_package_prefix(self):
        result = get_logger("MyComponent")
        assert result.name == "cleaner.MyComponent"

    def test_none_name_returns_root_logger(self):
        result = get_logger(None)
        assert result.name == "cleaner"

    def test_logger_has_handler(self):
        result = get_logger("TestHandler")
        assert len(result.handlers) > 0

    def test_logger_level_is_info(self):
        result = get_logger()
        assert result.level == logging.INFO

    def test_logger_propagate_is_false(self):
        result = get_logger()
        assert result.propagate is False

    def test_same_name_returns_same_logger(self):
        a = get_logger("same")
        b = get_logger("same")
        assert a is b


class TestModuleLevelLogger:
    def test_logger_is_logging_logger(self):
        assert isinstance(logger, logging.Logger)

    def test_logger_name_is_cleaner(self):
        assert logger.name == "cleaner"
