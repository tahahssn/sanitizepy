"""
Tests for sanitizepy.core (Cleaner) and sanitizepy public API
"""

from __future__ import annotations

import inspect as _inspect

import pandas as pd

import sanitizepy
from sanitizepy import (
    DEFAULT_CONFIG,
    VERSION,
    VERSION_INFO,
    Cleaner,
    CleanerConfig,
    CleanerError,
    CleaningPlan,
    CleaningResult,
    ConfigurationError,
    DatasetHealthReport,
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


# ---------------------------------------------------------------------------
# Task 20.3 - Regression tests for public API compatibility
# Validates: Requirements 15.1, 15.2, 15.3, 15.4
# ---------------------------------------------------------------------------


class TestPublicApiCompatibility:
    """
    Regression coverage guaranteeing the original public surface is intact
    and the new capabilities are exposed additively (opt-in only).
    """

    ORIGINAL_EXPORTS = (
        "Cleaner",
        "inspect",
        "plan",
        "clean",
        "DatasetHealthReport",
        "DatasetIssue",
        "CleaningPlan",
        "CleaningResult",
        "OperationResult",
        "CleaningEngine",
        "CleanerConfig",
        "DEFAULT_CONFIG",
        "CleanerError",
        "ConfigurationError",
        "DataValidationError",
        "EngineError",
        "VERSION",
        "VERSION_INFO",
        "get_version",
    )

    ADDITIVE_EXPORTS = (
        # Operations
        "MissingTokenOperation",
        "TypeCoercionOperation",
        "TextNormalizationOperation",
        "EncodingRepairOperation",
        "NearDuplicateRemovalOperation",
        "OperationRegistry",
        "registry",
        # Inspectors
        "DatasetProfiler",
        "profile_to_report",
        "AnomalyInspector",
        "NearDuplicateDetector",
        "TextQualityAnalyzer",
        # Models
        "DatasetProfile",
        "TextQualityResult",
        "ColumnContract",
        "DataContract",
        "ReplayablePlan",
        "ReplayOperation",
    )

    def test_all_original_exports_importable(self):
        for name in self.ORIGINAL_EXPORTS:
            assert hasattr(sanitizepy, name), f"missing original export: {name}"

    def test_original_exports_listed_in_dunder_all(self):
        for name in self.ORIGINAL_EXPORTS:
            assert name in sanitizepy.__all__, f"{name} not in __all__"

    def test_all_additive_exports_importable(self):
        for name in self.ADDITIVE_EXPORTS:
            assert hasattr(sanitizepy, name), f"missing additive export: {name}"

    def test_additive_exports_listed_in_dunder_all(self):
        for name in self.ADDITIVE_EXPORTS:
            assert name in sanitizepy.__all__, f"{name} not in __all__"

    def test_cleaner_core_method_signatures_unchanged(self):
        inspect_sig = _inspect.signature(Cleaner.inspect)
        assert list(inspect_sig.parameters) == ["self", "dataframe"]

        plan_sig = _inspect.signature(Cleaner.plan)
        assert list(plan_sig.parameters) == ["self", "target"]

        clean_sig = _inspect.signature(Cleaner.clean)
        clean_params = clean_sig.parameters
        assert list(clean_params) == ["self", "dataframe", "plan", "dry_run"]
        assert clean_params["plan"].default is None
        assert clean_params["dry_run"].default is False

    def test_cleaner_facade_methods_exist(self):
        # New additive facade methods are present but do not disturb the
        # existing method surface.
        assert callable(getattr(Cleaner, "profile", None))
        assert callable(getattr(Cleaner, "validate", None))

    def test_module_level_plan_signature_unchanged(self):
        # `sanitizepy.plan` still points at the original core helper — it
        # has no name collision with the Simple API.
        assert list(_inspect.signature(sanitizepy.plan).parameters) == ["target"]

    def test_module_level_inspect_is_simple_api(self):
        # `sanitizepy.inspect` / `sanitizepy.clean` are intentionally the
        # Simple API's one-liner versions (df in, DatasetProfile /
        # CleaningResult out). The original health-report/plan-based
        # behavior remains fully available, unchanged, via
        # `Cleaner().inspect(df)` and `Cleaner().clean(df)`.
        assert list(_inspect.signature(sanitizepy.inspect).parameters) == ["df"]
        assert list(_inspect.signature(sanitizepy.clean).parameters) == [
            "df",
            "verbose",
            "dry_run",
        ]

    def test_cleaner_instance_methods_retain_original_contract(self):
        # The expert-level Cleaner().inspect / Cleaner().plan / Cleaner().clean
        # instance methods are completely untouched by the Simple API.
        c = Cleaner()
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = c.inspect(df)
        assert isinstance(report, DatasetHealthReport)
        cleaning_plan = c.plan(report)
        assert isinstance(cleaning_plan, CleaningPlan)
        result = c.clean(df, plan=cleaning_plan, dry_run=True)
        assert isinstance(result, CleaningResult)
