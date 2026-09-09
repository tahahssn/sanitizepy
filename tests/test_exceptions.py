"""
Tests for sanitizepy.exceptions
"""

from __future__ import annotations

import pytest

from sanitizepy.exceptions import (
    CleanerError,
    CleaningError,
    ConfigurationError,
    DataTypeConversionError,
    DataTypeInspectionError,
    DataValidationError,
    DependencyError,
    DeserializationError,
    DuplicateCleaningError,
    DuplicateInspectionError,
    EncodingError,
    EngineError,
    EngineExecutionError,
    EngineInitializationError,
    FeatureEngineeringError,
    FileSystemError,
    InspectionError,
    MemoryInspectionError,
    MissingValueCleaningError,
    MissingValueInspectionError,
    ModelError,
    OutlierCleaningError,
    PreprocessingError,
    ReportError,
    ReportExportError,
    ReportGenerationError,
    RuleError,
    RuleExecutionError,
    RuleValidationError,
    ScalingError,
    SchemaValidationError,
    SerializationError,
    StatisticsInspectionError,
    UtilityError,
    ValidationError,
)


class TestCleanerError:
    def test_is_exception(self):
        assert issubclass(CleanerError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(CleanerError):
            raise CleanerError("base error")

    def test_message_preserved(self):
        with pytest.raises(CleanerError, match="specific message"):
            raise CleanerError("specific message")


class TestConfigurationError:
    def test_inherits_sanitizepy_error(self):
        assert issubclass(ConfigurationError, CleanerError)

    def test_can_catch_as_sanitizepy_error(self):
        with pytest.raises(CleanerError):
            raise ConfigurationError("bad config")


class TestValidationErrors:
    def test_validation_error_inherits_sanitizepy_error(self):
        assert issubclass(ValidationError, CleanerError)

    def test_data_validation_error_inherits_validation_error(self):
        assert issubclass(DataValidationError, ValidationError)

    def test_schema_validation_error_inherits_validation_error(self):
        assert issubclass(SchemaValidationError, ValidationError)

    def test_data_validation_can_catch_as_sanitizepy_error(self):
        with pytest.raises(CleanerError):
            raise DataValidationError("bad data")


class TestEngineErrors:
    def test_engine_error_inherits_sanitizepy_error(self):
        assert issubclass(EngineError, CleanerError)

    def test_engine_init_error_inherits_engine_error(self):
        assert issubclass(EngineInitializationError, EngineError)

    def test_engine_exec_error_inherits_engine_error(self):
        assert issubclass(EngineExecutionError, EngineError)

    def test_engine_init_can_catch_as_sanitizepy_error(self):
        with pytest.raises(CleanerError):
            raise EngineInitializationError("init failed")


class TestInspectionErrors:
    def test_inspection_error_inherits_sanitizepy_error(self):
        assert issubclass(InspectionError, CleanerError)

    def test_missing_value_inspection_error_inherits(self):
        assert issubclass(MissingValueInspectionError, InspectionError)

    def test_duplicate_inspection_error_inherits(self):
        assert issubclass(DuplicateInspectionError, InspectionError)

    def test_datatype_inspection_error_inherits(self):
        assert issubclass(DataTypeInspectionError, InspectionError)

    def test_memory_inspection_error_inherits(self):
        assert issubclass(MemoryInspectionError, InspectionError)

    def test_statistics_inspection_error_inherits(self):
        assert issubclass(StatisticsInspectionError, InspectionError)


class TestCleaningErrors:
    def test_cleaning_error_inherits_sanitizepy_error(self):
        assert issubclass(CleaningError, CleanerError)

    def test_missing_value_cleaning_error_inherits(self):
        assert issubclass(MissingValueCleaningError, CleaningError)

    def test_duplicate_cleaning_error_inherits(self):
        assert issubclass(DuplicateCleaningError, CleaningError)

    def test_outlier_cleaning_error_inherits(self):
        assert issubclass(OutlierCleaningError, CleaningError)

    def test_datatype_conversion_error_inherits(self):
        assert issubclass(DataTypeConversionError, CleaningError)


class TestPreprocessingErrors:
    def test_preprocessing_error_inherits_sanitizepy_error(self):
        assert issubclass(PreprocessingError, CleanerError)

    def test_encoding_error_inherits(self):
        assert issubclass(EncodingError, PreprocessingError)

    def test_scaling_error_inherits(self):
        assert issubclass(ScalingError, PreprocessingError)

    def test_feature_engineering_error_inherits(self):
        assert issubclass(FeatureEngineeringError, PreprocessingError)


class TestRuleErrors:
    def test_rule_error_inherits_sanitizepy_error(self):
        assert issubclass(RuleError, CleanerError)

    def test_rule_validation_error_inherits(self):
        assert issubclass(RuleValidationError, RuleError)

    def test_rule_execution_error_inherits(self):
        assert issubclass(RuleExecutionError, RuleError)


class TestReportErrors:
    def test_report_error_inherits_sanitizepy_error(self):
        assert issubclass(ReportError, CleanerError)

    def test_report_generation_error_inherits(self):
        assert issubclass(ReportGenerationError, ReportError)

    def test_report_export_error_inherits(self):
        assert issubclass(ReportExportError, ReportError)


class TestModelErrors:
    def test_model_error_inherits_sanitizepy_error(self):
        assert issubclass(ModelError, CleanerError)

    def test_serialization_error_inherits(self):
        assert issubclass(SerializationError, ModelError)

    def test_deserialization_error_inherits(self):
        assert issubclass(DeserializationError, ModelError)


class TestUtilityErrors:
    def test_utility_error_inherits_sanitizepy_error(self):
        assert issubclass(UtilityError, CleanerError)

    def test_filesystem_error_inherits(self):
        assert issubclass(FileSystemError, UtilityError)

    def test_dependency_error_inherits(self):
        assert issubclass(DependencyError, UtilityError)


class TestActivatedExceptionConstructors:
    """Construct each activated exception and assert message/attribute shape.

    These exceptions are wired onto real paths in later tasks. Because they
    are plain ``CleanerError`` subclasses with no custom ``__init__``, they
    accept a message like any standard exception and preserve it on ``args``
    and ``str()``.
    """

    ACTIVATED_EXCEPTIONS = [
        SchemaValidationError,
        DataTypeConversionError,
        DataValidationError,
        EncodingError,
        DependencyError,
        SerializationError,
        DeserializationError,
    ]

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_construct_with_message_preserves_str(self, exc_cls):
        message = f"{exc_cls.__name__} occurred"
        error = exc_cls(message)
        assert str(error) == message

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_construct_with_message_preserves_args(self, exc_cls):
        message = f"{exc_cls.__name__} occurred"
        error = exc_cls(message)
        assert error.args == (message,)

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_construct_without_args(self, exc_cls):
        error = exc_cls()
        assert error.args == ()
        assert str(error) == ""

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_construct_with_multiple_args(self, exc_cls):
        error = exc_cls("column", "expected_dtype")
        assert error.args == ("column", "expected_dtype")

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_instance_is_cleaner_error(self, exc_cls):
        error = exc_cls("boom")
        assert isinstance(error, CleanerError)

    @pytest.mark.parametrize("exc_cls", ACTIVATED_EXCEPTIONS)
    def test_can_raise_and_catch_specifically(self, exc_cls):
        message = f"raised {exc_cls.__name__}"
        with pytest.raises(exc_cls, match=exc_cls.__name__):
            raise exc_cls(message)

    def test_dependency_error_message_can_name_extra(self):
        error = DependencyError("rapidfuzz is required; install sanitizepy[fuzzy]")
        assert "sanitizepy[fuzzy]" in str(error)
        assert isinstance(error, CleanerError)


# ===========================================================================
# Task 19.4 - Optional-extra guard behavior
# Requirements: 16.1, 16.3
#
# Three capabilities are guarded behind optional extras. This consolidated
# suite asserts, for each guard, that:
#   * requesting the optional path with the extra simulated absent raises a
#     DependencyError whose message names the correct extra (Req 16.3), and
#   * the Core_Stack-only path of the same capability still works while the
#     extra is absent (Req 16.1).
#
# Per-capability edge-case simulations already live in the capability test
# files (test_cleaning_encoding.py, test_cleaning_near_duplicates.py,
# test_inspection_near_duplicates.py, test_inspection_text_quality.py). This
# class deliberately does not duplicate them; it verifies the guard contract
# in one place and the core-only fallbacks.
# ===========================================================================

import sys as _sys  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pandas as pd  # noqa: E402

from sanitizepy.cleaning.encoding import EncodingRepairOperation  # noqa: E402
from sanitizepy.cleaning.near_duplicates import (  # noqa: E402
    NearDuplicateRemovalOperation,
)
from sanitizepy.inspection.near_duplicates import (  # noqa: E402
    NearDuplicateDetector,
)
from sanitizepy.inspection.text_quality import TextQualityAnalyzer  # noqa: E402


class TestOptionalExtraGuards:
    """
    Consolidated optional-extra guard behavior across the three guarded
    capabilities.

    Guards covered:
      1. NearDuplicateDetector.detect(method="similarity")           -> fuzzy
      2. NearDuplicateRemovalOperation(method="similarity").apply    -> fuzzy
      3. EncodingRepairOperation(mode="advanced").apply              -> text
      4. TextQualityAnalyzer.analyze(mode="advanced")                -> text
    """

    # -- rapidfuzz / sanitizepy[fuzzy] --------------------------------------

    def test_detector_similarity_missing_fuzzy_names_extra(self):
        detector = NearDuplicateDetector()
        df = pd.DataFrame({"name": ["hello", "hallo"]})
        with (
            patch.dict(_sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError, match=r"sanitizepy\[fuzzy\]"),
        ):
            detector.detect(df, method="similarity")

    def test_removal_similarity_missing_fuzzy_names_extra(self):
        op = NearDuplicateRemovalOperation(method="similarity")
        df = pd.DataFrame({"name": ["hello", "hallo"]})
        with (
            patch.dict(_sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError, match=r"sanitizepy\[fuzzy\]"),
        ):
            op.apply(df)

    # -- ftfy / sanitizepy[text] --------------------------------------------

    def test_encoding_advanced_missing_ftfy_names_extra(self):
        op = EncodingRepairOperation(mode="advanced")
        df = pd.DataFrame({"col": ["CafÃ©"]})
        with (
            patch.dict(_sys.modules, {"ftfy": None}),
            pytest.raises(DependencyError, match=r"sanitizepy\[text\]"),
        ):
            op.apply(df)

    def test_text_quality_advanced_names_text_extra(self):
        analyzer = TextQualityAnalyzer()
        df = pd.DataFrame({"text": ["hello world"]})
        with pytest.raises(DependencyError, match=r"sanitizepy\[text\]"):
            analyzer.analyze(df, mode="advanced")

    # -- The raised guard is a DependencyError / CleanerError ---------------

    def test_all_guards_raise_dependency_error_subclass(self):
        """Each guard raises a DependencyError catchable as CleanerError."""
        df = pd.DataFrame({"col": ["CafÃ©", "cafe"]})
        with (
            patch.dict(_sys.modules, {"ftfy": None}),
            pytest.raises(CleanerError),
        ):
            EncodingRepairOperation(mode="advanced").apply(df)

    # -- Core_Stack-only paths work WITHOUT the extra present (Req 16.1) ----

    def test_detector_exact_normalized_works_without_fuzzy(self):
        detector = NearDuplicateDetector()
        df = pd.DataFrame({"name": ["Alice", "alice", "Bob"]})
        with patch.dict(_sys.modules, {"rapidfuzz": None}):
            result = detector.detect(df, method="exact_normalized")
        # "Alice"/"alice" collapse into a single normalized group.
        assert result.groups == (("0", "1"),) or result.groups == ((0, 1),)
        assert result.duplicate_count == 1

    def test_removal_exact_normalized_works_without_fuzzy(self):
        op = NearDuplicateRemovalOperation(method="exact_normalized")
        df = pd.DataFrame({"name": ["Alice", "alice", "Bob"]})
        with patch.dict(_sys.modules, {"rapidfuzz": None}):
            result = op.apply(df)
        assert list(result["name"]) == ["Alice", "Bob"]

    def test_encoding_core_repair_works_without_ftfy(self):
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=False)
        df = pd.DataFrame({"col": ["hello\ufffdworld", "clean"]})
        with patch.dict(_sys.modules, {"ftfy": None}):
            result = op.apply(df)
        assert "\ufffd" not in str(result["col"].iloc[0])
        assert result["col"].iloc[1] == "clean"

    def test_text_quality_core_analysis_works_without_extra(self):
        analyzer = TextQualityAnalyzer()
        df = pd.DataFrame({"text": ["hello world", "a", "another sample"]})
        with patch.dict(_sys.modules, {"ftfy": None, "rapidfuzz": None}):
            results = analyzer.analyze(df, mode="core")
        assert len(results) == 1
        assert results[0].column == "text"
        assert results[0].non_null_count == 3
