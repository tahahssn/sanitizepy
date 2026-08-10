"""
Tests for cleaner.exceptions
"""

from __future__ import annotations

import pytest

from cleaner.exceptions import (
    CleanerError,
    CleaningError,
    ConfigurationError,
    DataTypeConversionError,
    DataValidationError,
    DeserializationError,
    DependencyError,
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
    DataTypeInspectionError,
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
    def test_inherits_cleaner_error(self):
        assert issubclass(ConfigurationError, CleanerError)

    def test_can_catch_as_cleaner_error(self):
        with pytest.raises(CleanerError):
            raise ConfigurationError("bad config")


class TestValidationErrors:
    def test_validation_error_inherits_cleaner_error(self):
        assert issubclass(ValidationError, CleanerError)

    def test_data_validation_error_inherits_validation_error(self):
        assert issubclass(DataValidationError, ValidationError)

    def test_schema_validation_error_inherits_validation_error(self):
        assert issubclass(SchemaValidationError, ValidationError)

    def test_data_validation_can_catch_as_cleaner_error(self):
        with pytest.raises(CleanerError):
            raise DataValidationError("bad data")


class TestEngineErrors:
    def test_engine_error_inherits_cleaner_error(self):
        assert issubclass(EngineError, CleanerError)

    def test_engine_init_error_inherits_engine_error(self):
        assert issubclass(EngineInitializationError, EngineError)

    def test_engine_exec_error_inherits_engine_error(self):
        assert issubclass(EngineExecutionError, EngineError)

    def test_engine_init_can_catch_as_cleaner_error(self):
        with pytest.raises(CleanerError):
            raise EngineInitializationError("init failed")


class TestInspectionErrors:
    def test_inspection_error_inherits_cleaner_error(self):
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
    def test_cleaning_error_inherits_cleaner_error(self):
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
    def test_preprocessing_error_inherits_cleaner_error(self):
        assert issubclass(PreprocessingError, CleanerError)

    def test_encoding_error_inherits(self):
        assert issubclass(EncodingError, PreprocessingError)

    def test_scaling_error_inherits(self):
        assert issubclass(ScalingError, PreprocessingError)

    def test_feature_engineering_error_inherits(self):
        assert issubclass(FeatureEngineeringError, PreprocessingError)


class TestRuleErrors:
    def test_rule_error_inherits_cleaner_error(self):
        assert issubclass(RuleError, CleanerError)

    def test_rule_validation_error_inherits(self):
        assert issubclass(RuleValidationError, RuleError)

    def test_rule_execution_error_inherits(self):
        assert issubclass(RuleExecutionError, RuleError)


class TestReportErrors:
    def test_report_error_inherits_cleaner_error(self):
        assert issubclass(ReportError, CleanerError)

    def test_report_generation_error_inherits(self):
        assert issubclass(ReportGenerationError, ReportError)

    def test_report_export_error_inherits(self):
        assert issubclass(ReportExportError, ReportError)


class TestModelErrors:
    def test_model_error_inherits_cleaner_error(self):
        assert issubclass(ModelError, CleanerError)

    def test_serialization_error_inherits(self):
        assert issubclass(SerializationError, ModelError)

    def test_deserialization_error_inherits(self):
        assert issubclass(DeserializationError, ModelError)


class TestUtilityErrors:
    def test_utility_error_inherits_cleaner_error(self):
        assert issubclass(UtilityError, CleanerError)

    def test_filesystem_error_inherits(self):
        assert issubclass(FileSystemError, UtilityError)

    def test_dependency_error_inherits(self):
        assert issubclass(DependencyError, UtilityError)
