"""
sanitizepy.exceptions
~~~~~~~~~~~~~~~~~~

Centralized exception hierarchy for the Cleaner library.

Every exception raised anywhere inside the library must inherit from
CleanerError.

This guarantees consistent exception handling for both library users
and internal components.
"""

from __future__ import annotations


class CleanerError(Exception):
    """
    Base exception for the Cleaner library.

    Users can catch this exception to handle every library-specific
    error with a single except block.
    """

    pass


# ==========================================================
# Configuration
# ==========================================================


class ConfigurationError(CleanerError):
    """
    Raised when configuration is invalid, incomplete,
    or internally inconsistent.
    """

    pass


# ==========================================================
# Validation
# ==========================================================


class ValidationError(CleanerError):
    """
    Raised when user input fails validation.
    """

    pass


class DataValidationError(ValidationError):
    """
    Raised when dataset validation fails.
    """

    pass


class SchemaValidationError(ValidationError):
    """
    Raised when an expected schema does not match the dataset.
    """

    pass


# ==========================================================
# Engine
# ==========================================================


class EngineError(CleanerError):
    """
    Base exception for engine failures.
    """

    pass


class EngineInitializationError(EngineError):
    """
    Raised when an engine cannot be initialized.
    """

    pass


class EngineExecutionError(EngineError):
    """
    Raised when an engine fails during execution.
    """

    pass


# ==========================================================
# Inspection
# ==========================================================


class InspectionError(CleanerError):
    """
    Base exception for inspection failures.
    """

    pass


class MissingValueInspectionError(InspectionError):
    """
    Raised when missing value inspection fails.
    """

    pass


class DuplicateInspectionError(InspectionError):
    """
    Raised when duplicate inspection fails.
    """

    pass


class DataTypeInspectionError(InspectionError):
    """
    Raised when datatype inspection fails.
    """

    pass


class MemoryInspectionError(InspectionError):
    """
    Raised when memory inspection fails.
    """

    pass


class StatisticsInspectionError(InspectionError):
    """
    Raised when statistical inspection fails.
    """

    pass


# ==========================================================
# Cleaning
# ==========================================================


class CleaningError(CleanerError):
    """
    Base exception for cleaning failures.
    """

    pass


class MissingValueCleaningError(CleaningError):
    """
    Raised when missing value cleaning fails.
    """

    pass


class DuplicateCleaningError(CleaningError):
    """
    Raised when duplicate removal fails.
    """

    pass


class OutlierCleaningError(CleaningError):
    """
    Raised when outlier processing fails.
    """

    pass


class DataTypeConversionError(CleaningError):
    """
    Raised when datatype conversion fails.
    """

    pass


# ==========================================================
# Preprocessing
# ==========================================================


class PreprocessingError(CleanerError):
    """
    Base exception for preprocessing failures.
    """

    pass


class EncodingError(PreprocessingError):
    """
    Raised when categorical encoding fails.
    """

    pass


class ScalingError(PreprocessingError):
    """
    Raised when feature scaling fails.
    """

    pass


class FeatureEngineeringError(PreprocessingError):
    """
    Raised when feature engineering fails.
    """

    pass


# ==========================================================
# Rules
# ==========================================================


class RuleError(CleanerError):
    """
    Base exception for rule engine failures.
    """

    pass


class RuleValidationError(RuleError):
    """
    Raised when a rule definition is invalid.
    """

    pass


class RuleExecutionError(RuleError):
    """
    Raised when a rule fails during execution.
    """

    pass


# ==========================================================
# Reports
# ==========================================================


class ReportError(CleanerError):
    """
    Base exception for report generation failures.
    """

    pass


class ReportGenerationError(ReportError):
    """
    Raised when a report cannot be generated.
    """

    pass


class ReportExportError(ReportError):
    """
    Raised when exporting a report fails.
    """

    pass


# ==========================================================
# Models
# ==========================================================


class ModelError(CleanerError):
    """
    Base exception for internal model failures.
    """

    pass


class SerializationError(ModelError):
    """
    Raised when serialization fails.
    """

    pass


class DeserializationError(ModelError):
    """
    Raised when deserialization fails.
    """

    pass


# ==========================================================
# Utilities
# ==========================================================


class UtilityError(CleanerError):
    """
    Base exception for utility failures.
    """

    pass


class FileSystemError(UtilityError):
    """
    Raised when filesystem operations fail.
    """

    pass


class DependencyError(UtilityError):
    """
    Raised when a required dependency is unavailable.
    """

    pass
