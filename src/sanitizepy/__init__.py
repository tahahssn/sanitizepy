"""
sanitizepy

An open-source Python engine for automated tabular data quality inspection,
explainable cleaning, and preprocessing.
"""

from __future__ import annotations

# ── expert API — untouched ───────────────────────────────
from sanitizepy.cleaning import (
    CleaningEngine as CleaningEngine,
)
from sanitizepy.cleaning import (
    CleaningPlan as CleaningPlan,
)
from sanitizepy.cleaning import (
    CleaningResult as CleaningResult,
)
from sanitizepy.cleaning import (
    DropColumns as DropColumns,
)
from sanitizepy.cleaning import (
    DropDuplicates as DropDuplicates,
)
from sanitizepy.cleaning import (
    DropMissingColumns as DropMissingColumns,
)
from sanitizepy.cleaning import (
    DropMissingRows as DropMissingRows,
)
from sanitizepy.cleaning import (
    FillMissing as FillMissing,
)
from sanitizepy.cleaning import (
    OperationResult as OperationResult,
)
from sanitizepy.cleaning import (
    TypeCoercionOperation as TypeCoercionOperation,
)
from sanitizepy.cleaning.encoding import (
    EncodingRepairOperation as EncodingRepairOperation,
)
from sanitizepy.cleaning.missing_tokens import (
    MissingTokenOperation as MissingTokenOperation,
)
from sanitizepy.cleaning.near_duplicates import (
    NearDuplicateRemovalOperation as NearDuplicateRemovalOperation,
)
from sanitizepy.cleaning.registry import (
    OperationRegistry as OperationRegistry,
)
from sanitizepy.cleaning.registry import (
    registry as registry,
)
from sanitizepy.cleaning.text_normalization import (
    TextNormalizationOperation as TextNormalizationOperation,
)
from sanitizepy.config import (
    DEFAULT_CONFIG as DEFAULT_CONFIG,
)
from sanitizepy.config import (
    CleanerConfig as CleanerConfig,
)
from sanitizepy.core import (
    Cleaner as Cleaner,
)
from sanitizepy.core import (
    plan as plan,
)
from sanitizepy.exceptions import (
    CleanerError as CleanerError,
)
from sanitizepy.exceptions import (
    ConfigurationError as ConfigurationError,
)
from sanitizepy.exceptions import (
    DataValidationError as DataValidationError,
)
from sanitizepy.exceptions import (
    DependencyError as DependencyError,
)
from sanitizepy.exceptions import (
    EngineError as EngineError,
)
from sanitizepy.inspection import (
    AnomalyInspector as AnomalyInspector,
)
from sanitizepy.inspection import (
    DatatypeInspectionResult as DatatypeInspectionResult,
)
from sanitizepy.inspection import (
    DatatypeInspector as DatatypeInspector,
)
from sanitizepy.inspection import (
    DuplicateInspectionResult as DuplicateInspectionResult,
)
from sanitizepy.inspection import (
    DuplicateInspector as DuplicateInspector,
)
from sanitizepy.inspection import (
    MemoryInspectionResult as MemoryInspectionResult,
)
from sanitizepy.inspection import (
    MemoryInspector as MemoryInspector,
)
from sanitizepy.inspection import (
    MissingInspectionResult as MissingInspectionResult,
)
from sanitizepy.inspection import (
    MissingValueInspector as MissingValueInspector,
)
from sanitizepy.inspection import (
    NearDuplicateDetector as NearDuplicateDetector,
)
from sanitizepy.inspection import (
    NearDuplicateResult as NearDuplicateResult,
)
from sanitizepy.inspection import (
    StatisticsInspectionResult as StatisticsInspectionResult,
)
from sanitizepy.inspection import (
    StatisticsInspector as StatisticsInspector,
)
from sanitizepy.inspection import (
    TextQualityAnalyzer as TextQualityAnalyzer,
)
from sanitizepy.inspection import (
    TextQualityResult as TextQualityResult,
)
from sanitizepy.inspection.health import (
    DatasetHealthReport as DatasetHealthReport,
)
from sanitizepy.inspection.health import (
    DatasetIssue as DatasetIssue,
)
from sanitizepy.inspection.profile import (
    DatasetProfiler as DatasetProfiler,
)
from sanitizepy.inspection.profile import (
    profile_to_report as profile_to_report,
)
from sanitizepy.models.contracts import (
    ColumnContract as ColumnContract,
)
from sanitizepy.models.contracts import (
    DataContract as DataContract,
)
from sanitizepy.models.profile import (
    DatasetProfile as DatasetProfile,
)
from sanitizepy.models.replay import (
    ReplayablePlan as ReplayablePlan,
)
from sanitizepy.models.replay import (
    ReplayOperation as ReplayOperation,
)
from sanitizepy.pipeline import (
    CallableStep as CallableStep,
)
from sanitizepy.pipeline import (
    PipelineEngine as PipelineEngine,
)
from sanitizepy.pipeline import (
    PipelineResult as PipelineResult,
)
from sanitizepy.pipeline import (
    PipelineStep as PipelineStep,
)
from sanitizepy.pipeline import (
    PipelineStepResult as PipelineStepResult,
)
from sanitizepy.pipeline import (
    TransformStep as TransformStep,
)
from sanitizepy.preprocessing import (
    ColumnInteraction as ColumnInteraction,
)
from sanitizepy.preprocessing import (
    DatetimeFeatures as DatetimeFeatures,
)
from sanitizepy.preprocessing import (
    FeatureEngineeringEngine as FeatureEngineeringEngine,
)
from sanitizepy.preprocessing import (
    LogFeature as LogFeature,
)
from sanitizepy.preprocessing import (
    PolynomialFeature as PolynomialFeature,
)
from sanitizepy.preprocessing import (
    RatioFeature as RatioFeature,
)
from sanitizepy.reports import (
    FileExporter as FileExporter,
)
from sanitizepy.reports import (
    JSONRenderer as JSONRenderer,
)
from sanitizepy.reports import (
    Report as Report,
)
from sanitizepy.reports import (
    ReportEngine as ReportEngine,
)
from sanitizepy.reports import (
    StringExporter as StringExporter,
)
from sanitizepy.reports import (
    TextRenderer as TextRenderer,
)
from sanitizepy.rules import (
    Rule as Rule,
)
from sanitizepy.rules import (
    RuleCategory as RuleCategory,
)
from sanitizepy.rules import (
    RuleEngine as RuleEngine,
)
from sanitizepy.rules import (
    RuleRegistry as RuleRegistry,
)
from sanitizepy.rules import (
    RuleSeverity as RuleSeverity,
)
from sanitizepy.rules import (
    register_builtin_rules as register_builtin_rules,
)

# ── simple API ──────────────────────────────────────────
from sanitizepy.simple import (
    anomalies as anomalies,
)
from sanitizepy.simple import (
    cast as cast,
)
from sanitizepy.simple import (
    clean as clean,
)
from sanitizepy.simple import (
    clean_text as clean_text,
)
from sanitizepy.simple import (
    contract as contract,
)
from sanitizepy.simple import (
    datetime_features as datetime_features,
)
from sanitizepy.simple import (
    drop_cols as drop_cols,
)
from sanitizepy.simple import (
    drop_duplicates as drop_duplicates,
)
from sanitizepy.simple import (
    drop_missing_cols as drop_missing_cols,
)
from sanitizepy.simple import (
    drop_missing_rows as drop_missing_rows,
)
from sanitizepy.simple import (
    drop_near_duplicates as drop_near_duplicates,
)
from sanitizepy.simple import (
    dtypes as dtypes,
)
from sanitizepy.simple import (
    dummies as dummies,
)
from sanitizepy.simple import (
    duplicates as duplicates,
)
from sanitizepy.simple import (
    fill_missing as fill_missing,
)
from sanitizepy.simple import (
    fix_encoding as fix_encoding,
)
from sanitizepy.simple import (
    fix_tokens as fix_tokens,
)
from sanitizepy.simple import (
    fix_types as fix_types,
)
from sanitizepy.simple import (
    inspect as inspect,
)
from sanitizepy.simple import (
    interaction as interaction,
)
from sanitizepy.simple import (
    log as log,
)
from sanitizepy.simple import (
    lowercase as lowercase,
)
from sanitizepy.simple import (
    memory as memory,
)
from sanitizepy.simple import (
    missing as missing,
)
from sanitizepy.simple import (
    near_duplicates as near_duplicates,
)
from sanitizepy.simple import (
    normalize as normalize,
)
from sanitizepy.simple import (
    normalize_text as normalize_text,
)
from sanitizepy.simple import (
    poly as poly,
)
from sanitizepy.simple import (
    profile as profile,
)
from sanitizepy.simple import (
    ratio as ratio,
)
from sanitizepy.simple import (
    rename as rename,
)
from sanitizepy.simple import (
    replay as replay,
)
from sanitizepy.simple import (
    report as report,
)
from sanitizepy.simple import (
    select as select,
)
from sanitizepy.simple import (
    serialize as serialize,
)
from sanitizepy.simple import (
    standardize as standardize,
)
from sanitizepy.simple import (
    stats as stats,
)
from sanitizepy.simple import (
    text_quality as text_quality,
)
from sanitizepy.simple import (
    titlecase as titlecase,
)
from sanitizepy.simple import (
    uppercase as uppercase,
)
from sanitizepy.simple import (
    validate as validate,
)
from sanitizepy.version import (
    VERSION as VERSION,
)
from sanitizepy.version import (
    VERSION_INFO as VERSION_INFO,
)
from sanitizepy.version import (
    get_version as get_version,
)

__version__ = VERSION

__all__ = [
    "AnomalyInspector",
    "CallableStep",
    "Cleaner",
    "CleanerConfig",
    "CleanerError",
    "CleaningEngine",
    "CleaningPlan",
    "CleaningResult",
    "ColumnContract",
    "ColumnInteraction",
    "ConfigurationError",
    "DEFAULT_CONFIG",
    "DataContract",
    "DataValidationError",
    "DatasetHealthReport",
    "DatasetIssue",
    "DatasetProfile",
    "DatasetProfiler",
    "DatatypeInspectionResult",
    "DatatypeInspector",
    "DatetimeFeatures",
    "DependencyError",
    "DropColumns",
    "DropDuplicates",
    "DropMissingColumns",
    "DropMissingRows",
    "DuplicateInspectionResult",
    "DuplicateInspector",
    "EncodingRepairOperation",
    "EngineError",
    "FeatureEngineeringEngine",
    "FileExporter",
    "FillMissing",
    "JSONRenderer",
    "LogFeature",
    "MemoryInspectionResult",
    "MemoryInspector",
    "MissingInspectionResult",
    "MissingTokenOperation",
    "MissingValueInspector",
    "NearDuplicateDetector",
    "NearDuplicateRemovalOperation",
    "NearDuplicateResult",
    "OperationRegistry",
    "OperationResult",
    "PipelineEngine",
    "PipelineResult",
    "PipelineStep",
    "PipelineStepResult",
    "PolynomialFeature",
    "RatioFeature",
    "ReplayOperation",
    "ReplayablePlan",
    "Report",
    "ReportEngine",
    "Rule",
    "RuleCategory",
    "RuleEngine",
    "RuleRegistry",
    "RuleSeverity",
    "StatisticsInspectionResult",
    "StatisticsInspector",
    "StringExporter",
    "TextNormalizationOperation",
    "TextQualityAnalyzer",
    "TextQualityResult",
    "TextRenderer",
    "TransformStep",
    "TypeCoercionOperation",
    "VERSION",
    "VERSION_INFO",
    "anomalies",
    "cast",
    "clean",
    "clean_text",
    "contract",
    "datetime_features",
    "drop_cols",
    "drop_duplicates",
    "drop_missing_cols",
    "drop_missing_rows",
    "drop_near_duplicates",
    "dtypes",
    "dummies",
    "duplicates",
    "fill_missing",
    "fix_encoding",
    "fix_tokens",
    "fix_types",
    "get_version",
    "inspect",
    "interaction",
    "log",
    "lowercase",
    "memory",
    "missing",
    "near_duplicates",
    "normalize",
    "normalize_text",
    "plan",
    "profile_to_report",
    "poly",
    "profile",
    "ratio",
    "register_builtin_rules",
    "registry",
    "rename",
    "replay",
    "report",
    "select",
    "serialize",
    "standardize",
    "stats",
    "text_quality",
    "titlecase",
    "uppercase",
    "validate",
]
