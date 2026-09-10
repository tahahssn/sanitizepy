"""
sanitizepy

An open-source Python engine for automated tabular data quality inspection,
explainable cleaning, and preprocessing.
"""

from __future__ import annotations

# ── simple API ──────────────────────────────────────────
from sanitizepy.simple import *  # noqa: F403

# ── expert API — untouched ───────────────────────────────
from sanitizepy.cleaning import (
    CleaningEngine,
    CleaningPlan,
    CleaningResult,
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
    TypeCoercionOperation,
)
from sanitizepy.cleaning.encoding import EncodingRepairOperation
from sanitizepy.cleaning.missing_tokens import MissingTokenOperation
from sanitizepy.cleaning.near_duplicates import NearDuplicateRemovalOperation
from sanitizepy.cleaning.registry import OperationRegistry, registry
from sanitizepy.cleaning.text_normalization import TextNormalizationOperation
from sanitizepy.config import DEFAULT_CONFIG, CleanerConfig
from sanitizepy.core import Cleaner, plan
from sanitizepy.exceptions import (
    CleanerError,
    ConfigurationError,
    DataValidationError,
    DependencyError,
    EngineError,
)
from sanitizepy.inspection import (
    AnomalyInspector,
    DatatypeInspectionResult,
    DatatypeInspector,
    DuplicateInspectionResult,
    DuplicateInspector,
    MemoryInspectionResult,
    MemoryInspector,
    MissingInspectionResult,
    MissingValueInspector,
    NearDuplicateDetector,
    NearDuplicateResult,
    StatisticsInspectionResult,
    StatisticsInspector,
    TextQualityAnalyzer,
    TextQualityResult,
)
from sanitizepy.inspection.health import DatasetHealthReport, DatasetIssue
from sanitizepy.inspection.profile import DatasetProfiler, profile_to_report
from sanitizepy.models.contracts import ColumnContract, DataContract
from sanitizepy.models.profile import DatasetProfile
from sanitizepy.models.replay import ReplayablePlan, ReplayOperation
from sanitizepy.pipeline import (
    CallableStep,
    PipelineEngine,
    PipelineResult,
    PipelineStep,
    PipelineStepResult,
    TransformStep,
)
from sanitizepy.preprocessing import (
    ColumnInteraction,
    DatetimeFeatures,
    FeatureEngineeringEngine,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)
from sanitizepy.reports import (
    FileExporter,
    JSONRenderer,
    Report,
    ReportEngine,
    StringExporter,
    TextRenderer,
)
from sanitizepy.rules import (
    Rule,
    RuleCategory,
    RuleEngine,
    RuleRegistry,
    RuleSeverity,
    register_builtin_rules,
)
from sanitizepy.version import VERSION, VERSION_INFO, get_version

__version__ = VERSION
