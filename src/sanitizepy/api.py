from .cleaning import (
    CleaningEngine,
    CleaningPlan,
    CleaningResult,
    OperationRegistry,
    OperationResult,
    TypeCoercionOperation,
    registry,
)
from .cleaning.encoding import EncodingRepairOperation
from .cleaning.missing_tokens import MissingTokenOperation
from .cleaning.near_duplicates import NearDuplicateRemovalOperation
from .cleaning.text_normalization import TextNormalizationOperation
from .config import DEFAULT_CONFIG, CleanerConfig
from .core import Cleaner, clean, inspect, plan
from .exceptions import (
    CleanerError,
    ConfigurationError,
    DataValidationError,
    EngineError,
)
from .inspection.anomalies import AnomalyInspector
from .inspection.health import DatasetHealthReport, DatasetIssue
from .inspection.near_duplicates import NearDuplicateDetector
from .inspection.profile import DatasetProfiler, profile_to_report
from .inspection.text_quality import TextQualityAnalyzer
from .models.contracts import ColumnContract, DataContract
from .models.profile import DatasetProfile, TextQualityResult
from .models.replay import ReplayablePlan, ReplayOperation
from .version import VERSION, VERSION_INFO, get_version

__all__ = [
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
    # --- Additive exports (production-grade-evolution) ---
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
]
