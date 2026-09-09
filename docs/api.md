# API Reference

This document provides technical API documentation for all public entry points, engines, operations, models, dataclasses, and exceptions in `sanitizepy`.

> **Optional dependency extras.** The core library works with pandas + stdlib only. A few capabilities have opt-in extras:
> - `pip install "sanitizepy[fuzzy]"` enables `rapidfuzz`-backed similarity mode for near-duplicate detection/removal (`method="similarity"`).
> - `pip install "sanitizepy[text]"` enables `ftfy`-backed advanced encoding repair (`mode="advanced"`) and advanced text-quality analysis (`mode="advanced"`).
>
> These are optional; the core stack functions without them. When an opt-in path is used without its extra installed, a `DependencyError` is raised naming the required extra.

---

## Table of Contents

1. [Top-Level Package API](#1-top-level-package-api)
2. [Data Inspection (`sanitizepy.inspection`)](#2-data-inspection-sanitizepyinspection)
3. [Data Cleaning (`sanitizepy.cleaning`)](#3-data-cleaning-sanitizepycleaning)
4. [Preprocessing & Feature Engineering (`sanitizepy.preprocessing`)](#4-preprocessing--feature-engineering-sanitizepypreprocessing)
5. [Rule Engine (`sanitizepy.rules`)](#5-rule-engine-sanitizepyrules)
6. [Report Engine (`sanitizepy.reports`)](#6-report-engine-sanitizepyreports)
7. [Pipeline Engine (`sanitizepy.pipeline`)](#7-pipeline-engine-sanitizepypipeline)
8. [Base Engine & Data Models (`sanitizepy.engine`, `sanitizepy.models`)](#8-base-engine--data-models)
9. [Exceptions (`sanitizepy.exceptions`)](#9-exceptions-sanitizepyexceptions)
10. [Logging (`sanitizepy.logger`)](#10-logging-sanitizepylogger)

---

## 1. Top-Level Package API

Imports available directly from `sanitizepy`:

```python
from sanitizepy import (
    Cleaner,
    CleanerConfig,
    DEFAULT_CONFIG,
    CleanerError,
    ConfigurationError,
    DataValidationError,
    EngineError,
    VERSION,
    VERSION_INFO,
    get_version,
)
```

### `Cleaner`

Main entry point of the library. Holds global package configuration and logger instances shared across engines.

```python
class Cleaner:
    def __init__(self, config: CleanerConfig | None = None) -> None: ...
```

- **Parameters**:
  - `config` (`CleanerConfig | None`): Active configuration instance. Defaults to `DEFAULT_CONFIG` if `None`.
- **Properties**:
  - `config` (`CleanerConfig`): Return the active configuration.
  - `logger` (`logging.Logger`): Return the package logger instance.

### `CleanerConfig`

Dataclass storing global configuration settings shared across inspection, cleaning, preprocessing, reporting, and rules.

```python
@dataclass(slots=True, kw_only=True)
class CleanerConfig:
    encoding: str = "utf-8"
    float_precision: int = 6
    preview_rows: int = 10
    top_values: int = 10
    missing_value_tokens: frozenset[str] = DEFAULT_MISSING_VALUE_TOKENS
    report_directory: Path | None = None
    enable_logging: bool = True
```

- **Fields**:
  - `encoding` (`str`): Default text encoding for file I/O operations. Default: `"utf-8"`.
  - `float_precision` (`int`): Decimal precision for floating-point values. Default: `6`. Must be `>= 0`.
  - `preview_rows` (`int`): Number of rows displayed in previews. Default: `10`. Must be `> 0`.
  - `top_values` (`int`): Number of top frequent values tracked in statistics. Default: `10`. Must be `> 0`.
  - `missing_value_tokens` (`frozenset[str]`): Tokens recognized as missing values (case-folded during post-init). Default: `{"", " ", "na", "n/a", "nan", "null", "none", "nil", "?", "-"}`.
  - `report_directory` (`Path | None`): Path to output report directory. Expanded and resolved on initialization if supplied.
  - `enable_logging` (`bool`): Flag enabling/disabling library logging. Default: `True`.
- **Exceptions**:
  - `ValueError`: Raised if `preview_rows <= 0`, `top_values <= 0`, or `float_precision < 0`.

### `DEFAULT_CONFIG`

Pre-instantiated default instance of `CleanerConfig`.

### Version Constants & Functions

- `VERSION` (`str`): Current package version string (e.g., `"0.1.0"`).
- `VERSION_INFO` (`tuple[int, int, int]`): Version tuple (e.g., `(0, 1, 0)`).
- `get_version() -> str`: Function returning `VERSION`.
- `sanitizepy.__version__`: Package version string.

---

## 2. Data Inspection (`sanitizepy.inspection`)

Import path:

```python
from sanitizepy.inspection import (
    MissingValueInspector,
    DuplicateInspector,
    DatatypeInspector,
    MemoryInspector,
    StatisticsInspector,
)
```

### `MissingValueInspector`

Analyzes missing cell counts, percentages, and severity in a DataFrame without mutating it.

```python
class MissingValueInspector:
    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> MissingInspectionResult: ...
```

- **Parameters**:
  - `dataframe` (`pd.DataFrame`): Input DataFrame to inspect. Must not be empty.
  - `threshold` (`float | None`): Optional minimum missing percentage required for a column to appear in column reports.
- **Returns**: `MissingInspectionResult` dataclass containing:
  - `summary` (`MissingSummary`): Includes `total_rows`, `total_columns`, `total_cells`, `missing_cells`, `missing_percentage`, `complete_cells`, `complete_percentage`, `columns_with_missing`, `complete_columns`, `complete_rows`, `rows_with_missing`, and `severity` (`"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`).
  - `column_reports` (`tuple[MissingColumnReport, ...]`): Tuple of column-level reports sorted descending by missing percentage.
  - `missing_mask` (`pd.DataFrame`): Boolean mask DataFrame where `True` indicates missing values.
  - `missing_counts` (`pd.Series`): Series of missing counts per column.
  - `missing_percentages` (`pd.Series`): Series of missing percentages per column.
- **Raises**:
  - `ValueError`: If `dataframe.empty` is `True`.

### `DuplicateInspector`

Analyzes row-level and column-level duplicate records.

```python
class DuplicateInspector:
    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
        keep: Literal["first", "last", False] = "first",
    ) -> DuplicateInspectionResult: ...

    def has_duplicates(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
    ) -> bool: ...

    def duplicate_columns(self, dataframe: pd.DataFrame) -> tuple[str, ...]: ...

    def duplicate_column_count(self, dataframe: pd.DataFrame) -> int: ...
```

- **Methods**:
  - `inspect(...)`: Returns `DuplicateInspectionResult` containing `summary` (`DuplicateSummary`), `duplicate_mask` (`pd.Series`), `duplicate_indices` (`tuple[int, ...]`), `duplicate_dataframe` (`pd.DataFrame`), and `duplicate_count` (`int`). Severity classification: `<2%` LOW, `<5%` MEDIUM, `<15%` HIGH, `>=15%` CRITICAL.
  - `has_duplicates(...)`: Returns `True` if any duplicate rows exist for the given column subset.
  - `duplicate_columns(...)`: Returns tuple of duplicated column names.
  - `duplicate_column_count(...)`: Returns count of duplicated column names.

### `DatatypeInspector`

Analyzes physical dtypes, inferred semantic types, nullability, unique value counts, and recommendations.

```python
class DatatypeInspector:
    def inspect(self, dataframe: pd.DataFrame) -> DatatypeInspectionResult: ...
```

- **Returns**: `DatatypeInspectionResult` containing `summary` (`DatatypeSummary`) and `reports` (`tuple[ColumnTypeReport, ...]`).
- **Semantic Types**: `"integer"`, `"float"`, `"boolean"`, `"datetime"`, `"category"`, `"string"`, `"numeric"`, `"object"`, `"unknown"`.

### `MemoryInspector`

Calculates actual memory usage and estimates potential memory savings from dtype optimizations.

```python
class MemoryInspector:
    def inspect(self, dataframe: pd.DataFrame) -> MemoryInspectionResult: ...
    def largest_columns(self, dataframe: pd.DataFrame, n: int = 10) -> pd.Series: ...
```

- **Returns**: `MemoryInspectionResult` containing `summary` (`MemorySummary`), `reports` (`tuple[MemoryColumnReport, ...]`), and `memory_usage` (`pd.Series`).

### `StatisticsInspector`

Computes parametric and non-parametric statistical metrics for numeric columns.

```python
class StatisticsInspector:
    def inspect(self, dataframe: pd.DataFrame) -> StatisticsInspectionResult: ...
    def describe(self, dataframe: pd.DataFrame) -> pd.DataFrame: ...
    def correlation(self, dataframe: pd.DataFrame, *, method: Literal["pearson", "kendall", "spearman"] = "pearson") -> pd.DataFrame: ...
    def covariance(self, dataframe: pd.DataFrame) -> pd.DataFrame: ...
```

- **Returns**: `StatisticsInspectionResult` containing `summary` (`StatisticsSummary`) and `reports` (`tuple[NumericColumnStatistics, ...]`).

---

## 3. Data Cleaning (`sanitizepy.cleaning`)

Import path:

```python
from sanitizepy.cleaning import (
    CleaningEngine,
    CleaningResult,
    CleaningOperation,
    OperationResult,
    CleaningPlan,
    OperationRegistry,
    registry,
    DropMissingRows,
    DropMissingColumns,
    FillMissing,
    DropDuplicates,
    DropColumns,
    TypeCoercionOperation,
)

# Additional operations live in their own submodules and are also
# re-exported from the top-level ``sanitizepy`` package:
from sanitizepy import (
    MissingTokenOperation,
    TextNormalizationOperation,
    EncodingRepairOperation,
    NearDuplicateRemovalOperation,
)
```

### `CleaningEngine`

Orchestrates execution of an ordered sequence of `CleaningOperation` instances.

```python
class CleaningEngine:
    def __init__(self, operations: Iterable[CleaningOperation] | None = None) -> None: ...
    def add(self, operation: CleaningOperation) -> None: ...
    def clear(self) -> None: ...
    def run(
        self,
        dataframe: pd.DataFrame,
        dry_run: bool = False,
        chunk_size: int | None = None,
    ) -> pd.DataFrame: ...
    def run_with_result(
        self,
        dataframe: pd.DataFrame,
        dry_run: bool = False,
        chunk_size: int | None = None,
    ) -> CleaningResult: ...
    def describe(self) -> list[dict[str, object]]: ...
    @property
    def operations(self) -> tuple[CleaningOperation, ...]: ...
```

- **Methods**:
  - `run(dataframe, dry_run=False, chunk_size=None)`: Applies configured operations in order and returns the cleaned DataFrame. When `dry_run=True`, operations are evaluated without mutating the caller's DataFrame. When `chunk_size` is a positive integer, chunk-safe operations (`is_chunk_safe=True`) are applied in row-wise batches of that size; non-chunk-safe operations always run over the whole dataset.
  - `run_with_result(dataframe, dry_run=False, chunk_size=None)`: Executes the same operations but returns a `CleaningResult` containing the transformed `data`, per-operation `operations` metrics, the `dry_run` flag, `duration_seconds`, and an ordered `audit_log`.
  - `describe()`: Returns lightweight dictionary representations of all configured operations.
- **Raises**:
  - `TypeError`: If input object is not a `pd.DataFrame` or an added operation is not a `CleaningOperation`.
  - `ValueError`: If `chunk_size` is not a positive integer.

#### `CleaningResult`

Structured outcome returned by `CleaningEngine.run_with_result(...)` and `CleaningPlan.apply(...)`.

```python
@dataclass
class CleaningResult:
    data: pd.DataFrame
    operations: list[OperationResult]
    dry_run: bool
    duration_seconds: float
    audit_log: list[dict[str, Any]]

    def summary(self) -> str: ...
```

- **Fields**:
  - `data` (`pd.DataFrame`): Resulting DataFrame (or an unmodified copy of the input under `dry_run=True`).
  - `operations` (`list[OperationResult]`): Per-step results detailing affected rows/columns and before/after shapes.
  - `dry_run` (`bool`): Whether the execution was a dry run.
  - `duration_seconds` (`float`): Total execution time.
  - `audit_log` (`list[dict[str, Any]]`): Ordered, JSON-serializable log. Each entry records `order` (1-based index), UTC `timestamp`, `operation` name, `parameters` (the operation's `describe()` output), `affected_columns`, `rows_affected`, `columns_affected`, `before_shape`, `after_shape`, `dry_run`, and operation-specific `details`.
- **Methods**:
  - `summary()`: Returns a human-readable text summary of the executed operations.

### Concrete Cleaning Operations

All cleaning operations inherit from `CleaningOperation` and implement `.apply(dataframe) -> pd.DataFrame` and `.describe() -> dict[str, Any]`.

#### `DropMissingRows`

```python
DropMissingRows(subset: list[str] | None = None)
```

- Drops rows containing missing values in the specified column `subset` (or all columns if `subset` is `None`).
- **Raises**: `KeyError` if any column in `subset` is missing from the DataFrame.

#### `DropMissingColumns`

```python
DropMissingColumns(subset: list[str] | None = None)
```

- Drops columns containing missing values. If `subset` is provided, drops specified columns if they contain missing values.
- **Raises**: `KeyError` if any column in `subset` does not exist in the DataFrame.

#### `FillMissing`

```python
FillMissing(
    value: Any = None,
    subset: list[str] | None = None,
    strategy: Literal["median", "mean", "mode", "constant"] = "constant",
)
```

- Fills missing values in the targeted columns (or all columns when `subset` is `None`) using the selected `strategy`:
  - `"constant"` (default): replaces missing values with the explicit `value`.
  - `"median"` / `"mean"`: replaces missing values in each targeted numeric column with the column median/mean computed over its non-missing values.
  - `"mode"`: replaces missing values with the most frequent non-missing value, selecting the first in sorted order on ties.
- **Raises**:
  - `ValueError` if `strategy` is not one of `"median"`, `"mean"`, `"mode"`, `"constant"`.
  - `DataValidationError` if `"median"` or `"mean"` is applied to a non-numeric column.
  - `KeyError` if any column in `subset` does not exist in the DataFrame.

#### `DropDuplicates`

```python
DropDuplicates(subset: list[str] | None = None, keep: str | bool = "first")
```

- Removes duplicate rows based on `subset` columns. `keep` must be `"first"`, `"last"`, or `False`.
- **Raises**: `ValueError` if `keep` is invalid; `KeyError` if `subset` columns do not exist.

#### `DropColumns`

```python
DropColumns(columns: list[str])
```

- Drops specified `columns`. `columns` list must not be empty.
- **Raises**: `ValueError` if `columns` list is empty; `KeyError` if specified columns do not exist.

#### `TypeCoercionOperation`

```python
TypeCoercionOperation(
    target_dtypes: dict[str, str],
    error_policy: Literal["raise", "coerce"] = "raise",
)
```

- Converts named columns to target dtypes deterministically (for example `{"age": "int64", "signup": "datetime64[ns]"}`).
- `error_policy="raise"` (default) fails on the first non-convertible value; `error_policy="coerce"` replaces non-convertible values with the missing marker and records the affected count in the `OperationResult`.
- Chunk-safe (`is_chunk_safe = True`).
- **Raises**:
  - `ValueError` if `target_dtypes` is empty or `error_policy` is invalid.
  - `KeyError` if any listed column does not exist.
  - `DataTypeConversionError` under the strict policy when a value cannot be converted (identifies the column, target dtype, and first non-convertible value).
  - `DataValidationError` if a target dtype is not recognized by pandas.

#### `MissingTokenOperation`

```python
MissingTokenOperation(
    extra_tokens: frozenset[str] | set[str] | list[str] | None = None,
    subset: list[str] | None = None,
)
```

- Replaces sentinel/missing-token strings (for example `"n/a"`, `"null"`, `"nil"`, `"?"`, `"-"`) in object/string columns with the pandas missing marker.
- Comparison is case-insensitive and whitespace-trimmed. `extra_tokens` *extend* (do not replace) `DEFAULT_MISSING_VALUE_TOKENS`.
- Chunk-safe (`is_chunk_safe = True`).
- **Raises**: `KeyError` if any column in `subset` does not exist.

#### `TextNormalizationOperation`

```python
TextNormalizationOperation(
    subset: list[str] | None = None,
    unicode_form: Literal["NFC", "NFKC", "none"] = "NFC",
    normalize_whitespace: bool = True,
    case: Literal["lower", "upper", "title", "none"] = "none",
)
```

- Applies deterministic, local text normalization to object/string columns (or all such columns when `subset` is `None`) using only stdlib (`unicodedata`, `re`) plus pandas/numpy.
- Supports Unicode normalization (NFC/NFKC), whitespace normalization (strip, collapse internal runs, convert non-breaking spaces to spaces), and case transforms. A value that becomes empty after normalization is set to missing.
- Chunk-safe (`is_chunk_safe = True`).
- **Raises**:
  - `ValueError` if `unicode_form` or `case` is invalid.
  - `KeyError` if any column in `subset` does not exist.

#### `EncodingRepairOperation`

```python
EncodingRepairOperation(
    subset: list[str] | None = None,
    mode: Literal["core", "advanced"] = "core",
    error_on_unrepaired: bool = True,
)
```

- Detects and repairs encoding artifacts (mojibake, replacement characters, control characters) in object/string columns.
- `mode="core"` (default) uses only stdlib + pandas. `mode="advanced"` additionally invokes `ftfy` and requires the `sanitizepy[text]` extra.
- When `error_on_unrepaired=True` (default), cells that cannot be deterministically repaired raise `EncodingError`; when `False`, they are left in place and recorded in the result details.
- Chunk-safe (`is_chunk_safe = True`).
- **Raises**:
  - `ValueError` if `mode` is invalid.
  - `KeyError` if any column in `subset` does not exist.
  - `EncodingError` when an artifact cannot be safely repaired in strict mode.
  - `DependencyError` when `mode="advanced"` and `ftfy` is not installed (names the `sanitizepy[text]` extra).

#### `NearDuplicateRemovalOperation`

```python
NearDuplicateRemovalOperation(
    subset: list[str] | None = None,
    method: Literal["exact_normalized", "similarity"] = "exact_normalized",
    threshold: float = 90.0,
    keep: Literal["first", "last"] = "first",
)
```

- Removes near-duplicate records, retaining one representative per near-duplicate group (`keep="first"` keeps the earliest group member, `keep="last"` the latest).
- `method="exact_normalized"` (default) is stdlib + pandas only. `method="similarity"` performs fuzzy matching via `rapidfuzz` and requires the `sanitizepy[fuzzy]` extra; `threshold` is the inclusive minimum similarity score (0-100).
- Not chunk-safe (`is_chunk_safe = False`), because grouping is dataset-wide.
- **Raises**:
  - `ValueError` if `keep` is invalid.
  - `DependencyError` when `method="similarity"` and `rapidfuzz` is not installed (names the `sanitizepy[fuzzy]` extra).

### `OperationRegistry`

Registry mapping operation `name` values to their `CleaningOperation` classes. Used to reconstruct operations from serialized plans (see [Reproducible Plans & Replay](#313-reproducible-plans--replay)). Custom operations can be registered here so they behave like built-ins.

```python
class OperationRegistry:
    def register(self, name: str, operation_cls: type[CleaningOperation]) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> type[CleaningOperation]: ...
    def contains(self, name: str) -> bool: ...
    def clear(self) -> None: ...
    def values(self) -> tuple[type[CleaningOperation], ...]: ...
    def names(self) -> tuple[str, ...]: ...
    def items(self) -> tuple[tuple[str, type[CleaningOperation]], ...]: ...
```

- `registry`: Module-level `OperationRegistry` instance with the built-in operations pre-registered under their `name` (e.g. `"fill_missing"`, `"type_coercion"`, `"near_duplicate_removal"`).
- **Raises**: `ValueError` from `register(...)` if a name is already registered; `KeyError` from `get(...)` for an unknown name.

### 3.11. High-Level Facade & Plan (`sanitizepy` / `sanitizepy.cleaning`)

The top-level `Cleaner` facade and the `CleaningPlan` provide a cohesive inspect → plan → clean workflow.

```python
class Cleaner:
    def inspect(self, dataframe: pd.DataFrame) -> DatasetHealthReport: ...
    def plan(self, target: pd.DataFrame | DatasetHealthReport) -> CleaningPlan: ...
    def clean(
        self,
        dataframe: pd.DataFrame,
        plan: CleaningPlan | None = None,
        dry_run: bool = False,
    ) -> CleaningResult: ...
    def profile(self, dataframe: pd.DataFrame) -> DatasetProfile: ...
    def validate(
        self,
        dataframe: pd.DataFrame,
        contract: DataContract,
    ) -> tuple[RuleResult, ...]: ...
```

- `profile(dataframe)`: Convenience facade over `DatasetProfiler().profile(...)` (see [Dataset Profiling](#310-dataset-profiling)).
- `validate(dataframe, contract)`: Convenience facade over `RuleEngine().validate_contract(...)` (see [Data Contracts](#311-data-contracts)).

```python
class CleaningPlan:
    def apply(self, dataframe: pd.DataFrame, dry_run: bool = False) -> CleaningResult: ...
    def serialize(self) -> ReplayablePlan: ...
    @classmethod
    def deserialize(cls, plan: ReplayablePlan) -> CleaningPlan: ...
```

### 3.10. Dataset Profiling

Import path:

```python
# DatasetProfiler / profile_to_report are re-exported from the top-level package
from sanitizepy import DatasetProfiler, profile_to_report, DatasetProfile
# ...or from their submodule:
from sanitizepy.inspection.profile import DatasetProfiler, profile_to_report
from sanitizepy.models.profile import DatasetProfile, TextQualityResult
```

#### `DatasetProfiler`

```python
class DatasetProfiler:
    def profile(self, dataframe: pd.DataFrame) -> DatasetProfile: ...
```

- Aggregates the standalone inspectors (`DatatypeInspector`, `MissingValueInspector`, `DuplicateInspector`, `MemoryInspector`, `StatisticsInspector`) into an immutable `DatasetProfile` without recomputing any analysis. Core-only; requires no optional extra.
- **Raises**: `ValueError` if the DataFrame is empty.

#### `profile_to_report`

```python
def profile_to_report(profile: DatasetProfile, *, title: str = "Dataset Profile") -> Report: ...
```

- Renders a `DatasetProfile` through the existing reports subsystem, returning a canonical `Report` that can be rendered/exported by the existing renderers and exporters.

#### `DatasetProfile`

Frozen dataclass with fields `row_count`, `column_count`, `datatypes`, `missing_values`, `duplicates`, `memory`, `statistics`, and forward-compatible slots `text_quality` (`tuple[TextQualityResult, ...] | None`), `anomalies`, and `near_duplicate` (default `None`).

### 3.11. Data Contracts

Declarative, column-level expectations validated deterministically against a DataFrame.

Import path:

```python
from sanitizepy.models.contracts import ColumnContract, DataContract
from sanitizepy.rules import RuleEngine
```

#### `ColumnContract`

Pydantic model (all fields optional; an unset field declares no expectation):

- `dtype` (`str | None`): Expected column dtype.
- `nullable` (`bool | None`): Whether missing values are permitted.
- `allowed_values` (`tuple[Any, ...] | None`): Exhaustive set of permitted values.
- `min_value` / `max_value` (`float | None`): Inclusive numeric bounds.
- `regex` (`str | None`): Pattern every non-missing value must match.
- `unique` (`bool | None`): Whether values must be unique.

#### `DataContract`

```python
class DataContract(BaseCleanerModel):
    columns: dict[str, ColumnContract]
```

#### `RuleEngine.validate_contract`

```python
def validate_contract(
    self,
    dataframe: pd.DataFrame,
    contract: DataContract,
) -> tuple[RuleResult, ...]: ...
```

- Produces one `RuleResult` per declared expectation, in deterministic (column-name-sorted) order. A missing declared column yields a failing `RuleResult` rather than an exception.
- **Raises**:
  - `TypeError` if `dataframe` is not a DataFrame or `contract` is not a `DataContract`.
  - `SchemaValidationError` if an expectation is structurally unusable (for example an invalid regex or `min_value > max_value`).

`Cleaner.validate(dataframe, contract)` is a convenience facade over this method.

### 3.12. Near-Duplicate, Text-Quality & Anomaly Analysis

Import path:

```python
from sanitizepy.inspection import (
    NearDuplicateDetector,
    TextQualityAnalyzer,
    AnomalyInspector,
)
```

#### `NearDuplicateDetector`

```python
class NearDuplicateDetector:
    def detect(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
        method: Literal["exact_normalized", "similarity"] = "exact_normalized",
        threshold: float = 90.0,
    ) -> NearDuplicateResult: ...
```

- Read-only, deterministic detection. `"exact_normalized"` is core-only; `"similarity"` requires the `sanitizepy[fuzzy]` extra.
- Returns a `NearDuplicateResult` with `columns`, `method`, `groups` (tuples of grouped index labels), and `duplicate_count`.
- **Raises**: `ValueError` if the DataFrame is empty or a subset column is absent; `DependencyError` if `method="similarity"` and `rapidfuzz` is missing.

#### `TextQualityAnalyzer`

```python
class TextQualityAnalyzer:
    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | None = None,
        mode: Literal["core", "advanced"] = "core",
    ) -> tuple[TextQualityResult, ...]: ...
```

- Computes per-column deterministic text statistics (token-count and character-length min/max/mean/median, empty-after-strip count, boilerplate and encoding-garbage counts) over object/string columns.
- **Raises**: `ValueError` if the DataFrame is empty; `DependencyError` if `mode="advanced"` (names the `sanitizepy[text]` extra).

#### `AnomalyInspector`

```python
class AnomalyInspector:
    def inspect(
        self,
        dataframe: pd.DataFrame,
        *,
        method: Literal["iqr", "zscore"] = "iqr",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        seed: int | None = None,
    ) -> AnomalyResult: ...
```

- Deterministic numeric-column anomaly detection. `"iqr"` uses Tukey fences (preserving existing behavior); `"zscore"` flags values whose absolute standard score exceeds `zscore_threshold`. A fixed `seed` is honored for reproducibility.
- Returns an `AnomalyResult` with `method`, `seed`, `analyzed_columns`, `total_anomalies`, and per-column `reports`.
- **Raises**: `ValueError` if the DataFrame is empty.

### 3.13. Reproducible Plans & Replay

A `CleaningPlan` can be serialized to a frozen, JSON-serializable snapshot and later reconstructed and re-run deterministically.

Import path:

```python
from sanitizepy.models.replay import ReplayablePlan, ReplayOperation
```

- `CleaningPlan.serialize() -> ReplayablePlan`: Emits an ordered snapshot of the enabled steps.
- `CleaningPlan.deserialize(plan: ReplayablePlan) -> CleaningPlan`: Reconstructs the plan by looking up each operation class in the `OperationRegistry` and instantiating it from the recorded parameters.
- `ReplayablePlan`: Frozen model with `version`, `operations` (`tuple[ReplayOperation, ...]`), `configuration`, and `seeds`. `to_json()` / `from_json(...)` provide the JSON boundary.
- `ReplayOperation`: A single entry with `name` (the operation's registry key) and `parameters` (JSON-serializable reconstruction parameters).
- **Raises**: `SerializationError` (non-JSON-serializable parameters) and `DeserializationError` (malformed payload or unknown operation name).

---

## 4. Preprocessing & Feature Engineering (`sanitizepy.preprocessing`)

Import path:

```python
from sanitizepy.preprocessing import (
    FeatureEngineeringEngine,
    FeatureOperation,
    ColumnInteraction,
    RatioFeature,
    PolynomialFeature,
    LogFeature,
    DatetimeFeatures,
)
```

### `FeatureEngineeringEngine`

Orchestrates sequential execution of stateful `FeatureOperation` instances.

```python
class FeatureEngineeringEngine:
    def __init__(self, operations: Iterable[FeatureOperation] | None = None) -> None: ...
    def add_operation(self, operation: FeatureOperation) -> FeatureEngineeringEngine: ...
    def add_operations(self, operations: Iterable[FeatureOperation]) -> FeatureEngineeringEngine: ...
    def clear(self) -> None: ...
    def fit(self, data: pd.DataFrame) -> FeatureEngineeringEngine: ...
    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def get_params(self) -> dict[str, Any]: ...
```

- **Methods**:
  - `fit(data)`: Fits all operations against `data`.
  - `transform(data)`: Applies fitted transformations to `data` and returns the transformed DataFrame.
  - `fit_transform(data)`: Fits and transforms in a single pass.

### Built-in Feature Operations

All operations inherit from `FeatureOperation` and implement `fit`, `transform`, `fit_transform`, and `get_params`.

#### `ColumnInteraction`

```python
ColumnInteraction(column_a: str, column_b: str, output_column: str | None = None)
```

- Computes element-wise multiplication: `output_column = column_a * column_b`. Default output name: `"{column_a}_x_{column_b}"`.
- **Raises**: `ValueError` if columns are identical or empty; `TypeError` if columns are non-numeric; `KeyError` if columns are missing.

#### `RatioFeature`

```python
RatioFeature(numerator: str, denominator: str, output_column: str | None = None, zero_division: str = "nan")
```

- Computes element-wise division: `output_column = numerator / denominator`. `zero_division` must be `"nan"` (replaces inf with NaN) or `"raise"`.
- **Raises**: `ValueError` if `zero_division` is invalid; `ZeroDivisionError` if `zero_division="raise"` and zeros exist in denominator.

#### `PolynomialFeature`

```python
PolynomialFeature(column: str, degree: int, include_bias: bool = False, output_prefix: str | None = None)
```

- Generates polynomial power columns up to `degree` (degree `>= 2`).
- Generated column names: `{output_prefix}^{power}` (and optionally `{output_prefix}_bias`).

#### `LogFeature`

```python
LogFeature(column: str, output_column: str | None = None, offset: float = 0.0)
```

- Computes natural logarithm: `output_column = log(column + offset)`. Default output name: `"log_{column}"`.
- **Raises**: `ValueError` if `column + offset <= 0`.

#### `DatetimeFeatures`

```python
DatetimeFeatures(column: str, features: Sequence[str], prefix: str | None = None)
```

- Extracts specified calendar components from a datetime column.
- Supported feature names: `"year"`, `"month"`, `"day"`, `"day_of_week"`, `"day_of_year"`, `"week"`, `"quarter"`, `"hour"`, `"minute"`, `"second"`.
- **Raises**: `TypeError` if column dtype is not datetime64; `ValueError` if unsupported features are specified.

---

## 5. Rule Engine (`sanitizepy.rules`)

Import path:

```python
from sanitizepy.rules import (
    RuleEngine,
    RuleRegistry,
    BaseRule,
    Rule,
    RuleSeverity,
    RuleCategory,
    RuleResult,
    register_builtin_rules,
)
```

### `RuleEngine`

Executes rules registered in a `RuleRegistry` against a DataFrame.

```python
class RuleEngine(BaseEngine):
    def __init__(self, registry: RuleRegistry | None = None, config: Any = None) -> None: ...
    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> tuple[RuleResult, ...]: ...
    @property
    def registry(self) -> RuleRegistry: ...
```

### `RuleRegistry`

Registry for managing validation rules.

```python
class RuleRegistry:
    def register(self, rule: BaseRule) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> BaseRule: ...
    def contains(self, name: str) -> bool: ...
    def clear(self) -> None: ...
    def values(self) -> tuple[BaseRule, ...]: ...
    def names(self) -> tuple[str, ...]: ...
    def items(self) -> tuple[tuple[str, BaseRule], ...]: ...
```

### Enums & Data Models

- `RuleSeverity` (`str, Enum`): `"info"`, `"warning"`, `"error"`, `"critical"`.
- `RuleCategory` (`str, Enum`): `"data_quality"`, `"completeness"`, `"consistency"`, `"uniqueness"`, `"validity"`, `"structure"`, `"custom"`.
- `RuleResult` (`BaseModel`): Pydantic model with fields:
  - `rule` (`str`): Rule name.
  - `passed` (`bool`): Outcome boolean.
  - `severity` (`RuleSeverity`): Assigned severity.
  - `category` (`RuleCategory`): Assigned category.
  - `message` (`str`): Human-readable message.
  - `affected_columns` (`tuple[str, ...]`): Tuple of affected column names.
  - `affected_rows` (`int`): Count of affected rows.
  - `metadata` (`dict[str, Any]`): Extra metadata.

### `register_builtin_rules`

Registers 10 standard built-in rules into the supplied `RuleRegistry`: `"missing_values"`, `"duplicate_rows"`, `"duplicate_columns"`, `"invalid_dtypes"`, `"outliers"`, `"constant_columns"`, `"high_cardinality"`, `"whitespace"`, `"string_case"`, `"column_names"`.

---

## 6. Report Engine (`sanitizepy.reports`)

Import path:

```python
from sanitizepy.reports import (
    ReportEngine,
    Report,
    ReportSection,
    BaseRenderer,
    TextRenderer,
    JSONRenderer,
    BaseExporter,
    StringExporter,
    FileExporter,
)
```

### `ReportEngine`

Converts processing results into a canonical immutable `Report`.

```python
class ReportEngine(BaseEngine):
    def run(
        self,
        results: Mapping[str, Any] | Iterable[ReportSection],
        *,
        title: str = "Cleaner Report",
        metadata: Mapping[str, Any] | None = None,
    ) -> Report: ...
```

### Report Dataclasses

- `ReportSection(name: str, title: str, content: Any)`: Represents a section within a report.
- `Report(title: str, sections: tuple[ReportSection, ...], metadata: Mapping[str, Any], created_at: datetime)`: Immutable report object.
  - Methods: `get_section(name: str) -> ReportSection | None`, `has_section(name: str) -> bool`.

### Renderers

- `TextRenderer()`: Renders report as formatted plain text string.
- `JSONRenderer()`: Renders report as formatted JSON string.

### Exporters

- `StringExporter()`: Exports rendered report to an in-memory string.
- `FileExporter(path: str | Path)`: Renders report and writes output to the specified filesystem `path` (creating parent directories as needed).

---

## 7. Pipeline Engine (`sanitizepy.pipeline`)

Import path:

```python
from sanitizepy.pipeline import (
    PipelineEngine,
    PipelineStep,
    CallableStep,
    TransformStep,
    PipelineResult,
    PipelineStepResult,
)
```

### `PipelineEngine`

Executes an ordered sequence of `PipelineStep` instances and tracks execution metrics.

```python
class PipelineEngine:
    def __init__(self, steps: Iterable[PipelineStep] | None = None) -> None: ...
    def add_step(self, step: PipelineStep) -> PipelineEngine: ...
    def remove_step(self, name: str) -> PipelineEngine: ...
    def clear(self) -> None: ...
    def run(self, data: pd.DataFrame) -> PipelineResult: ...
    @property
    def steps(self) -> tuple[PipelineStep, ...]: ...
```

- **Returns**: `PipelineResult` containing:
  - `data` (`pd.DataFrame`): Transformed output DataFrame.
  - `steps` (`tuple[PipelineStepResult, ...]`): Per-step execution metadata.
  - `duration_seconds` (`float`): Total pipeline duration in seconds.

### Pipeline Steps

- `CallableStep(name: str, operation: Callable[[pd.DataFrame], pd.DataFrame])`: Step wrapping a callable function (e.g., `CleaningEngine.run`).
- `TransformStep(name: str, transformer: Any)`: Step wrapping an object exposing a `.transform(data)` method (e.g., `FeatureEngineeringEngine`).

---

## 8. Base Engine & Data Models

### `BaseEngine` (`sanitizepy.engine.base`)

Abstract base class for all processing engines.

```python
class BaseEngine(ABC):
    def __init__(self, config: CleanerConfig | None = None) -> None: ...
    @property
    def config(self) -> CleanerConfig: ...
    @property
    def logger(self) -> logging.Logger: ...
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any: ...
```

### Core Models (`sanitizepy.models.base`)

Pydantic models for validated internal data structures:

- `BaseCleanerModel`: Root immutable Pydantic model (`frozen=True`, `strict=True`, `extra="forbid"`).
- `IdentifiedModel`: Model adding UUID string `id`.
- `TimestampedModel`: Model adding UTC creation timestamp `created_at`.
- `MetadataModel`: Model adding `metadata` dict container.
- `ResourceUsage`: Model for memory statistics (`memory_bytes`, `memory_mb`, `memory_gb`).
- `ExecutionTime`: Model for timing statistics (`seconds`).
- `DataShape`: Model representing DataFrame dimensions (`rows`, `columns`, `size`).
- `ColumnReference`: Model referencing a column (`name`, `dtype`).

---

## 9. Exceptions (`sanitizepy.exceptions`)

Import path:

```python
from sanitizepy.exceptions import CleanerError, ConfigurationError, DataValidationError, EngineError
```

All exceptions inherit from `CleanerError`:

```text
CleanerError
├── ConfigurationError
├── ValidationError
│   ├── DataValidationError
│   └── SchemaValidationError
├── EngineError
│   ├── EngineInitializationError
│   └── EngineExecutionError
├── InspectionError
│   ├── MissingValueInspectionError
│   ├── DuplicateInspectionError
│   ├── DataTypeInspectionError
│   ├── MemoryInspectionError
│   └── StatisticsInspectionError
├── CleaningError
│   ├── MissingValueCleaningError
│   ├── DuplicateCleaningError
│   ├── OutlierCleaningError
│   └── DataTypeConversionError
├── PreprocessingError
│   ├── EncodingError
│   ├── ScalingError
│   └── FeatureEngineeringError
├── RuleError
│   ├── RuleValidationError
│   └── RuleExecutionError
├── ReportError
│   ├── ReportGenerationError
│   └── ReportExportError
├── ModelError
│   ├── SerializationError
│   └── DeserializationError
└── UtilityError
    ├── FileSystemError
    └── DependencyError
```

---

## 10. Logging (`sanitizepy.logger`)

Import path:

```python
from sanitizepy.logger import get_logger, logger
```

- `get_logger(name: str | None = None) -> logging.Logger`: Returns a configured logger instance under the `"sanitizepy"` namespace (e.g. `"sanitizepy.CleaningEngine"`).
- `logger`: Root package logger instance (`logging.Logger`).
- **Format**: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
- **Date Format**: `%Y-%m-%d %H:%M:%S`
