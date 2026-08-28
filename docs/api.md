# API Reference

This document provides technical API documentation for all public entry points, engines, operations, models, dataclasses, and exceptions in `sanitizepy`.

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
    CleaningOperation,
    DropMissingRows,
    DropMissingColumns,
    FillMissing,
    DropDuplicates,
    DropColumns,
)
```

### `CleaningEngine`

Orchestrates execution of an ordered sequence of `CleaningOperation` instances.

```python
class CleaningEngine:
    def __init__(self, operations: Iterable[CleaningOperation] | None = None) -> None: ...
    def add(self, operation: CleaningOperation) -> None: ...
    def clear(self) -> None: ...
    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame: ...
    def describe(self) -> list[dict[str, object]]: ...
    @property
    def operations(self) -> tuple[CleaningOperation, ...]: ...
```

- **Methods**:
  - `run(dataframe)`: Applies configured operations in order to a copy of `dataframe` and returns the cleaned result.
  - `describe()`: Returns lightweight dictionary representations of all configured operations.
- **Raises**:
  - `TypeError`: If input object is not a `pd.DataFrame` or an added operation is not a `CleaningOperation`.

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
FillMissing(value: Any, subset: list[str] | None = None)
```

- Fills missing values with the explicit `value` parameter.
- **Raises**: `KeyError` if any column in `subset` does not exist in the DataFrame.

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
