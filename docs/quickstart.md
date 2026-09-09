# Quick Start Guide

This guide introduces the core workflows of `sanitizepy`: data inspection, cleaning, preprocessing, feature engineering, rule execution, report generation, and pipeline composition.

---

## 1. Prerequisites & Installation

Ensure `sanitizepy` is installed in your Python environment:

```bash
pip install sanitizepy
```

### Optional Extras

The core library runs on pandas + the standard library. Two opt-in extras unlock additional capabilities:

```bash
# rapidfuzz-backed similarity mode for near-duplicate detection/removal
pip install "sanitizepy[fuzzy]"

# ftfy-backed advanced encoding repair and advanced text-quality analysis
pip install "sanitizepy[text]"
```

These extras are optional; everything in this guide except the explicitly noted `method="similarity"` / `mode="advanced"` paths works without them. Using an opt-in path without its extra installed raises a `DependencyError` naming the required extra.

`sanitizepy` operates primarily on `pandas.DataFrame` objects. Verify your environment:

```python
import pandas as pd
import sanitizepy

print(f"Cleaner Version: {sanitizepy.__version__}")
```

---

## 2. Public API Imports

`sanitizepy` provides clean import paths for both high-level entry points and modular engine components:

```python
# Core Entry Point & Configuration
from sanitizepy import Cleaner, CleanerConfig, DEFAULT_CONFIG

# Inspection Submodule
from sanitizepy.inspection import (
    DatatypeInspector,
    DuplicateInspector,
    MemoryInspector,
    MissingValueInspector,
    StatisticsInspector,
)

# Cleaning Submodule
from sanitizepy.cleaning import (
    CleaningEngine,
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
    TypeCoercionOperation,
)

# Additional operations (re-exported from the top-level package)
from sanitizepy import (
    Cleaner,
    MissingTokenOperation,
    TextNormalizationOperation,
    EncodingRepairOperation,
    NearDuplicateRemovalOperation,
)

# Preprocessing & Feature Engineering Submodule
from sanitizepy.preprocessing import (
    ColumnInteraction,
    DatetimeFeatures,
    FeatureEngineeringEngine,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)

# Rule Submodule
from sanitizepy.rules import (
    Rule,
    RuleCategory,
    RuleEngine,
    RuleRegistry,
    RuleSeverity,
    register_builtin_rules,
)

# Report Submodule
from sanitizepy.reports import (
    FileExporter,
    JSONRenderer,
    ReportEngine,
    StringExporter,
    TextRenderer,
)

# Pipeline Submodule
from sanitizepy.pipeline import (
    CallableStep,
    PipelineEngine,
    TransformStep,
)
```

---

## 3. Data Inspection Workflow

The inspection package provides **read-only dataset analysis**. Inspectors inspect a `pandas.DataFrame` and return structured result objects without modifying the original DataFrame.

```python
import pandas as pd
from sanitizepy.inspection import (
    MissingValueInspector,
    DuplicateInspector,
    DatatypeInspector,
    MemoryInspector,
    StatisticsInspector,
)

# Load data
df = pd.read_csv("your_data.csv")

# Missing value inspection
missing_inspector = MissingValueInspector()
missing_result = missing_inspector.inspect(df, threshold=5.0)

print(f"Total Rows: {missing_result.summary.total_rows}")
print(f"Missing Cells: {missing_result.summary.missing_cells} ({missing_result.summary.missing_percentage}%)")
print(f"Missing Severity: {missing_result.summary.severity}")
for report in missing_result.column_reports:
    if report.has_missing:
        print(f" - Column '{report.column}': {report.missing_count} missing ({report.missing_percentage}%)")

# Duplicate row inspection
duplicate_inspector = DuplicateInspector()
dup_result = duplicate_inspector.inspect(df, keep="first")
print(f"Duplicate Rows: {dup_result.duplicate_count} ({dup_result.summary.duplicate_percentage}%)")

# Datatype inspection
datatype_inspector = DatatypeInspector()
type_result = datatype_inspector.inspect(df)
print(f"Numeric Columns: {type_result.summary.numeric_columns}")
print(f"Mixed Object Columns: {type_result.summary.mixed_object_columns}")

# Memory inspection
memory_inspector = MemoryInspector()
memory_result = memory_inspector.inspect(df)
print(f"Current Memory: {memory_result.summary.total_memory_mb:.2f} MB")
print(f"Estimated Savings: {memory_result.summary.estimated_saved_mb:.2f} MB ({memory_result.summary.estimated_saved_percentage}%)")

# Statistical inspection
stats_inspector = StatisticsInspector()
stats_result = stats_inspector.inspect(df)
for col_stat in stats_result.reports:
    print(f"Column '{col_stat.column}': Mean={col_stat.mean:.2f}, Median={col_stat.median:.2f}, Outliers={col_stat.outlier_count}")
```

---

## 4. Data Cleaning Workflow

The `CleaningEngine` applies an ordered sequence of deterministic cleaning operations to a DataFrame. It returns a new cleaned copy and does not mutate input data in-place.

```python
import pandas as pd
from sanitizepy.cleaning import (
    CleaningEngine,
    DropMissingRows,
    DropMissingColumns,
    FillMissing,
    DropDuplicates,
    DropColumns,
)

df = pd.read_csv("your_data.csv")

# Configure cleaning engine with operations
engine = CleaningEngine([
    # Fill strategies: "constant" (default), "median", "mean", "mode"
    FillMissing(strategy="median", subset=["numeric_score"]),
    FillMissing(value="Unknown", subset=["category_name"]),
    DropDuplicates(subset=["user_id"], keep="first"),
    DropColumns(columns=["internal_notes"]),
])

# Execute cleaning operations
cleaned_df = engine.run(df)

# Inspect executed steps
for op_info in engine.describe():
    print(f"Applied operation: {op_info['name']}")
```

### Detailed Results & Audit Log

Use `run_with_result(...)` to get a `CleaningResult` containing per-operation metrics and an ordered, JSON-serializable audit log. Pass `dry_run=True` to evaluate the plan without mutating your DataFrame, or `chunk_size=...` to process chunk-safe operations in row-wise batches.

```python
result = engine.run_with_result(df, dry_run=True)

print(result.summary())
for entry in result.audit_log:
    print(f"{entry['order']}. {entry['operation']}: "
          f"{entry['before_shape']} -> {entry['after_shape']} "
          f"({entry['rows_affected']} row(s) affected)")
```

### New Cleaning Operations

Additional deterministic operations cover common raw-data problems:

```python
engine = CleaningEngine([
    # Normalize sentinel strings ("n/a", "null", "?", "-", ...) to missing.
    MissingTokenOperation(extra_tokens={"missing", "unknown"}, subset=["city"]),

    # Unicode + whitespace + case normalization for text columns.
    TextNormalizationOperation(
        subset=["name"],
        unicode_form="NFKC",
        normalize_whitespace=True,
        case="title",
    ),

    # Detect/repair encoding artifacts (mojibake, replacement chars).
    # mode="advanced" requires: pip install "sanitizepy[text]"
    EncodingRepairOperation(subset=["description"], mode="core"),

    # Safe, deterministic type coercion.
    TypeCoercionOperation(
        target_dtypes={"age": "int64", "signup": "datetime64[ns]"},
        error_policy="coerce",
    ),

    # Remove near-duplicate rows, keeping one representative per group.
    # method="similarity" requires: pip install "sanitizepy[fuzzy]"
    NearDuplicateRemovalOperation(subset=["name", "email"], method="exact_normalized"),
])

cleaned_df = engine.run(df)
```

---

## 5. High-Level Facade, Profiling & Data Contracts

The top-level `Cleaner` facade offers a cohesive inspect → plan → clean workflow plus profiling and contract validation.

```python
import pandas as pd
from sanitizepy import Cleaner, DatasetProfiler, profile_to_report
from sanitizepy.models.contracts import ColumnContract, DataContract
from sanitizepy.reports import TextRenderer

df = pd.read_csv("your_data.csv")
cleaner = Cleaner()

# Inspect -> plan -> clean
report = cleaner.inspect(df)
plan = cleaner.plan(report)
result = cleaner.clean(df, plan=plan)      # returns a CleaningResult
cleaned_df = result.data

# Dataset profile (aggregates the standalone inspectors; core-only)
profile = cleaner.profile(df)              # == DatasetProfiler().profile(df)
print(f"Rows: {profile.row_count}, Columns: {profile.column_count}")

# Render the profile through the reports subsystem
profile_report = profile_to_report(profile, title="Dataset Profile")
print(TextRenderer().render(profile_report))

# Validate against a declarative data contract
contract = DataContract(columns={
    "age": ColumnContract(dtype="int64", nullable=False, min_value=0, max_value=120),
    "status": ColumnContract(allowed_values=("active", "inactive")),
    "email": ColumnContract(regex=r".+@.+\..+", unique=True),
})
results = cleaner.validate(df, contract)   # one RuleResult per expectation
for rr in results:
    status = "PASSED" if rr.passed else "FAILED"
    print(f"[{status}] {rr.rule}: {rr.message}")
```

---

## 6. Near-Duplicate, Text-Quality & Anomaly Analysis

Read-only, deterministic analyzers complement the inspectors.

```python
import pandas as pd
from sanitizepy.inspection import (
    NearDuplicateDetector,
    TextQualityAnalyzer,
    AnomalyInspector,
)

df = pd.read_csv("your_data.csv")

# Near-duplicate detection ("exact_normalized" is core-only)
detector = NearDuplicateDetector()
near_dup = detector.detect(df, subset=["name", "email"], method="exact_normalized")
print(f"Near-duplicate records: {near_dup.duplicate_count} across {len(near_dup.groups)} group(s)")

# Text-quality analysis (one result per text column)
for tq in TextQualityAnalyzer().analyze(df):
    print(f"Column '{tq.column}': mean length {tq.character_length_mean:.1f}, "
          f"empty-after-strip {tq.empty_after_strip_count}, "
          f"encoding-garbage {tq.encoding_garbage_count}")

# Anomaly detection ("iqr" or "zscore")
anomalies = AnomalyInspector().inspect(df, method="zscore", zscore_threshold=3.0, seed=42)
print(f"Total anomalies: {anomalies.total_anomalies}")
for col_report in anomalies.reports:
    print(f" - {col_report.column}: {col_report.anomaly_count} anomalies")
```

---

## 7. Reproducible Plans & Replay

A `CleaningPlan` can be serialized to a frozen, JSON-serializable snapshot and reconstructed later to replay the exact cleaning sequence.

```python
from sanitizepy import Cleaner

cleaner = Cleaner()
plan = cleaner.plan(df)

# Serialize to a replayable snapshot, then persist as JSON
replayable = plan.serialize()
plan_json = replayable.to_json()

# Later / elsewhere: reconstruct and re-run deterministically
from sanitizepy.models.replay import ReplayablePlan
from sanitizepy.cleaning import CleaningPlan

restored = ReplayablePlan.from_json(plan_json)
rebuilt_plan = CleaningPlan.deserialize(restored)
result = rebuilt_plan.apply(df)
```

---

## 8. Preprocessing & Feature Engineering Workflow

The `FeatureEngineeringEngine` manages stateful transformation operations using a standard `fit`/`transform` contract.

```python
import pandas as pd
from sanitizepy.preprocessing import (
    FeatureEngineeringEngine,
    ColumnInteraction,
    RatioFeature,
    PolynomialFeature,
    LogFeature,
    DatetimeFeatures,
)

df = pd.read_csv("your_data.csv")

# Ensure datetime dtype before DatetimeFeatures operation
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# Configure feature engineering engine
engine = FeatureEngineeringEngine([
    ColumnInteraction(column_a="price", column_b="quantity", output_column="total_amount"),
    RatioFeature(numerator="total_amount", denominator="quantity", output_column="calculated_unit_price", zero_division="nan"),
    PolynomialFeature(column="price", degree=2, include_bias=False),
    LogFeature(column="income", output_column="log_income", offset=1.0),
    DatetimeFeatures(column="timestamp", features=["year", "month", "day", "day_of_week"]),
])

# Fit on dataset and apply transformations
engine.fit(df)
engineered_df = engine.transform(df)

# Or in a single call:
# engineered_df = engine.fit_transform(df)

print(f"Generated columns: {list(engineered_df.columns)}")
```

---

## 9. Rule Validation Workflow

The `RuleEngine` evaluates validation rules registered in a `RuleRegistry` against a DataFrame, returning `RuleResult` instances.

```python
import pandas as pd
from sanitizepy.rules import (
    RuleEngine,
    RuleRegistry,
    register_builtin_rules,
    Rule,
    RuleSeverity,
    RuleCategory,
    RuleResult,
)

df = pd.read_csv("your_data.csv")

# Create registry and populate with built-in rules
registry = RuleRegistry()
register_builtin_rules(registry)

# Register a custom rule
class CustomNullCheck(Rule):
    def evaluate(self, dataframe: pd.DataFrame, **kwargs) -> RuleResult:
        has_nulls = dataframe.isna().any().any()
        return RuleResult(
            rule=self.name,
            passed=not has_nulls,
            severity=RuleSeverity.WARNING,
            category=RuleCategory.DATA_QUALITY,
            message="Dataset contains missing values" if has_nulls else "No missing values found",
            affected_rows=int(dataframe.isna().any(axis=1).sum()),
        )

registry.register(CustomNullCheck(name="custom_null_check", description="Verify dataset completeness"))

# Execute rule engine
rule_engine = RuleEngine(registry=registry)
results = rule_engine.run(df)

for result in results:
    status = "PASSED" if result.passed else "FAILED"
    print(f"[{result.severity.value.upper()}] Rule '{result.rule}': {status} - {result.message}")
```

---

## 10. Report Generation & Exporting Workflow

The `ReportEngine` accepts processing results or dictionaries and constructs a canonical, immutable `Report`. Renderers format reports into string or JSON representations, and exporters save or return them.

```python
import pandas as pd
from sanitizepy.reports import (
    ReportEngine,
    TextRenderer,
    JSONRenderer,
    StringExporter,
    FileExporter,
)

df = pd.read_csv("your_data.csv")

# Generate structured report
report_engine = ReportEngine()
report = report_engine.run(
    results={
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    },
    title="Data Ingestion Report",
    metadata={"environment": "production"},
)

# Render as plain text
text_rendered = TextRenderer().render(report)
print(text_rendered)

# Render as JSON string
json_rendered = JSONRenderer().render(report)

# Export to file
file_exporter = FileExporter("reports/ingestion_report.json")
output_path = file_exporter.export(report, JSONRenderer())
print(f"Report written to {output_path}")
```

---

## 11. Pipeline Composition Workflow

The `PipelineEngine` sequences processing steps. Use `CallableStep` for functions (like `CleaningEngine.run`) and `TransformStep` for transformers exposing a `.transform()` method (like `FeatureEngineeringEngine`).

```python
import pandas as pd
from sanitizepy.cleaning import CleaningEngine, DropDuplicates, FillMissing
from sanitizepy.preprocessing import ColumnInteraction, FeatureEngineeringEngine
from sanitizepy.pipeline import PipelineEngine, CallableStep, TransformStep

df = pd.read_csv("your_data.csv")

# 1. Setup cleaning
clean_engine = CleaningEngine([
    FillMissing(value=0.0, subset=["price"]),
    DropDuplicates(),
])

# 2. Setup feature engineering
feat_engine = FeatureEngineeringEngine([
    ColumnInteraction(column_a="price", column_b="tax_rate", output_column="tax_amount"),
])
feat_engine.fit(df)

# 3. Build and execute pipeline
pipeline = PipelineEngine()
pipeline.add_step(CallableStep("cleaning", clean_engine.run))
pipeline.add_step(TransformStep("features", feat_engine))

result = pipeline.run(df)

# Result contains final data and step metrics
final_df = result.data
print(f"Pipeline finished in {result.duration_seconds:.4f}s")
for step_metric in result.steps:
    print(f"Step '{step_metric.name}': {step_metric.input_rows}x{step_metric.input_columns} -> {step_metric.output_rows}x{step_metric.output_columns} ({step_metric.duration_seconds:.4f}s)")
```

---

## Next Steps

To explore parameter options, method signatures, dataclass fields, and exception hierarchies in detail, proceed to the [API Reference](file:///d:/sanitizepy/docs/api.md).
