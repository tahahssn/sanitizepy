# Quick Start Guide

This guide introduces the core workflows of `cleaner`: data inspection, cleaning, preprocessing, feature engineering, rule execution, report generation, and pipeline composition.

---

## 1. Prerequisites & Installation

Ensure `cleaner` is installed in your Python environment:

```bash
pip install cleaner
```

`cleaner` operates primarily on `pandas.DataFrame` objects. Verify your environment:

```python
import pandas as pd
import cleaner

print(f"Cleaner Version: {cleaner.__version__}")
```

---

## 2. Public API Imports

`cleaner` provides clean import paths for both high-level entry points and modular engine components:

```python
# Core Entry Point & Configuration
from cleaner import Cleaner, CleanerConfig, DEFAULT_CONFIG

# Inspection Submodule
from cleaner.inspection import (
    DatatypeInspector,
    DuplicateInspector,
    MemoryInspector,
    MissingValueInspector,
    StatisticsInspector,
)

# Cleaning Submodule
from cleaner.cleaning import (
    CleaningEngine,
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)

# Preprocessing & Feature Engineering Submodule
from cleaner.preprocessing import (
    ColumnInteraction,
    DatetimeFeatures,
    FeatureEngineeringEngine,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)

# Rule Submodule
from cleaner.rules import (
    Rule,
    RuleCategory,
    RuleEngine,
    RuleRegistry,
    RuleSeverity,
    register_builtin_rules,
)

# Report Submodule
from cleaner.reports import (
    FileExporter,
    JSONRenderer,
    ReportEngine,
    StringExporter,
    TextRenderer,
)

# Pipeline Submodule
from cleaner.pipeline import (
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
from cleaner.inspection import (
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
from cleaner.cleaning import (
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
    FillMissing(value=0.0, subset=["numeric_score"]),
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

---

## 5. Preprocessing & Feature Engineering Workflow

The `FeatureEngineeringEngine` manages stateful transformation operations using a standard `fit`/`transform` contract.

```python
import pandas as pd
from cleaner.preprocessing import (
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

## 6. Rule Validation Workflow

The `RuleEngine` evaluates validation rules registered in a `RuleRegistry` against a DataFrame, returning `RuleResult` instances.

```python
import pandas as pd
from cleaner.rules import (
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

## 7. Report Generation & Exporting Workflow

The `ReportEngine` accepts processing results or dictionaries and constructs a canonical, immutable `Report`. Renderers format reports into string or JSON representations, and exporters save or return them.

```python
import pandas as pd
from cleaner.reports import (
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

## 8. Pipeline Composition Workflow

The `PipelineEngine` sequences processing steps. Use `CallableStep` for functions (like `CleaningEngine.run`) and `TransformStep` for transformers exposing a `.transform()` method (like `FeatureEngineeringEngine`).

```python
import pandas as pd
from cleaner.cleaning import CleaningEngine, DropDuplicates, FillMissing
from cleaner.preprocessing import ColumnInteraction, FeatureEngineeringEngine
from cleaner.pipeline import PipelineEngine, CallableStep, TransformStep

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

To explore parameter options, method signatures, dataclass fields, and exception hierarchies in detail, proceed to the [API Reference](file:///d:/cleaner/docs/api.md).
