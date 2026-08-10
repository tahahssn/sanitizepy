
# cleaner

A production-ready Python library for automated data inspection, cleaning, preprocessing, feature engineering, rule execution, reporting, and pipeline orchestration.

---

## Overview

`cleaner` provides modular, high-performance data engineering components built on top of `pandas`, `numpy`, `scipy`, `rich`, and `pydantic`. The library is designed around independent engines that can be used individually or composed into structured processing pipelines.

`cleaner` separates responsibilities into dedicated subsystems:
- **Core & Config**: Centralized configuration (`CleanerConfig`) and logging (`logger`).
- **Inspection Engine**: Read-only dataset analysis covering missing values, duplicates, datatypes, memory consumption, and statistical distributions.
- **Cleaning Engine**: Deterministic dataset transformations including row/column dropping, duplicate removal, value imputation, and column selection.
- **Preprocessing & Feature Engineering**: Stateful fit/transform operations for interactions, ratio features, polynomial terms, logarithmic transformations, and datetime extraction.
- **Rule Engine**: Validation framework with built-in data quality rules, severity levels, and category classifications.
- **Report Engine**: Structured report generation, rendering (Text, JSON), and exporting (String, File).
- **Pipeline Engine**: Execution workflow orchestration with step timing and metadata tracking.

---

## Capabilities

| Area | Component | Key Functionality |
| :--- | :--- | :--- |
| **Core & Config** | `Cleaner`, `CleanerConfig` | Package entry point, global runtime settings, logging configuration |
| **Inspection** | `MissingValueInspector`, `DuplicateInspector`, `DatatypeInspector`, `MemoryInspector`, `StatisticsInspector` | Read-only dataset analysis, severity scoring, memory estimation, distribution stats |
| **Cleaning** | `CleaningEngine`, `DropMissingRows`, `DropMissingColumns`, `FillMissing`, `DropDuplicates`, `DropColumns` | Ordered row and column cleaning operations without silent mutations |
| **Preprocessing** | `FeatureEngineeringEngine`, `ColumnInteraction`, `RatioFeature`, `PolynomialFeature`, `LogFeature`, `DatetimeFeatures` | Stateful fit/transform feature generation preserving dataset indices |
| **Rules** | `RuleEngine`, `RuleRegistry`, `Rule`, `register_builtin_rules` | Data quality rules, severity levels (`info`, `warning`, `error`, `critical`), custom rules |
| **Reporting** | `ReportEngine`, `TextRenderer`, `JSONRenderer`, `StringExporter`, `FileExporter` | Structured immutable reports with multi-format rendering and exporting |
| **Pipeline** | `PipelineEngine`, `CallableStep`, `TransformStep` | Sequenced workflow execution with step duration and row/column metrics |

---

## Requirements

- **Python**: `>=3.11`
- **Core Dependencies**:
  - `numpy >= 2.5.1`
  - `pandas >= 3.0.5`
  - `scipy >= 1.18.0`
  - `rich >= 15.0.0`
  - `pydantic >= 2.13.4`

---

## Installation

### Standard User Installation

Install `cleaner` using `pip`:

```bash
pip install cleaner
```

Or via `python -m pip`:

```bash
python -m pip install cleaner
```

### Developer / Contributor Installation

For local development or contributing to the codebase, clone the repository and perform an editable installation with development dependencies:

```bash
git clone https://github.com/cleaner-dev/cleaner.git
cd cleaner
pip install -e .[dev]
```

> **Note**: `pip install -e .` is reserved for local development and testing. Standard library users should install via `pip install cleaner`.

---

## Quick Start

Below is a complete workflow demonstrating data inspection, cleaning, feature engineering, rule execution, report generation, and pipeline composition.

```python
import pandas as pd

from cleaner import Cleaner, CleanerConfig
from cleaner.cleaning import CleaningEngine, DropDuplicates, FillMissing
from cleaner.inspection import MissingValueInspector
from cleaner.pipeline import CallableStep, PipelineEngine, TransformStep
from cleaner.preprocessing import ColumnInteraction, FeatureEngineeringEngine, RatioFeature
from cleaner.reports import ReportEngine, StringExporter, TextRenderer
from cleaner.rules import RuleEngine, RuleRegistry, register_builtin_rules

# 1. Initialize global configuration
cleaner = Cleaner(config=CleanerConfig(float_precision=4, preview_rows=10))

# Load your dataset
df = pd.read_csv("your_data.csv")

# 2. Perform read-only missing value inspection
inspector = MissingValueInspector()
inspection_result = inspector.inspect(df)
print(f"Missing Severity: {inspection_result.summary.severity}")
print(f"Missing Cells: {inspection_result.summary.missing_cells}")

# 3. Configure data cleaning
clean_engine = CleaningEngine([
    FillMissing(value=0.0, subset=["numeric_column"]),
    DropDuplicates(keep="first"),
])

# 4. Configure feature engineering (fit/transform contract)
feat_engine = FeatureEngineeringEngine([
    ColumnInteraction("feature_a", "feature_b", output_column="a_x_b"),
    RatioFeature("feature_a", "feature_b", output_column="a_div_b", zero_division="nan"),
])

# 5. Compose a processing pipeline
pipeline = PipelineEngine([
    CallableStep("cleaning_step", clean_engine.run),
    TransformStep("feature_step", feat_engine),
])

# Execute pipeline
pipeline_result = pipeline.run(df)
processed_df = pipeline_result.data
print(f"Pipeline executed in {pipeline_result.duration_seconds:.4f} seconds")

# 6. Evaluate data quality rules
registry = RuleRegistry()
register_builtin_rules(registry)
rule_engine = RuleEngine(registry=registry)
rule_results = rule_engine.run(processed_df)

# 7. Generate and render structured report
report_engine = ReportEngine()
report = report_engine.run(
    results={
        "inspection": inspection_result.summary,
        "cleaning_operations": clean_engine.describe(),
        "pipeline_steps": [s.name for s in pipeline_result.steps],
    },
    title="Dataset Processing Summary",
)

rendered_report = StringExporter().export(report, TextRenderer())
print(rendered_report)
```

---

## Architecture & Design

`cleaner` adopts a decoupled architecture where inspection, cleaning, feature engineering, validation, and reporting are separate single-responsibility components:

```text
               ┌───────────────────────┐
               │     Input Data        │
               └───────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             │    Inspection Engine      │ (Read-Only Analysis)
             └─────────────┬─────────────┘
                           │
             ┌─────────────┴─────────────┐
             │     Cleaning Engine       │ (Deterministic Cleaning)
             └─────────────┬─────────────┘
                           │
             ┌─────────────┴─────────────┐
             │ Preprocessing / Features  │ (Fit/Transform Operations)
             └─────────────┬─────────────┘
                           │
             ┌─────────────┴─────────────┐
             │       Rule Engine         │ (Quality Validation)
             └─────────────┬─────────────┘
                           │
             ┌─────────────┴─────────────┐
             │     Pipeline Engine       │ (Workflow Orchestration)
             └─────────────┬─────────────┘
                           │
             ┌─────────────┴─────────────┐
             │      Report Engine        │ (Rendering & Exporting)
             └───────────────────────────┘
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Installation Guide](file:///d:/cleaner/docs/installation.md): Requirements, virtual environments, installation commands, verification, and upgrade procedures.
- [Quick Start Guide](file:///d:/cleaner/docs/quickstart.md): Step-by-step examples for inspection, cleaning, feature engineering, rules, reporting, and pipelines.
- [API Reference](file:///d:/cleaner/docs/api.md): Complete technical API documentation for classes, functions, dataclasses, models, and exceptions.

---

## Development & Testing

To run the project test suite or linting tools:

### Run Tests

```bash
pytest
```

### Code Formatting & Linting

```bash
black --check src tests
ruff check src tests
mypy src
```

---

## License

`cleaner` is distributed under the terms of the [MIT License](file:///d:/cleaner/pyproject.toml).
