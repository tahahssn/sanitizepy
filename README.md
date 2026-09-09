<a id="readme-top"></a>

<div align="center">

<img src="assets/banner.svg" alt="sanitizepy banner" width="70%" />

# sanitizepy

### *Find what is wrong with your data, understand why, and fix it, without guessing.*

Automated tabular **data quality inspection**, **explainable cleaning**, and **preprocessing** in pure Python.

<br/>

[![PyPI Version](https://img.shields.io/pypi/v/sanitizepy?style=for-the-badge&color=4f46e5&label=pypi)](https://pypi.org/project/sanitizepy)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-3b82f6?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-10b981?style=for-the-badge)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/tahahssn/sanitizepy?style=for-the-badge&color=f59e0b)](https://github.com/tahahssn/sanitizepy/stargazers)

<a href="#quick-start"><strong>Quick Start</strong></a>
&middot;
<a href="./docs">Documentation</a>
&middot;
<a href="https://github.com/tahahssn/sanitizepy/issues/new?labels=bug">Report Bug</a>
&middot;
<a href="https://github.com/tahahssn/sanitizepy/issues/new?labels=enhancement">Request Feature</a>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#why-sanitizepy">Why sanitizepy</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#inspect-understand-whats-wrong">Inspect</a></li>
        <li><a href="#plan-preview-before-you-touch-anything">Plan</a></li>
        <li><a href="#clean-dry-run-then-apply">Clean</a></li>
        <li><a href="#profile-and-validate">Profile &amp; Validate</a></li>
      </ul>
    </li>
    <li><a href="#how-it-works">How It Works</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#support">Support</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Why sanitizepy

Most data-cleaning code is a pile of one-off `pandas` snippets: drop these rows, fill those nulls, strip that whitespace. It works once, then rots. Six months later nobody remembers *why* a column was dropped, and there is no record of what changed.

`sanitizepy` treats cleaning as a transparent, reviewable process instead of a black box:

- **You see what is wrong first.** Every dataset gets a health score and a list of concrete issues before anything is modified.
- **You get told why.** Each issue carries a plain-language explanation, a severity, and the evidence behind it.
- **Nothing changes without your say-so.** Cleaning runs as a previewable plan with a dry-run mode, so you inspect the impact before applying it.
- **Everything is recorded.** Each run produces a JSON-serializable audit log of exactly which operations touched which rows and columns.

The whole library is pure Python on top of `pandas`, with a deliberately small dependency footprint so it installs fast and stays out of your way.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

- **Dataset health inspection** — a composite **0–100 health score** across completeness, uniqueness, consistency, validity, datatypes, and memory.
- **Explainable issue detection** — every finding reports *what*, *why*, *severity*, and *evidence*, not just a boolean.
- **Previewable cleaning plans** — enable/disable individual steps and preview them before running.
- **Safe, deterministic cleaning** — dry-run mode, before/after impact metrics, and a full audit trail. Your input DataFrame is never mutated unexpectedly.
- **Real cleaning operations** — statistical fills, safe type coercion, missing-token normalization, text normalization, encoding repair, and near-duplicate removal.
- **Data profiling & contracts** — build an immutable dataset profile or validate a DataFrame against a declarative data contract.
- **Anomaly & text-quality analysis** — deterministic IQR / z-score outlier detection and per-column text-quality metrics.
- **Replayable plans** — serialize a cleaning plan and re-apply it to future data.
- **Typed** — ships a `py.typed` marker, so your editor and type checker see real types.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

Requires **Python 3.11+**. Core dependencies are just `numpy`, `pandas`, `rich`, and `pydantic`.

```bash
pip install sanitizepy
```

Prefer a quiet install without the dependency-resolution noise? Use `-q`:

```bash
pip install -q sanitizepy
```

### Optional extras

Some features rely on optional libraries and are only imported when you use them:

```bash
pip install "sanitizepy[fuzzy]"   # similarity-based near-duplicate detection (rapidfuzz)
pip install "sanitizepy[text]"    # advanced encoding repair (ftfy)
```

### From source (contributors)

```bash
git clone https://github.com/tahahssn/sanitizepy.git
cd sanitizepy
pip install -e ".[dev]"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Quick Start

```python
import pandas as pd
from sanitizepy import Cleaner

df = pd.read_csv("your_data.csv")
cleaner = Cleaner()

# 1. See what is wrong
report = cleaner.inspect(df)
report.show()                       # rich terminal health report

# 2. Preview a cleaning plan (nothing changes yet)
plan = cleaner.plan(report)
plan.show()

# 3. Apply it and keep the cleaned frame
result = cleaner.clean(df, plan=plan, dry_run=False)
cleaned_df = result.data
print(result.summary())
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Inspect: understand what's wrong

`inspect()` is read-only. It returns a `DatasetHealthReport` with a score, issues, and recommendations.

```python
report = cleaner.inspect(df)

print(f"Health score: {report.health_score}/100")
print(f"Critical issues: {len(report.critical_issues)}")
print(f"Recommendations: {len(report.recommendations)}")

report.show()   # formatted table of issues + recommendations
```

### Plan: preview before you touch anything

A `CleaningPlan` is a previewable, editable list of steps. Steps are 1-based.

```python
plan = cleaner.plan(report)
plan.show()

plan.disable(2)   # skip step #2
plan.enable(2)    # put it back
```

### Clean: dry-run, then apply

```python
# Dry run: compute impact without changing data
preview = cleaner.clean(df, plan=plan, dry_run=True)
print(preview.summary())

# Real run: returns a new cleaned DataFrame
result = cleaner.clean(df, plan=plan, dry_run=False)
cleaned_df = result.data

print(result.summary())     # human-readable summary
print(result.audit_log)     # JSON-serializable record of every operation
```

Prefer the module-level shortcuts?

```python
from sanitizepy import inspect, plan, clean

report = inspect(df)
cleaning_plan = plan(report)
result = clean(df, cleaning_plan=cleaning_plan, dry_run=True)
```

### Profile and validate

```python
from sanitizepy import Cleaner, DataContract, ColumnContract

cleaner = Cleaner()

# Immutable dataset profile (datatypes, missing values, duplicates, memory, stats)
profile = cleaner.profile(df)

# Validate against a declarative contract.
# ColumnContract fields are all optional; DataContract.columns is a
# mapping of column name -> its expectations.
contract = DataContract(
    columns={
        "order_id": ColumnContract(nullable=False, unique=True),
        "amount": ColumnContract(dtype="float64", min_value=0),
    },
)
results = cleaner.validate(df, contract)
for result in results:
    print(result)
```

### Direct engine access

For granular control, compose operations yourself:

```python
from sanitizepy.cleaning import CleaningEngine, DropDuplicates, FillMissing

engine = CleaningEngine([
    FillMissing(value=0.0, subset=["amount"]),
    DropDuplicates(keep="first"),
])

result = engine.run_with_result(df, dry_run=False)
print(result.summary())
print(result.audit_log)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How It Works

`sanitizepy` follows a transparent pipeline: **Detect → Explain → Recommend → Preview → Apply → Audit.**

```
   Dataset
      |
      v
  Inspect / Profile      (read-only)
      |
      v
  Explainable issues     (what / why / severity / evidence)
      |
      v
  Cleaning plan          (previewable, editable)
      |
   you approve
      |
      v
  Transformation engine  (deterministic, dry-run capable)
      |
      v
  Result + audit log     (auditable, JSON-serializable)
```

The input DataFrame is treated as immutable; cleaning returns a new frame and records every step.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation

Full docs live in the [`docs/`](./docs) directory:

- [Installation Guide](./docs/installation.md) — requirements, virtual environments, verification, upgrades.
- [Quick Start Guide](./docs/quickstart.md) — worked examples for inspection, cleaning, profiling, rules, and pipelines.
- [API Reference](./docs/api.md) — classes, functions, models, and exceptions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome and appreciated. If you have an idea that would make `sanitizepy` better, open an issue or a pull request.

1. Fork the project
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Stage only the files you changed: `git add path/to/file.py`
4. Commit your changes: `git commit -m "Add your feature"`
5. Push the branch: `git push origin feature/your-feature`
6. Open a pull request

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md) first. Found a security issue? See [SECURITY.md](./SECURITY.md).

### Development

```bash
pip install -e ".[dev]"

pytest                                  # run the test suite
ruff check src tests                    # lint
black --check src tests                 # format check
mypy src                                # type check
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Support

<div align="center">
<br/>
<a href="https://www.patreon.com/cw/SyedTahaHassan">
  <img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron" height="48" />
</a>
<br/><br/>

If `sanitizepy` saved you hours of painful data cleaning, consider supporting its development.
Every contribution helps keep it open-source, maintained, and free.

**[→ Become a Patron](https://www.patreon.com/cw/SyedTahaHassan)**
<br/>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
