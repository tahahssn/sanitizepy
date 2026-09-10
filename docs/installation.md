# Installation Guide

This guide describes how to install, verify, upgrade, and configure the `sanitizepy` Python package for standard usage and local development.

---

## Requirements

### Supported Python Versions

`sanitizepy` requires **Python 3.11** or newer (`>=3.11`). Tested against 3.11, 3.12, and 3.13.

### Runtime Dependencies

When installed via `pip`, the following core dependencies are automatically resolved:

| Dependency | Version Range | Purpose |
| :--- | :--- | :--- |
| `numpy` | `>=1.24.0,<2.3.0` | Numerical array operations and mathematical primitives |
| `pandas` | `>=2.0.0,<2.4.0` | Tabular data structure and DataFrame operations |
| `rich` | `>=13.0.0,<15.0.0` | Rich terminal output and formatted rendering |
| `pydantic` | `>=2.0.0,<3.0.0` | Data model validation and schema enforcement |

The core install is intentionally minimal: `scipy` is **not** a dependency (nothing in the library imports it), so a standard `pip install sanitizepy` stays light and fast.

### Optional Extras

Two capabilities are opt-in and only pulled in when explicitly requested:

| Extra | Adds | Unlocks |
| :--- | :--- | :--- |
| `sanitizepy[fuzzy]` | `rapidfuzz>=3.0.0,<4.0.0` | Similarity-based near-duplicate detection/removal (`method="similarity"`) |
| `sanitizepy[text]` | `ftfy>=6.0.0,<7.0.0` | Advanced encoding repair (`mode="advanced"`) |

```bash
pip install "sanitizepy[fuzzy]"
pip install "sanitizepy[text]"
pip install "sanitizepy[fuzzy,text]"   # both
```

Using an opt-in code path (`method="similarity"`, `mode="advanced"`) without its extra installed raises a `DependencyError` naming the missing package — it never fails silently or falls back unexpectedly.

---

## Standard Installation

To install the latest release of `sanitizepy` from PyPI, run:

```bash
pip install sanitizepy
```

Alternatively, use the module syntax to ensure installation into the active Python environment:

```bash
python -m pip install sanitizepy
```

### Quiet Installation

`pip` prints a full dependency-resolution log by default. For a clean install with minimal output, use `-q`:

```bash
pip install -q sanitizepy
```

---

## Virtual Environment Setup (Recommended)

To isolate `sanitizepy` and its dependencies from system-level packages, create and activate a Python virtual environment:

### On Linux / macOS

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install sanitizepy
pip install sanitizepy
```

### On Windows (PowerShell)

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install sanitizepy
pip install sanitizepy
```

### On Windows (Command Prompt)

```cmd
:: Create a virtual environment named .venv
python -m venv .venv

:: Activate the virtual environment
.\.venv\Scripts\activate.bat

:: Install sanitizepy
pip install sanitizepy
```

---

## Verification

After installation, verify that `sanitizepy` is correctly installed and accessible by printing the package version:

```bash
python -c "import sanitizepy; print(sanitizepy.__version__)"
```

Or using the `get_version()` helper function:

```bash
python -c "from sanitizepy import get_version; print(get_version())"
```

Expected output:

```text
0.2.1
```

You can also verify that the main `Cleaner` entry point imports cleanly:

```bash
python -c "from sanitizepy import Cleaner; c = Cleaner(); print(c.config.encoding)"
```

Expected output:

```text
utf-8
```

---

## Simple API Usage (v0.2.1+)

As of version 0.2.1, `sanitizepy` provides a high-level, pandas-style Simple API accessed via `sp.*` functions for one-liner operations on DataFrames.

### Quick Example

```python
import sanitizepy as sp

df = pd.read_csv("data.csv")

# Inspect
sp.inspect(df)          # overview
sp.missing(df)          # missing values
sp.duplicates(df)       # duplicates

# Clean
df_clean = sp.clean(df)  # auto-clean: missing tokens → text norm → encoding → dedup

# Transform
df = sp.fill_missing(df, strategy="median")
df = sp.fix_types(df)
df = sp.dummies(df, cols=["city"])

# Validate
sp.validate(df)

# Report
sp.report(df, save="report.json")
```

All functions accept a DataFrame as the first argument, use smart auto-detection for column types, and return rich, printable result objects that render beautifully in the terminal or notebook:

```python
result = sp.missing(df)      # MissingInspectionResult
result.to_dict()              # → dict
result.to_json()              # → JSON string
```

See [Simple API Reference](../api.md#2-simple-api-sanitizepysimple) for the full list of functions.

---

## Package Upgrade

To upgrade `sanitizepy` to the latest version, run:

```bash
pip install --upgrade sanitizepy
```

---

## Version Pinning

To install a specific version of `sanitizepy`:

```bash
pip install sanitizepy==0.2.1
```

---

## Developer / Contributor Installation

If you intend to contribute to `sanitizepy`, modify the source code, or run the test suite, perform an editable installation:

1. Clone the repository:

   ```bash
   git clone https://github.com/tahahssn/sanitizepy.git
   cd sanitizepy
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. Install `sanitizepy` in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   Add the optional extras too if you need them for local testing:

   ```bash
   pip install -e ".[dev,fuzzy,text]"
   ```

Development dependencies installed by `.[dev]` include:

- `black`: Code formatting (`>=24.0.0,<25.0.0`)
- `ruff`: Fast Python linter (`>=0.4.0,<1.0.0`)
- `mypy`: Static type checker (`>=1.10.0,<2.0.0`)
- `pandas-stubs`: Type stubs for pandas (`>=2.0.0`)
- `pytest`: Testing framework (`>=8.0.0,<9.0.0`)
- `pytest-cov`: Test coverage plugin (`>=5.0.0,<7.0.0`)
- `hypothesis`: Property-based testing (`>=6.100.0,<7.0.0`)

4. Verify the developer setup by running the test suite and static checks:

   ```bash
   pytest
   ruff check src tests
   black --check src tests
   mypy src
   ```

> **Important**: Editable installation (`pip install -e .`) is strictly intended for local source development. Standard users should always use `pip install sanitizepy`.
