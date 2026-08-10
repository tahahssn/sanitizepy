# Installation Guide

This guide describes how to install, verify, upgrade, and configure the `cleaner` Python package for standard usage and local development.

---

## Requirements

### Supported Python Versions

`cleaner` requires **Python 3.11** or newer (`>=3.11`).

### Runtime Dependencies

When installed via `pip`, the following core dependencies are automatically resolved:

| Dependency | Minimum Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `>= 2.5.1` | Numerical array operations and mathematical primitives |
| `pandas` | `>= 3.0.5` | Tabular data structure and DataFrame operations |
| `scipy` | `>= 1.18.0` | Statistical calculations and distribution analysis |
| `rich` | `>= 15.0.0` | Rich terminal output and formatted rendering |
| `pydantic` | `>= 2.13.4` | Data model validation and schema enforcement |

---

## Standard Installation

To install the latest release of `cleaner` from PyPI, run:

```bash
pip install cleaner
```

Alternatively, use the module syntax to ensure installation into the active Python environment:

```bash
python -m pip install cleaner
```

---

## Virtual Environment Setup (Recommended)

To isolate `cleaner` and its dependencies from system-level packages, create and activate a Python virtual environment:

### On Linux / macOS

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install cleaner
pip install cleaner
```

### On Windows (PowerShell)

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install cleaner
pip install cleaner
```

### On Windows (Command Prompt)

```cmd
:: Create a virtual environment named .venv
python -m venv .venv

:: Activate the virtual environment
.\.venv\Scripts\activate.bat

:: Install cleaner
pip install cleaner
```

---

## Verification

After installation, verify that `cleaner` is correctly installed and accessible by printing the package version:

```bash
python -c "import cleaner; print(cleaner.__version__)"
```

Or using the `get_version()` helper function:

```bash
python -c "from cleaner import get_version; print(get_version())"
```

Expected output:

```text
0.1.0
```

You can also verify that the main `Cleaner` entry point imports cleanly:

```bash
python -c "from cleaner import Cleaner; c = Cleaner(); print(c.config.encoding)"
```

Expected output:

```text
utf-8
```

---

## Package Upgrade

To upgrade `cleaner` to the latest version, run:

```bash
pip install --upgrade cleaner
```

---

## Version Pinning

To install a specific version of `cleaner`:

```bash
pip install cleaner==0.1.0
```

---

## Developer / Contributor Installation

If you intend to contribute to `cleaner`, modify the source code, or run the test suite, perform an editable installation:

1. Clone the repository:

   ```bash
   git clone https://github.com/cleaner-dev/cleaner.git
   cd cleaner
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. Install `cleaner` in editable mode with development dependencies:

   ```bash
   pip install -e .[dev]
   ```

Development dependencies installed by `.[dev]` include:

- `black`: Code formatting (`>= 26.5.1`)
- `ruff`: Fast Python linter (`>= 0.16.0`)
- `mypy`: Static type checker (`>= 1.18.0`)
- `pytest`: Testing framework (`>= 9.1.1`)
- `pytest-cov`: Test coverage plugin (`>= 7.1.0`)

4. Verify the developer setup by running tests:

   ```bash
   pytest
   ```

> **Important**: Editable installation (`pip install -e .`) is strictly intended for local source development. Standard users should always use `pip install cleaner`.
