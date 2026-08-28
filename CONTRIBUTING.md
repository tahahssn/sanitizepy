# Contributing to sanitizepy

Thank you for your interest in contributing to `sanitizepy`! We welcome contributions of all kinds, including bug fixes, feature improvements, documentation updates, and test coverage enhancements.

## Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/tahahssn/sanitizepy.git
   cd sanitizepy
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .[dev]
   ```

## Code Quality Standards

Before submitting a Pull Request, ensure all quality checks pass locally:

- **Run Unit Tests**:
  ```bash
  pytest
  ```
- **Code Formatting**:
  ```bash
  black src tests
  ```
- **Linting**:
  ```bash
  ruff check src tests
  ```
- **Static Type Checking**:
  ```bash
  mypy src
  ```

## Design Principles

1. **Never Silently Mutate Data**: Transformations must be transparent, deterministic, and explainable.
2. **Explainability First**: Every recommendation or operation must include a human-readable explanation (`WHAT`, `WHY`, `SEVERITY`, `EVIDENCE`).
3. **Decoupled Architecture**: Keep inspection, cleaning, rules, preprocessing, and reporting as independent modules.
