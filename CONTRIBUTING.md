<div align="center">

# Contributing to sanitizepy

### *Bug fixes, new inspectors, documentation, tests: all of it matters. Here's how to get started.*

</div>

## Requirements

- **Python**: `>=3.11`
- **Core Dependencies**: `numpy`, `pandas`, `scipy`, `rich`, `pydantic`
- **Dev Dependencies**: `pytest`, `black`, `ruff`, `mypy`

## Development Setup

**Fork and clone the repository:**

```bash
git clone https://github.com/tahahssn/sanitizepy.git
cd sanitizepy
```

**Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**Install with dev dependencies:**

```bash
pip install -e .[dev]
```

## Code Quality Standards

Before opening a pull request, make sure all checks pass locally:

| Check | Command |
|---|---|
| **Unit Tests** | `pytest` |
| **Formatting** | `black src tests` |
| **Linting** | `ruff check src tests` |
| **Type Checking** | `mypy src` |

All four must pass cleanly. PRs that fail any check will not be merged until resolved.

## Design Principles

Every contribution to `sanitizepy` should respect the core architecture:

- **Never silently mutate data:** Transformations must be transparent, deterministic, and explainable. If something changes, the user should be able to see exactly what and why.
- **Explainability first:** Every recommendation or operation must include a human-readable explanation (`WHAT`, `WHY`, `SEVERITY`, `EVIDENCE`). Black-box behaviour is a bug.
- **Decoupled architecture:** Inspection, cleaning, rules, preprocessing, and reporting are independent modules. Keep them that way.

## Submitting Changes

1. Create a feature branch off `main`
2. Make your changes with clear, descriptive commit messages
3. Run the full quality suite (see above)
4. Open a pull request with a summary of what changed and why
5. Reference any related issues in the PR description

## What to Work On

Not sure where to start? Look for issues tagged `good first issue` or `help wanted` in the [issue tracker](https://github.com/tahahssn/sanitizepy/issues). Documentation improvements and additional test coverage are always welcome.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).