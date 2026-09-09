<!--
*** Thanks for checking out sanitizepy. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or
*** simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/tahahssn/sanitizepy">
    <img src="https://raw.githubusercontent.com/tahahssn/sanitizepy/main/assets/banner.svg" alt="Logo" width="70%">
  </a>

  <h3 align="center">sanitizepy</h3>

  <!-- PROJECT SHIELDS -->
  <p align="center">
    <a href="https://github.com/tahahssn/sanitizepy/graphs/contributors"><img src="https://img.shields.io/github/contributors/tahahssn/sanitizepy.svg?style=for-the-badge" alt="Contributors"></a>
    <a href="https://github.com/tahahssn/sanitizepy/network/members"><img src="https://img.shields.io/github/forks/tahahssn/sanitizepy.svg?style=for-the-badge" alt="Forks"></a>
    <a href="https://github.com/tahahssn/sanitizepy/stargazers"><img src="https://img.shields.io/github/stars/tahahssn/sanitizepy.svg?style=for-the-badge" alt="Stargazers"></a>
    <a href="https://github.com/tahahssn/sanitizepy/issues"><img src="https://img.shields.io/github/issues/tahahssn/sanitizepy.svg?style=for-the-badge" alt="Issues"></a>
    <a href="https://github.com/tahahssn/sanitizepy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tahahssn/sanitizepy.svg?style=for-the-badge" alt="MIT License"></a>
    <a href="https://pypi.org/project/sanitizepy"><img src="https://img.shields.io/pypi/v/sanitizepy.svg?style=for-the-badge&label=pypi" alt="PyPI"></a>
  </p>

  <p align="center">
    Automated data quality inspection, explainable cleaning, and preprocessing for pandas.
    <br />
    <a href="./docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://pypi.org/project/sanitizepy">View on PyPI</a>
    &middot;
    <a href="https://github.com/tahahssn/sanitizepy/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/tahahssn/sanitizepy/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Most data-cleaning code is a pile of one-off `pandas` snippets: drop these rows, fill those nulls, strip that whitespace. It works once, then rots. Six months later nobody remembers *why* a column was dropped, and there is no record of what changed.

`sanitizepy` treats cleaning as a transparent, reviewable process instead of a black box:

* You see what is wrong first. Every dataset gets a health score and a list of concrete issues before anything is modified.
* You get told why. Each issue carries a plain-language explanation, a severity, and the evidence behind it.
* Nothing changes without your say-so. Cleaning runs as a previewable plan with a dry-run mode, so you inspect the impact before applying it.
* Everything is recorded. Each run produces a JSON-serializable audit log of exactly which operations touched which rows and columns.

Core capabilities:

* **Dataset health inspection:** a composite 0–100 health score across completeness, uniqueness, consistency, validity, datatypes, and memory.
* **Explainable issue detection:** every finding reports *what*, *why*, *severity*, and *evidence*.
* **Previewable cleaning plans:** enable/disable individual steps before running anything.
* **Safe, deterministic cleaning:** dry-run mode, before/after impact metrics, full audit trail, no unexpected mutation of your input.
* **Real cleaning operations:** statistical fills, safe type coercion, missing-token normalization, text normalization, encoding repair, near-duplicate removal.
* **Data profiling & contracts:** build an immutable dataset profile or validate a DataFrame against a declarative data contract.
* **Anomaly & text-quality analysis:** deterministic IQR / z-score outlier detection and per-column text-quality metrics.
* **Typed:** ships a `py.typed` marker.


### Built With

* [![Python][Python-badge]][Python-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![Pandas][Pandas-badge]][Pandas-url]
* [![Pydantic][Pydantic-badge]][Pydantic-url]
* [![Rich][Rich-badge]][Rich-url]


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

* Python 3.11 or later
  ```sh
  python --version
  ```

### Installation

1. Install from PyPI
   ```sh
   pip install sanitizepy
   ```
   Prefer a quiet install without pip's dependency-resolution noise?
   ```sh
   pip install -q sanitizepy
   ```
2. Or clone the repo for local development
   ```sh
   git clone https://github.com/tahahssn/sanitizepy.git
   cd sanitizepy
   ```
3. Install with dev dependencies
   ```sh
   pip install -e ".[dev]"
   ```
4. Change git remote url to avoid accidental pushes to the base project (if you forked it)
   ```sh
   git remote set-url origin github_username/sanitizepy
   git remote -v # confirm the changes
   ```

Optional extras, only imported when you use them:

```sh
pip install "sanitizepy[fuzzy]"   # similarity-based near-duplicate detection (rapidfuzz)
pip install "sanitizepy[text]"    # advanced encoding repair (ftfy)
```


<!-- USAGE EXAMPLES. -->
## Usage

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
plan.disable(1)                     # skip step #1 (steps are 1-based)
plan.enable(1)                      # put it back

# 3. Dry-run, then apply for real
preview = cleaner.clean(df, plan=plan, dry_run=True)
result = cleaner.clean(df, plan=plan, dry_run=False)

cleaned_df = result.data
print(result.summary())             # human-readable summary
print(result.audit_log)             # JSON-serializable record of every operation
```

Profiling and data contracts:

```python
from sanitizepy import Cleaner, DataContract, ColumnContract

cleaner = Cleaner()
profile = cleaner.profile(df)

contract = DataContract(
    columns={
        "order_id": ColumnContract(nullable=False, unique=True),
        "amount": ColumnContract(dtype="float64", min_value=0),
    },
)
results = cleaner.validate(df, contract)
```

*For the full API — direct engine access, replayable plans, anomaly/text-quality analysis — see the [Documentation](./docs).*


<!-- ROADMAP -->
## Roadmap

- [x] Dataset health scoring and explainable issue detection
- [x] Previewable, editable cleaning plans with dry-run mode and audit log
- [x] Statistical fills, type coercion, text normalization, encoding repair, near-duplicate removal
- [x] Dataset profiling and declarative data contracts
- [x] Anomaly detection (IQR / z-score) and text-quality analysis
- [x] `py.typed` marker for downstream type checking
- [ ] Command-line interface for running plans against files
- [ ] Additional built-in rules and connectors
- [ ] Multi-language documentation

See the [open issues](https://github.com/tahahssn/sanitizepy/issues) for a full list of proposed features and known issues.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Stage only the files you changed (`git add path/to/file.py`)
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Before opening a pull request, make sure these all pass locally:

| Check | Command |
|---|---|
| Unit Tests | `pytest` |
| Formatting | `black --check src tests` |
| Linting | `ruff check src tests` |
| Type Checking | `mypy src` |

Please also read [CONTRIBUTING.md](./CONTRIBUTING.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md). Found a security issue? See [SECURITY.md](./SECURITY.md) — do not open a public issue for vulnerabilities.

### Top contributors:

<a href="https://github.com/tahahssn/sanitizepy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tahahssn/sanitizepy" alt="contrib.rocks image" />
</a>


<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.


<!-- CONTACT -->
## Contact

Syed Muhammad Taha Hassan - [@tahahssn](https://github.com/tahahssn)

Project Link: [https://github.com/tahahssn/sanitizepy](https://github.com/tahahssn/sanitizepy)

If `sanitizepy` saved you hours of painful data cleaning, consider [supporting its development on Patreon](https://www.patreon.com/cw/SyedTahaHassan).


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/), the foundation everything here is built on
* [Pydantic](https://docs.pydantic.dev/) for the data models and contracts
* [Rich](https://github.com/Textualize/rich) for the terminal reports
* [Choose an Open Source License](https://choosealicense.com)
* [Img Shields](https://shields.io)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template), which this README's structure is based on


<!-- MARKDOWN LINKS & IMAGES -->
[Python-badge]: https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[NumPy-badge]: https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Pydantic-badge]: https://img.shields.io/badge/pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[Rich-badge]: https://img.shields.io/badge/rich-000000?style=for-the-badge
[Rich-url]: https://github.com/Textualize/rich
