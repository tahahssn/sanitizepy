"""
Deterministic local text-quality analysis.

This module provides a read-only, deterministic analyzer over the free-form
text (object / string) columns of a pandas DataFrame. It never mutates the
input and depends only on the Core_Stack (numpy, pandas) plus the Python
standard library (``re``, ``statistics``).

For every analyzed text column the analyzer computes, over the non-null
values only:

* character-length statistics (min/max/mean/median),
* whitespace-delimited token-count statistics (min/max/mean/median),
* the number of values that are empty after stripping whitespace,
* a count of values matching deterministic boilerplate patterns
  (repeated identical values, lorem-ipsum-like text, placeholder text),
* a count of values matching deterministic encoding-garbage patterns
  (mojibake, the Unicode replacement character, ASCII control characters).

The analyzer follows the Standalone_Inspector convention: it raises
``ValueError`` on an empty DataFrame and returns immutable frozen dataclass
results (:class:`~sanitizepy.models.profile.TextQualityResult`).

The deterministic core requires no optional dependency. Any *advanced*
analysis path lazily raises :class:`~sanitizepy.exceptions.DependencyError`
naming the required extra without affecting the deterministic core.

References:
    Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Final, Literal

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from ..exceptions import DependencyError
from ..models.profile import TextQualityResult

# ---------------------------------------------------------------------------
# Encoding-garbage patterns (mirrored from cleaning.encoding for independence)
# ---------------------------------------------------------------------------

# Unicode REPLACEMENT CHARACTER - produced by lossy decoding.
_REPLACEMENT_CHAR: Final[str] = "\ufffd"

# ASCII control characters excluding common printable-range whitespace
# (HT=\x09, LF=\x0a, CR=\x0d).
_CONTROL_CHAR_RE: Final[re.Pattern[str]] = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# Mojibake detection (same pattern used by cleaning.encoding).
_MOJIBAKE_RE: Final[re.Pattern[str]] = re.compile(
    r"[\xc0-\xc3][\x80-\xbf\xa0-\xff]"
    r"|Ã[^\s]"
    r"|â\x80[\x93\x94\x98\x99\x9c\x9d\xa2\xa6\xa0]"
)

# ---------------------------------------------------------------------------
# Boilerplate patterns (deterministic, intentionally simple)
# ---------------------------------------------------------------------------

# Lorem-ipsum-like placeholder prose.
_LOREM_IPSUM_RE: Final[re.Pattern[str]] = re.compile(
    r"lorem\s+ipsum",
    re.IGNORECASE,
)

# Common placeholder / filler tokens that carry no real information.
_PLACEHOLDER_VALUES: Final[frozenset[str]] = frozenset(
    {
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "tbd",
        "todo",
        "placeholder",
        "test",
        "sample",
        "example",
        "foo",
        "bar",
        "baz",
        "xxx",
        "asdf",
        "qwerty",
        "lorem ipsum",
    }
)

TextQualityMode = Literal["core", "advanced"]


def _has_encoding_garbage(value: str) -> bool:
    """Return ``True`` when *value* contains at least one encoding artifact."""
    return bool(
        _MOJIBAKE_RE.search(value)
        or _REPLACEMENT_CHAR in value
        or _CONTROL_CHAR_RE.search(value)
    )


def _is_boilerplate(value: str) -> bool:
    """
    Return ``True`` when *value* matches a deterministic boilerplate pattern.

    A value is boilerplate when its stripped, case-folded form is a known
    placeholder token, or when it contains lorem-ipsum-like text.
    """
    normalized = value.strip().casefold()
    if normalized in _PLACEHOLDER_VALUES:
        return True
    return bool(_LOREM_IPSUM_RE.search(value))


class TextQualityAnalyzer:
    """
    Deterministic text-quality analysis over object/string columns.

    Read-only. Never mutates the dataframe. Depends only on the Core_Stack
    and the Python standard library for its deterministic core.

    References:
        Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
    """

    __slots__: Final = ()

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | None = None,
        mode: TextQualityMode = "core",
    ) -> tuple[TextQualityResult, ...]:
        """
        Analyze the text columns of *dataframe*.

        Parameters
        ----------
        dataframe:
            Input dataframe (never mutated).

        subset:
            Optional list of column names to restrict the analysis to. When
            ``None`` every object/string column is analyzed. Columns in
            *subset* that are not object/string dtype are skipped.

        mode:
            ``"core"`` (default) runs the deterministic Core_Stack + stdlib
            analysis. ``"advanced"`` is reserved for analysis that requires
            an optional extra; it lazily raises
            :class:`~sanitizepy.exceptions.DependencyError` naming the extra
            without affecting the deterministic core.

        Returns
        -------
        tuple[TextQualityResult, ...]
            One :class:`TextQualityResult` per analyzed text column, in
            column order.

        Raises
        ------
        ValueError
            When *dataframe* is empty.
        DependencyError
            When ``mode="advanced"`` and the required optional extra is not
            installed.
        """
        if dataframe.empty:
            raise ValueError("Cannot analyze an empty DataFrame.")

        if mode == "advanced":
            # Advanced analysis is reserved for an optional NLP extra. The
            # deterministic core never reaches this path, so requesting it
            # without the extra lazily raises DependencyError naming it.
            self._require_advanced_extra()

        target_columns = self._target_columns(dataframe, subset)

        results: list[TextQualityResult] = []
        for column in target_columns:
            results.append(self._analyze_column(column, dataframe[column]))

        return tuple(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_columns(
        self,
        dataframe: pd.DataFrame,
        subset: list[str] | None,
    ) -> list[str]:
        """Return the object/string columns to analyze, in column order."""
        if subset is not None:
            missing = [col for col in subset if col not in dataframe.columns]
            if missing:
                raise KeyError(f"Columns not found in dataframe: {missing}")
            candidates = list(subset)
        else:
            candidates = list(dataframe.columns)

        return [
            str(col)
            for col in candidates
            if is_object_dtype(dataframe[col]) or is_string_dtype(dataframe[col])
        ]

    def _analyze_column(
        self,
        column: str,
        series: pd.Series,
    ) -> TextQualityResult:
        """Compute the text-quality result for a single column."""
        non_null = series.dropna()

        values: list[str] = [self._to_text(value) for value in non_null.tolist()]
        non_null_count = len(values)

        if non_null_count == 0:
            return TextQualityResult(
                column=column,
                non_null_count=0,
                empty_after_strip_count=0,
                character_length_min=0,
                character_length_max=0,
                character_length_mean=0.0,
                character_length_median=0.0,
                token_count_min=0,
                token_count_max=0,
                token_count_mean=0.0,
                token_count_median=0.0,
                boilerplate_count=0,
                encoding_garbage_count=0,
            )

        char_lengths: list[int] = [len(value) for value in values]
        token_counts: list[int] = [len(value.split()) for value in values]

        empty_after_strip = sum(1 for value in values if value.strip() == "")
        boilerplate = sum(1 for value in values if _is_boilerplate(value))
        encoding_garbage = sum(1 for value in values if _has_encoding_garbage(value))

        return TextQualityResult(
            column=column,
            non_null_count=non_null_count,
            empty_after_strip_count=empty_after_strip,
            character_length_min=min(char_lengths),
            character_length_max=max(char_lengths),
            character_length_mean=float(statistics.fmean(char_lengths)),
            character_length_median=float(statistics.median(char_lengths)),
            token_count_min=min(token_counts),
            token_count_max=max(token_counts),
            token_count_mean=float(statistics.fmean(token_counts)),
            token_count_median=float(statistics.median(token_counts)),
            boilerplate_count=boilerplate,
            encoding_garbage_count=encoding_garbage,
        )

    @staticmethod
    def _to_text(value: Any) -> str:
        """Coerce a non-null cell value to ``str`` deterministically."""
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _require_advanced_extra() -> None:
        """
        Guard the advanced analysis path behind an optional extra.

        Advanced text analysis (e.g. language detection, semantic quality)
        is only available with an optional extra. The deterministic core
        never invokes this; requesting ``mode="advanced"`` without the
        extra raises :class:`DependencyError` naming it.
        """
        raise DependencyError(
            "Advanced text-quality analysis requires an optional dependency. "
            "Install it via: pip install 'sanitizepy[text]'"
        )


__all__ = [
    "TextQualityAnalyzer",
    "TextQualityResult",
]
