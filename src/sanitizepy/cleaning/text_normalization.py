"""
sanitizepy.cleaning.text_normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic, local text normalization for object/string columns.

:class:`TextNormalizationOperation` applies a configurable combination of:

* **Unicode normalization** (NFC or NFKC) via :mod:`unicodedata` (Req 4.2).
* **Whitespace normalization**: strip leading/trailing whitespace, collapse
  internal whitespace runs to a single ASCII space, replace U+00A0
  non-breaking spaces with regular spaces (Req 4.3).
* **Case transform**: lower, upper, or title (Req 4.4).

A value that becomes empty after normalization is replaced with the pandas
missing marker (Req 4.5).

Normalization steps are applied in the order: Unicode normalization,
whitespace normalization, case transform, then a final Unicode
re-normalization pass. The re-normalization is required because casing can
produce combining sequences that fall out of the requested normal form; it
guarantees the operation is idempotent (Req 4.7).

The operation is **chunk-safe** because each cell is processed independently
(Req 7.2) and depends only on Core_Stack + stdlib (Req 4.8).

References:
    Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 7.2, 13.1
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .base import CleaningOperation, OperationResult

# Internal whitespace collapse: replace one or more whitespace characters
# (including U+00A0 after the earlier substitution step) with a single space.
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")

CaseTransform = Literal["lower", "upper", "title", "none"]
UnicodeForm = Literal["NFC", "NFKC", "none"]


class TextNormalizationOperation(CleaningOperation):
    """
    Deterministic, local text normalization for object/string columns.

    Parameters
    ----------
    subset:
        Column names to restrict the operation to. When ``None`` every
        object/string column is processed.
    unicode_form:
        ``"NFC"`` or ``"NFKC"`` to apply Unicode normalization (via
        :mod:`unicodedata`); ``"none"`` to skip (default ``"NFC"``).
    normalize_whitespace:
        When ``True`` (default) strip leading/trailing whitespace, collapse
        internal whitespace runs, and replace non-breaking spaces
        (``U+00A0``) with regular spaces.
    case:
        ``"lower"``, ``"upper"``, or ``"title"`` to apply case
        transformation; ``"none"`` to skip (default ``"none"``).

    Notes
    -----
    * Cells that become empty strings after normalization are replaced with
      ``np.nan`` (Req 4.5).
    * ``is_chunk_safe = True`` (Req 7.2, 4.8).
    * Only :mod:`unicodedata`, :mod:`re`, numpy, and pandas are required.
    """

    name = "text_normalization"
    is_chunk_safe: bool = True
    is_inplace_safe: bool = False

    def __init__(
        self,
        subset: list[str] | None = None,
        unicode_form: UnicodeForm = "NFC",
        normalize_whitespace: bool = True,
        case: CaseTransform = "none",
    ) -> None:
        valid_unicode = {"NFC", "NFKC", "none"}
        if unicode_form not in valid_unicode:
            raise ValueError(
                f"unicode_form must be one of {valid_unicode}; got {unicode_form!r}"
            )
        valid_case = {"lower", "upper", "title", "none"}
        if case not in valid_case:
            raise ValueError(f"case must be one of {valid_case}; got {case!r}")

        self.subset: list[str] | None = subset
        self.unicode_form: UnicodeForm = unicode_form
        self.normalize_whitespace: bool = normalize_whitespace
        self.case: CaseTransform = case

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """Return the object/string column names to process."""
        if self.subset is not None:
            missing_columns = [
                col for col in self.subset if col not in dataframe.columns
            ]
            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")
            return list(self.subset)

        return [
            col
            for col in dataframe.columns
            if is_object_dtype(dataframe[col]) or is_string_dtype(dataframe[col])
        ]

    def _normalize_value(self, value: object) -> object:
        """
        Apply the configured normalization steps to a single cell value.

        Non-string (including ``None`` / ``NaN``) values are returned
        unchanged.

        Returns
        -------
        str | float
            The normalized string, or ``np.nan`` if normalization produces
            an empty string.
        """
        if not isinstance(value, str):
            return value

        s: str = value

        # Step 1: Unicode normalization.
        if self.unicode_form != "none":
            s = unicodedata.normalize(self.unicode_form, s)

        # Step 2: Whitespace normalization.
        if self.normalize_whitespace:
            # Replace non-breaking spaces with regular spaces first so
            # that the regex collapse step catches them uniformly.
            s = s.replace("\u00a0", " ")
            s = _WHITESPACE_RE.sub(" ", s).strip()

        # Step 3: Case transform.
        if self.case == "lower":
            s = s.lower()
        elif self.case == "upper":
            s = s.upper()
        elif self.case == "title":
            s = s.title()

        # Step 4: Re-apply Unicode normalization after the case transform.
        # Casing operations (upper/lower/title) can produce combining-mark
        # sequences that are no longer in the requested normal form. For
        # example, NFKC(U+1FE2) → "U+03C5 U+0308 U+0300"; upper() →
        # "U+03A5 U+0308 U+0300", but "U+03A5 U+0308" has an NFKC-precomposed
        # form (U+03AB), so a subsequent pass would re-normalize and change
        # the output. Normalizing last guarantees the final value is already
        # in the requested normal form, making f(f(x)) == f(x) (Req 4.7).
        if self.unicode_form != "none":
            s = unicodedata.normalize(self.unicode_form, s)

        # Step 5: Empty-after-normalization → missing.
        if s == "":
            return np.nan

        return s

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Return a new Series with each string value normalized."""
        if not (is_object_dtype(series) or is_string_dtype(series)):
            return series
        return series.apply(self._normalize_value)

    # ------------------------------------------------------------------
    # CleaningOperation interface
    # ------------------------------------------------------------------

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        columns = self._target_columns(dataframe)
        result = dataframe.copy()
        for col in columns:
            result[col] = self._normalize_series(result[col])
        return result

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        before_shape = dataframe.shape
        columns = self._target_columns(dataframe)

        transformed = self.apply(dataframe)

        after_shape = transformed.shape

        # Count cells where the value actually changed (including cells that
        # became NaN after normalization).
        values_modified = 0
        for col in columns:
            original_col = dataframe[col]
            transformed_col = transformed[col]
            for orig, new in zip(original_col, transformed_col, strict=True):
                if pd.isna(orig) and pd.isna(new):
                    continue  # both missing — unchanged
                if pd.isna(orig) != pd.isna(new):
                    values_modified += 1  # became missing (or vice-versa)
                elif orig != new:
                    values_modified += 1

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(columns),
            rows_affected=values_modified,
            columns_affected=0,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(f"{self.name}: modified {values_modified} cell(s)"),
            dry_run=dry_run,
            details={**self.describe(), "values_modified": values_modified},
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
            "unicode_form": self.unicode_form,
            "normalize_whitespace": self.normalize_whitespace,
            "case": self.case,
        }
