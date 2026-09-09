"""
sanitizepy.cleaning.missing_tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`MissingTokenOperation` replaces configured sentinel strings in
object/string columns with the pandas missing marker (``pd.NA`` / ``None``).

This closes the "missing-token normalisation" gap for messy/raw datasets
where columns may contain strings like ``"n/a"``, ``"null"``, ``"nil"``,
``"?"``, ``"-"``, or any caller-supplied extension token.

The comparison is case-insensitive and whitespace-trimmed, so ``"  N/A  "``
and ``"na"`` are both matched by the default token ``"n/a"``.

References:
    Requirement 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.2, 13.1
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import DEFAULT_MISSING_VALUE_TOKENS
from .base import CleaningOperation, OperationResult


class MissingTokenOperation(CleaningOperation):
    """
    Replace sentinel/missing-token strings with ``pd.NA``.

    Parameters
    ----------
    extra_tokens:
        Additional tokens to treat as missing **in addition to**
        ``DEFAULT_MISSING_VALUE_TOKENS``.  The union is casefolded at
        construction time so runtime matching is O(1) per cell.
    subset:
        Column names to restrict the operation to.  When ``None`` every
        object/string column is processed.

    Notes
    -----
    * Comparison is case-insensitive and whitespace-trimmed (Req 3.4).
    * Configured tokens *extend*, not replace, the defaults (Req 3.3).
    * ``is_chunk_safe = True`` because the per-cell replacement does not
      depend on the rest of the dataset (Req 7.2).
    """

    name = "missing_token_normalization"
    is_chunk_safe: bool = True
    is_inplace_safe: bool = False

    def __init__(
        self,
        extra_tokens: frozenset[str] | set[str] | list[str] | None = None,
        subset: list[str] | None = None,
    ) -> None:
        extra: frozenset[str] = (
            frozenset(extra_tokens) if extra_tokens is not None else frozenset()
        )
        # Casefold the complete token set once at construction time.
        self._token_set: frozenset[str] = frozenset(
            t.casefold() for t in (*DEFAULT_MISSING_VALUE_TOKENS, *extra)
        )
        self.extra_tokens: frozenset[str] = extra
        self.subset: list[str] | None = subset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """Return the object/string columns to process."""
        if self.subset is not None:
            missing_columns = [
                col for col in self.subset if col not in dataframe.columns
            ]
            if missing_columns:
                raise KeyError(f"Columns not found in dataframe: {missing_columns}")
            return list(self.subset)

        # Auto-detect: only operate on object/string-typed columns.
        return [
            col
            for col in dataframe.columns
            if dataframe[col].dtype == object
            or pd.api.types.is_string_dtype(dataframe[col])
        ]

    def _is_token(self, value: object) -> bool:
        """Return True when *value* normalises to a missing-token string."""
        if not isinstance(value, str):
            return False
        return value.strip().casefold() in self._token_set

    def _replace_tokens_in_series(self, series: pd.Series) -> pd.Series:
        """
        Return a new Series with token strings replaced by ``np.nan``.

        Only object/string columns contain candidate tokens; numeric or
        boolean columns are returned unchanged.
        """
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            return series

        # Vectorised path: build a boolean mask, then assign NaN.
        mask = series.apply(self._is_token)
        if not mask.any():
            return series

        result = series.copy()
        result[mask] = np.nan
        return result

    # ------------------------------------------------------------------
    # CleaningOperation interface
    # ------------------------------------------------------------------

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        columns = self._target_columns(dataframe)

        result = dataframe.copy()
        for col in columns:
            result[col] = self._replace_tokens_in_series(result[col])
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

        # Count missing before applying on targeted columns only.
        missing_before = int(dataframe[columns].isna().to_numpy().sum())

        transformed = self.apply(dataframe)

        after_shape = transformed.shape
        missing_after = int(transformed[columns].isna().to_numpy().sum())
        cells_converted = missing_after - missing_before

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(columns),
            rows_affected=cells_converted,
            columns_affected=0,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(
                f"{self.name}: converted {cells_converted} cell(s) to missing"
            ),
            dry_run=dry_run,
            details={**self.describe(), "cells_converted_to_missing": cells_converted},
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "extra_tokens": sorted(self.extra_tokens),
            "subset": list(self.subset) if self.subset is not None else None,
            "token_count": len(self._token_set),
        }
