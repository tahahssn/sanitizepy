"""
sanitizepy.cleaning.encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`EncodingRepairOperation` detects and repairs encoding artifacts
(mojibake, replacement characters, control characters) in object/string
columns.

**Core repair** (Core_Stack + stdlib only):
* Replaces the Unicode replacement character ``U+FFFD`` with ``np.nan``
  when the entire cell is a replacement character, or strips it when
  mixed with legitimate text.
* Removes ASCII control characters (excluding HT, LF, CR).

**Advanced repair** (optional *ftfy* extra):
* Invokes ``ftfy.fix_text()`` to deterministically repair mojibake and
  other encoding-corruption patterns.
* Raises :class:`~sanitizepy.exceptions.DependencyError` naming
  ``sanitizepy[text]`` when *ftfy* is not installed.

:class:`~sanitizepy.exceptions.EncodingError` is raised when a detected
artifact cannot be safely/deterministically repaired (e.g. a cell that
consists entirely of the replacement character and the caller has requested
strict mode).

References:
    Requirements 5.2, 5.3, 5.4, 7.2, 13.1, 16.3, 18.5
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from ..exceptions import DependencyError, EncodingError
from .base import CleaningOperation, OperationResult

# ---------------------------------------------------------------------------
# Internal helpers mirrored from inspection.detector for independence
# ---------------------------------------------------------------------------

# Unicode REPLACEMENT CHARACTER – produced by lossy decoding.
_REPLACEMENT_CHAR: str = "\ufffd"

# ASCII control characters excluding common printable-range whitespace
# (HT=\x09, LF=\x0a, CR=\x0d).
_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Mojibake detection (same pattern as inspection.detector)
_MOJIBAKE_RE: re.Pattern[str] = re.compile(
    r"[\xc0-\xc3][\x80-\xbf\xa0-\xff]"
    r"|Ã[^\s]"
    r"|â\x80[\x93\x94\x98\x99\x9c\x9d\xa2\xa6\xa0]"
)

RepairMode = Literal["core", "advanced"]


def _lazy_import_ftfy() -> Any:
    """
    Lazily import *ftfy* and raise :class:`DependencyError` when absent.

    Returns
    -------
    module
        The ``ftfy`` module.

    Raises
    ------
    DependencyError
        When *ftfy* is not installed, with the extra name ``sanitizepy[text]``.
    """
    try:
        import ftfy  # noqa: PLC0415

        return ftfy
    except ImportError as exc:
        raise DependencyError(
            "The 'ftfy' package is required for advanced encoding repair. "
            "Install it via: pip install 'sanitizepy[text]'"
        ) from exc


def _core_repair_value(value: object) -> object:
    """
    Apply deterministic core repair to a single cell value.

    Steps
    -----
    1. Return non-string values unchanged (including existing ``NaN``).
    2. Strip ASCII control characters.
    3. If the result is entirely replacement characters (``U+FFFD``),
       return ``np.nan`` (the value is unrecoverable without the extra).
    4. Remove any remaining replacement characters embedded in otherwise
       valid text.
    5. Return the repaired string, or ``np.nan`` if the string becomes empty.

    Returns
    -------
    str | float
        Repaired string or ``np.nan``.
    """
    if not isinstance(value, str):
        return value

    # Step 2: strip control characters
    repaired = _CONTROL_CHAR_RE.sub("", value)

    # Step 3: cell is entirely replacement characters → unrecoverable
    stripped_replacements = repaired.replace(_REPLACEMENT_CHAR, "")
    if len(stripped_replacements) == 0 and _REPLACEMENT_CHAR in repaired:
        return np.nan

    # Step 4: remove embedded replacement characters
    repaired = repaired.replace(_REPLACEMENT_CHAR, "")

    # Step 5: empty after repair → missing
    if repaired == "":
        return np.nan

    return repaired


def _advanced_repair_value(value: object, ftfy: Any) -> object:
    """
    Apply advanced encoding repair via *ftfy* to a single cell value.

    Returns
    -------
    str | float
        Repaired string or ``np.nan`` if the value becomes empty.
    """
    if not isinstance(value, str):
        return value

    # First apply core repairs (control chars, replacement chars)
    core = _core_repair_value(value)
    if not isinstance(core, str):
        return core

    repaired: str = ftfy.fix_text(core)
    return repaired if repaired else np.nan


def _has_artifact(value: object) -> bool:
    """Return True when *value* contains at least one encoding artifact."""
    if not isinstance(value, str):
        return False
    return bool(
        _MOJIBAKE_RE.search(value)
        or _REPLACEMENT_CHAR in value
        or _CONTROL_CHAR_RE.search(value)
    )


class EncodingRepairOperation(CleaningOperation):
    """
    Detect and repair encoding artifacts in object/string columns.

    Parameters
    ----------
    subset:
        Column names to restrict the operation to.  When ``None`` every
        object/string column is processed.
    mode:
        ``"core"`` (default) uses only the Core_Stack + stdlib for
        deterministic repair.  ``"advanced"`` additionally invokes *ftfy*
        for mojibake repair; raises :class:`DependencyError` when *ftfy*
        is not installed.
    error_on_unrepaired:
        When ``True`` (default), raise :class:`EncodingError` if an
        artifact cell cannot be repaired deterministically (i.e. it
        remains an artifact after the repair step).  When ``False``,
        leave unrepaired artifacts in place and record them in the result
        details.

    Notes
    -----
    * ``is_chunk_safe = True`` because each cell is processed independently.
    * Advanced mode lazily imports *ftfy* only when first invoked.
    * Raises :class:`DependencyError` naming ``sanitizepy[text]`` when
      advanced mode requires *ftfy* but it is absent (Req 16.3, 18.5).
    * Raises :class:`EncodingError` when an artifact cannot be safely
      repaired in strict mode (Req 5.3).

    References:
        Requirements 5.2, 5.3, 5.4, 7.2, 13.1
    """

    name = "encoding_repair"
    is_chunk_safe: bool = True
    is_inplace_safe: bool = False

    def __init__(
        self,
        subset: list[str] | None = None,
        mode: RepairMode = "core",
        error_on_unrepaired: bool = True,
    ) -> None:
        valid_modes: set[str] = {"core", "advanced"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}; got {mode!r}")
        self.subset: list[str] | None = subset
        self.mode: RepairMode = mode
        self.error_on_unrepaired: bool = error_on_unrepaired

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

        return [
            col
            for col in dataframe.columns
            if is_object_dtype(dataframe[col]) or is_string_dtype(dataframe[col])
        ]

    def _repair_value(self, value: object, ftfy: Any | None) -> object:
        """Dispatch to core or advanced repair based on *mode*."""
        if self.mode == "advanced" and ftfy is not None:
            return _advanced_repair_value(value, ftfy)
        return _core_repair_value(value)

    def _repair_series(
        self, series: pd.Series, ftfy: Any | None
    ) -> tuple[pd.Series, int, list[object]]:
        """
        Repair a single Series.

        Returns
        -------
        repaired_series, count_repaired, unrepaired_indices
        """
        count_repaired = 0
        unrepaired_indices: list[object] = []
        values: list[object] = []

        for idx, value in series.items():
            had_artifact = _has_artifact(value)
            repaired = self._repair_value(value, ftfy)

            if had_artifact:
                if _has_artifact(repaired):
                    # Still has artifact after repair
                    unrepaired_indices.append(idx)
                    values.append(repaired)
                else:
                    count_repaired += 1
                    values.append(repaired)
            else:
                values.append(repaired)

        result_series = pd.Series(values, index=series.index, dtype=object)
        return result_series, count_repaired, unrepaired_indices

    # ------------------------------------------------------------------
    # CleaningOperation interface
    # ------------------------------------------------------------------

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        # Eagerly import ftfy if advanced mode (raises DependencyError if missing)
        ftfy: Any | None = None
        if self.mode == "advanced":
            ftfy = _lazy_import_ftfy()

        columns = self._target_columns(dataframe)
        result = dataframe.copy()

        for col in columns:
            repaired_series, _count, unrepaired = self._repair_series(result[col], ftfy)
            if unrepaired and self.error_on_unrepaired:
                raise EncodingError(
                    f"Column '{col}' contains {len(unrepaired)} artifact(s) that "
                    f"cannot be deterministically repaired in mode={self.mode!r}. "
                    f"Affected indices: {unrepaired[:5]}"
                )
            result[col] = repaired_series

        return result

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        # Eagerly import ftfy if advanced mode (raises DependencyError if missing)
        ftfy: Any | None = None
        if self.mode == "advanced":
            ftfy = _lazy_import_ftfy()

        before_shape = dataframe.shape
        columns = self._target_columns(dataframe)

        result = dataframe.copy()
        total_repaired = 0
        total_unrepaired: list[tuple[str, object]] = []

        for col in columns:
            repaired_series, count, unrepaired = self._repair_series(result[col], ftfy)
            total_repaired += count
            if unrepaired and self.error_on_unrepaired:
                raise EncodingError(
                    f"Column '{col}' contains {len(unrepaired)} artifact(s) that "
                    f"cannot be deterministically repaired in mode={self.mode!r}. "
                    f"Affected indices: {unrepaired[:5]}"
                )
            for idx in unrepaired:
                total_unrepaired.append((col, idx))
            result[col] = repaired_series

        after_shape = result.shape

        operation_result = OperationResult(
            operation_name=self.name,
            affected_columns=list(columns),
            rows_affected=total_repaired,
            columns_affected=0,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(
                f"{self.name}: repaired {total_repaired} cell(s) "
                f"in mode={self.mode!r}"
            ),
            dry_run=dry_run,
            details={
                **self.describe(),
                "values_repaired": total_repaired,
                "unrepaired_count": len(total_unrepaired),
            },
        )

        final_df = dataframe.copy() if dry_run else result
        return final_df, operation_result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
            "mode": self.mode,
            "error_on_unrepaired": self.error_on_unrepaired,
        }
