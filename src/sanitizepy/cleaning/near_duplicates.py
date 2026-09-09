"""
sanitizepy.cleaning.near_duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`NearDuplicateRemovalOperation` removes near-duplicate records while
retaining one representative per near-duplicate group.

The operation delegates detection to
:class:`~sanitizepy.inspection.near_duplicates.NearDuplicateDetector`, which
groups rows that are equivalent under a normalization/similarity criterion
rather than exact byte equality. For each detected group the operation keeps a
single representative according to the ``keep`` policy (``"first"`` retains the
earliest row in DataFrame order, ``"last"`` retains the latest) and drops the
remaining members.

Because grouping depends on the whole dataset (two equivalent rows may live in
different row-wise chunks), the operation is *not* chunk-safe:
``is_chunk_safe = False``. The resulting
:class:`~sanitizepy.cleaning.base.OperationResult` records the number of removed
records.

References:
    Requirements 10.5, 10.6, 7.2, 13.1
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from ..inspection.near_duplicates import (
    DEFAULT_SIMILARITY_THRESHOLD,
    NearDuplicateDetector,
    NearDuplicateMethod,
)
from .base import CleaningOperation, OperationResult

KeepPolicy = Literal["first", "last"]


class NearDuplicateRemovalOperation(CleaningOperation):
    """
    Remove near-duplicate records, keeping one representative per group.

    Parameters
    ----------
    subset:
        Columns used for near-duplicate comparison. When ``None`` all columns
        are compared.
    method:
        Detection method forwarded to
        :class:`~sanitizepy.inspection.near_duplicates.NearDuplicateDetector`:
        ``"exact_normalized"`` (Core_Stack-only, default) or ``"similarity"``
        (requires the ``sanitizepy[fuzzy]`` extra).
    threshold:
        For ``"similarity"`` mode, the inclusive minimum similarity score
        (0-100) at which two normalized records are treated as duplicates.
    keep:
        ``"first"`` (default) retains the earliest row of each group in
        DataFrame order and drops the rest; ``"last"`` retains the latest.

    Notes
    -----
    * Grouping is dataset-wide, so ``is_chunk_safe = False`` (Req 7.2).
    * The input DataFrame is never mutated; a new DataFrame is returned.
    * Detected groups preserve DataFrame row order, so ``"first"`` and
      ``"last"`` select the group's first and last members respectively.
    """

    name = "near_duplicate_removal"
    is_chunk_safe: bool = False
    is_inplace_safe: bool = False

    def __init__(
        self,
        subset: list[str] | None = None,
        method: NearDuplicateMethod = "exact_normalized",
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        keep: KeepPolicy = "first",
    ) -> None:
        if keep not in ("first", "last"):
            raise ValueError("keep must be one of: 'first', 'last'")

        self.subset: list[str] | None = subset
        self.method: NearDuplicateMethod = method
        self.threshold: float = threshold
        self.keep: KeepPolicy = keep
        self._detector = NearDuplicateDetector()

    def _labels_to_drop(self, dataframe: pd.DataFrame) -> list[object]:
        """
        Return the index labels to drop across all near-duplicate groups.

        For each detected group one representative is retained per the ``keep``
        policy and every remaining member is scheduled for removal. Detected
        groups follow DataFrame row order, so ``"first"`` keeps the earliest
        member and ``"last"`` keeps the latest.
        """
        result = self._detector.detect(
            dataframe,
            subset=self.subset,
            method=self.method,
            threshold=self.threshold,
        )

        labels_to_drop: list[object] = []
        for group in result.groups:
            members = list(group)
            if self.keep == "first":
                labels_to_drop.extend(members[1:])
            else:
                labels_to_drop.extend(members[:-1])

        return labels_to_drop

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        if dataframe.empty:
            return dataframe.copy()

        labels_to_drop = self._labels_to_drop(dataframe)

        if not labels_to_drop:
            return dataframe.copy()

        return dataframe.drop(index=labels_to_drop).copy()

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        before_shape = dataframe.shape
        transformed = self.apply(dataframe)
        after_shape = transformed.shape

        removed = before_shape[0] - after_shape[0]

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(self.subset) if self.subset is not None else [],
            rows_affected=removed,
            columns_affected=0,
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(
                f"{self.name} ({self.method}, keep={self.keep}) removed "
                f"{removed} near-duplicate record(s)"
            ),
            dry_run=dry_run,
            details={**self.describe(), "records_removed": removed},
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subset": list(self.subset) if self.subset is not None else None,
            "method": self.method,
            "threshold": self.threshold,
            "keep": self.keep,
        }
