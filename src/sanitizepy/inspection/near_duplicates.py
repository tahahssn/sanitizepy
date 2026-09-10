"""
Deterministic near-duplicate detection.

This module provides read-only, deterministic detection of records that are
equivalent under a normalization/similarity criterion rather than exact byte
equality. It never mutates the input DataFrame.

Two modes are provided:

- ``exact_normalized`` (default): the Core_Stack-only mode. Selected columns
  are normalized (string coercion, lowercase, whitespace strip/collapse) and
  joined into a single normalized key per row. Rows sharing an identical
  normalized key form a near-duplicate group. Grouping is O(n) via a
  normalized-key dictionary, avoiding any unrestricted O(n^2) comparison.
- ``similarity``: the optional mode. Rows are first bucketed by a coarse
  normalized key, then compared pairwise *within* each bucket using
  ``rapidfuzz`` so that near-identical (but not exactly normalized-equal)
  records are grouped. ``rapidfuzz`` is imported lazily; when it is not
  installed a :class:`~sanitizepy.exceptions.DependencyError` naming
  ``sanitizepy[fuzzy]`` is raised. The exact-normalized mode remains available
  without the optional extra.

Both modes are fully deterministic: repeated detection over the same data with
the same configuration yields identical duplicate groups.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final, Literal

import pandas as pd

NearDuplicateMethod = Literal["exact_normalized", "similarity"]

DEFAULT_SIMILARITY_THRESHOLD: Final = 90.0

_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class NearDuplicateResult:
    """
    Immutable near-duplicate detection result.

    Never references the caller's DataFrame.

    Attributes
    ----------
    columns:
        The columns compared, in order. Empty means the full row was used.
    method:
        The detection method used (``"exact_normalized"`` or ``"similarity"``).
    groups:
        Duplicate groups. Each group is a tuple of the row index labels that
        were judged equivalent. Only groups with two or more members are
        included, and groups are ordered by first appearance for determinism.
    duplicate_count:
        The number of redundant records: the total group membership minus one
        retained representative per group.
    """

    columns: tuple[str, ...]

    method: NearDuplicateMethod

    groups: tuple[tuple[object, ...], ...]

    duplicate_count: int

    def __repr__(self) -> str:
        return (
            f"NearDuplicateResult(groups={len(self.groups)}, "
            f"duplicates={self.duplicate_count})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "method": self.method,
            "groups": [list(g) for g in self.groups],
            "duplicate_count": self.duplicate_count,
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from sanitizepy.ui import (
            COLOR_META,
            SYMBOL_OK,
            SYMBOL_WARN,
            Text,
            render_footer,
            render_header,
            render_metric,
            render_status_row,
        )

        yield render_header("near duplicates")
        yield Text("")

        yield Text("  Columns checked", style=f"bold {COLOR_META}")
        cols_str = ", ".join(self.columns) if self.columns else "all columns"
        yield Text(f"    {cols_str}")
        yield Text("")

        rows_involved = sum(len(g) for g in self.groups)
        method_str = "normalized matching" if self.method == "exact_normalized" else "similarity matching"

        yield render_metric("Groups detected", f"{len(self.groups):,}")
        yield render_metric("Rows involved", f"{rows_involved:,}")
        yield render_metric("Method", method_str)
        yield Text("")

        if self.groups:
            yield render_status_row(
                SYMBOL_WARN,
                f"{len(self.groups):,} near-duplicate groups detected",
            )
        else:
            yield render_status_row(SYMBOL_OK, "No near-duplicates detected")

        yield Text("")
        yield render_footer(f"{len(self.columns)} columns checked")


def _lazy_import_rapidfuzz() -> Any:
    """
    Lazily import *rapidfuzz* and raise :class:`DependencyError` when absent.

    Returns
    -------
    module
        The ``rapidfuzz.fuzz`` module.

    Raises
    ------
    DependencyError
        When *rapidfuzz* is not installed, naming the extra ``sanitizepy[fuzzy]``.
    """

    # Imported here rather than at exception time to keep the failure message
    # local to the capability that requires the optional extra.
    from ..exceptions import DependencyError

    try:
        from rapidfuzz import fuzz  # noqa: PLC0415

        return fuzz
    except ImportError as exc:
        raise DependencyError(
            "The 'rapidfuzz' package is required for similarity-based "
            "near-duplicate detection. "
            "Install it via: pip install 'sanitizepy[fuzzy]'"
        ) from exc


def _normalize_column(series: pd.Series) -> pd.Series:
    """
    Normalize a column into deterministic comparable strings.

    Missing values normalize to the empty string. All other values are coerced
    to ``str``, lowercased, stripped, and have internal whitespace runs
    collapsed to a single space. Fully vectorized for determinism and speed.
    """

    missing = series.isna()

    text = series.astype(str).str.strip().str.lower()
    text = text.str.replace(_WHITESPACE_RUN, " ", regex=True)

    return text.mask(missing, "")


class NearDuplicateDetector:
    """
    Deterministic near-duplicate detection.

    Read-only. Never mutates the dataframe.
    """

    __slots__: Final = ()

    def detect(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: list[str] | tuple[str, ...] | None = None,
        method: NearDuplicateMethod = "exact_normalized",
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> NearDuplicateResult:
        """
        Detect near-duplicate records.

        Parameters
        ----------
        dataframe:
            Input dataframe (never mutated).
        subset:
            Columns used for comparison. When ``None`` all columns are used.
        method:
            ``"exact_normalized"`` (Core_Stack-only, default) or
            ``"similarity"`` (requires the ``sanitizepy[fuzzy]`` extra).
        threshold:
            For ``"similarity"`` mode, the inclusive minimum similarity score
            (0-100) at which two normalized records are treated as duplicates.

        Returns
        -------
        NearDuplicateResult

        Raises
        ------
        ValueError
            When ``dataframe`` is empty, or a requested subset column is absent.
        DependencyError
            When ``method="similarity"`` and *rapidfuzz* is not installed.
        """

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        columns = self._resolve_columns(dataframe, subset)

        keys = self._normalized_keys(dataframe, columns)

        if method == "exact_normalized":
            groups = self._exact_groups(keys)
        else:
            groups = self._similarity_groups(keys, threshold)

        duplicate_count = sum(len(group) - 1 for group in groups)

        return NearDuplicateResult(
            columns=columns,
            method=method,
            groups=groups,
            duplicate_count=duplicate_count,
        )

    def _resolve_columns(
        self,
        dataframe: pd.DataFrame,
        subset: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """
        Resolve and validate the comparison columns, preserving order.
        """

        if subset is None:
            return tuple(str(column) for column in dataframe.columns)

        missing = [column for column in subset if column not in dataframe.columns]

        if missing:
            raise ValueError(f"Subset columns not found in DataFrame: {missing!r}")

        return tuple(str(column) for column in subset)

    def _normalized_keys(
        self,
        dataframe: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> list[tuple[object, str]]:
        """
        Build a deterministic ``(index_label, normalized_key)`` list.

        The normalized key joins each column's normalized value with a unit
        separator so that values cannot collide across column boundaries.
        Row order follows the DataFrame's index order.
        """

        normalized = pd.DataFrame(
            {column: _normalize_column(dataframe[column]) for column in columns},
            index=dataframe.index,
        )

        # Join per-column normalized values with a unit separator so values
        # cannot collide across column boundaries.
        joined = normalized.apply(lambda row: "\x1f".join(row), axis=1)

        return [
            (index_label, str(key))
            for index_label, key in zip(joined.index, joined.tolist(), strict=True)
        ]

    def _exact_groups(
        self,
        keys: list[tuple[object, str]],
    ) -> tuple[tuple[object, ...], ...]:
        """
        Group rows sharing an identical normalized key.

        Uses an insertion-ordered dictionary so groups are ordered by first
        appearance. Runs in O(n) over the number of rows.
        """

        buckets: OrderedDict[str, list[object]] = OrderedDict()

        for index_label, key in keys:
            buckets.setdefault(key, []).append(index_label)

        return tuple(tuple(members) for members in buckets.values() if len(members) > 1)

    def _similarity_groups(
        self,
        keys: list[tuple[object, str]],
        threshold: float,
    ) -> tuple[tuple[object, ...], ...]:
        """
        Group rows by fuzzy similarity within coarse normalized buckets.

        Rows are first bucketed by their exact normalized key; within each
        bucket, remaining ungrouped rows are compared pairwise with
        ``rapidfuzz`` so that near-identical records collapse together. The
        coarse bucketing keeps the pairwise comparison local rather than
        performing an unrestricted O(n^2) scan over the whole dataset.
        """

        fuzz = _lazy_import_rapidfuzz()

        buckets: OrderedDict[str, list[tuple[object, str]]] = OrderedDict()

        for index_label, key in keys:
            buckets.setdefault(key, []).append((index_label, key))

        groups: list[tuple[object, ...]] = []

        for members in buckets.values():
            groups.extend(self._cluster_bucket(members, fuzz, threshold))

        return tuple(groups)

    def _cluster_bucket(
        self,
        members: list[tuple[object, str]],
        fuzz: Any,
        threshold: float,
    ) -> list[tuple[object, ...]]:
        """
        Cluster a single bucket into similarity groups.

        Deterministic: members are considered in their original row order and
        each unassigned member seeds a new group that absorbs later members
        whose similarity score meets the threshold.
        """

        assigned: set[int] = set()
        clusters: list[tuple[object, ...]] = []

        for i, (index_label, key) in enumerate(members):
            if i in assigned:
                continue

            cluster: list[object] = [index_label]
            assigned.add(i)

            for j in range(i + 1, len(members)):
                if j in assigned:
                    continue

                other_label, other_key = members[j]
                score = float(fuzz.ratio(key, other_key))

                if score >= threshold:
                    cluster.append(other_label)
                    assigned.add(j)

            if len(cluster) > 1:
                clusters.append(tuple(cluster))

        return clusters
