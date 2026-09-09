from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import CleaningOperation, OperationResult


def _json_safe(value: Any) -> Any:
    """
    Return a JSON-serializable representation of *value*.

    The audit log is documented as a list of JSON-serializable dicts
    (Requirement 7.3). Operation ``describe()`` outputs and
    ``OperationResult.details`` may contain numpy scalars, pandas missing
    markers, tuples, sets, or nested containers. This helper converts them
    to plain Python primitives so ``json.dumps`` never fails, while
    preserving structure and values.
    """
    # None passes through directly.
    if value is None:
        return None

    # Preserve native JSON primitives, coercing non-finite floats to None so
    # the result stays strictly JSON-serializable (NaN/Infinity are not).
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value

    # numpy scalars -> native Python scalars.
    if isinstance(value, np.generic):
        return _json_safe(value.item())

    # pandas / numpy missing markers.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Mappings: recurse over values, stringify non-string keys.
    if isinstance(value, Mapping):
        return {
            (key if isinstance(key, str) else str(key)): _json_safe(item)
            for key, item in value.items()
        }

    # Ordered/unordered collections -> lists of sanitized items.
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]

    # Fallback: string representation keeps the entry serializable.
    return str(value)


@dataclass
class CleaningResult:
    """
    Structured outcome of running a sequence of cleaning operations.

    Attributes
    ----------
    data:
        Resulting (or original if dry-run) pandas DataFrame.
    operations:
        Sequence of OperationResult instances detailing each step.
    dry_run:
        Whether the execution was performed in dry-run mode.
    duration_seconds:
        Total execution time in seconds.
    audit_log:
        Ordered list of JSON-serializable execution-log dicts. Each entry
        records, for a single executed operation, its ``order`` (1-based
        execution index), UTC ``timestamp``, ``operation`` name,
        ``parameters`` (the operation's ``describe()`` output), affected
        columns/row counts, before/after shape, ``dry_run`` flag, and
        operation-specific ``details``.
    """

    data: pd.DataFrame
    operations: list[OperationResult] = field(default_factory=list)
    dry_run: bool = False
    duration_seconds: float = 0.0
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """
        Return a human-readable text summary of all executed operations.
        """
        mode = " [DRY RUN]" if self.dry_run else ""
        lines = [
            f"Cleaning Result{mode} - {len(self.operations)} operation(s) "
            f"in {self.duration_seconds:.4f}s:"
        ]
        for idx, op in enumerate(self.operations, start=1):
            lines.append(
                f"  {idx}. {op.operation_name}: affected {op.rows_affected} row(s), "
                f"{op.columns_affected} col(s) ({op.before_shape} -> {op.after_shape})"
            )
        return "\n".join(lines)


def _apply_chunked(
    operation: CleaningOperation,
    dataframe: pd.DataFrame,
    chunk_size: int,
    dry_run: bool,
) -> tuple[pd.DataFrame, OperationResult]:
    """
    Apply a chunk-safe operation to *dataframe* in row-wise chunks and
    reassemble the result.

    The returned :class:`OperationResult` aggregates metrics across all
    chunks so callers see the same information as in whole-dataset execution.
    """
    before_shape = dataframe.shape

    result_chunks: list[pd.DataFrame] = []
    total_rows_affected = 0
    total_cols_affected = 0
    first_affected_cols: list[str] = []
    last_desc: dict[str, Any] = {}

    for start in range(0, len(dataframe), chunk_size):
        chunk = dataframe.iloc[start : start + chunk_size]
        next_chunk, chunk_res = operation.apply_with_result(chunk, dry_run=dry_run)
        result_chunks.append(next_chunk)
        total_rows_affected += chunk_res.rows_affected
        total_cols_affected = max(total_cols_affected, chunk_res.columns_affected)
        if not first_affected_cols:
            first_affected_cols = chunk_res.affected_columns
        last_desc = chunk_res.details

    if result_chunks:
        combined = pd.concat(result_chunks, ignore_index=False)
    else:
        combined = dataframe.copy() if dry_run else dataframe

    after_shape = combined.shape

    aggregated_result = OperationResult(
        operation_name=operation.name,
        affected_columns=first_affected_cols,
        rows_affected=total_rows_affected,
        columns_affected=total_cols_affected,
        before_shape=before_shape,
        after_shape=after_shape,
        strategy_description=(
            f"{operation.name} (chunked, chunk_size={chunk_size}) "
            f"{before_shape} -> {after_shape}"
        ),
        dry_run=dry_run,
        details={**last_desc, "chunked": True, "chunk_size": chunk_size},
    )

    return combined, aggregated_result


class CleaningEngine:
    """
    Execute an ordered sequence of cleaning operations.

    The engine is intentionally orchestration-only. Individual cleaning
    operations are responsible for their own transformation logic.

    Chunked Execution
    -----------------
    When *chunk_size* is supplied to :meth:`run` or :meth:`run_with_result`,
    the engine processes chunk-safe operations (``is_chunk_safe=True``) in
    row-wise chunks of that size.  Operations with ``is_chunk_safe=False``
    always execute over the whole dataset, regardless of *chunk_size*.

    In-Place-Safe Optimization
    --------------------------
    When ``dry_run=False`` the engine skips the initial defensive full copy
    of the caller's DataFrame for operations flagged ``is_inplace_safe=True``.
    The caller's DataFrame is therefore passed directly into the operation
    chain, saving one full copy per run.  When ``dry_run=True`` the engine
    always works on a copy so the caller's DataFrame is never mutated.
    """

    def __init__(
        self,
        operations: Iterable[CleaningOperation] | None = None,
    ) -> None:
        self._operations: list[CleaningOperation] = []

        if operations is not None:
            for operation in operations:
                self.add(operation)

    @property
    def operations(self) -> tuple[CleaningOperation, ...]:
        """Return the configured operations as an immutable sequence."""
        return tuple(self._operations)

    def add(self, operation: CleaningOperation) -> None:
        """
        Add a cleaning operation to the execution sequence.

        Operations execute in the exact order in which they are added.
        """
        if not isinstance(operation, CleaningOperation):
            raise TypeError("operation must be an instance of CleaningOperation")

        self._operations.append(operation)

    def clear(self) -> None:
        """Remove all configured cleaning operations."""
        self._operations.clear()

    def run(
        self,
        dataframe: pd.DataFrame,
        dry_run: bool = False,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Execute all configured cleaning operations and return the resultant DataFrame.

        Parameters
        ----------
        dataframe:
            Input dataframe to clean.
        dry_run:
            If True, evaluates operations without applying permanent data changes.
        chunk_size:
            When set to a positive integer, chunk-safe operations are applied in
            row-wise batches of this size.  Non-chunk-safe operations always run
            over the entire dataset.  When ``None`` (default), all operations run
            over the entire dataset regardless of their ``is_chunk_safe`` flag.

        Returns
        -------
        pandas.DataFrame
            The cleaned dataframe (or original copy if dry_run=True).
        """
        result = self.run_with_result(dataframe, dry_run=dry_run, chunk_size=chunk_size)
        return result.data

    def run_with_result(
        self,
        dataframe: pd.DataFrame,
        dry_run: bool = False,
        chunk_size: int | None = None,
    ) -> CleaningResult:
        """
        Execute all configured cleaning operations and return a detailed
        CleaningResult containing transformed data, operation metrics, and audit log.

        Parameters
        ----------
        dataframe:
            Input dataframe to clean.
        dry_run:
            If True, evaluates operations without applying permanent data changes.
            The caller's DataFrame is guaranteed not to be mutated.
        chunk_size:
            When set to a positive integer, chunk-safe operations are applied in
            row-wise batches of this size.  Non-chunk-safe operations always run
            over the entire dataset.  When ``None`` (default), all operations run
            over the entire dataset regardless of their ``is_chunk_safe`` flag.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        start_time = time.perf_counter()

        # Dry-run: always work on a copy so the caller's DataFrame is
        # never mutated, regardless of operation flags.
        # Non-dry-run: if every queued operation is inplace_safe we can
        # skip the defensive copy entirely and work on the caller's object
        # directly; otherwise copy once up-front.
        if dry_run:
            current_df: pd.DataFrame = dataframe.copy()
        elif self._operations and all(op.is_inplace_safe for op in self._operations):
            current_df = dataframe
        else:
            current_df = dataframe.copy()

        operation_results: list[OperationResult] = []
        audit_log: list[dict[str, Any]] = []

        for order, operation in enumerate(self._operations, start=1):
            use_chunks = (
                chunk_size is not None
                and chunk_size > 0
                and operation.is_chunk_safe
                and len(current_df) > 0
            )

            if use_chunks:
                next_df, op_res = _apply_chunked(
                    operation, current_df, chunk_size, dry_run  # type: ignore[arg-type]
                )
            else:
                next_df, op_res = operation.apply_with_result(
                    current_df, dry_run=dry_run
                )

            operation_results.append(op_res)
            # Each executed operation records exactly one JSON-serializable,
            # UTC-timestamped audit entry containing the operation name, its
            # parameters (via describe()), the execution order, before/after
            # shape, and affected rows/columns (Requirements 7.2, 7.3, 7.5).
            audit_log.append(
                {
                    "order": order,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "operation": op_res.operation_name,
                    "parameters": _json_safe(operation.describe()),
                    "affected_columns": _json_safe(op_res.affected_columns),
                    "rows_affected": op_res.rows_affected,
                    "columns_affected": op_res.columns_affected,
                    "before_shape": list(op_res.before_shape),
                    "after_shape": list(op_res.after_shape),
                    "dry_run": dry_run,
                    "details": _json_safe(op_res.details),
                }
            )
            if not dry_run:
                current_df = next_df

        duration = time.perf_counter() - start_time

        return CleaningResult(
            data=dataframe.copy() if dry_run else current_df,
            operations=operation_results,
            dry_run=dry_run,
            duration_seconds=duration,
            audit_log=audit_log,
        )

    def describe(self) -> list[dict[str, object]]:
        """
        Return descriptions of the configured operations in execution order.
        """
        return [operation.describe() for operation in self._operations]
