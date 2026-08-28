from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .base import CleaningOperation, OperationResult


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
        Immutable list of json-serializable execution log dicts.
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


class CleaningEngine:
    """
    Execute an ordered sequence of cleaning operations.

    The engine is intentionally orchestration-only. Individual cleaning
    operations are responsible for their own transformation logic.
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

    def run(self, dataframe: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
        """
        Execute all configured cleaning operations and return the resultant DataFrame.

        Parameters
        ----------
        dataframe:
            Input dataframe to clean.
        dry_run:
            If True, evaluates operations without applying permanent data changes.

        Returns
        -------
        pandas.DataFrame
            The cleaned dataframe (or original copy if dry_run=True).
        """
        result = self.run_with_result(dataframe, dry_run=dry_run)
        return result.data

    def run_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> CleaningResult:
        """
        Execute all configured cleaning operations and return a detailed
        CleaningResult containing transformed data, operation metrics, and audit log.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        start_time = time.perf_counter()
        current_df = dataframe.copy()
        operation_results: list[OperationResult] = []
        audit_log: list[dict[str, Any]] = []

        for operation in self._operations:
            next_df, op_res = operation.apply_with_result(current_df, dry_run=dry_run)
            operation_results.append(op_res)
            audit_log.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "operation": op_res.operation_name,
                    "affected_columns": op_res.affected_columns,
                    "rows_affected": op_res.rows_affected,
                    "columns_affected": op_res.columns_affected,
                    "before_shape": list(op_res.before_shape),
                    "after_shape": list(op_res.after_shape),
                    "dry_run": dry_run,
                    "details": op_res.details,
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
