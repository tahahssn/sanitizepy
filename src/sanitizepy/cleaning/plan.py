"""
CleaningPlan Module.

Represents an interactive, previewable, and executable sequence of operations
generated from DatasetHealthReport recommendations or manual user configuration.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from sanitizepy.cleaning.base import CleaningOperation
from sanitizepy.cleaning.engine import CleaningEngine, CleaningResult
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingRows,
    FillMissing,
    FillStrategy,
)
from sanitizepy.cleaning.registry import registry as operation_registry
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation
from sanitizepy.exceptions import DeserializationError
from sanitizepy.inspection.health import DatasetHealthReport
from sanitizepy.models.recommendations import Recommendation, RecommendationAction
from sanitizepy.models.replay import ReplayablePlan, ReplayOperation
from sanitizepy.version import VERSION


@dataclass
class PlanStep:
    """
    Individual step within a CleaningPlan.
    """

    index: int
    title: str
    operation: CleaningOperation
    recommendation: Recommendation | None = None
    enabled: bool = True


class CleaningPlan:
    """
    Execution plan containing ordered PlanStep instances.

    Supports interactive preview via `.show()`, enabling/disabling specific steps,
    and executing against a DataFrame via `.apply(df, dry_run=...)`.
    """

    def __init__(self, steps: list[PlanStep] | None = None) -> None:
        self.steps: list[PlanStep] = steps or []

    @classmethod
    def from_report(cls, report: DatasetHealthReport) -> CleaningPlan:
        """
        Construct a CleaningPlan automatically from a DatasetHealthReport's
        recommendations.
        """
        steps: list[PlanStep] = []
        idx = 1

        fill_strategy_map: dict[RecommendationAction, FillStrategy] = {
            RecommendationAction.FILL_MEDIAN: "median",
            RecommendationAction.FILL_MEAN: "mean",
            RecommendationAction.FILL_MODE: "mode",
            RecommendationAction.FILL_CONSTANT: "constant",
        }

        for rec in report.recommendations:
            if not rec.enabled:
                continue

            op: CleaningOperation | None = None
            col_name = rec.column.name if rec.column else None

            if rec.action == RecommendationAction.REMOVE_DUPLICATES:
                op = DropDuplicates(keep="first")
            elif rec.action == RecommendationAction.DROP_COLUMN and col_name:
                op = DropColumns(columns=[col_name])
            elif rec.action in fill_strategy_map and col_name:
                strategy = fill_strategy_map[rec.action]
                op = FillMissing(strategy=strategy, subset=[col_name])
            elif (
                rec.action == RecommendationAction.CONVERT_TYPE
                and col_name
                and rec.target_dtype
            ):
                op = TypeCoercionOperation(target_dtypes={col_name: rec.target_dtype})
            elif rec.action == RecommendationAction.DROP_ROWS and col_name:
                op = DropMissingRows(subset=[col_name])

            if op is not None:
                steps.append(
                    PlanStep(
                        index=idx,
                        title=rec.title,
                        operation=op,
                        recommendation=rec,
                        enabled=rec.enabled,
                    )
                )
                idx += 1

        return cls(steps=steps)

    def enable(self, index: int) -> None:
        """Enable a plan step by index (1-based)."""
        for step in self.steps:
            if step.index == index:
                step.enabled = True
                return
        raise KeyError(f"Plan step with index {index} not found.")

    def disable(self, index: int) -> None:
        """Disable a plan step by index (1-based)."""
        for step in self.steps:
            if step.index == index:
                step.enabled = False
                return
        raise KeyError(f"Plan step with index {index} not found.")

    def show(self) -> None:
        """Display formatted cleaning plan preview in terminal."""
        console = Console()
        if not self.steps:
            console.print("[yellow]CleaningPlan contains no steps.[/yellow]")
            return

        table = Table(
            title="PROPOSED CLEANING PLAN",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="bold", width=4)
        table.add_column("Status", width=10)
        table.add_column("Action / Step", style="white")
        table.add_column("Operation Details", style="dim green")

        for step in self.steps:
            status = (
                "[green]ENABLED[/green]"
                if step.enabled
                else "[dim red]DISABLED[/dim red]"
            )
            table.add_row(
                str(step.index),
                status,
                step.title,
                str(step.operation.describe()),
            )

        console.print(table)

    def apply(self, dataframe: pd.DataFrame, dry_run: bool = False) -> CleaningResult:
        """
        Execute active steps in the cleaning plan against the target DataFrame.
        """
        engine = CleaningEngine()
        for step in self.steps:
            if step.enabled:
                engine.add(step.operation)

        return engine.run_with_result(dataframe, dry_run=dry_run)

    def serialize(self) -> ReplayablePlan:
        """
        Emit a :class:`ReplayablePlan` capturing the enabled steps.

        Each enabled step contributes one
        :class:`~sanitizepy.models.replay.ReplayOperation` built from the
        operation's :meth:`~sanitizepy.cleaning.base.CleaningOperation.describe`
        payload. The redundant ``name`` key is stripped from the recorded
        parameters (it is carried on the ``ReplayOperation`` itself), and any
        derived, non-reconstructable keys (for example ``columns`` on a type
        coercion or ``token_count`` on missing-token normalization) are
        dropped so the parameters map cleanly back onto each operation's
        constructor during :meth:`deserialize`.

        Seeds for operations involving randomness are recorded in the plan's
        ``seeds`` mapping keyed by operation name. The built-in cleaning
        operations are deterministic, so ``seeds`` is typically empty; the
        plumbing exists for randomized operations (for example anomaly-driven
        steps) to populate it.

        Returns
        -------
        ReplayablePlan
            A frozen, JSON-serializable snapshot of the enabled steps.
        """
        operations: list[ReplayOperation] = []
        seeds: dict[str, int] = {}

        for step in self.steps:
            if not step.enabled:
                continue

            operation = step.operation
            parameters = self._reconstruction_parameters(operation)
            operations.append(
                ReplayOperation(name=operation.name, parameters=parameters)
            )

            seed = getattr(operation, "seed", None)
            if isinstance(seed, int) and not isinstance(seed, bool):
                seeds[operation.name] = seed

        return ReplayablePlan(
            version=VERSION,
            operations=tuple(operations),
            configuration={},
            seeds=seeds,
        )

    @staticmethod
    def _reconstruction_parameters(operation: CleaningOperation) -> dict[str, Any]:
        """
        Build the JSON-serializable reconstruction parameters for *operation*.

        The operation's ``describe()`` payload is used as the source of truth,
        with the redundant ``name`` key removed and any keys that are not
        accepted by the operation's ``__init__`` filtered out. Filtering by the
        constructor signature transparently handles operations whose
        ``describe()`` exposes derived fields (such as ``columns`` on
        :class:`TypeCoercionOperation` or ``token_count`` on
        ``MissingTokenOperation``) that are not constructor arguments.
        """
        described = operation.describe()
        accepted = _constructor_parameter_names(type(operation))
        return {
            key: value
            for key, value in described.items()
            if key != "name" and key in accepted
        }

    @classmethod
    def deserialize(cls, plan: ReplayablePlan) -> CleaningPlan:
        """
        Reconstruct a :class:`CleaningPlan` from a :class:`ReplayablePlan`.

        Each recorded operation is turned back into a live
        :class:`~sanitizepy.cleaning.base.CleaningOperation` by looking up its
        class in the module-level
        :class:`~sanitizepy.cleaning.registry.OperationRegistry` and
        instantiating it from the recorded parameters. Recorded seeds are
        forwarded to any operation that accepts a ``seed`` argument so
        randomized operations replay deterministically.

        Parameters
        ----------
        plan:
            The serialized plan produced by :meth:`serialize`.

        Returns
        -------
        CleaningPlan
            A plan whose steps mirror the recorded operations in order.

        Raises
        ------
        DeserializationError
            If an operation name is not registered, or an operation cannot be
            reconstructed from its recorded parameters.
        """
        steps: list[PlanStep] = []

        for index, replay_op in enumerate(plan.operations, start=1):
            try:
                operation_cls = operation_registry.get(replay_op.name)
            except KeyError as exc:
                raise DeserializationError(
                    f"Unknown operation '{replay_op.name}': not registered in the "
                    f"OperationRegistry."
                ) from exc

            parameters = dict(replay_op.parameters)

            seed = plan.seeds.get(replay_op.name)
            if seed is not None and "seed" in _constructor_parameter_names(
                operation_cls
            ):
                parameters.setdefault("seed", seed)

            try:
                operation = operation_cls(**parameters)
            except (TypeError, ValueError, KeyError) as exc:
                raise DeserializationError(
                    f"Failed to reconstruct operation '{replay_op.name}' from "
                    f"parameters {parameters!r}: {exc}"
                ) from exc

            steps.append(
                PlanStep(
                    index=index,
                    title=replay_op.name,
                    operation=operation,
                    recommendation=None,
                    enabled=True,
                )
            )

        return cls(steps=steps)


def _constructor_parameter_names(operation_cls: type[CleaningOperation]) -> set[str]:
    """
    Return the accepted keyword-parameter names of *operation_cls*'s ``__init__``.

    ``self`` is excluded. ``*args``/``**kwargs`` catch-alls are ignored because
    they do not correspond to concrete reconstruction parameters.
    """
    signature = inspect.signature(operation_cls.__init__)
    return {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
