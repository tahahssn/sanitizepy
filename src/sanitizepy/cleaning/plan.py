"""
CleaningPlan Module.

Represents an interactive, previewable, and executable sequence of operations
generated from DatasetHealthReport recommendations or manual user configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

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
)
from sanitizepy.inspection.health import DatasetHealthReport
from sanitizepy.models.recommendations import Recommendation, RecommendationAction


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

        for rec in report.recommendations:
            if not rec.enabled:
                continue

            op: CleaningOperation | None = None
            col_name = rec.column.name if rec.column else None

            if rec.action == RecommendationAction.REMOVE_DUPLICATES:
                op = DropDuplicates(keep="first")
            elif rec.action == RecommendationAction.DROP_COLUMN and col_name:
                op = DropColumns(columns=[col_name])
            elif (
                rec.action
                in (
                    RecommendationAction.FILL_MEDIAN,
                    RecommendationAction.FILL_MEAN,
                    RecommendationAction.FILL_CONSTANT,
                    RecommendationAction.FILL_MODE,
                )
                and col_name
            ):
                op = FillMissing(value=0.0, subset=[col_name])
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
