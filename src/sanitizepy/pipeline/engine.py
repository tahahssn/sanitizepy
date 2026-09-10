"""Execution engine for Cleaner pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from sanitizepy.pipeline.base import PipelineStep


@dataclass(frozen=True, slots=True)
class PipelineStepResult:
    """Execution metadata for a completed pipeline step."""

    name: str
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result produced by a completed pipeline execution."""

    data: pd.DataFrame
    steps: tuple[PipelineStepResult, ...]
    duration_seconds: float

    def __repr__(self) -> str:
        return (
            f"PipelineResult(steps={len(self.steps)}, "
            f"duration={self.duration_seconds:.2f}s)"
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "duration_seconds": self.duration_seconds,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from sanitizepy.ui import (
            COLOR_OK,
            SYMBOL_OK,
            Text,
            render_footer,
            render_header,
            render_table,
        )

        yield render_header("pipeline")
        yield Text("")

        headers = ["Step", "Rows", "Time"]
        rows = []
        for s in self.steps:
            rows.append([s.name, f"{s.output_rows:,}", f"{s.duration_seconds:.2f}s"])
        rows.append(["Total", "", f"{self.duration_seconds:.2f}s"])

        yield render_table(headers, rows)
        yield Text("")

        status_text = Text("  ")
        status_text.append(f"{SYMBOL_OK} ", style=COLOR_OK)
        status_text.append("Pipeline completed")
        yield status_text

        yield Text("")
        rows_cnt = len(self.data)
        cols_cnt = len(self.data.columns)
        yield render_footer(f"{rows_cnt:,} rows × {cols_cnt:,} columns  •  {self.duration_seconds:.2f}s")


class PipelineEngine:
    """Execute an ordered collection of pipeline steps.

    The engine is responsible only for orchestration. Individual operations
    remain implemented by their respective Cleaner engines or transformers.
    """

    def __init__(
        self,
        steps: Iterable[PipelineStep[pd.DataFrame, Any]] | None = None,
    ) -> None:
        self._steps: list[PipelineStep[pd.DataFrame, Any]] = []

        if steps is not None:
            for step in steps:
                self.add_step(step)

    @property
    def steps(self) -> tuple[PipelineStep[pd.DataFrame, Any], ...]:
        """Return the configured pipeline steps."""
        return tuple(self._steps)

    def add_step(self, step: PipelineStep[pd.DataFrame, Any]) -> PipelineEngine:
        """Append a pipeline step.

        Parameters
        ----------
        step:
            Pipeline step to append.

        Returns
        -------
        PipelineEngine
            The current engine instance, allowing fluent configuration.
        """
        if not isinstance(step, PipelineStep):
            raise TypeError("step must be an instance of PipelineStep.")

        self._steps.append(step)
        return self

    def remove_step(self, name: str) -> PipelineEngine:
        """Remove a pipeline step by name.

        Raises
        ------
        ValueError
            If no configured step has the supplied name.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Step name must be a non-empty string.")

        normalized_name = name.strip()

        for index, step in enumerate(self._steps):
            if step.name == normalized_name:
                del self._steps[index]
                return self

        raise ValueError(f"No pipeline step named {normalized_name!r} exists.")

    def clear(self) -> None:
        """Remove all configured pipeline steps."""
        self._steps.clear()

    def run(self, data: pd.DataFrame) -> PipelineResult:
        """Execute all configured steps in order.

        Parameters
        ----------
        data:
            Input dataframe.

        Returns
        -------
        PipelineResult
            Final dataframe and execution metadata.

        Raises
        ------
        TypeError
            If the input is not a pandas DataFrame.
        RuntimeError
            If a pipeline step fails during execution.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Pipeline input must be a pandas DataFrame.")

        current = data
        results: list[PipelineStepResult] = []

        pipeline_started = perf_counter()

        for step in self._steps:
            input_rows, input_columns = current.shape
            step_started = perf_counter()

            try:
                transformed = step.execute(current)
            except Exception as exc:
                raise RuntimeError(f"Pipeline step '{step.name}' failed.") from exc

            if not isinstance(transformed, pd.DataFrame):
                raise TypeError(
                    f"Pipeline step '{step.name}' returned "
                    f"{type(transformed).__name__}; expected DataFrame."
                )

            current = transformed

            output_rows, output_columns = current.shape

            results.append(
                PipelineStepResult(
                    name=step.name,
                    input_rows=input_rows,
                    output_rows=output_rows,
                    input_columns=input_columns,
                    output_columns=output_columns,
                    duration_seconds=perf_counter() - step_started,
                )
            )

        return PipelineResult(
            data=current,
            steps=tuple(results),
            duration_seconds=perf_counter() - pipeline_started,
        )

    def __len__(self) -> int:
        """Return the number of configured steps."""
        return len(self._steps)

    def __iter__(self) -> Iterator[PipelineStep[pd.DataFrame, Any]]:
        """Iterate over configured steps."""
        return iter(self._steps)
