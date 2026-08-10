"""Execution engine for Cleaner pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from cleaner.pipeline.base import PipelineStep


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


class PipelineEngine:
    """Execute an ordered collection of pipeline steps.

    The engine is responsible only for orchestration. Individual operations
    remain implemented by their respective Cleaner engines or transformers.
    """

    def __init__(self, steps: Iterable[PipelineStep] | None = None) -> None:
        self._steps: list[PipelineStep] = []

        if steps is not None:
            for step in steps:
                self.add_step(step)

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        """Return the configured pipeline steps."""
        return tuple(self._steps)

    def add_step(self, step: PipelineStep) -> PipelineEngine:
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
            raise TypeError(
                "step must be an instance of PipelineStep."
            )

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

        raise ValueError(
            f"No pipeline step named {normalized_name!r} exists."
        )

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
                raise RuntimeError(
                    f"Pipeline step '{step.name}' failed."
                ) from exc

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

    def __iter__(self) -> Iterator[PipelineStep]:
        """Iterate over configured steps."""
        return iter(self._steps)