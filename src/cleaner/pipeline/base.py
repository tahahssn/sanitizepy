"""Base abstractions for the Cleaner pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import pandas as pd

DataT = TypeVar("DataT", bound=pd.DataFrame)
ResultT = TypeVar("ResultT")


class PipelineStep(ABC, Generic[DataT, ResultT]):
    """Base contract for a single pipeline execution step.

    A pipeline step receives the current pipeline data and returns the
    transformed data or an execution result.

    Implementations should remain focused on orchestration. Domain-specific
    logic belongs to the corresponding Cleaner engine.
    """

    name: str

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Pipeline step name must be a non-empty string.")

        self.name = name.strip()

    @abstractmethod
    def execute(self, data: DataT) -> ResultT:
        """Execute the step against the supplied data.

        Parameters
        ----------
        data:
            Current pipeline dataset.

        Returns
        -------
        ResultT
            Result produced by this pipeline step.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return an unambiguous representation of the pipeline step."""
        return f"{self.__class__.__name__}(name={self.name!r})"