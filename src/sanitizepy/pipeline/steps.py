"""Pipeline adapters for Cleaner engines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from sanitizepy.pipeline.base import PipelineStep


class CallableStep(PipelineStep[pd.DataFrame, pd.DataFrame]):
    """Pipeline step backed by a callable.

    This provides a small integration boundary for existing or external
    engine implementations without coupling the pipeline to their internal
    implementation details.
    """

    def __init__(
        self,
        name: str,
        operation: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        super().__init__(name)

        if not callable(operation):
            raise TypeError("operation must be callable.")

        self._operation = operation

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the wrapped operation."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Pipeline steps require a pandas DataFrame.")

        result = self._operation(data)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Pipeline step '{self.name}' must return a pandas DataFrame, "
                f"got {type(result).__name__}."
            )

        return result


class TransformStep(PipelineStep[pd.DataFrame, pd.DataFrame]):
    """Pipeline step for a dataframe transformation object.

    The supplied transformer must expose a callable ``transform`` method.
    This allows existing preprocessing, cleaning, or feature-engineering
    components to participate in a pipeline without duplicating their logic.
    """

    def __init__(self, name: str, transformer: Any) -> None:
        super().__init__(name)

        transform = getattr(transformer, "transform", None)

        if not callable(transform):
            raise TypeError(
                f"Transformer for pipeline step '{self.name}' must expose "
                "a callable 'transform' method."
            )

        self._transformer = transformer

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the current dataframe."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Pipeline steps require a pandas DataFrame.")

        result = self._transformer.transform(data)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Pipeline step '{self.name}' must return a pandas DataFrame, "
                f"got {type(result).__name__}."
            )

        return result
