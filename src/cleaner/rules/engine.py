"""
cleaner.rules.engine
~~~~~~~~~~~~~~~~~~~~

Execution engine for Cleaner rules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..engine.base import BaseEngine
from .registry import RuleRegistry
from .rule import RuleResult


class RuleEngine(BaseEngine):
    """
    Engine responsible for executing registered rules against a DataFrame.
    """

    def __init__(
        self,
        registry: RuleRegistry | None = None,
        config: Any = None,
    ) -> None:
        super().__init__(config=config)
        self._registry = registry or RuleRegistry()

    @property
    def registry(self) -> RuleRegistry:
        """Return the active rule registry."""
        return self._registry

    def run(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> tuple[RuleResult, ...]:
        """
        Execute all registered rules against the DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")

        results: list[RuleResult] = []
        for rule in self._registry:
            result = rule.evaluate(dataframe, **kwargs)
            results.append(result)

        return tuple(results)


__all__ = [
    "RuleEngine",
]
