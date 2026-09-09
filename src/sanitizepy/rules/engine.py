"""
sanitizepy.rules.engine
~~~~~~~~~~~~~~~~~~~~

Execution engine for Cleaner rules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..engine.base import BaseEngine
from ..models.contracts import DataContract
from .registry import RuleRegistry
from .rule import RuleResult
from .validators import validate_column_contract


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

    def validate_contract(
        self,
        dataframe: pd.DataFrame,
        contract: DataContract,
    ) -> tuple[RuleResult, ...]:
        """
        Validate a DataFrame against a :class:`DataContract`.

        Each declared column contract is evaluated deterministically,
        producing one :class:`RuleResult` per declared expectation. Columns
        are processed in ascending name order so that the aggregated result
        ordering is stable regardless of the contract's insertion order.

        Parameters
        ----------
        dataframe:
            The DataFrame to validate.
        contract:
            The declarative :class:`DataContract` describing per-column
            expectations.

        Returns
        -------
        tuple[RuleResult, ...]
            One ``RuleResult`` per declared expectation, in deterministic
            order.

        Raises
        ------
        TypeError
            If ``dataframe`` is not a pandas DataFrame or ``contract`` is not
            a :class:`DataContract`.
        SchemaValidationError
            If any declared expectation is structurally unusable (for
            example an invalid regex or a ``min_value`` greater than
            ``max_value``). The error is propagated from the underlying
            contract validators.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if not isinstance(contract, DataContract):
            raise TypeError("contract must be a DataContract.")

        results: list[RuleResult] = []
        for column in sorted(contract.columns):
            column_contract = contract.columns[column]
            results.extend(validate_column_contract(dataframe, column, column_contract))

        return tuple(results)


__all__ = [
    "RuleEngine",
]
