"""
cleaner.rules.base
~~~~~~~~~~~~~~~~~~

Abstract base class for all validation rules.

Every rule in Cleaner must inherit from ``BaseRule`` and implement
the ``evaluate`` method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..logger import get_logger
from .rule import RuleCategory, RuleResult, RuleSeverity


class BaseRule(ABC):
    """
    Base class for every validation rule.

    A rule inspects a DataFrame and returns a RuleResult
    describing whether the rule passed or failed.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
    ) -> None:
        self._name = name
        self._description = description
        self._logger = get_logger(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Human-readable rule name."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable rule description."""
        return self._description

    @property
    def logger(self) -> logging.Logger:
        """Rule logger."""
        return self._logger

    @abstractmethod
    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        """
        Execute the rule against a DataFrame.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" f"(name={self.name!r})"


class Rule(BaseRule):
    """
    Standard concrete rule implementation.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        return RuleResult(
            rule=self.name,
            passed=True,
            severity=RuleSeverity.INFO,
            category=RuleCategory.DATA_QUALITY,
            message=self.description,
        )


__all__ = [
    "BaseRule",
    "Rule",
]
