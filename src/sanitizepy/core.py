from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from sanitizepy.cleaning.engine import CleaningResult
from sanitizepy.cleaning.plan import CleaningPlan
from sanitizepy.config import DEFAULT_CONFIG, CleanerConfig
from sanitizepy.inspection.detector import IssueDetector
from sanitizepy.inspection.health import DatasetHealthReport
from sanitizepy.logger import get_logger

if TYPE_CHECKING:
    from sanitizepy.models.contracts import DataContract
    from sanitizepy.models.profile import DatasetProfile
    from sanitizepy.rules.rule import RuleResult


class Cleaner:
    """
    Main entry point of the sanitizepy library.

    Provides a high-level, cohesive API for dataset health inspection,
    explainable plan generation, and safe execution with dry-run support.
    """

    def __init__(
        self,
        config: CleanerConfig | None = None,
    ) -> None:
        self._config: CleanerConfig = config or DEFAULT_CONFIG
        self._logger = get_logger(self.__class__.__name__)
        self._detector = IssueDetector()

    @property
    def config(self) -> CleanerConfig:
        """
        Return the active configuration.
        """
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """
        Return the package logger.
        """
        return self._logger

    def inspect(self, dataframe: pd.DataFrame) -> DatasetHealthReport:
        """
        Inspect dataset quality and return a DatasetHealthReport containing
        health score, identified issues, and explainable recommendations.
        """
        self._logger.info("Executing dataset health inspection.")
        return self._detector.inspect(dataframe)

    def plan(self, target: pd.DataFrame | DatasetHealthReport) -> CleaningPlan:
        """
        Generate a previewable CleaningPlan from a DatasetHealthReport or DataFrame.
        """
        self._logger.info("Generating cleaning plan.")
        report = (
            target if isinstance(target, DatasetHealthReport) else self.inspect(target)
        )
        return CleaningPlan.from_report(report)

    def clean(
        self,
        dataframe: pd.DataFrame,
        plan: CleaningPlan | None = None,
        dry_run: bool = False,
    ) -> CleaningResult:
        """
        Apply a CleaningPlan or automatically generated plan to a DataFrame.
        """
        self._logger.info(f"Executing data cleaning (dry_run={dry_run}).")
        active_plan = plan or self.plan(dataframe)
        return active_plan.apply(dataframe, dry_run=dry_run)

    def profile(self, dataframe: pd.DataFrame) -> DatasetProfile:
        """
        Build an immutable dataset profile by aggregating the standalone
        inspectors (datatypes, missing values, duplicates, memory, statistics).

        This is an additive convenience facade over
        :class:`~sanitizepy.inspection.profile.DatasetProfiler`; it never
        mutates the input DataFrame.
        """
        from sanitizepy.inspection.profile import DatasetProfiler

        self._logger.info("Building dataset profile.")
        return DatasetProfiler().profile(dataframe)

    def validate(
        self,
        dataframe: pd.DataFrame,
        contract: DataContract,
    ) -> tuple[RuleResult, ...]:
        """
        Validate a DataFrame against a declarative
        :class:`~sanitizepy.models.contracts.DataContract`, producing one
        ``RuleResult`` per declared expectation.

        This is an additive convenience facade over
        :meth:`~sanitizepy.rules.engine.RuleEngine.validate_contract`.
        """
        from sanitizepy.rules.engine import RuleEngine

        self._logger.info("Validating DataFrame against data contract.")
        return RuleEngine().validate_contract(dataframe, contract)


# Module-level convenience functions
_default_sanitizepy = Cleaner()


def inspect(dataframe: pd.DataFrame) -> DatasetHealthReport:
    """Convenience wrapper for Cleaner().inspect(df)."""
    return _default_sanitizepy.inspect(dataframe)


def plan(target: pd.DataFrame | DatasetHealthReport) -> CleaningPlan:
    """Convenience wrapper for Cleaner().plan(target)."""
    return _default_sanitizepy.plan(target)


def clean(
    dataframe: pd.DataFrame,
    cleaning_plan: CleaningPlan | None = None,
    dry_run: bool = False,
) -> CleaningResult:
    """Convenience wrapper for Cleaner().clean(df, plan, dry_run)."""
    return _default_sanitizepy.clean(dataframe, plan=cleaning_plan, dry_run=dry_run)


__all__ = [
    "Cleaner",
    "inspect",
    "plan",
    "clean",
]
