"""
sanitizepy.rules.validators
~~~~~~~~~~~~~~~~~~~~~~~~

Validation utilities for rule definitions and data contracts.

This module hosts two related concerns:

1. Small helpers that validate rule *definitions* (name, description,
   priority, enabled flag, callback). These are unchanged historical
   utilities.

2. Deterministic *data-contract* validators. Each validator inspects a
   single column against a :class:`~sanitizepy.models.contracts.ColumnContract`
   expectation and produces a :class:`~sanitizepy.rules.rule.RuleResult`
   describing whether the expectation holds.

Contract validators follow two firm rules:

- An ordinary validation failure (including a declared column being
  absent) yields a *failing* ``RuleResult`` -- never an exception.
- A malformed or structurally unusable contract (for example an invalid
  regex pattern or ``min_value`` greater than ``max_value``) raises
  :class:`~sanitizepy.exceptions.SchemaValidationError`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd
from pandas.api import types as pdtypes

from ..exceptions import SchemaValidationError
from ..models.contracts import ColumnContract
from .rule import RuleCategory, RuleResult, RuleSeverity

# Maximum number of example violating values embedded in a RuleResult message.
_MAX_EXAMPLES = 5


# ==========================================================
# Rule-definition validators (historical utilities)
# ==========================================================


def validate_rule_name(name: str) -> None:
    """
    Validate a rule name.

    Parameters
    ----------
    name:
        Rule name.

    Raises
    ------
    TypeError
        If name is not a string.

    ValueError
        If the name is invalid.
    """
    if not isinstance(name, str):
        raise TypeError("Rule name must be a string.")

    name = name.strip()

    if not name:
        raise ValueError("Rule name cannot be empty.")


def validate_description(description: str) -> None:
    """
    Validate a rule description.

    Parameters
    ----------
    description:
        Human-readable description.
    """
    if not isinstance(description, str):
        raise TypeError("Rule description must be a string.")


def validate_priority(priority: int) -> None:
    """
    Validate rule priority.

    Parameters
    ----------
    priority:
        Rule execution priority.
    """
    if not isinstance(priority, int):
        raise TypeError("Rule priority must be an integer.")


def validate_enabled(enabled: bool) -> None:
    """
    Validate enabled flag.
    """
    if not isinstance(enabled, bool):
        raise TypeError("Rule enabled flag must be a boolean.")


def validate_callback(
    callback: Callable[..., object],
) -> None:
    """
    Validate rule callback.

    Parameters
    ----------
    callback:
        Callable executed by the rule.
    """
    if not callable(callback):
        raise TypeError("Rule callback must be callable.")


# ==========================================================
# Contract validation helpers
# ==========================================================


def _format_examples(values: list[Any]) -> str:
    """
    Render a short, deterministic preview of violating values.
    """
    preview = values[:_MAX_EXAMPLES]
    rendered = ", ".join(repr(value) for value in preview)
    if len(values) > _MAX_EXAMPLES:
        rendered = f"{rendered}, ..."
    return rendered


def _result(
    *,
    rule: str,
    passed: bool,
    category: RuleCategory,
    message: str,
    column: str,
    affected_rows: int = 0,
    metadata: dict[str, Any] | None = None,
) -> RuleResult:
    """
    Build a ``RuleResult`` with consistent severity and column wiring.

    A passing expectation is informational; a failing expectation is an
    error. Affected columns always reference the validated column.
    """
    return RuleResult(
        rule=rule,
        passed=passed,
        severity=RuleSeverity.INFO if passed else RuleSeverity.ERROR,
        category=category,
        message=message,
        affected_columns=(column,),
        affected_rows=affected_rows,
        metadata=metadata or {},
    )


def _non_missing_mask(series: pd.Series) -> pd.Series:
    """
    Boolean mask selecting non-missing entries of a series.
    """
    return series.notna()


# ==========================================================
# Individual contract validators
# ==========================================================


def validate_column_exists(
    dataframe: pd.DataFrame,
    column: str,
) -> RuleResult:
    """
    Validate that a declared column is present in the DataFrame.

    A missing column is an ordinary validation failure and yields a
    failing ``RuleResult`` rather than raising.
    """
    exists = column in dataframe.columns
    if exists:
        message = f"Column '{column}' exists."
    else:
        message = f"Column '{column}' is missing from the DataFrame."
    return _result(
        rule=f"contract.column_exists[{column}]",
        passed=exists,
        category=RuleCategory.STRUCTURE,
        message=message,
        column=column,
    )


def validate_dtype(
    dataframe: pd.DataFrame,
    column: str,
    expected_dtype: str,
) -> RuleResult:
    """
    Validate that a column's dtype matches the expected dtype string.
    """
    actual_dtype = str(dataframe[column].dtype)
    passed = actual_dtype == expected_dtype
    if passed:
        message = f"Column '{column}' has expected dtype '{expected_dtype}'."
    else:
        message = (
            f"Column '{column}' expected dtype '{expected_dtype}' "
            f"but found '{actual_dtype}'."
        )
    return _result(
        rule=f"contract.dtype[{column}]",
        passed=passed,
        category=RuleCategory.VALIDITY,
        message=message,
        column=column,
        metadata={"expected_dtype": expected_dtype, "actual_dtype": actual_dtype},
    )


def validate_nullable(
    dataframe: pd.DataFrame,
    column: str,
    nullable: bool,
) -> RuleResult:
    """
    Validate a column's nullability expectation.

    When ``nullable`` is ``False`` any missing value is a violation. When
    ``nullable`` is ``True`` the expectation always holds.
    """
    series = dataframe[column]
    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())

    if nullable or missing_count == 0:
        message = f"Column '{column}' satisfies nullability expectation."
        return _result(
            rule=f"contract.nullable[{column}]",
            passed=True,
            category=RuleCategory.COMPLETENESS,
            message=message,
            column=column,
        )

    message = (
        f"Column '{column}' must not contain missing values "
        f"but has {missing_count}."
    )
    return _result(
        rule=f"contract.nullable[{column}]",
        passed=False,
        category=RuleCategory.COMPLETENESS,
        message=message,
        column=column,
        affected_rows=missing_count,
        metadata={"missing_count": missing_count},
    )


def validate_allowed_values(
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: tuple[Any, ...],
) -> RuleResult:
    """
    Validate that every non-missing value is within the allowed set.
    """
    series = dataframe[column]
    non_missing = series[_non_missing_mask(series)]
    allowed_set = set(allowed_values)
    violating_mask = ~non_missing.isin(allowed_set)
    violating = non_missing[violating_mask]
    count = int(violating.shape[0])

    if count == 0:
        message = f"Column '{column}' contains only allowed values."
        return _result(
            rule=f"contract.allowed_values[{column}]",
            passed=True,
            category=RuleCategory.VALIDITY,
            message=message,
            column=column,
        )

    examples = _format_examples(list(dict.fromkeys(violating.tolist())))
    message = (
        f"Column '{column}' has {count} value(s) outside the allowed set "
        f"(e.g. {examples})."
    )
    return _result(
        rule=f"contract.allowed_values[{column}]",
        passed=False,
        category=RuleCategory.VALIDITY,
        message=message,
        column=column,
        affected_rows=count,
        metadata={"violating_count": count},
    )


def validate_range(
    dataframe: pd.DataFrame,
    column: str,
    min_value: float | None,
    max_value: float | None,
) -> RuleResult:
    """
    Validate that numeric values fall within an inclusive ``[min, max]``.

    Either bound may be ``None`` to leave that side unbounded.

    Raises
    ------
    SchemaValidationError
        If the column is not numeric, or if ``min_value`` exceeds
        ``max_value`` (a structurally unusable contract).
    """
    if min_value is not None and max_value is not None and min_value > max_value:
        raise SchemaValidationError(
            f"Invalid range contract for column '{column}': "
            f"min_value ({min_value}) is greater than max_value ({max_value})."
        )

    series = dataframe[column]
    if not pdtypes.is_numeric_dtype(series):
        raise SchemaValidationError(
            f"Range contract for column '{column}' requires a numeric dtype "
            f"but found '{series.dtype}'."
        )

    non_missing = series[_non_missing_mask(series)]
    below = (
        non_missing < min_value
        if min_value is not None
        else pd.Series(False, index=non_missing.index)
    )
    above = (
        non_missing > max_value
        if max_value is not None
        else pd.Series(False, index=non_missing.index)
    )
    violating_mask = below | above
    violating = non_missing[violating_mask]
    count = int(violating.shape[0])

    bounds = f"[{min_value}, {max_value}]"
    if count == 0:
        message = f"Column '{column}' values are within range {bounds}."
        return _result(
            rule=f"contract.range[{column}]",
            passed=True,
            category=RuleCategory.VALIDITY,
            message=message,
            column=column,
        )

    examples = _format_examples(violating.tolist())
    message = (
        f"Column '{column}' has {count} value(s) outside range {bounds} "
        f"(e.g. {examples})."
    )
    return _result(
        rule=f"contract.range[{column}]",
        passed=False,
        category=RuleCategory.VALIDITY,
        message=message,
        column=column,
        affected_rows=count,
        metadata={
            "violating_count": count,
            "min_value": min_value,
            "max_value": max_value,
        },
    )


def validate_regex(
    dataframe: pd.DataFrame,
    column: str,
    pattern: str,
) -> RuleResult:
    """
    Validate that every non-missing value fully matches ``pattern``.

    Raises
    ------
    SchemaValidationError
        If ``pattern`` is not a valid regular expression (a malformed
        contract).
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise SchemaValidationError(
            f"Invalid regex contract for column '{column}': {exc}."
        ) from exc

    series = dataframe[column]
    non_missing = series[_non_missing_mask(series)]

    def _matches(value: Any) -> bool:
        return compiled.fullmatch(str(value)) is not None

    match_mask = non_missing.map(_matches)
    violating = non_missing[~match_mask.astype(bool)]
    count = int(violating.shape[0])

    if count == 0:
        message = f"Column '{column}' values match pattern '{pattern}'."
        return _result(
            rule=f"contract.regex[{column}]",
            passed=True,
            category=RuleCategory.VALIDITY,
            message=message,
            column=column,
        )

    examples = _format_examples(violating.tolist())
    message = (
        f"Column '{column}' has {count} value(s) not matching pattern "
        f"'{pattern}' (e.g. {examples})."
    )
    return _result(
        rule=f"contract.regex[{column}]",
        passed=False,
        category=RuleCategory.VALIDITY,
        message=message,
        column=column,
        affected_rows=count,
        metadata={"violating_count": count, "pattern": pattern},
    )


def validate_unique(
    dataframe: pd.DataFrame,
    column: str,
) -> RuleResult:
    """
    Validate that a column contains only unique (non-missing) values.
    """
    series = dataframe[column]
    non_missing = series[_non_missing_mask(series)]
    duplicated_mask = non_missing.duplicated(keep=False)
    duplicated = non_missing[duplicated_mask]
    duplicate_count = int(duplicated.shape[0])

    if duplicate_count == 0:
        message = f"Column '{column}' contains only unique values."
        return _result(
            rule=f"contract.unique[{column}]",
            passed=True,
            category=RuleCategory.UNIQUENESS,
            message=message,
            column=column,
        )

    examples = _format_examples(list(dict.fromkeys(duplicated.tolist())))
    message = (
        f"Column '{column}' has {duplicate_count} duplicated value(s) "
        f"(e.g. {examples})."
    )
    return _result(
        rule=f"contract.unique[{column}]",
        passed=False,
        category=RuleCategory.UNIQUENESS,
        message=message,
        column=column,
        affected_rows=duplicate_count,
        metadata={"duplicate_count": duplicate_count},
    )


def validate_column_contract(
    dataframe: pd.DataFrame,
    column: str,
    contract: ColumnContract,
) -> list[RuleResult]:
    """
    Validate a single column against every declared expectation.

    Column existence is always checked first. When the column is absent,
    a single failing ``RuleResult`` is returned and no further
    expectations are evaluated (there is nothing to inspect).

    Otherwise one ``RuleResult`` is produced per declared (non-``None``)
    expectation, in a fixed, deterministic order.

    Raises
    ------
    SchemaValidationError
        If a declared expectation is structurally unusable (invalid regex
        or ``min_value`` greater than ``max_value``).
    """
    existence = validate_column_exists(dataframe, column)
    if not existence.passed:
        return [existence]

    results: list[RuleResult] = [existence]

    if contract.dtype is not None:
        results.append(validate_dtype(dataframe, column, contract.dtype))

    if contract.nullable is not None:
        results.append(validate_nullable(dataframe, column, contract.nullable))

    if contract.allowed_values is not None:
        results.append(
            validate_allowed_values(dataframe, column, contract.allowed_values)
        )

    if contract.min_value is not None or contract.max_value is not None:
        results.append(
            validate_range(
                dataframe,
                column,
                contract.min_value,
                contract.max_value,
            )
        )

    if contract.regex is not None:
        results.append(validate_regex(dataframe, column, contract.regex))

    if contract.unique is not None and contract.unique:
        results.append(validate_unique(dataframe, column))

    return results


__all__ = [
    "validate_rule_name",
    "validate_description",
    "validate_priority",
    "validate_enabled",
    "validate_callback",
    "validate_column_exists",
    "validate_dtype",
    "validate_nullable",
    "validate_allowed_values",
    "validate_range",
    "validate_regex",
    "validate_unique",
    "validate_column_contract",
]
