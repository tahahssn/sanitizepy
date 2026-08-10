"""
cleaner.rules.validators
~~~~~~~~~~~~~~~~~~~~~~~~

Validation utilities for rule definitions.
"""

from __future__ import annotations

from collections.abc import Callable


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
        raise TypeError(
            "Rule name must be a string."
        )

    name = name.strip()

    if not name:
        raise ValueError(
            "Rule name cannot be empty."
        )


def validate_description(description: str) -> None:
    """
    Validate a rule description.

    Parameters
    ----------
    description:
        Human-readable description.
    """
    if not isinstance(description, str):
        raise TypeError(
            "Rule description must be a string."
        )


def validate_priority(priority: int) -> None:
    """
    Validate rule priority.

    Parameters
    ----------
    priority:
        Rule execution priority.
    """
    if not isinstance(priority, int):
        raise TypeError(
            "Rule priority must be an integer."
        )


def validate_enabled(enabled: bool) -> None:
    """
    Validate enabled flag.
    """
    if not isinstance(enabled, bool):
        raise TypeError(
            "Rule enabled flag must be a boolean."
        )


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
        raise TypeError(
            "Rule callback must be callable."
        )


__all__ = [
    "validate_rule_name",
    "validate_description",
    "validate_priority",
    "validate_enabled",
    "validate_callback",
]