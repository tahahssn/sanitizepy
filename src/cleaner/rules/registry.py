"""
cleaner.rules.registry
~~~~~~~~~~~~~~~~~~~~~~

Registry responsible for managing validation rules.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import BaseRule


class RuleRegistry:
    """
    Registry of validation rules.

    Rules are uniquely identified by their name.
    """

    def __init__(self) -> None:
        self._rules: dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """
        Register a rule.
        """
        if rule.name in self._rules:
            raise ValueError(
                f"Rule '{rule.name}' is already registered."
            )

        self._rules[rule.name] = rule

    def unregister(self, name: str) -> None:
        """
        Remove a rule.
        """
        del self._rules[name]

    def get(self, name: str) -> BaseRule:
        """
        Return a registered rule.
        """
        return self._rules[name]

    def contains(self, name: str) -> bool:
        """
        Check whether a rule exists.
        """
        return name in self._rules

    def clear(self) -> None:
        """
        Remove every registered rule.
        """
        self._rules.clear()

    def values(self) -> tuple[BaseRule, ...]:
        """
        Return all registered rules.
        """
        return tuple(self._rules.values())

    def names(self) -> tuple[str, ...]:
        """
        Return registered rule names.
        """
        return tuple(self._rules.keys())

    def items(self) -> tuple[tuple[str, BaseRule], ...]:
        """
        Return (name, rule) pairs.
        """
        return tuple(self._rules.items())

    def __contains__(self, name: object) -> bool:
        return (
            isinstance(name, str)
            and name in self._rules
        )

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[BaseRule]:
        return iter(self._rules.values())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(rules={len(self)})"
        )


__all__ = [
    "RuleRegistry",
]