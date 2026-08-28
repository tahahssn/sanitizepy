"""
cleaner.rules.builtins
~~~~~~~~~~~~~~~~~~~~~~

Built-in rules shipped with Cleaner.
"""

from __future__ import annotations

from .base import Rule
from .registry import RuleRegistry


def register_builtin_rules(
    registry: RuleRegistry,
) -> None:
    """
    Register all built-in Cleaner rules.

    Parameters
    ----------
    registry:
        Rule registry instance.
    """

    registry.register(
        Rule(
            name="missing_values",
            description="Handle missing values.",
        )
    )

    registry.register(
        Rule(
            name="duplicate_rows",
            description="Remove duplicated rows.",
        )
    )

    registry.register(
        Rule(
            name="duplicate_columns",
            description="Remove duplicated columns.",
        )
    )

    registry.register(
        Rule(
            name="invalid_dtypes",
            description="Correct invalid data types.",
        )
    )

    registry.register(
        Rule(
            name="outliers",
            description="Handle statistical outliers.",
        )
    )

    registry.register(
        Rule(
            name="constant_columns",
            description="Remove constant columns.",
        )
    )

    registry.register(
        Rule(
            name="high_cardinality",
            description="Handle high-cardinality features.",
        )
    )

    registry.register(
        Rule(
            name="whitespace",
            description="Normalize whitespace.",
        )
    )

    registry.register(
        Rule(
            name="string_case",
            description="Normalize string casing.",
        )
    )

    registry.register(
        Rule(
            name="column_names",
            description="Normalize column names.",
        )
    )


__all__ = [
    "register_builtin_rules",
]
