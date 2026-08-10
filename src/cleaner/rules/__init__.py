"""
cleaner.rules
~~~~~~~~~~~~~

Rule Engine for Cleaner.
"""

from .base import BaseRule, Rule
from .builtins import register_builtin_rules
from .engine import RuleEngine
from .registry import RuleRegistry
from .rule import RuleCategory, RuleResult, RuleSeverity

__all__ = [
    "BaseRule",
    "Rule",
    "RuleCategory",
    "RuleResult",
    "RuleSeverity",
    "RuleRegistry",
    "RuleEngine",
    "register_builtin_rules",
]