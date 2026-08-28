"""
Tests for sanitizepy.rules
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from sanitizepy.rules.base import Rule
from sanitizepy.rules.builtins import register_builtin_rules
from sanitizepy.rules.registry import RuleRegistry
from sanitizepy.rules.rule import RuleCategory, RuleResult, RuleSeverity
from sanitizepy.rules.validators import (
    validate_callback,
    validate_description,
    validate_enabled,
    validate_priority,
    validate_rule_name,
)

# ---------------------------------------------------------------------------
# RuleSeverity
# ---------------------------------------------------------------------------


class TestRuleSeverity:
    def test_info_value(self):
        assert RuleSeverity.INFO.value == "info"

    def test_warning_value(self):
        assert RuleSeverity.WARNING.value == "warning"

    def test_error_value(self):
        assert RuleSeverity.ERROR.value == "error"

    def test_critical_value(self):
        assert RuleSeverity.CRITICAL.value == "critical"

    def test_is_string_enum(self):
        assert isinstance(RuleSeverity.INFO, str)


# ---------------------------------------------------------------------------
# RuleCategory
# ---------------------------------------------------------------------------


class TestRuleCategory:
    def test_data_quality_value(self):
        assert RuleCategory.DATA_QUALITY.value == "data_quality"

    def test_custom_value(self):
        assert RuleCategory.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------------


class TestRuleResult:
    def _make_result(self, **kwargs):
        defaults = {
            "rule": "test_rule",
            "passed": True,
            "severity": RuleSeverity.INFO,
            "category": RuleCategory.DATA_QUALITY,
            "message": "All good",
        }
        defaults.update(kwargs)
        return RuleResult(**defaults)

    def test_creation(self):
        r = self._make_result()
        assert r.rule == "test_rule"

    def test_passed_true(self):
        r = self._make_result(passed=True)
        assert r.passed is True

    def test_passed_false(self):
        r = self._make_result(passed=False)
        assert r.passed is False

    def test_default_affected_columns_empty(self):
        r = self._make_result()
        assert r.affected_columns == ()

    def test_default_affected_rows_zero(self):
        r = self._make_result()
        assert r.affected_rows == 0

    def test_affected_rows_non_negative(self):
        r = self._make_result(affected_rows=5)
        assert r.affected_rows == 5

    def test_negative_affected_rows_raises(self):
        with pytest.raises(ValidationError):
            self._make_result(affected_rows=-1)

    def test_metadata_default_empty(self):
        r = self._make_result()
        assert r.metadata == {}

    def test_is_frozen(self):
        r = self._make_result()
        with pytest.raises(ValidationError):
            r.rule = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            self._make_result(unknown_field="value")


# ---------------------------------------------------------------------------
# BaseRule / Rule
# ---------------------------------------------------------------------------


class TestBaseRule:
    def test_rule_name_property(self):
        rule = Rule(name="my_rule", description="test desc")
        assert rule.name == "my_rule"

    def test_rule_description_property(self):
        rule = Rule(name="my_rule", description="test desc")
        assert rule.description == "test desc"

    def test_rule_repr(self):
        rule = Rule(name="my_rule", description="desc")
        assert "my_rule" in repr(rule)

    def test_evaluate_returns_rule_result(self):
        rule = Rule(name="sample", description="A sample rule")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = rule.evaluate(df)
        assert isinstance(result, RuleResult)

    def test_evaluate_passes_by_default(self):
        rule = Rule(name="sample", description="A sample rule")
        df = pd.DataFrame({"a": [1, 2]})
        result = rule.evaluate(df)
        assert result.passed is True

    def test_evaluate_sets_rule_name(self):
        rule = Rule(name="my_rule", description="desc")
        df = pd.DataFrame({"a": [1]})
        result = rule.evaluate(df)
        assert result.rule == "my_rule"


# ---------------------------------------------------------------------------
# RuleRegistry
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    def _make_registry(self):
        return RuleRegistry()

    def test_empty_registry(self):
        registry = self._make_registry()
        assert len(registry) == 0

    def test_register_rule(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(rule)

    def test_get_rule(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        assert registry.get("r1") is rule

    def test_get_missing_rule_raises(self):
        registry = self._make_registry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_contains_registered_rule(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        assert registry.contains("r1") is True

    def test_not_contains_unregistered_rule(self):
        registry = self._make_registry()
        assert registry.contains("missing") is False

    def test_in_operator(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        assert "r1" in registry
        assert "missing" not in registry

    def test_unregister_rule(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="desc")
        registry.register(rule)
        registry.unregister("r1")
        assert len(registry) == 0

    def test_clear_all_rules(self):
        registry = self._make_registry()
        registry.register(Rule(name="r1", description="a"))
        registry.register(Rule(name="r2", description="b"))
        registry.clear()
        assert len(registry) == 0

    def test_names(self):
        registry = self._make_registry()
        registry.register(Rule(name="r1", description="a"))
        registry.register(Rule(name="r2", description="b"))
        names = registry.names()
        assert "r1" in names
        assert "r2" in names

    def test_values_returns_all_rules(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="a")
        registry.register(rule)
        assert rule in registry.values()

    def test_items_returns_name_rule_pairs(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="a")
        registry.register(rule)
        items = dict(registry.items())
        assert items["r1"] is rule

    def test_iter_yields_rules(self):
        registry = self._make_registry()
        rule = Rule(name="r1", description="a")
        registry.register(rule)
        rules = list(registry)
        assert rule in rules

    def test_repr_contains_count(self):
        registry = self._make_registry()
        assert "0" in repr(registry)


# ---------------------------------------------------------------------------
# BuiltIn Rules
# ---------------------------------------------------------------------------


class TestBuiltinRules:
    def test_register_builtin_rules_populates_registry(self):
        registry = RuleRegistry()
        register_builtin_rules(registry)
        assert len(registry) > 0

    def test_missing_values_rule_registered(self):
        registry = RuleRegistry()
        register_builtin_rules(registry)
        assert registry.contains("missing_values")

    def test_duplicate_rows_rule_registered(self):
        registry = RuleRegistry()
        register_builtin_rules(registry)
        assert registry.contains("duplicate_rows")

    def test_duplicate_columns_rule_registered(self):
        registry = RuleRegistry()
        register_builtin_rules(registry)
        assert registry.contains("duplicate_columns")

    def test_all_builtin_rules_are_evaluable(self):
        registry = RuleRegistry()
        register_builtin_rules(registry)
        df = pd.DataFrame({"a": [1, 2, 3]})
        for rule in registry:
            result = rule.evaluate(df)
            assert isinstance(result, RuleResult)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidateRuleName:
    def test_valid_name_passes(self):
        validate_rule_name("my_rule")  # should not raise

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_rule_name(123)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_rule_name("   ")


class TestValidateDescription:
    def test_valid_description_passes(self):
        validate_description("a description")  # should not raise

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_description(42)  # type: ignore[arg-type]


class TestValidatePriority:
    def test_valid_priority_passes(self):
        validate_priority(1)

    def test_non_int_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_priority(1.5)  # type: ignore[arg-type]


class TestValidateEnabled:
    def test_valid_enabled_passes(self):
        validate_enabled(True)

    def test_non_bool_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_enabled(1)  # type: ignore[arg-type]


class TestValidateCallback:
    def test_callable_passes(self):
        validate_callback(lambda: None)

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_callback("not_callable")  # type: ignore[arg-type]
