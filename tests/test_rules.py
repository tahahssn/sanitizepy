"""
Tests for sanitizepy.rules
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from sanitizepy.exceptions import SchemaValidationError
from sanitizepy.models.contracts import ColumnContract, DataContract
from sanitizepy.rules.base import BaseRule, Rule
from sanitizepy.rules.builtins import (
    ColumnNamesRule,
    ConstantColumnsRule,
    DuplicateColumnsRule,
    DuplicateRowsRule,
    HighCardinalityRule,
    InvalidDtypesRule,
    MissingValuesRule,
    OutliersRule,
    StringCaseRule,
    WhitespaceRule,
    register_builtin_rules,
)
from sanitizepy.rules.engine import RuleEngine
from sanitizepy.rules.registry import RuleRegistry
from sanitizepy.rules.rule import RuleCategory, RuleResult, RuleSeverity
from sanitizepy.rules.validators import (
    validate_allowed_values,
    validate_callback,
    validate_column_contract,
    validate_column_exists,
    validate_description,
    validate_dtype,
    validate_enabled,
    validate_nullable,
    validate_priority,
    validate_range,
    validate_regex,
    validate_rule_name,
    validate_unique,
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


# ===========================================================================
# Contract validators (task 12.7)
#
# Covers each expectation type pass/fail, missing-column failing result,
# malformed-contract SchemaValidationError, RuleEngine.validate_contract,
# and each replaced built-in rule, plus the shared edge-case matrix
# (empty, single-row, all-null, mixed-type, infinite, wide, tall).
#
# Validates: Requirements 9.1, 9.2, 9.4, 9.5, 9.6
# ===========================================================================


# ---------------------------------------------------------------------------
# Edge-case DataFrame matrix (shared helpers)
# ---------------------------------------------------------------------------


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1], "b": ["x"]})


def _all_null_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [None, None, None], "b": [np.nan, np.nan, np.nan]})


def _mixed_type_df() -> pd.DataFrame:
    return pd.DataFrame({"mixed": [1, "two", 3.0, None]})


def _infinite_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})


def _wide_df() -> pd.DataFrame:
    return pd.DataFrame({f"c{i}": [i, i + 1] for i in range(60)})


def _tall_df() -> pd.DataFrame:
    return pd.DataFrame({"a": list(range(1000)), "b": ["v"] * 1000})


_EDGE_CASE_FACTORIES = [
    _empty_df,
    _single_row_df,
    _all_null_df,
    _mixed_type_df,
    _infinite_df,
    _wide_df,
    _tall_df,
]


# ---------------------------------------------------------------------------
# validate_column_exists
# ---------------------------------------------------------------------------


class TestValidateColumnExists:
    def test_present_column_passes(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_column_exists(df, "a")
        assert result.passed is True
        assert result.affected_columns == ("a",)

    def test_missing_column_fails_not_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_column_exists(df, "missing")
        assert result.passed is False
        assert result.severity is RuleSeverity.ERROR
        assert "missing" in result.message


# ---------------------------------------------------------------------------
# validate_dtype
# ---------------------------------------------------------------------------


class TestValidateDtype:
    def test_matching_dtype_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_dtype(df, "a", str(df["a"].dtype))
        assert result.passed is True

    def test_mismatched_dtype_fails(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_dtype(df, "a", "float64")
        assert result.passed is False
        assert result.metadata["actual_dtype"] == str(df["a"].dtype)
        assert result.metadata["expected_dtype"] == "float64"


# ---------------------------------------------------------------------------
# validate_nullable
# ---------------------------------------------------------------------------


class TestValidateNullable:
    def test_nullable_true_always_passes(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        result = validate_nullable(df, "a", True)
        assert result.passed is True

    def test_nullable_false_no_missing_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_nullable(df, "a", False)
        assert result.passed is True

    def test_nullable_false_with_missing_fails(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        result = validate_nullable(df, "a", False)
        assert result.passed is False
        assert result.affected_rows == 1
        assert result.metadata["missing_count"] == 1


# ---------------------------------------------------------------------------
# validate_allowed_values
# ---------------------------------------------------------------------------


class TestValidateAllowedValues:
    def test_all_allowed_passes(self):
        df = pd.DataFrame({"a": ["x", "y", "x"]})
        result = validate_allowed_values(df, "a", ("x", "y"))
        assert result.passed is True

    def test_disallowed_value_fails(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = validate_allowed_values(df, "a", ("x", "y"))
        assert result.passed is False
        assert result.affected_rows == 1
        assert "z" in result.message

    def test_missing_values_are_ignored(self):
        df = pd.DataFrame({"a": ["x", None, "y"]})
        result = validate_allowed_values(df, "a", ("x", "y"))
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_range
# ---------------------------------------------------------------------------


class TestValidateRange:
    def test_within_range_passes(self):
        df = pd.DataFrame({"a": [1, 5, 10]})
        result = validate_range(df, "a", 0.0, 10.0)
        assert result.passed is True

    def test_below_min_fails(self):
        df = pd.DataFrame({"a": [-1, 5, 10]})
        result = validate_range(df, "a", 0.0, 10.0)
        assert result.passed is False
        assert result.affected_rows == 1

    def test_above_max_fails(self):
        df = pd.DataFrame({"a": [1, 5, 99]})
        result = validate_range(df, "a", 0.0, 10.0)
        assert result.passed is False
        assert result.affected_rows == 1

    def test_unbounded_min_only(self):
        df = pd.DataFrame({"a": [1, 5, 10]})
        result = validate_range(df, "a", None, 10.0)
        assert result.passed is True

    def test_min_greater_than_max_raises_schema_error(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(SchemaValidationError):
            validate_range(df, "a", 10.0, 0.0)

    def test_non_numeric_column_raises_schema_error(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(SchemaValidationError):
            validate_range(df, "a", 0.0, 10.0)


# ---------------------------------------------------------------------------
# validate_regex
# ---------------------------------------------------------------------------


class TestValidateRegex:
    def test_matching_values_pass(self):
        df = pd.DataFrame({"a": ["123", "456"]})
        result = validate_regex(df, "a", r"\d+")
        assert result.passed is True

    def test_non_matching_value_fails(self):
        df = pd.DataFrame({"a": ["123", "abc"]})
        result = validate_regex(df, "a", r"\d+")
        assert result.passed is False
        assert result.affected_rows == 1
        assert "abc" in result.message

    def test_invalid_pattern_raises_schema_error(self):
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(SchemaValidationError):
            validate_regex(df, "a", r"([unclosed")


# ---------------------------------------------------------------------------
# validate_unique
# ---------------------------------------------------------------------------


class TestValidateUnique:
    def test_unique_column_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_unique(df, "a")
        assert result.passed is True

    def test_duplicated_values_fail(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        result = validate_unique(df, "a")
        assert result.passed is False
        assert result.affected_rows == 2
        assert result.metadata["duplicate_count"] == 2

    def test_missing_values_do_not_count_as_duplicates(self):
        df = pd.DataFrame({"a": [1, None, None]})
        result = validate_unique(df, "a")
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_column_contract (aggregation + missing column short-circuit)
# ---------------------------------------------------------------------------


class TestValidateColumnContract:
    def test_missing_column_returns_single_failing_result(self):
        df = pd.DataFrame({"a": [1, 2]})
        results = validate_column_contract(df, "missing", ColumnContract(dtype="int64"))
        assert len(results) == 1
        assert results[0].passed is False

    def test_one_result_per_declared_expectation(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        contract = ColumnContract(
            dtype=str(df["a"].dtype),
            nullable=False,
            min_value=0.0,
            max_value=10.0,
            unique=True,
        )
        results = validate_column_contract(df, "a", contract)
        # existence + dtype + nullable + range + unique
        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_unique_false_does_not_add_result(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        results = validate_column_contract(df, "a", ColumnContract(unique=False))
        # only existence is checked
        assert len(results) == 1
        assert results[0].passed is True

    def test_malformed_regex_propagates_schema_error(self):
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(SchemaValidationError):
            validate_column_contract(df, "a", ColumnContract(regex=r"([unclosed"))


# ---------------------------------------------------------------------------
# RuleEngine.validate_contract
# ---------------------------------------------------------------------------


class TestRuleEngineValidateContract:
    def test_returns_result_per_expectation_sorted_by_column(self):
        df = pd.DataFrame({"b": [1, 2], "a": ["x", "y"]})
        contract = DataContract(
            columns={
                "b": ColumnContract(nullable=False),
                "a": ColumnContract(dtype=str(df["a"].dtype)),
            }
        )
        engine = RuleEngine()
        results = engine.validate_contract(df, contract)
        # column 'a' processed before 'b'
        assert results[0].affected_columns == ("a",)
        assert all(r.passed for r in results)

    def test_missing_column_yields_failing_result(self):
        df = pd.DataFrame({"a": [1, 2]})
        contract = DataContract(columns={"ghost": ColumnContract(dtype="int64")})
        engine = RuleEngine()
        results = engine.validate_contract(df, contract)
        assert len(results) == 1
        assert results[0].passed is False

    def test_malformed_contract_raises_schema_error(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        contract = DataContract(
            columns={"a": ColumnContract(min_value=10.0, max_value=0.0)}
        )
        engine = RuleEngine()
        with pytest.raises(SchemaValidationError):
            engine.validate_contract(df, contract)

    def test_deterministic_repeated_validation(self):
        df = pd.DataFrame({"a": [1, 2, 2]})
        contract = DataContract(columns={"a": ColumnContract(unique=True)})
        engine = RuleEngine()
        first = engine.validate_contract(df, contract)
        second = engine.validate_contract(df, contract)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    def test_non_dataframe_raises_type_error(self):
        engine = RuleEngine()
        with pytest.raises(TypeError):
            engine.validate_contract([1, 2, 3], DataContract())  # type: ignore[arg-type]

    def test_non_contract_raises_type_error(self):
        engine = RuleEngine()
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            engine.validate_contract(df, {"a": "x"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Replaced built-in rules: pass/fail behaviour
# ---------------------------------------------------------------------------


def _missing_values_rule() -> MissingValuesRule:
    return MissingValuesRule(name="missing_values", description="d")


def _duplicate_rows_rule() -> DuplicateRowsRule:
    return DuplicateRowsRule(name="duplicate_rows", description="d")


def _duplicate_columns_rule() -> DuplicateColumnsRule:
    return DuplicateColumnsRule(name="duplicate_columns", description="d")


def _invalid_dtypes_rule() -> InvalidDtypesRule:
    return InvalidDtypesRule(name="invalid_dtypes", description="d")


def _outliers_rule() -> OutliersRule:
    return OutliersRule(name="outliers", description="d")


def _constant_columns_rule() -> ConstantColumnsRule:
    return ConstantColumnsRule(name="constant_columns", description="d")


def _high_cardinality_rule() -> HighCardinalityRule:
    return HighCardinalityRule(name="high_cardinality", description="d")


def _whitespace_rule() -> WhitespaceRule:
    return WhitespaceRule(name="whitespace", description="d")


def _string_case_rule() -> StringCaseRule:
    return StringCaseRule(name="string_case", description="d")


def _column_names_rule() -> ColumnNamesRule:
    return ColumnNamesRule(name="column_names", description="d")


class TestMissingValuesRule:
    def test_clean_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _missing_values_rule().evaluate(df)
        assert result.passed is True

    def test_missing_fails_with_counts(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})
        result = _missing_values_rule().evaluate(df)
        assert result.passed is False
        assert set(result.affected_columns) == {"a", "b"}
        assert result.metadata["missing_cells"] == 2


class TestDuplicateRowsRule:
    def test_unique_rows_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _duplicate_rows_rule().evaluate(df)
        assert result.passed is True

    def test_duplicate_rows_fail(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        result = _duplicate_rows_rule().evaluate(df)
        assert result.passed is False
        assert result.affected_rows == 2
        assert result.metadata["redundant_rows"] == 1


class TestDuplicateColumnsRule:
    def test_distinct_columns_pass(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _duplicate_columns_rule().evaluate(df)
        assert result.passed is True

    def test_content_duplicate_columns_fail(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1, 2]})
        result = _duplicate_columns_rule().evaluate(df)
        assert result.passed is False
        assert "b" in result.affected_columns


class TestInvalidDtypesRule:
    def test_proper_dtypes_pass(self):
        df = pd.DataFrame({"a": ["x", "y"], "n": [1, 2]})
        result = _invalid_dtypes_rule().evaluate(df)
        assert result.passed is True

    def test_numeric_object_column_fails(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = _invalid_dtypes_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns


class TestOutliersRule:
    def test_no_outliers_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = _outliers_rule().evaluate(df)
        assert result.passed is True

    def test_outlier_fails(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 1000]})
        result = _outliers_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns
        assert result.affected_rows >= 1


class TestConstantColumnsRule:
    def test_varied_column_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _constant_columns_rule().evaluate(df)
        assert result.passed is True

    def test_constant_column_fails(self):
        df = pd.DataFrame({"a": [7, 7, 7]})
        result = _constant_columns_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns


class TestHighCardinalityRule:
    def test_low_cardinality_passes(self):
        df = pd.DataFrame({"a": ["x", "x", "y", "y"]})
        result = _high_cardinality_rule().evaluate(df)
        assert result.passed is True

    def test_high_cardinality_fails(self):
        df = pd.DataFrame({"a": [f"id-{i}" for i in range(10)]})
        result = _high_cardinality_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns


class TestWhitespaceRule:
    def test_clean_strings_pass(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        result = _whitespace_rule().evaluate(df)
        assert result.passed is True

    def test_irregular_whitespace_fails(self):
        df = pd.DataFrame({"a": [" leading", "trailing ", "two  spaces"]})
        result = _whitespace_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns
        assert result.affected_rows == 3


class TestStringCaseRule:
    def test_consistent_case_passes(self):
        df = pd.DataFrame({"a": ["yes", "no", "yes"]})
        result = _string_case_rule().evaluate(df)
        assert result.passed is True

    def test_inconsistent_case_fails(self):
        df = pd.DataFrame({"a": ["Yes", "yes", "no"]})
        result = _string_case_rule().evaluate(df)
        assert result.passed is False
        assert "a" in result.affected_columns


class TestColumnNamesRule:
    def test_normalized_names_pass(self):
        df = pd.DataFrame({"first_name": [1], "age2": [2]})
        result = _column_names_rule().evaluate(df)
        assert result.passed is True

    def test_non_normalized_names_fail(self):
        df = pd.DataFrame({"First Name": [1], "AGE": [2]})
        result = _column_names_rule().evaluate(df)
        assert result.passed is False
        assert set(result.affected_columns) == {"First Name", "AGE"}


# ---------------------------------------------------------------------------
# Edge-case matrix: every built-in rule stays evaluable and read-only
# ---------------------------------------------------------------------------


_ALL_BUILTIN_FACTORIES = [
    _missing_values_rule,
    _duplicate_rows_rule,
    _duplicate_columns_rule,
    _invalid_dtypes_rule,
    _outliers_rule,
    _constant_columns_rule,
    _high_cardinality_rule,
    _whitespace_rule,
    _string_case_rule,
    _column_names_rule,
]


class TestBuiltinRuleEdgeCases:
    @pytest.mark.parametrize("factory", _EDGE_CASE_FACTORIES)
    @pytest.mark.parametrize("rule_factory", _ALL_BUILTIN_FACTORIES)
    def test_rule_returns_result_and_does_not_mutate(self, factory, rule_factory):
        df = factory()
        before = df.copy()
        rule = rule_factory()
        result = rule.evaluate(df)
        assert isinstance(result, RuleResult)
        pd.testing.assert_frame_equal(df, before)

    @pytest.mark.parametrize("factory", _EDGE_CASE_FACTORIES)
    def test_rules_are_deterministic(self, factory):
        df = factory()
        for rule_factory in _ALL_BUILTIN_FACTORIES:
            first = rule_factory().evaluate(df)
            second = rule_factory().evaluate(df)
            assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# Edge-case matrix: contract validation stays stable across shapes
# ---------------------------------------------------------------------------


class TestContractValidationEdgeCases:
    @pytest.mark.parametrize("factory", _EDGE_CASE_FACTORIES)
    def test_missing_column_always_fails(self, factory):
        df = factory()
        contract = DataContract(
            columns={"__definitely_absent__": ColumnContract(dtype="int64")}
        )
        results = RuleEngine().validate_contract(df, contract)
        assert len(results) == 1
        assert results[0].passed is False

    def test_all_null_column_nullable_false_fails(self):
        df = _all_null_df()
        contract = DataContract(columns={"a": ColumnContract(nullable=False)})
        results = RuleEngine().validate_contract(df, contract)
        nullable_results = [r for r in results if "nullable" in r.rule]
        assert nullable_results
        assert nullable_results[0].passed is False

    def test_single_row_unique_passes(self):
        df = _single_row_df()
        contract = DataContract(columns={"a": ColumnContract(unique=True)})
        results = RuleEngine().validate_contract(df, contract)
        unique_results = [r for r in results if "unique" in r.rule]
        assert unique_results
        assert unique_results[0].passed is True

    def test_infinite_values_within_open_range_pass(self):
        df = _infinite_df()
        contract = DataContract(columns={"a": ColumnContract(min_value=None)})
        # No range declared (min/max both None) -> only existence check.
        results = RuleEngine().validate_contract(df, contract)
        assert all(r.passed for r in results)


# ===========================================================================
# Property tests: DataContract validation
# ===========================================================================


@st.composite
def _dataframe_with_columns(draw: st.DrawFn) -> pd.DataFrame:
    """
    Draw a small DataFrame with a stable, deterministic set of columns.

    Columns are drawn from a fixed pool so that contracts can reliably
    reference columns that may or may not be present. Each drawn column is
    populated with a mix of integers, strings, and missing values.
    """
    column_pool = ["a", "b", "c", "d"]
    selected = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=1,
            max_size=len(column_pool),
            unique=True,
        )
    )
    n_rows = draw(st.integers(min_value=0, max_value=15))

    cell = st.one_of(
        st.none(),
        st.integers(min_value=-100, max_value=100),
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=0,
            max_size=6,
        ),
    )

    data: dict[str, list[object]] = {}
    for column in selected:
        data[column] = draw(st.lists(cell, min_size=n_rows, max_size=n_rows))
    return pd.DataFrame(data)


@st.composite
def _valid_column_contract(draw: st.DrawFn) -> ColumnContract:
    """
    Draw a structurally valid ColumnContract.

    Only expectations that can never raise SchemaValidationError are drawn
    (no regex patterns, no numeric ranges) so that repeated validation of
    the contract is well-defined and comparable across runs.
    """
    kwargs: dict[str, object] = {}
    if draw(st.booleans()):
        kwargs["nullable"] = draw(st.booleans())
    if draw(st.booleans()):
        kwargs["allowed_values"] = tuple(
            draw(
                st.lists(
                    st.integers(min_value=-100, max_value=100),
                    min_size=1,
                    max_size=5,
                    unique=True,
                )
            )
        )
    if draw(st.booleans()):
        kwargs["unique"] = draw(st.booleans())
    return ColumnContract(**kwargs)


@st.composite
def _valid_data_contract(draw: st.DrawFn) -> DataContract:
    """Draw a DataContract referencing columns from the shared pool."""
    column_pool = ["a", "b", "c", "d", "missing_1", "missing_2"]
    names = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=1,
            max_size=len(column_pool),
            unique=True,
        )
    )
    columns = {name: draw(_valid_column_contract()) for name in names}
    return DataContract(columns=columns)


class TestContractDeterminismProperty:
    """
    **Property 9: DataContract validation is deterministic**
    **Validates: Requirements 9.7**

    For any DataFrame and any structurally valid DataContract, running
    ``RuleEngine.validate_contract`` twice yields identical results.
    """

    @given(_dataframe_with_columns(), _valid_data_contract())
    @settings(max_examples=150)
    def test_repeated_validation_is_identical(
        self, dataframe: pd.DataFrame, contract: DataContract
    ) -> None:
        engine = RuleEngine()
        first = engine.validate_contract(dataframe, contract)
        second = engine.validate_contract(dataframe, contract)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    @given(_dataframe_with_columns(), _valid_data_contract())
    @settings(max_examples=150)
    def test_independent_engines_produce_identical_results(
        self, dataframe: pd.DataFrame, contract: DataContract
    ) -> None:
        first = RuleEngine().validate_contract(dataframe, contract)
        second = RuleEngine().validate_contract(dataframe.copy(), contract)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]


class TestMissingContractColumnsProperty:
    """
    **Property 10: Missing contract columns always fail**
    **Validates: Requirements 9.3**

    For any DataFrame and any contract declaring a column that is NOT
    present in the DataFrame, the result for that column is a failing
    RuleResult (``passed`` is ``False``).
    """

    @given(
        _dataframe_with_columns(),
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=12,
        ),
        _valid_column_contract(),
    )
    @settings(max_examples=150)
    def test_declared_absent_column_fails(
        self,
        dataframe: pd.DataFrame,
        missing_name: str,
        column_contract: ColumnContract,
    ) -> None:
        # Guarantee the drawn name is absent from the DataFrame.
        absent = missing_name
        while absent in dataframe.columns:
            absent = f"{absent}_x"

        contract = DataContract(columns={absent: column_contract})
        results = RuleEngine().validate_contract(dataframe, contract)

        # A missing column short-circuits to exactly one failing result.
        assert len(results) == 1
        result = results[0]
        assert result.passed is False
        assert result.affected_columns == (absent,)

    @given(_dataframe_with_columns(), _valid_column_contract())
    @settings(max_examples=100)
    def test_missing_columns_fail_even_when_present_columns_pass(
        self, dataframe: pd.DataFrame, column_contract: ColumnContract
    ) -> None:
        absent = "__guaranteed_absent__"
        while absent in dataframe.columns:
            absent = f"{absent}_x"

        contract = DataContract(columns={absent: column_contract})
        results = RuleEngine().validate_contract(dataframe, contract)

        # Every result concerning the absent column must be a failure.
        absent_results = [r for r in results if absent in r.affected_columns]
        assert absent_results
        assert all(r.passed is False for r in absent_results)


# ---------------------------------------------------------------------------
# Property tests — Task 18.3
# **Property 18: Custom rules integrate through RuleRegistry**
# **Validates: Requirements 14.2**
# ---------------------------------------------------------------------------


class _CustomColumnPresenceRule(BaseRule):
    """
    A user-defined validation rule, defined outside the library.

    It subclasses the existing :class:`BaseRule` ABC only and checks whether
    a required column is present. Registering it through the existing
    :class:`RuleRegistry` and executing it via :class:`RuleEngine` must work
    without any engine modification and must yield a genuine
    :class:`RuleResult` (Requirement 14.2), identical in shape to a built-in
    rule's result.
    """

    def __init__(self, *, required_column: str) -> None:
        super().__init__(
            name=f"custom_requires_{required_column}",
            description=f"Requires column '{required_column}' to be present.",
        )
        self._required_column = required_column

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: object,
    ) -> RuleResult:
        present = self._required_column in dataframe.columns
        return RuleResult(
            rule=self.name,
            passed=present,
            severity=RuleSeverity.INFO if present else RuleSeverity.ERROR,
            category=RuleCategory.CUSTOM,
            message=(
                f"Column '{self._required_column}' is present."
                if present
                else f"Column '{self._required_column}' is missing."
            ),
            affected_columns=() if present else (self._required_column,),
        )


class TestCustomRulesIntegrateThroughRuleRegistry:
    """
    **Property 18: Custom rules integrate through RuleRegistry**
    **Validates: Requirements 14.2**

    A custom ``BaseRule`` subclass registers through the existing
    ``RuleRegistry`` and executes via ``RuleEngine`` yielding a
    ``RuleResult`` like built-in rules, without any engine modification.
    """

    def test_custom_rule_registers_and_executes_like_builtin(self) -> None:
        registry = RuleRegistry()
        rule = _CustomColumnPresenceRule(required_column="id")
        registry.register(rule)

        # Retrievable by name, exactly like a built-in rule.
        assert registry.get(rule.name) is rule

        engine = RuleEngine(registry=registry)
        results = engine.run(pd.DataFrame({"id": [1, 2], "v": [3, 4]}))

        # One RuleResult per registered rule, like built-ins.
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, RuleResult)
        assert result.rule == rule.name
        assert result.passed is True
        assert result.category is RuleCategory.CUSTOM

    @given(
        columns=st.lists(
            st.sampled_from(["id", "name", "value", "ts", "flag"]),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        required=st.sampled_from(["id", "name", "value", "ts", "flag", "missing_col"]),
    )
    @settings(max_examples=100)
    def test_custom_rule_result_shape_matches_builtin(
        self, columns: list[str], required: str
    ) -> None:
        frame = pd.DataFrame({col: [1, 2, 3] for col in columns})

        # Run a built-in rule to capture the canonical RuleResult field set.
        builtin_registry = RuleRegistry()
        builtin_registry.register(
            MissingValuesRule(name="missing_values", description="d")
        )
        builtin_result = RuleEngine(registry=builtin_registry).run(frame)[0]

        # Run the custom rule through the same engine mechanism.
        custom_registry = RuleRegistry()
        custom_rule = _CustomColumnPresenceRule(required_column=required)
        custom_registry.register(custom_rule)
        custom_result = RuleEngine(registry=custom_registry).run(frame)[0]

        # Shape parity: identical field names on the frozen pydantic model.
        assert set(custom_result.model_dump().keys()) == set(
            builtin_result.model_dump().keys()
        )

        # It yields a real RuleResult whose passed flag reflects the data.
        assert isinstance(custom_result, RuleResult)
        assert custom_result.rule == custom_rule.name
        assert custom_result.passed is (required in frame.columns)
        if required not in frame.columns:
            assert custom_result.affected_columns == (required,)

    def test_multiple_custom_rules_execute_in_registry_order(self) -> None:
        registry = RuleRegistry()
        registry.register(_CustomColumnPresenceRule(required_column="a"))
        registry.register(_CustomColumnPresenceRule(required_column="b"))

        engine = RuleEngine(registry=registry)
        results = engine.run(pd.DataFrame({"a": [1]}))

        # Both custom rules ran and produced RuleResults, like built-ins.
        assert len(results) == 2
        assert all(isinstance(r, RuleResult) for r in results)
        by_rule = {r.rule: r for r in results}
        assert by_rule["custom_requires_a"].passed is True
        assert by_rule["custom_requires_b"].passed is False

    def test_custom_rule_alongside_builtin_rules(self) -> None:
        # A custom rule coexists with built-in rules in the same registry
        # and the engine executes them identically.
        registry = RuleRegistry()
        register_builtin_rules(registry)
        n_builtins = len(registry)
        registry.register(_CustomColumnPresenceRule(required_column="id"))

        engine = RuleEngine(registry=registry)
        results = engine.run(pd.DataFrame({"id": [1, 2, 3]}))

        assert len(results) == n_builtins + 1
        custom = next(r for r in results if r.rule == "custom_requires_id")
        assert isinstance(custom, RuleResult)
        assert custom.passed is True
