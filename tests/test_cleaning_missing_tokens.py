"""
Tests for sanitizepy.cleaning.missing_tokens.MissingTokenOperation

Task 4.5 – unit tests for MissingTokenOperation
Requirements: 3.1, 3.4, 3.5

Covers:
  - Default tokens converted to missing (Req 3.1, 3.2)
  - Configured/extra tokens extend the defaults (Req 3.3)
  - Case-insensitive matching (Req 3.4)
  - Whitespace-trimmed matching (Req 3.4)
  - Empty strings converted to missing
  - Already-missing values unchanged
  - Non-string columns untouched
  - OperationResult records correct count (Req 3.5)
  - describe() exposes name, extra_tokens, subset, token_count
  - is_chunk_safe flag
  - Registry presence
  - Edge cases: empty DF, single-row, all-null, mixed-type, wide, tall
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.missing_tokens import MissingTokenOperation
from sanitizepy.constants import DEFAULT_MISSING_VALUE_TOKENS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# 1. Default tokens (Requirement 3.1, 3.2)
# ---------------------------------------------------------------------------


class TestDefaultTokens:
    """MissingTokenOperation must recognise DEFAULT_MISSING_VALUE_TOKENS."""

    def test_default_tokens_are_a_superset_of_common_sentinels(self) -> None:
        """Spot-check that the most common sentinels are in the default set."""
        for token in ("na", "n/a", "nan", "null", "none", "nil", "?", "-", ""):
            assert token in DEFAULT_MISSING_VALUE_TOKENS

    @pytest.mark.parametrize(
        "sentinel",
        sorted(DEFAULT_MISSING_VALUE_TOKENS),
    )
    def test_each_default_token_is_replaced(self, sentinel: str) -> None:
        """Every token in DEFAULT_MISSING_VALUE_TOKENS should become NaN."""
        df = _make_df(col=[sentinel, "keep"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(
            result["col"].iloc[0]
        ), f"Expected token {sentinel!r} to become NaN, got {result['col'].iloc[0]!r}"
        assert result["col"].iloc[1] == "keep"

    def test_non_token_string_not_replaced(self) -> None:
        df = _make_df(col=["hello", "world", "foo"])
        result = MissingTokenOperation().apply(df)
        assert list(result["col"]) == ["hello", "world", "foo"]

    def test_all_default_tokens_in_single_column_all_become_nan(self) -> None:
        tokens = list(DEFAULT_MISSING_VALUE_TOKENS)
        df = _make_df(col=tokens)
        result = MissingTokenOperation().apply(df)
        assert result["col"].isna().all()


# ---------------------------------------------------------------------------
# 2. Configured / extra tokens extend defaults (Requirement 3.3)
# ---------------------------------------------------------------------------


class TestConfiguredTokens:
    """Extra tokens are added *in addition* to defaults, not instead of them."""

    def test_extra_token_is_replaced(self) -> None:
        df = _make_df(col=["missing_value", "real_data"])
        op = MissingTokenOperation(extra_tokens=["missing_value"])
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "real_data"

    def test_default_tokens_still_replaced_when_extra_provided(self) -> None:
        df = _make_df(col=["na", "n/a", "custom_missing"])
        op = MissingTokenOperation(extra_tokens=["custom_missing"])
        result = op.apply(df)
        assert result["col"].isna().all()

    def test_multiple_extra_tokens(self) -> None:
        df = _make_df(col=["TBD", "N.A.", "real"])
        op = MissingTokenOperation(extra_tokens=["tbd", "n.a."])
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert pd.isna(result["col"].iloc[1])
        assert result["col"].iloc[2] == "real"

    def test_extra_tokens_as_list(self) -> None:
        df = _make_df(col=["placeholder", "real"])
        op = MissingTokenOperation(extra_tokens=["placeholder"])
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_extra_tokens_as_frozenset(self) -> None:
        df = _make_df(col=["absent", "real"])
        op = MissingTokenOperation(extra_tokens=frozenset(["absent"]))
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_extra_tokens_as_set(self) -> None:
        df = _make_df(col=["unknown", "real"])
        op = MissingTokenOperation(extra_tokens={"unknown"})
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_no_extra_tokens_only_defaults_apply(self) -> None:
        """When no extra tokens provided, only default set applies."""
        df = _make_df(col=["na", "real_value"])
        op = MissingTokenOperation()
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "real_value"


# ---------------------------------------------------------------------------
# 3. Case-insensitive matching (Requirement 3.4)
# ---------------------------------------------------------------------------


class TestCaseInsensitiveMatching:
    """Comparison must be case-insensitive (Req 3.4)."""

    @pytest.mark.parametrize(
        "variant",
        [
            "NA",
            "Na",
            "nA",
            "N/A",
            "N/a",
            "n/A",
            "NaN",
            "Nan",
            "nAn",
            "NULL",
            "Null",
            "None",
            "NONE",
            "NIL",
            "Nil",
        ],
    )
    def test_uppercase_and_mixed_case_default_tokens(self, variant: str) -> None:
        df = _make_df(col=[variant, "keep"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0]), f"Expected {variant!r} to become NaN"

    def test_extra_token_case_insensitive(self) -> None:
        df = _make_df(col=["TBD", "tbd", "Tbd", "real"])
        op = MissingTokenOperation(extra_tokens=["TBD"])
        result = op.apply(df)
        assert result["col"].iloc[:3].isna().all()
        assert result["col"].iloc[3] == "real"


# ---------------------------------------------------------------------------
# 4. Whitespace-trimmed matching (Requirement 3.4)
# ---------------------------------------------------------------------------


class TestWhitespaceTrimmingMatching:
    """Leading/trailing whitespace must be stripped before comparison (Req 3.4)."""

    @pytest.mark.parametrize(
        "padded",
        ["  na", "na  ", "  na  ", "\tna\t", "\nna\n", " N/A ", "  NULL  "],
    )
    def test_padded_default_token_is_replaced(self, padded: str) -> None:
        df = _make_df(col=[padded, "real"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(
            result["col"].iloc[0]
        ), f"Expected {padded!r} to become NaN after strip"

    def test_padded_extra_token_is_replaced(self) -> None:
        df = _make_df(col=["  missing  ", "  real  "])
        op = MissingTokenOperation(extra_tokens=["missing"])
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "  real  "

    def test_internal_whitespace_not_collapsed(self) -> None:
        """Only strip — internal spaces are not collapsed (not a token match)."""
        df = _make_df(col=["n a", "real"])
        result = MissingTokenOperation().apply(df)
        # "n a" (with internal space) is not a default token
        assert result["col"].iloc[0] == "n a"


# ---------------------------------------------------------------------------
# 5. Empty strings (Requirement 3.1, 3.2)
# ---------------------------------------------------------------------------


class TestEmptyStrings:
    def test_empty_string_is_replaced_by_default(self) -> None:
        """The empty string '' is in DEFAULT_MISSING_VALUE_TOKENS."""
        df = _make_df(col=["", "real"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_whitespace_only_string_is_replaced_by_default(self) -> None:
        """' ' (single space) is in DEFAULT_MISSING_VALUE_TOKENS."""
        df = _make_df(col=[" ", "real"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_multi_space_string_stripped_to_empty_is_replaced(self) -> None:
        """'   ' strips to '' which is in the default token set."""
        df = _make_df(col=["   ", "real"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])


# ---------------------------------------------------------------------------
# 6. Already-missing values (NaN / None) unchanged
# ---------------------------------------------------------------------------


class TestAlreadyMissingValues:
    def test_existing_nan_unchanged(self) -> None:
        df = _make_df(col=[np.nan, "real"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "real"

    def test_none_value_unchanged(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, "real"], dtype="object")})
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "real"

    def test_mixed_existing_nan_and_token(self) -> None:
        df = _make_df(col=[np.nan, "na", "real"])
        result = MissingTokenOperation().apply(df)
        assert result["col"].isna().iloc[0]
        assert result["col"].isna().iloc[1]
        assert result["col"].iloc[2] == "real"

    def test_apply_with_result_does_not_count_pre_existing_nan(self) -> None:
        """Pre-existing NaN must not inflate the converted count."""
        df = _make_df(col=[np.nan, "na", "real"])
        # 1 pre-existing NaN + 1 token "na" = 1 newly converted
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.rows_affected == 1
        assert res.details["cells_converted_to_missing"] == 1


# ---------------------------------------------------------------------------
# 7. Non-string columns untouched
# ---------------------------------------------------------------------------


class TestNonStringColumnsUntouched:
    def test_integer_column_not_modified(self) -> None:
        df = _make_df(a=["na", "real"], b=[1, 2])
        result = MissingTokenOperation().apply(df)
        # Object-like "na" in 'a' becomes NaN; int column 'b' unchanged
        assert pd.isna(result["a"].iloc[0])
        assert list(result["b"]) == [1, 2]

    def test_float_column_not_modified(self) -> None:
        df = _make_df(a=["null", "ok"], b=[1.0, 2.0])
        result = MissingTokenOperation().apply(df)
        assert list(result["b"]) == pytest.approx([1.0, 2.0])

    def test_boolean_column_not_modified(self) -> None:
        df = pd.DataFrame({"a": ["na"], "flag": pd.array([True], dtype=bool)})
        result = MissingTokenOperation().apply(df)
        assert list(result["flag"]) == [True]

    def test_numeric_column_with_nan_float_not_treated_as_token(self) -> None:
        """float('nan') in a float column must not be 'replaced' (it's already NaN)."""
        df = pd.DataFrame({"n": [1.0, float("nan"), 3.0]})
        result = MissingTokenOperation().apply(df)
        assert list(result["n"][:1]) == pytest.approx([1.0])
        assert list(result["n"][2:]) == pytest.approx([3.0])
        assert pd.isna(result["n"].iloc[1])

    def test_datetime_column_not_modified(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-06-15"])
        df = pd.DataFrame({"d": dates, "label": ["na", "ok"]})
        result = MissingTokenOperation().apply(df)
        pd.testing.assert_series_equal(result["d"], df["d"])


# ---------------------------------------------------------------------------
# 8. subset parameter
# ---------------------------------------------------------------------------


class TestSubsetParameter:
    def test_subset_restricts_to_specified_columns(self) -> None:
        df = _make_df(a=["na", "real"], b=["null", "data"])
        op = MissingTokenOperation(subset=["a"])
        result = op.apply(df)
        assert pd.isna(result["a"].iloc[0])
        # Column b not in subset — 'null' is NOT replaced
        assert result["b"].iloc[0] == "null"

    def test_subset_invalid_column_raises_key_error(self) -> None:
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            MissingTokenOperation(subset=["nonexistent"]).apply(df)

    def test_subset_multiple_columns(self) -> None:
        df = _make_df(a=["na", "real"], b=["null", "data"], c=["none", "value"])
        op = MissingTokenOperation(subset=["a", "b"])
        result = op.apply(df)
        assert pd.isna(result["a"].iloc[0])
        assert pd.isna(result["b"].iloc[0])
        assert result["c"].iloc[0] == "none"  # not in subset


# ---------------------------------------------------------------------------
# 9. OperationResult / apply_with_result (Requirement 3.5)
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_cells_converted_count_is_correct(self) -> None:
        df = _make_df(col=["na", "null", "real"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.rows_affected == 2
        assert res.details["cells_converted_to_missing"] == 2

    def test_before_after_shape_unchanged(self) -> None:
        df = _make_df(col=["na", "real"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.before_shape == (2, 1)
        assert res.after_shape == (2, 1)

    def test_operation_name_in_result(self) -> None:
        df = _make_df(col=["na"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.operation_name == "missing_token_normalization"

    def test_affected_columns_listed(self) -> None:
        df = _make_df(a=["na"], b=["null"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert set(res.affected_columns) == {"a", "b"}

    def test_dry_run_true_does_not_modify_input(self) -> None:
        df = _make_df(col=["na", "real"])
        original_col = df["col"].copy()
        result_df, res = MissingTokenOperation().apply_with_result(df, dry_run=True)
        # Input must not be mutated
        pd.testing.assert_series_equal(df["col"], original_col)
        # Result should still be the unmodified copy in dry_run mode
        assert res.dry_run is True

    def test_dry_run_false_returns_transformed_data(self) -> None:
        df = _make_df(col=["na", "real"])
        result_df, res = MissingTokenOperation().apply_with_result(df, dry_run=False)
        assert pd.isna(result_df["col"].iloc[0])
        assert res.dry_run is False

    def test_zero_count_when_no_tokens_present(self) -> None:
        df = _make_df(col=["hello", "world"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.rows_affected == 0
        assert res.details["cells_converted_to_missing"] == 0

    def test_apply_raises_on_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            MissingTokenOperation().apply_with_result([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. describe() output
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_required_keys(self) -> None:
        op = MissingTokenOperation(extra_tokens=["tbd"], subset=["col"])
        desc = op.describe()
        assert desc["name"] == "missing_token_normalization"
        assert "tbd" in desc["extra_tokens"]
        assert desc["subset"] == ["col"]
        assert isinstance(desc["token_count"], int)
        assert desc["token_count"] > 0

    def test_describe_subset_none_when_not_set(self) -> None:
        desc = MissingTokenOperation().describe()
        assert desc["subset"] is None

    def test_describe_extra_tokens_empty_when_none_provided(self) -> None:
        desc = MissingTokenOperation().describe()
        assert desc["extra_tokens"] == []

    def test_token_count_includes_defaults_and_extras(self) -> None:
        defaults_count = len(DEFAULT_MISSING_VALUE_TOKENS)
        op = MissingTokenOperation(extra_tokens=["custom1", "custom2"])
        desc = op.describe()
        # casefolded defaults + 2 new unique extras
        assert desc["token_count"] >= defaults_count

    def test_describe_does_not_mutate_internal_state(self) -> None:
        op = MissingTokenOperation(extra_tokens=["abc"])
        desc = op.describe()
        desc["extra_tokens"].append("injected")
        assert "injected" not in op.describe()["extra_tokens"]


# ---------------------------------------------------------------------------
# 11. Classification attributes
# ---------------------------------------------------------------------------


class TestClassificationAttributes:
    def test_is_chunk_safe(self) -> None:
        assert MissingTokenOperation.is_chunk_safe is True

    def test_is_inplace_safe_default(self) -> None:
        assert MissingTokenOperation.is_inplace_safe is False

    def test_name_attribute(self) -> None:
        assert MissingTokenOperation.name == "missing_token_normalization"


# ---------------------------------------------------------------------------
# 12. Registry presence
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_module_registry(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.contains("missing_token_normalization")

    def test_registry_returns_correct_class(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.get("missing_token_normalization") is MissingTokenOperation


# ---------------------------------------------------------------------------
# 13. Non-mutation guarantee
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_apply_does_not_mutate_input(self) -> None:
        df = _make_df(col=["na", "null", "real"])
        original = df.copy()
        MissingTokenOperation().apply(df)
        pd.testing.assert_frame_equal(df, original)

    def test_apply_with_result_does_not_mutate_input_live_run(self) -> None:
        df = _make_df(col=["na", "real"])
        original = df.copy()
        MissingTokenOperation().apply_with_result(df, dry_run=False)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# 14. Edge case: Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_apply_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        result = MissingTokenOperation().apply(df)
        assert result.shape == (0, 1)

    def test_apply_with_result_empty_dataframe_zero_conversions(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.before_shape == (0, 1)
        assert res.after_shape == (0, 1)
        assert res.details["cells_converted_to_missing"] == 0

    def test_apply_completely_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = MissingTokenOperation().apply(df)
        assert result.empty


# ---------------------------------------------------------------------------
# 15. Edge case: Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_token_converted(self) -> None:
        df = _make_df(col=["na"])
        result = MissingTokenOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_single_row_real_value_kept(self) -> None:
        df = _make_df(col=["hello"])
        result = MissingTokenOperation().apply(df)
        assert result["col"].iloc[0] == "hello"

    def test_single_row_apply_with_result_count(self) -> None:
        df = _make_df(col=["null"])
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.rows_affected == 1


# ---------------------------------------------------------------------------
# 16. Edge case: All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_column_no_additional_conversions(self) -> None:
        """If the column is already all-NaN, no tokens are converted."""
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        result = MissingTokenOperation().apply(df)
        assert result["col"].isna().all()

    def test_all_null_apply_with_result_records_zero(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None], dtype="object")})
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.details["cells_converted_to_missing"] == 0

    def test_all_default_token_strings_become_all_null(self) -> None:
        tokens = list(DEFAULT_MISSING_VALUE_TOKENS)
        df = _make_df(col=tokens)
        result = MissingTokenOperation().apply(df)
        assert result["col"].isna().all()


# ---------------------------------------------------------------------------
# 17. Edge case: Mixed-type column (object dtype with various Python types)
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_integer_objects_in_object_column_not_replaced(self) -> None:
        """Non-string objects in an object column are not matched as tokens."""
        df = pd.DataFrame({"col": pd.array([1, "na", 3.0, None], dtype="object")})
        result = MissingTokenOperation().apply(df)
        # Integer 1 and float 3.0 should be unchanged
        assert result["col"].iloc[0] == 1
        assert result["col"].iloc[2] == 3.0
        # "na" should be replaced
        assert pd.isna(result["col"].iloc[1])
        # None was already missing
        assert pd.isna(result["col"].iloc[3])

    def test_mixed_type_count_only_counts_string_tokens(self) -> None:
        df = pd.DataFrame({"col": pd.array([1, "na", "null", 3.0], dtype="object")})
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.details["cells_converted_to_missing"] == 2

    def test_boolean_objects_in_object_column_not_replaced(self) -> None:
        df = pd.DataFrame({"col": pd.array([True, "none", False], dtype="object")})
        result = MissingTokenOperation().apply(df)
        assert result["col"].iloc[0] is True
        assert result["col"].iloc[2] is False
        assert pd.isna(result["col"].iloc[1])


# ---------------------------------------------------------------------------
# 18. Edge case: Wide DataFrame (many columns)
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_dataframe_all_string_columns_processed(self) -> None:
        n_cols = 30
        data = {f"col{i}": ["na", "real"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = MissingTokenOperation().apply(df)
        for col in df.columns:
            assert pd.isna(result[col].iloc[0])
            assert result[col].iloc[1] == "real"

    def test_wide_dataframe_only_subset_processed(self) -> None:
        n_cols = 30
        data = {f"col{i}": ["na", "real"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        subset_cols = ["col0", "col1", "col2"]
        op = MissingTokenOperation(subset=subset_cols)
        result = op.apply(df)
        # Only subset columns are processed
        for col in subset_cols:
            assert pd.isna(result[col].iloc[0])
        # Non-subset columns remain unchanged
        for col in [f"col{i}" for i in range(3, n_cols)]:
            assert result[col].iloc[0] == "na"

    def test_wide_apply_with_result_correct_count(self) -> None:
        n_cols = 10
        data = {f"c{i}": ["na", "null", "real"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        _, res = MissingTokenOperation().apply_with_result(df)
        # 2 tokens per column × 10 columns
        assert res.details["cells_converted_to_missing"] == 2 * n_cols


# ---------------------------------------------------------------------------
# 19. Edge case: Tall DataFrame (many rows)
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_dataframe_tokens_converted(self) -> None:
        n_rows = 10_000
        tokens = list(DEFAULT_MISSING_VALUE_TOKENS)
        # Cycle through default tokens to fill the column
        col_values = [tokens[i % len(tokens)] for i in range(n_rows)]
        df = _make_df(col=col_values)
        result = MissingTokenOperation().apply(df)
        assert result["col"].isna().all()

    def test_tall_dataframe_mixed_tokens_and_real_values(self) -> None:
        n_rows = 5_000
        col_values = ["na" if i % 2 == 0 else "real" for i in range(n_rows)]
        df = _make_df(col=col_values)
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.details["cells_converted_to_missing"] == n_rows // 2

    def test_tall_dataframe_no_tokens_zero_conversions(self) -> None:
        n_rows = 5_000
        df = _make_df(col=["real_data"] * n_rows)
        _, res = MissingTokenOperation().apply_with_result(df)
        assert res.details["cells_converted_to_missing"] == 0


# ---------------------------------------------------------------------------
# 20. Idempotency (Requirement 3.6)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_applying_twice_leaves_data_unchanged_after_first(self) -> None:
        df = _make_df(col=["na", "null", "none", "real"])
        first = MissingTokenOperation().apply(df)
        second = MissingTokenOperation().apply(first)
        pd.testing.assert_frame_equal(first, second)

    def test_idempotency_with_extra_tokens(self) -> None:
        df = _make_df(col=["placeholder", "na", "real"])
        op = MissingTokenOperation(extra_tokens=["placeholder"])
        first = op.apply(df)
        second = op.apply(first)
        pd.testing.assert_frame_equal(first, second)

    def test_idempotency_zero_new_conversions_on_second_pass(self) -> None:
        df = _make_df(col=["na", "real"])
        first, _ = MissingTokenOperation().apply_with_result(df)
        _, res2 = MissingTokenOperation().apply_with_result(first)
        assert res2.details["cells_converted_to_missing"] == 0


# ===========================================================================
# Property tests (task 4.4)
# **Property 4: Missing-token normalization is idempotent**
# **Validates: Requirements 3.6**
# ===========================================================================


@st.composite
def _object_column_with_tokens(draw: st.DrawFn) -> list[object]:
    """
    Draw a list of mixed objects that includes:
    - A random selection of DEFAULT_MISSING_VALUE_TOKENS (pre-casefolded matches)
    - Real non-token strings
    - None / NaN (pre-existing missing)
    - Non-string scalars (int, float) to verify they are passed through
    """
    tokens = sorted(DEFAULT_MISSING_VALUE_TOKENS)
    token_value = st.sampled_from(tokens)
    real_string = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=3,
        max_size=8,
    ).filter(lambda s: s.strip().casefold() not in DEFAULT_MISSING_VALUE_TOKENS)
    scalar = st.one_of(st.integers(0, 100), st.floats(0.0, 1.0, allow_nan=False))
    item = st.one_of(st.none(), token_value, real_string, scalar)
    return draw(st.lists(item, min_size=1, max_size=30))


class TestMissingTokenIdempotency:
    """
    **Property 4: Missing-token normalization is idempotent**
    **Validates: Requirements 3.6**

    After the first application all sentinel tokens have been replaced with
    NaN. Applying the operation a second time must leave the data unchanged
    because there are no remaining string-token cells to convert.
    """

    @given(_object_column_with_tokens())
    @settings(max_examples=100)
    def test_default_tokens_idempotent(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = MissingTokenOperation()
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice, check_like=False)

    @given(
        _object_column_with_tokens(),
        st.frozensets(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                min_size=1,
                max_size=6,
            ),
            max_size=4,
        ),
    )
    @settings(max_examples=80)
    def test_extra_tokens_idempotent(
        self, values: list[object], extra: frozenset[str]
    ) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = MissingTokenOperation(extra_tokens=extra)
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice, check_like=False)

    @given(_object_column_with_tokens())
    @settings(max_examples=60)
    def test_second_application_converts_zero_additional_cells(
        self, values: list[object]
    ) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = MissingTokenOperation()

        first_df, first_res = op.apply_with_result(df)
        second_df, second_res = op.apply_with_result(first_df.copy())

        assert second_res.details["cells_converted_to_missing"] == 0, (
            f"Expected 0 cells converted on second pass, got "
            f"{second_res.details['cells_converted_to_missing']}"
        )

    @given(_object_column_with_tokens())
    @settings(max_examples=60)
    def test_apply_with_result_idempotent_metadata(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = MissingTokenOperation()

        first_df, _ = op.apply_with_result(df)
        _, second_res = op.apply_with_result(first_df.copy())

        # After the first pass, shapes are unchanged by definition (tokens →
        # NaN does not change the shape). On the second pass the shape must
        # also be unchanged and no cells should be newly affected.
        assert second_res.before_shape == second_res.after_shape
        assert second_res.rows_affected == 0
