"""
Tests for sanitizepy.cleaning.type_coercion.TypeCoercionOperation

This module contains:

1. Property tests (task 3.4)
   **Property 3: Type coercion is deterministic**
   **Validates: Requirements 2.3, 2.4, 17.1**

2. Unit tests (task 3.5) — Requirements: 2.1, 2.4, 2.5, 2.6
   Covers: successful conversion, mixed valid/invalid, strict failure,
   coerce-to-missing, already-correct dtype, missing values, multiple
   columns, affected count, and edge cases (empty, single-row, all-null,
   mixed-type, infinite, wide, tall).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.type_coercion import TypeCoercionOperation
from sanitizepy.exceptions import DataTypeConversionError

# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


@st.composite
def _mixed_string_col(draw: st.DrawFn) -> list[str | None]:
    """Draw a list mixing numeric strings and non-convertible strings."""
    convertible = st.integers(min_value=-100, max_value=100).map(str)
    non_convertible = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=5,
    )
    values: list[str | None] = draw(
        st.lists(
            st.one_of(st.none(), convertible, non_convertible),
            min_size=1,
            max_size=20,
        )
    )
    return values


# ===========================================================================
# Property tests (task 3.4)
# **Property 3: Type coercion is deterministic**
# **Validates: Requirements 2.3, 2.4, 17.1**
# ===========================================================================


class TestTypeCoercionDeterminism:
    """
    **Property 3: Type coercion is deterministic**
    **Validates: Requirements 2.3, 2.4, 17.1**
    """

    @given(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=80)
    def test_raise_policy_same_output_across_runs(self, values: list[float]) -> None:
        df1 = pd.DataFrame({"val": values})
        df2 = df1.copy()
        op = TypeCoercionOperation({"val": "float64"}, error_policy="raise")
        result1, meta1 = op.apply_with_result(df1)
        result2, meta2 = op.apply_with_result(df2)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False)
        assert meta1.before_shape == meta2.before_shape
        assert meta1.after_shape == meta2.after_shape
        assert meta1.rows_affected == meta2.rows_affected
        assert meta1.columns_affected == meta2.columns_affected

    @given(_mixed_string_col())
    @settings(max_examples=80)
    def test_coerce_policy_same_output_across_runs(
        self, values: list[str | None]
    ) -> None:
        df1 = pd.DataFrame({"val": values})
        df2 = df1.copy()
        op = TypeCoercionOperation({"val": "float64"}, error_policy="coerce")
        result1, meta1 = op.apply_with_result(df1)
        result2, meta2 = op.apply_with_result(df2)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False)
        assert meta1.before_shape == meta2.before_shape
        assert meta1.after_shape == meta2.after_shape
        assert meta1.rows_affected == meta2.rows_affected
        assert meta1.columns_affected == meta2.columns_affected

    @given(_mixed_string_col())
    @settings(max_examples=80)
    def test_coerce_policy_missing_positions_are_reproducible(
        self, values: list[str | None]
    ) -> None:
        df_a = pd.DataFrame({"val": values})
        df_b = df_a.copy()
        op = TypeCoercionOperation({"val": "float64"}, error_policy="coerce")
        out_a, _ = op.apply_with_result(df_a)
        out_b, _ = op.apply_with_result(df_b)
        assert out_a["val"].isna().tolist() == out_b["val"].isna().tolist()

    @given(st.data())
    @settings(max_examples=80)
    def test_raise_policy_failure_is_deterministic(self, data: st.DataObject) -> None:
        bad_value = data.draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                min_size=1,
                max_size=5,
            )
        )
        numeric_values = data.draw(
            st.lists(
                st.integers(min_value=-100, max_value=100).map(str),
                min_size=0,
                max_size=10,
            )
        )
        values = numeric_values + [bad_value]
        df1 = pd.DataFrame({"val": values})
        df2 = df1.copy()
        op = TypeCoercionOperation({"val": "float64"}, error_policy="raise")
        raised1 = False
        raised2 = False
        try:
            op.apply_with_result(df1)
        except Exception:
            raised1 = True
        try:
            op.apply_with_result(df2)
        except Exception:
            raised2 = True
        assert (
            raised1 == raised2
        ), "raise policy is non-deterministic: one run raised and the other did not"

    @given(st.data())
    @settings(max_examples=80)
    def test_multiple_columns_deterministic(self, data: st.DataObject) -> None:
        int_values = data.draw(
            st.lists(st.integers(min_value=0, max_value=1_000), min_size=1, max_size=20)
        )
        float_values = data.draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=1_000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=len(int_values),
                max_size=len(int_values),
            )
        )
        df1 = pd.DataFrame({"a": int_values, "b": float_values})
        df2 = df1.copy()
        op = TypeCoercionOperation(
            {"a": "float64", "b": "float32"}, error_policy="coerce"
        )
        result1, meta1 = op.apply_with_result(df1)
        result2, meta2 = op.apply_with_result(df2)
        pd.testing.assert_frame_equal(result1, result2, check_exact=False)
        assert meta1.before_shape == meta2.before_shape
        assert meta1.after_shape == meta2.after_shape
        assert meta1.rows_affected == meta2.rows_affected
        assert meta1.columns_affected == meta2.columns_affected

    @given(
        st.lists(
            st.one_of(
                st.none(),
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=60)
    def test_dry_run_and_live_run_produce_same_metadata(
        self, values: list[float | None]
    ) -> None:
        df = pd.DataFrame({"val": values})
        op = TypeCoercionOperation({"val": "float64"}, error_policy="coerce")
        _, meta_dry = op.apply_with_result(df.copy(), dry_run=True)
        _, meta_live = op.apply_with_result(df.copy(), dry_run=False)
        assert meta_dry.before_shape == meta_live.before_shape
        assert meta_dry.after_shape == meta_live.after_shape
        assert meta_dry.rows_affected == meta_live.rows_affected
        assert meta_dry.columns_affected == meta_live.columns_affected


# ===========================================================================
# Unit tests (task 3.5) — Requirements: 2.1, 2.4, 2.5, 2.6
# ===========================================================================


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_empty_target_dtypes_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            TypeCoercionOperation(target_dtypes={})

    def test_invalid_error_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="error_policy"):
            TypeCoercionOperation(
                target_dtypes={"a": "float64"},
                error_policy="ignore",  # type: ignore[arg-type]
            )

    def test_default_error_policy_is_raise(self) -> None:
        op = TypeCoercionOperation(target_dtypes={"a": "float64"})
        assert op.error_policy == "raise"

    def test_columns_attribute_matches_keys(self) -> None:
        op = TypeCoercionOperation(target_dtypes={"x": "int64", "y": "float64"})
        assert set(op.columns) == {"x", "y"}


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_required_keys(self) -> None:
        op = TypeCoercionOperation(
            target_dtypes={"age": "int64"}, error_policy="coerce"
        )
        desc = op.describe()
        assert desc["name"] == "type_coercion"
        assert desc["columns"] == ["age"]
        assert desc["target_dtypes"] == {"age": "int64"}
        assert desc["error_policy"] == "coerce"

    def test_describe_does_not_mutate_internal_state(self) -> None:
        op = TypeCoercionOperation(target_dtypes={"a": "float64"})
        desc = op.describe()
        desc["columns"].append("EXTRA")
        assert op.columns == ["a"]


# ---------------------------------------------------------------------------
# Classification attributes
# ---------------------------------------------------------------------------


class TestClassificationAttributes:
    def test_is_chunk_safe(self) -> None:
        op = TypeCoercionOperation(target_dtypes={"a": "float64"})
        assert op.is_chunk_safe is True

    def test_is_inplace_safe_default(self) -> None:
        op = TypeCoercionOperation(target_dtypes={"a": "float64"})
        assert op.is_inplace_safe is False


# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_module_registry(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.contains("type_coercion")

    def test_registry_returns_correct_class(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.get("type_coercion") is TypeCoercionOperation


# ---------------------------------------------------------------------------
# Successful conversion (Requirement 2.1, 2.3)
# ---------------------------------------------------------------------------


class TestSuccessfulConversion:
    def test_string_to_float64(self) -> None:
        df = _make_df(score=["1.5", "2.0", "3.7"])
        result = TypeCoercionOperation(target_dtypes={"score": "float64"}).apply(df)
        assert result["score"].dtype == np.float64
        assert list(result["score"]) == pytest.approx([1.5, 2.0, 3.7])

    def test_float_to_float64(self) -> None:
        df = _make_df(val=[1.0, 2.0, 3.0])
        result = TypeCoercionOperation(target_dtypes={"val": "float64"}).apply(df)
        assert result["val"].dtype == np.float64

    def test_float_to_int64(self) -> None:
        df = _make_df(val=[1.0, 2.0, 3.0])
        result = TypeCoercionOperation(target_dtypes={"val": "int64"}).apply(df)
        assert result["val"].dtype == np.int64

    def test_string_to_datetime(self) -> None:
        df = _make_df(ts=["2024-01-01", "2024-06-15", "2024-12-31"])
        result = TypeCoercionOperation(target_dtypes={"ts": "datetime64[ns]"}).apply(df)
        assert pd.api.types.is_datetime64_any_dtype(result["ts"])

    def test_does_not_mutate_input_dataframe(self) -> None:
        df = _make_df(x=["1", "2", "3"])
        original_dtype = df["x"].dtype
        TypeCoercionOperation(target_dtypes={"x": "float64"}).apply(df)
        assert df["x"].dtype == original_dtype


# ---------------------------------------------------------------------------
# Already-correct dtype — idempotent pass-through
# ---------------------------------------------------------------------------


class TestAlreadyCorrectDtype:
    def test_float_column_already_float64(self) -> None:
        df = _make_df(score=[1.1, 2.2, 3.3])
        result = TypeCoercionOperation(target_dtypes={"score": "float64"}).apply(df)
        assert result["score"].dtype == np.float64
        assert list(result["score"]) == pytest.approx([1.1, 2.2, 3.3])

    def test_int_column_already_int64(self) -> None:
        df = pd.DataFrame({"n": pd.array([1, 2, 3], dtype="int64")})
        result = TypeCoercionOperation(target_dtypes={"n": "int64"}).apply(df)
        assert result["n"].dtype == np.int64

    def test_result_records_zero_coerced_on_already_correct(self) -> None:
        df = _make_df(v=[1.0, 2.0, 3.0])
        _, res = TypeCoercionOperation(
            target_dtypes={"v": "float64"}
        ).apply_with_result(df)
        assert res.details["values_coerced_to_missing"] == 0


# ---------------------------------------------------------------------------
# Columns containing missing values (Requirement 2.6)
# ---------------------------------------------------------------------------


class TestColumnsWithMissingValues:
    def test_missing_values_preserved_under_raise_policy(self) -> None:
        df = _make_df(x=["1.0", None, "3.0"])
        result = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="raise"
        ).apply(df)
        assert pd.isna(result["x"].iloc[1])
        assert result["x"].iloc[0] == pytest.approx(1.0)
        assert result["x"].iloc[2] == pytest.approx(3.0)

    def test_missing_values_preserved_under_coerce_policy(self) -> None:
        df = _make_df(x=["1.0", None, "3.0"])
        result = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="coerce"
        ).apply(df)
        assert pd.isna(result["x"].iloc[1])

    def test_coerce_count_excludes_pre_existing_missing(self) -> None:
        """Pre-existing NaN must not inflate the coerced-to-missing count."""
        df = _make_df(x=[None, "bad", "2.0"])
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        # "bad" → NaN (newly missing); original None was already missing
        assert res.details["values_coerced_to_missing"] == 1


# ---------------------------------------------------------------------------
# Mixed valid/invalid values (Requirement 2.4)
# ---------------------------------------------------------------------------


class TestMixedValidInvalid:
    def test_coerce_policy_replaces_unconvertible_with_nan(self) -> None:
        df = _make_df(n=["1", "two", "3"])
        result = TypeCoercionOperation(
            target_dtypes={"n": "float64"}, error_policy="coerce"
        ).apply(df)
        assert result["n"].iloc[0] == pytest.approx(1.0)
        assert pd.isna(result["n"].iloc[1])
        assert result["n"].iloc[2] == pytest.approx(3.0)

    def test_coerce_policy_records_newly_missing_count(self) -> None:
        df = _make_df(n=["1", "two", "three"])
        _, res = TypeCoercionOperation(
            target_dtypes={"n": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["values_coerced_to_missing"] == 2

    def test_coerce_per_column_count_tracked_separately(self) -> None:
        df = _make_df(a=["1", "bad"], b=["ok", "2"])
        _, res = TypeCoercionOperation(
            target_dtypes={"a": "float64", "b": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["coerced_to_missing"].get("a", 0) == 1
        assert res.details["coerced_to_missing"].get("b", 0) == 1


# ---------------------------------------------------------------------------
# Strict policy failures (Requirement 2.4, 2.5)
# ---------------------------------------------------------------------------


class TestStrictPolicyFailures:
    def test_raises_data_type_conversion_error_on_bad_value(self) -> None:
        df = _make_df(n=["1", "abc", "3"])
        with pytest.raises(DataTypeConversionError):
            TypeCoercionOperation(
                target_dtypes={"n": "float64"}, error_policy="raise"
            ).apply(df)

    def test_error_message_names_column(self) -> None:
        df = _make_df(price=["10.0", "not_a_number"])
        with pytest.raises(DataTypeConversionError, match="price"):
            TypeCoercionOperation(
                target_dtypes={"price": "float64"}, error_policy="raise"
            ).apply(df)

    def test_raises_on_partial_datetime_failure(self) -> None:
        df = _make_df(ts=["2024-01-01", "not-a-date"])
        with pytest.raises(DataTypeConversionError):
            TypeCoercionOperation(
                target_dtypes={"ts": "datetime64[ns]"}, error_policy="raise"
            ).apply(df)

    def test_missing_column_raises_key_error(self) -> None:
        df = _make_df(a=[1, 2, 3])
        with pytest.raises(KeyError):
            TypeCoercionOperation(target_dtypes={"nonexistent": "float64"}).apply(df)


# ---------------------------------------------------------------------------
# apply_with_result and OperationResult contract (Requirement 2.5, 2.6)
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_result_before_after_shape(self) -> None:
        df = _make_df(x=["1", "2", "3"])
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df)
        assert res.before_shape == (3, 1)
        assert res.after_shape == (3, 1)

    def test_result_operation_name(self) -> None:
        df = _make_df(x=[1.0, 2.0])
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df)
        assert res.operation_name == "type_coercion"

    def test_result_affected_columns_listed(self) -> None:
        df = _make_df(a=["1", "2"], b=["3", "4"])
        _, res = TypeCoercionOperation(
            target_dtypes={"a": "float64", "b": "float64"}
        ).apply_with_result(df)
        assert set(res.affected_columns) == {"a", "b"}

    def test_result_dry_run_flag_false(self) -> None:
        df = _make_df(x=["1"])
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df, dry_run=False)
        assert res.dry_run is False

    def test_result_dry_run_flag_true(self) -> None:
        df = _make_df(x=["1"])
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df, dry_run=True)
        assert res.dry_run is True

    def test_dry_run_does_not_modify_original(self) -> None:
        df = _make_df(x=["1", "2"])
        original_dtype = df["x"].dtype
        result_df, _ = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df, dry_run=True)
        # Original frame dtype unchanged
        assert df["x"].dtype == original_dtype
        # Returned frame in dry_run is also unchanged
        assert result_df["x"].dtype == original_dtype

    def test_live_run_returns_transformed_data(self) -> None:
        df = _make_df(x=["1.5", "2.5"])
        result_df, _ = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df, dry_run=False)
        assert result_df["x"].dtype == np.float64

    def test_result_details_include_error_policy(self) -> None:
        df = _make_df(v=[1.0])
        _, res = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["error_policy"] == "coerce"

    def test_affected_count_is_zero_on_clean_data(self) -> None:
        df = _make_df(v=["1.0", "2.0", "3.0"])
        _, res = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["values_coerced_to_missing"] == 0


# ---------------------------------------------------------------------------
# Multiple columns (Requirement 2.1)
# ---------------------------------------------------------------------------


class TestMultipleColumns:
    def test_convert_two_columns_simultaneously(self) -> None:
        df = _make_df(a=["1", "2"], b=["3.5", "4.5"])
        result = TypeCoercionOperation(
            target_dtypes={"a": "float64", "b": "float64"}
        ).apply(df)
        assert result["a"].dtype == np.float64
        assert result["b"].dtype == np.float64

    def test_independent_conversion_per_column_coerce(self) -> None:
        df = _make_df(a=["1", "bad"], b=["ok", "2"])
        result = TypeCoercionOperation(
            target_dtypes={"a": "float64", "b": "float64"}, error_policy="coerce"
        ).apply(df)
        assert pd.isna(result["a"].iloc[1])
        assert pd.isna(result["b"].iloc[0])

    def test_convert_numeric_and_datetime(self) -> None:
        df = _make_df(n=["1", "2"], ts=["2024-01-01", "2024-06-01"])
        result = TypeCoercionOperation(
            target_dtypes={"n": "float64", "ts": "datetime64[ns]"}
        ).apply(df)
        assert result["n"].dtype == np.float64
        assert pd.api.types.is_datetime64_any_dtype(result["ts"])

    def test_untargeted_columns_unchanged(self) -> None:
        df = _make_df(a=["1", "2"], b=["x", "y"])
        result = TypeCoercionOperation(target_dtypes={"a": "float64"}).apply(df)
        assert list(result["b"]) == ["x", "y"]

    def test_affected_count_sums_across_columns(self) -> None:
        df = _make_df(a=["bad1", "bad2"], b=["ok", "2"])
        _, res = TypeCoercionOperation(
            target_dtypes={"a": "float64", "b": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        # 2 bad in a, 1 bad in b = 3 total
        assert res.details["values_coerced_to_missing"] == 3


# ===========================================================================
# Edge cases
# ===========================================================================


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_apply_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame({"x": pd.Series([], dtype="object")})
        result = TypeCoercionOperation(target_dtypes={"x": "float64"}).apply(df)
        assert result.empty
        assert result["x"].dtype == np.float64

    def test_apply_with_result_empty_dataframe(self) -> None:
        df = pd.DataFrame({"x": pd.Series([], dtype="object")})
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}
        ).apply_with_result(df)
        assert res.before_shape == (0, 1)
        assert res.after_shape == (0, 1)
        assert res.details["values_coerced_to_missing"] == 0

    def test_empty_dataframe_missing_column_raises(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(KeyError):
            TypeCoercionOperation(target_dtypes={"x": "float64"}).apply(df)


# ---------------------------------------------------------------------------
# Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_valid_row(self) -> None:
        df = _make_df(v=["42.0"])
        result = TypeCoercionOperation(target_dtypes={"v": "float64"}).apply(df)
        assert result["v"].iloc[0] == pytest.approx(42.0)

    def test_single_invalid_row_raise_policy(self) -> None:
        df = _make_df(v=["not_a_number"])
        with pytest.raises(DataTypeConversionError):
            TypeCoercionOperation(
                target_dtypes={"v": "float64"}, error_policy="raise"
            ).apply(df)

    def test_single_invalid_row_coerce_policy(self) -> None:
        df = _make_df(v=["not_a_number"])
        result = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply(df)
        assert pd.isna(result["v"].iloc[0])

    def test_single_row_affected_count(self) -> None:
        df = _make_df(v=["bad"])
        _, res = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["values_coerced_to_missing"] == 1


# ---------------------------------------------------------------------------
# All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_column_coerce(self) -> None:
        df = pd.DataFrame({"x": pd.Series([None, None, None], dtype="object")})
        result = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="coerce"
        ).apply(df)
        assert result["x"].isna().all()

    def test_all_null_coerce_records_zero_newly_missing(self) -> None:
        """All values are already missing, so coerced_to_missing must be 0."""
        df = pd.DataFrame({"x": pd.Series([None, None], dtype="object")})
        _, res = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        assert res.details["values_coerced_to_missing"] == 0

    def test_all_null_column_raise_policy_does_not_raise(self) -> None:
        """Pre-existing NaN values should not trigger the strict policy."""
        df = pd.DataFrame({"x": pd.Series([None, None], dtype="object")})
        result = TypeCoercionOperation(
            target_dtypes={"x": "float64"}, error_policy="raise"
        ).apply(df)
        assert result["x"].isna().all()


# ---------------------------------------------------------------------------
# Mixed-type column (object dtype)
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_mixed_int_and_string_coerce(self) -> None:
        df = pd.DataFrame({"v": pd.Series([1, "two", 3], dtype="object")})
        result = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply(df)
        assert result["v"].iloc[0] == pytest.approx(1.0)
        assert pd.isna(result["v"].iloc[1])
        assert result["v"].iloc[2] == pytest.approx(3.0)

    def test_mixed_type_strict_raises(self) -> None:
        df = pd.DataFrame({"v": pd.Series([1, "two", 3], dtype="object")})
        with pytest.raises(DataTypeConversionError):
            TypeCoercionOperation(
                target_dtypes={"v": "float64"}, error_policy="raise"
            ).apply(df)


# ---------------------------------------------------------------------------
# Infinite values
# ---------------------------------------------------------------------------


class TestEdgeCaseInfinite:
    def test_inf_values_preserved_on_float_target(self) -> None:
        df = _make_df(v=[1.0, math.inf, -math.inf])
        result = TypeCoercionOperation(target_dtypes={"v": "float64"}).apply(df)
        assert math.isinf(result["v"].iloc[1])
        assert math.isinf(result["v"].iloc[2])

    def test_inf_string_coerced_on_float_target(self) -> None:
        """'inf' as a string should coerce to float inf (pandas behavior)."""
        df = _make_df(v=["1.0", "inf", "-inf"])
        result = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply(df)
        assert math.isinf(result["v"].iloc[1])
        assert math.isinf(result["v"].iloc[2])


# ---------------------------------------------------------------------------
# Wide DataFrame (many columns)
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_dataframe_only_targeted_columns_converted(self) -> None:
        n_cols = 50
        data = {f"col_{i}": ["1", "2", "3"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        target = {"col_0": "float64", "col_49": "float64"}
        original_untargeted_dtype = df["col_1"].dtype
        result = TypeCoercionOperation(target_dtypes=target).apply(df)
        assert result["col_0"].dtype == np.float64
        assert result["col_49"].dtype == np.float64
        # Untargeted columns retain their original dtype
        assert result["col_1"].dtype == original_untargeted_dtype

    def test_wide_dataframe_result_affected_columns(self) -> None:
        data = {f"c{i}": [str(i)] for i in range(40)}
        df = pd.DataFrame(data)
        target = {f"c{i}": "float64" for i in range(0, 40, 2)}
        _, res = TypeCoercionOperation(target_dtypes=target).apply_with_result(df)
        assert set(res.affected_columns) == set(target.keys())


# ---------------------------------------------------------------------------
# Tall DataFrame (many rows)
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_dataframe_all_rows_converted(self) -> None:
        n_rows = 10_000
        df = pd.DataFrame({"v": [str(i) for i in range(n_rows)]})
        result = TypeCoercionOperation(target_dtypes={"v": "float64"}).apply(df)
        assert result["v"].dtype == np.float64
        assert result["v"].notna().all()

    def test_tall_dataframe_coerce_count(self) -> None:
        n_rows = 5_000
        values = [str(i) if i % 100 != 0 else "bad" for i in range(n_rows)]
        df = pd.DataFrame({"v": values})
        _, res = TypeCoercionOperation(
            target_dtypes={"v": "float64"}, error_policy="coerce"
        ).apply_with_result(df)
        expected_bad = sum(1 for i in range(n_rows) if i % 100 == 0)
        assert res.details["values_coerced_to_missing"] == expected_bad

    def test_tall_dataframe_raise_policy_first_bad_value(self) -> None:
        n_rows = 3_000
        values = [str(i) for i in range(n_rows)]
        values[1500] = "INVALID"
        df = pd.DataFrame({"v": values})
        with pytest.raises(DataTypeConversionError, match="INVALID"):
            TypeCoercionOperation(
                target_dtypes={"v": "float64"}, error_policy="raise"
            ).apply(df)
