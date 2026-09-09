"""
Tests for sanitizepy.cleaning.text_normalization.TextNormalizationOperation

This module contains:

1. Property tests (task 5.2)
   **Property 5: Text normalization is idempotent**
   **Validates: Requirements 4.7**

2. Unit tests (task 5.3) — Requirements: 4.2, 4.3, 4.4, 4.5, 4.6
   Covers: leading/trailing/internal whitespace, non-breaking spaces,
   NFC, NFKC, lower/upper/title, empty-after-normalization, missing values.
   Edge cases: empty, single-row, all-null, mixed-type, infinite, wide, tall.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.text_normalization import TextNormalizationOperation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ===========================================================================
# Property tests (task 5.2)
# **Property 5: Text normalization is idempotent**
# **Validates: Requirements 4.7**
# ===========================================================================


@st.composite
def _text_column(draw: st.DrawFn) -> list[object]:
    """
    Draw a column with a mix of regular strings, strings with extra
    whitespace, None/NaN, and non-string scalars.
    """
    real_str = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
            whitelist_characters="\u00a0",
        ),
        min_size=0,
        max_size=20,
    )
    scalar = st.one_of(st.none(), st.integers(0, 100))
    item = st.one_of(real_str, scalar)
    return draw(st.lists(item, min_size=1, max_size=30))


class TestTextNormalizationIdempotency:
    """
    **Property 5: Text normalization is idempotent**
    **Validates: Requirements 4.7**

    Applying the operation twice with the same configuration must produce
    identical output to a single application.
    """

    @given(_text_column())
    @settings(max_examples=100)
    def test_nfc_whitespace_lower_idempotent(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="NFC", normalize_whitespace=True, case="lower"
        )
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice)

    @given(_text_column())
    @settings(max_examples=100)
    def test_nfkc_whitespace_upper_idempotent(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="NFKC", normalize_whitespace=True, case="upper"
        )
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice)

    @given(_text_column())
    @settings(max_examples=80)
    def test_whitespace_only_idempotent(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="none", normalize_whitespace=True, case="none"
        )
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice)

    @given(_text_column())
    @settings(max_examples=80)
    def test_case_lower_only_idempotent(self, values: list[object]) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="none", normalize_whitespace=False, case="lower"
        )
        once = op.apply(df)
        twice = op.apply(once.copy())
        pd.testing.assert_frame_equal(once, twice)

    @given(_text_column())
    @settings(max_examples=80)
    def test_second_application_modifies_zero_values(
        self, values: list[object]
    ) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="NFC", normalize_whitespace=True, case="lower"
        )
        first_df, _ = op.apply_with_result(df)
        _, second_res = op.apply_with_result(first_df.copy())

        assert second_res.details["values_modified"] == 0, (
            f"Expected 0 modifications on second pass, "
            f"got {second_res.details['values_modified']}"
        )

    @given(_text_column())
    @settings(max_examples=60)
    def test_idempotency_with_apply_with_result_metadata(
        self, values: list[object]
    ) -> None:
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        op = TextNormalizationOperation(
            unicode_form="NFC", normalize_whitespace=True, case="lower"
        )
        first_df, first_res = op.apply_with_result(df)
        second_df, second_res = op.apply_with_result(first_df.copy())

        # Shapes must be equal across both passes.
        assert first_res.after_shape == second_res.before_shape
        assert second_res.before_shape == second_res.after_shape

        # On the second pass, no additional values should be modified.
        assert second_res.rows_affected == 0


# ===========================================================================
# Unit tests (task 5.3)
# Requirements: 4.2, 4.3, 4.4, 4.5, 4.6
# ===========================================================================


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_unicode_form_raises(self) -> None:
        with pytest.raises(ValueError, match="unicode_form"):
            TextNormalizationOperation(unicode_form="INVALID")  # type: ignore[arg-type]

    def test_invalid_case_raises(self) -> None:
        with pytest.raises(ValueError, match="case"):
            TextNormalizationOperation(case="sentence")  # type: ignore[arg-type]

    def test_defaults_are_sensible(self) -> None:
        op = TextNormalizationOperation()
        assert op.unicode_form == "NFC"
        assert op.normalize_whitespace is True
        assert op.case == "none"
        assert op.subset is None

    def test_subset_stored(self) -> None:
        op = TextNormalizationOperation(subset=["a", "b"])
        assert op.subset == ["a", "b"]


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_all_keys(self) -> None:
        op = TextNormalizationOperation(
            subset=["text"],
            unicode_form="NFKC",
            normalize_whitespace=True,
            case="lower",
        )
        desc = op.describe()
        assert desc["name"] == "text_normalization"
        assert desc["subset"] == ["text"]
        assert desc["unicode_form"] == "NFKC"
        assert desc["normalize_whitespace"] is True
        assert desc["case"] == "lower"

    def test_describe_subset_none_when_unset(self) -> None:
        op = TextNormalizationOperation()
        assert op.describe()["subset"] is None


# ---------------------------------------------------------------------------
# Classification attributes
# ---------------------------------------------------------------------------


class TestClassificationAttributes:
    def test_is_chunk_safe(self) -> None:
        assert TextNormalizationOperation().is_chunk_safe is True

    def test_is_inplace_safe_default_false(self) -> None:
        assert TextNormalizationOperation().is_inplace_safe is False

    def test_name_attribute(self) -> None:
        assert TextNormalizationOperation.name == "text_normalization"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_module_registry(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.contains("text_normalization")

    def test_registry_returns_correct_class(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.get("text_normalization") is TextNormalizationOperation


# ---------------------------------------------------------------------------
# Whitespace normalization (Requirement 4.3)
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    def test_leading_whitespace_stripped(self) -> None:
        df = _make_df(col=["   hello"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello"

    def test_trailing_whitespace_stripped(self) -> None:
        df = _make_df(col=["hello   "])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello"

    def test_both_ends_stripped(self) -> None:
        df = _make_df(col=["  hello  "])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello"

    def test_internal_whitespace_run_collapsed(self) -> None:
        df = _make_df(col=["hello   world"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello world"

    def test_multiple_internal_runs_collapsed(self) -> None:
        df = _make_df(col=["a   b   c"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "a b c"

    def test_non_breaking_space_replaced(self) -> None:
        # U+00A0 non-breaking space should be converted to regular space.
        df = _make_df(col=["hello\u00a0world"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello world"

    def test_tab_and_newline_collapsed(self) -> None:
        df = _make_df(col=["hello\t\nworld"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hello world"

    def test_normalize_whitespace_false_leaves_spaces(self) -> None:
        df = _make_df(col=["  hello  "])
        result = TextNormalizationOperation(normalize_whitespace=False).apply(df)
        assert result["col"].iloc[0] == "  hello  "

    def test_non_breaking_space_not_replaced_when_disabled(self) -> None:
        df = _make_df(col=["hello\u00a0world"])
        result = TextNormalizationOperation(normalize_whitespace=False).apply(df)
        assert "\u00a0" in result["col"].iloc[0]


# ---------------------------------------------------------------------------
# Empty-after-normalization → missing (Requirement 4.5)
# ---------------------------------------------------------------------------


class TestEmptyAfterNormalization:
    def test_whitespace_only_string_becomes_missing(self) -> None:
        df = _make_df(col=["   "])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_tab_only_string_becomes_missing(self) -> None:
        df = _make_df(col=["\t"])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_empty_string_with_whitespace_on_becomes_missing(self) -> None:
        df = _make_df(col=[""])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_empty_string_without_whitespace_becomes_missing(self) -> None:
        # Even without whitespace normalization, an already-empty string
        # should be treated as missing.
        df = _make_df(col=[""])
        result = TextNormalizationOperation(normalize_whitespace=False).apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_empty_after_normalization_counted_in_result(self) -> None:
        df = _make_df(col=["   ", "hello", "   "])
        _, res = TextNormalizationOperation(
            normalize_whitespace=True
        ).apply_with_result(df)
        assert res.details["values_modified"] == 2


# ---------------------------------------------------------------------------
# Unicode normalization (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestUnicodeNormalization:
    def test_nfc_applied(self) -> None:
        # é can be represented as NFC (U+00E9) or NFD (e + combining accent).
        nfd_e = "e\u0301"  # NFD form: e + combining acute accent
        nfc_e = "\u00e9"  # NFC form: precomposed é
        df = _make_df(col=[nfd_e])
        result = TextNormalizationOperation(unicode_form="NFC").apply(df)
        assert result["col"].iloc[0] == nfc_e

    def test_nfkc_applied(self) -> None:
        # U+2126 OHM SIGN → U+03A9 GREEK CAPITAL LETTER OMEGA under NFKC.
        ohm_sign = "\u2126"
        omega = unicodedata.normalize("NFKC", ohm_sign)
        df = _make_df(col=[ohm_sign])
        result = TextNormalizationOperation(unicode_form="NFKC").apply(df)
        assert result["col"].iloc[0] == omega

    def test_nfkc_decomposes_ligature(self) -> None:
        # ﬁ (U+FB01 LATIN SMALL LIGATURE FI) decomposes to "fi" under NFKC.
        df = _make_df(col=["\ufb01"])
        result = TextNormalizationOperation(unicode_form="NFKC").apply(df)
        assert result["col"].iloc[0] == "fi"

    def test_unicode_none_skips_normalization(self) -> None:
        nfd_e = "e\u0301"
        df = _make_df(col=[nfd_e])
        result = TextNormalizationOperation(unicode_form="none").apply(df)
        # Value should remain in NFD form (unchanged).
        assert result["col"].iloc[0] == nfd_e

    def test_nfc_already_normalized_unchanged(self) -> None:
        # Already-NFC string should be returned as-is.
        s = "café"
        df = _make_df(col=[s])
        result = TextNormalizationOperation(unicode_form="NFC").apply(df)
        assert result["col"].iloc[0] == s


# ---------------------------------------------------------------------------
# Case normalization (Requirement 4.4)
# ---------------------------------------------------------------------------


class TestCaseNormalization:
    def test_lower_case(self) -> None:
        df = _make_df(col=["Hello World", "FOO", "Bar"])
        result = TextNormalizationOperation(case="lower").apply(df)
        assert list(result["col"]) == ["hello world", "foo", "bar"]

    def test_upper_case(self) -> None:
        df = _make_df(col=["Hello World", "foo", "bar"])
        result = TextNormalizationOperation(case="upper").apply(df)
        assert list(result["col"]) == ["HELLO WORLD", "FOO", "BAR"]

    def test_title_case(self) -> None:
        df = _make_df(col=["hello world", "FOO BAR"])
        result = TextNormalizationOperation(case="title").apply(df)
        assert list(result["col"]) == ["Hello World", "Foo Bar"]

    def test_case_none_leaves_casing_unchanged(self) -> None:
        df = _make_df(col=["Hello", "WORLD", "foo"])
        result = TextNormalizationOperation(case="none").apply(df)
        assert list(result["col"]) == ["Hello", "WORLD", "foo"]

    def test_lower_with_already_lowercase_unchanged(self) -> None:
        df = _make_df(col=["hello", "world"])
        result = TextNormalizationOperation(case="lower").apply(df)
        assert list(result["col"]) == ["hello", "world"]


# ---------------------------------------------------------------------------
# Missing values pass through (Requirement 4.6)
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_nan_passes_through(self) -> None:
        df = _make_df(col=[np.nan, "hello", np.nan])
        result = TextNormalizationOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "hello"
        assert pd.isna(result["col"].iloc[2])

    def test_none_passes_through(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, "hello", None], dtype="object")})
        result = TextNormalizationOperation().apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "hello"
        assert pd.isna(result["col"].iloc[2])

    def test_missing_not_counted_as_modified(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, np.nan], dtype="object")})
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert res.details["values_modified"] == 0


# ---------------------------------------------------------------------------
# apply_with_result and OperationResult contract
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_result_before_after_shape_preserved(self) -> None:
        df = _make_df(col=["  hello  ", "world"])
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert res.before_shape == (2, 1)
        assert res.after_shape == (2, 1)

    def test_result_operation_name(self) -> None:
        df = _make_df(col=["hello"])
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert res.operation_name == "text_normalization"

    def test_result_affected_columns_listed(self) -> None:
        df = _make_df(a=["hello"], b=["world"])
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert set(res.affected_columns) == {"a", "b"}

    def test_result_values_modified_count(self) -> None:
        df = _make_df(col=["  hello  ", "clean", "  again  "])
        _, res = TextNormalizationOperation(
            normalize_whitespace=True
        ).apply_with_result(df)
        # "  hello  " → "hello" (modified) and "  again  " → "again" (modified)
        assert res.details["values_modified"] == 2

    def test_dry_run_does_not_modify_input(self) -> None:
        df = _make_df(col=["  hello  "])
        original_val = df["col"].iloc[0]
        result_df, _ = TextNormalizationOperation().apply_with_result(df, dry_run=True)
        # Returned df in dry_run should be a copy with original values.
        assert result_df["col"].iloc[0] == original_val
        # Original DataFrame is unchanged.
        assert df["col"].iloc[0] == original_val

    def test_live_run_returns_transformed_data(self) -> None:
        df = _make_df(col=["  HELLO  "])
        result_df, _ = TextNormalizationOperation(
            normalize_whitespace=True, case="lower"
        ).apply_with_result(df, dry_run=False)
        assert result_df["col"].iloc[0] == "hello"

    def test_result_dry_run_flag(self) -> None:
        df = _make_df(col=["hello"])
        _, res_dry = TextNormalizationOperation().apply_with_result(
            df.copy(), dry_run=True
        )
        _, res_live = TextNormalizationOperation().apply_with_result(
            df.copy(), dry_run=False
        )
        assert res_dry.dry_run is True
        assert res_live.dry_run is False

    def test_input_not_mutated(self) -> None:
        df = _make_df(col=["  hello  "])
        original_val = df["col"].iloc[0]
        TextNormalizationOperation().apply(df)
        assert df["col"].iloc[0] == original_val


# ---------------------------------------------------------------------------
# Subset column restriction (Requirement 4.1)
# ---------------------------------------------------------------------------


class TestSubset:
    def test_subset_restricts_to_specified_columns(self) -> None:
        df = _make_df(a=["  hello  "], b=["  world  "])
        op = TextNormalizationOperation(subset=["a"])
        result = op.apply(df)
        assert result["a"].iloc[0] == "hello"
        assert result["b"].iloc[0] == "  world  "  # untouched

    def test_subset_missing_column_raises_key_error(self) -> None:
        df = _make_df(a=["hello"])
        with pytest.raises(KeyError):
            TextNormalizationOperation(subset=["nonexistent"]).apply(df)

    def test_subset_only_those_columns_in_affected_columns(self) -> None:
        df = _make_df(a=["hello"], b=["world"])
        _, res = TextNormalizationOperation(subset=["a"]).apply_with_result(df)
        assert res.affected_columns == ["a"]


# ---------------------------------------------------------------------------
# Non-string columns are not modified
# ---------------------------------------------------------------------------


class TestNonStringColumns:
    def test_integer_column_untouched(self) -> None:
        df = _make_df(text=["  hello  "], n=[1, 2] if False else [42])
        df = pd.DataFrame({"text": ["  hello  "], "n": [42]})
        result = TextNormalizationOperation().apply(df)
        assert result["n"].iloc[0] == 42

    def test_float_column_untouched(self) -> None:
        df = pd.DataFrame({"text": ["  hello  "], "f": [3.14]})
        result = TextNormalizationOperation().apply(df)
        assert result["f"].iloc[0] == pytest.approx(3.14)

    def test_datetime_column_untouched(self) -> None:
        dates = pd.to_datetime(["2024-01-01"])
        df = pd.DataFrame({"text": ["  hello  "], "d": dates})
        result = TextNormalizationOperation().apply(df)
        pd.testing.assert_series_equal(result["d"], df["d"])


# ===========================================================================
# Edge cases
# ===========================================================================


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_apply_empty_object_column_returns_empty(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        result = TextNormalizationOperation().apply(df)
        assert result.shape == (0, 1)

    def test_apply_with_result_empty_zero_modified(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert res.before_shape == (0, 1)
        assert res.after_shape == (0, 1)
        assert res.details["values_modified"] == 0

    def test_completely_empty_dataframe_no_error(self) -> None:
        df = pd.DataFrame()
        result = TextNormalizationOperation().apply(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_whitespace_trimmed(self) -> None:
        df = _make_df(col=["  hi  "])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert result["col"].iloc[0] == "hi"

    def test_single_row_all_whitespace_becomes_missing(self) -> None:
        df = _make_df(col=["     "])
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_single_row_real_value_lowercased(self) -> None:
        df = _make_df(col=["HELLO"])
        result = TextNormalizationOperation(case="lower").apply(df)
        assert result["col"].iloc[0] == "hello"


# ---------------------------------------------------------------------------
# All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_column_no_modification(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        result = TextNormalizationOperation().apply(df)
        assert result["col"].isna().all()

    def test_all_null_apply_with_result_zero_modified(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None], dtype="object")})
        _, res = TextNormalizationOperation().apply_with_result(df)
        assert res.details["values_modified"] == 0


# ---------------------------------------------------------------------------
# Mixed-type column (object dtype with non-string values)
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_non_string_values_in_object_column_not_modified(self) -> None:
        df = pd.DataFrame(
            {"col": pd.array([1, "  hello  ", 3.14, None], dtype="object")}
        )
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        # Non-string scalars pass through unchanged.
        assert result["col"].iloc[0] == 1
        assert result["col"].iloc[1] == "hello"
        assert result["col"].iloc[2] == pytest.approx(3.14)
        assert pd.isna(result["col"].iloc[3])

    def test_only_string_cells_counted_as_modified(self) -> None:
        df = pd.DataFrame(
            {"col": pd.array([1, "  hello  ", "  world  "], dtype="object")}
        )
        _, res = TextNormalizationOperation(
            normalize_whitespace=True
        ).apply_with_result(df)
        # Only the two string cells "  hello  " and "  world  " are modified.
        assert res.details["values_modified"] == 2


# ---------------------------------------------------------------------------
# Wide DataFrame (many columns)
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_all_string_columns_normalized(self) -> None:
        n_cols = 30
        data = {f"col{i}": ["  hello  ", "world"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = TextNormalizationOperation(normalize_whitespace=True).apply(df)
        for col in df.columns:
            assert result[col].iloc[0] == "hello"
            assert result[col].iloc[1] == "world"

    def test_wide_subset_only_specified_columns_normalized(self) -> None:
        n_cols = 20
        data = {f"col{i}": ["  hello  "] for i in range(n_cols)}
        df = pd.DataFrame(data)
        subset_cols = ["col0", "col1"]
        op = TextNormalizationOperation(subset=subset_cols, normalize_whitespace=True)
        result = op.apply(df)
        for col in subset_cols:
            assert result[col].iloc[0] == "hello"
        for col in [f"col{i}" for i in range(2, n_cols)]:
            assert result[col].iloc[0] == "  hello  "  # unchanged


# ---------------------------------------------------------------------------
# Tall DataFrame (many rows)
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_dataframe_all_rows_normalized(self) -> None:
        n_rows = 10_000
        df = pd.DataFrame({"col": ["  HELLO  "] * n_rows})
        result = TextNormalizationOperation(
            normalize_whitespace=True, case="lower"
        ).apply(df)
        assert (result["col"] == "hello").all()

    def test_tall_dataframe_values_modified_count(self) -> None:
        n_rows = 5_000
        # Half with whitespace (modified), half already clean.
        values = ["  hi  "] * (n_rows // 2) + ["hi"] * (n_rows // 2)
        df = pd.DataFrame({"col": values})
        _, res = TextNormalizationOperation(
            normalize_whitespace=True
        ).apply_with_result(df)
        assert res.details["values_modified"] == n_rows // 2
