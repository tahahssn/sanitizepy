"""
Tests for sanitizepy.cleaning.operations
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)
from sanitizepy.exceptions import DataValidationError


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# Property 1: Fill strategies are semantically correct
# Validates: Requirements 1.2, 1.3, 1.4, 1.5
# ---------------------------------------------------------------------------

# Hypothesis strategies for generating numeric lists that contain at least one
# non-missing value so that median/mean/mode have a well-defined statistic to
# compute.

# A finite float that is safe to use as a data value (no NaN; inf is fine
# because it participates in pandas statistics as-is).
_finite_or_inf_float = st.one_of(
    st.floats(
        min_value=-1e9,
        max_value=1e9,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.just(float("inf")),
    st.just(float("-inf")),
)

# A list that has at least one real float (not NaN/None) so the stat can be
# computed, plus optional None sentinels mixed in.
_numeric_with_some_missing = st.lists(
    st.one_of(_finite_or_inf_float, st.none()),
    min_size=2,
    max_size=30,
).filter(lambda xs: any(x is not None for x in xs))

# A list of small integers (easier for mode tie-breaking assertions) that
# contains at least one non-None value.
_int_with_some_missing = st.lists(
    st.one_of(st.integers(min_value=0, max_value=9), st.none()),
    min_size=2,
    max_size=30,
).filter(lambda xs: any(x is not None for x in xs))

# A list of short strings (for mode on text columns) with at least one
# non-None entry.
_str_with_some_missing = st.lists(
    st.one_of(st.text(min_size=1, max_size=5), st.none()),
    min_size=2,
    max_size=30,
).filter(lambda xs: any(x is not None for x in xs))


def _expected_mode(series: pd.Series) -> object:
    """Return the deterministic mode: first value in sorted order among ties."""
    non_missing = series.dropna()
    if non_missing.empty:
        return None
    counts = non_missing.value_counts()
    top_count = counts.iloc[0]
    tied = [v for v, c in counts.items() if c == top_count]
    return sorted(tied)[0]  # type: ignore[return-value]


class TestFillStrategyCorrectness:
    """
    **Property 1: Fill strategies are semantically correct**

    For any numeric column with missing values:
      - median strategy fills missing cells with the column median over
        non-missing values (Requirement 1.2).
      - mean strategy fills missing cells with the column mean over
        non-missing values (Requirement 1.3).

    For any supported column:
      - mode strategy fills missing cells with the first sorted value
        among tied modes of non-missing values (Requirement 1.4).

    For any configured constant:
      - constant strategy fills every missing cell with exactly that
        value (Requirement 1.5).
    """

    # ------------------------------------------------------------------
    # Req 1.3 – mean strategy is semantically correct for numeric columns
    # ------------------------------------------------------------------

    @given(values=_numeric_with_some_missing)
    @settings(max_examples=200)
    def test_mean_strategy_fills_with_column_mean(self, values: list) -> None:
        """mean fill matches pandas mean over non-missing values."""
        df = pd.DataFrame({"col": pd.array(values, dtype="Float64")})
        non_missing = df["col"].dropna()
        raw_mean = non_missing.mean()

        # If the mean itself is NA (e.g. [inf, -inf] → NaN), the fill is a
        # no-op and there is nothing else to verify about the fill value.
        if pd.isna(raw_mean):
            return

        expected_mean = float(raw_mean)  # type: ignore[arg-type]

        result = FillMissing(strategy="mean").apply(df)

        # Every previously-missing cell must now hold exactly the expected mean.
        original_missing_mask = df["col"].isna()
        if original_missing_mask.any() and not math.isnan(expected_mean):
            filled_values = result.loc[original_missing_mask, "col"]
            for v in filled_values:
                assert (
                    pytest.approx(float(v), rel=1e-9) == expected_mean
                ), f"Expected mean={expected_mean} but got {v}"

        # No new missing values are introduced.
        assert result["col"].isna().sum() <= df["col"].isna().sum()

    # ------------------------------------------------------------------
    # Req 1.2 – median strategy is semantically correct for numeric columns
    # ------------------------------------------------------------------

    @given(values=_numeric_with_some_missing)
    @settings(max_examples=200)
    def test_median_strategy_fills_with_column_median(self, values: list) -> None:
        """median fill matches pandas median over non-missing values."""
        df = pd.DataFrame({"col": pd.array(values, dtype="Float64")})
        non_missing = df["col"].dropna()
        raw_median = non_missing.median()

        # Same guard as for mean: [inf, -inf] etc. produce NA median.
        if pd.isna(raw_median):
            return

        expected_median = float(raw_median)  # type: ignore[arg-type]

        result = FillMissing(strategy="median").apply(df)

        original_missing_mask = df["col"].isna()
        if original_missing_mask.any() and not math.isnan(expected_median):
            filled_values = result.loc[original_missing_mask, "col"]
            for v in filled_values:
                assert (
                    pytest.approx(float(v), rel=1e-9) == expected_median
                ), f"Expected median={expected_median} but got {v}"

        assert result["col"].isna().sum() <= df["col"].isna().sum()

    # ------------------------------------------------------------------
    # Req 1.4 – mode strategy: deterministic first-sorted-value tie-break
    # ------------------------------------------------------------------

    @given(values=_int_with_some_missing)
    @settings(max_examples=200)
    def test_mode_strategy_fills_with_first_sorted_tied_mode_integers(
        self, values: list
    ) -> None:
        """mode on integer column fills with first sorted tied mode."""
        df = pd.DataFrame({"col": pd.array(values, dtype="Int64")})
        expected = _expected_mode(df["col"])

        result = FillMissing(strategy="mode").apply(df)

        original_missing_mask = df["col"].isna()
        if original_missing_mask.any() and expected is not None:
            filled_values = result.loc[original_missing_mask, "col"]
            for v in filled_values:
                assert v == expected, f"Expected mode={expected!r} but got {v!r}"

    @given(values=_str_with_some_missing)
    @settings(max_examples=200)
    def test_mode_strategy_fills_with_first_sorted_tied_mode_strings(
        self, values: list
    ) -> None:
        """mode on object column fills with first sorted tied mode."""
        df = pd.DataFrame({"col": pd.array(values, dtype="object")})
        expected = _expected_mode(df["col"])

        result = FillMissing(strategy="mode").apply(df)

        original_missing_mask = df["col"].isna()
        if original_missing_mask.any() and expected is not None:
            filled_values = result.loc[original_missing_mask, "col"]
            for v in filled_values:
                assert v == expected, f"Expected mode={expected!r} but got {v!r}"

    # ------------------------------------------------------------------
    # Req 1.5 – constant strategy fills every missing cell with the
    #           exact caller-supplied value
    # ------------------------------------------------------------------

    @given(
        fill_val=st.one_of(
            st.integers(min_value=-100, max_value=100),
            st.text(
                alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                min_size=1,
                max_size=10,
            ),
        ),
        n_rows=st.integers(min_value=2, max_value=20),
        missing_indices=st.data(),
    )
    @settings(max_examples=200)
    def test_constant_strategy_fills_every_missing_with_exact_value(
        self,
        fill_val: object,
        n_rows: int,
        missing_indices: st.DataObject,
    ) -> None:
        """constant fill places exactly fill_val in every previously-missing cell.

        The column is constructed as object dtype so that any fill_val type
        (int, str, …) can coexist with existing non-missing values.
        """
        # Draw a non-empty subset of row indices that will be None.
        all_indices = list(range(n_rows))
        none_idx_set = missing_indices.draw(
            st.frozensets(
                st.sampled_from(all_indices),
                min_size=1,
                max_size=max(1, n_rows - 1),
            )
        )

        # Build an object-typed column with sentinels at the selected indices.
        # Use pd.Series with dtype=object (NumPy object array) so that the column
        # accepts mixed types (string non-missing values + any fill_val type).
        # pd.array("object") on newer pandas may infer Arrow-backed StringDtype,
        # which rejects integer fill values.
        col_values: list[object] = ["present"] * n_rows
        for idx in none_idx_set:
            col_values[idx] = None

        df = pd.DataFrame({"col": pd.Series(col_values, dtype=object)})
        original_missing_mask = df["col"].isna()

        result = FillMissing(value=fill_val).apply(df)

        # Every cell that was missing must now equal fill_val.
        filled_cells = result.loc[original_missing_mask, "col"]
        for v in filled_cells:
            assert v == fill_val, f"Expected fill_val={fill_val!r} but got {v!r}"

        # No non-missing cell should have been altered.
        non_missing_mask = ~original_missing_mask
        pd.testing.assert_series_equal(
            df.loc[non_missing_mask, "col"],
            result.loc[non_missing_mask, "col"],
            check_names=False,
        )

    # ------------------------------------------------------------------
    # Cross-cutting: filled values are within the non-missing value range
    # for mean and median (sanity bound).
    # ------------------------------------------------------------------

    @given(values=_numeric_with_some_missing)
    @settings(max_examples=100)
    def test_mean_fill_value_is_within_data_range(self, values: list) -> None:
        """The mean fill value must equal the recomputed mean from the same data."""
        df = pd.DataFrame({"col": pd.array(values, dtype="Float64")})
        non_missing = df["col"].dropna()

        raw_mean = non_missing.mean()
        if pd.isna(raw_mean):
            return  # [inf, -inf] case — no fill expected

        expected_mean = float(raw_mean)  # type: ignore[arg-type]
        if math.isnan(expected_mean) or math.isinf(expected_mean):
            return

        result = FillMissing(strategy="mean").apply(df)
        original_missing_mask = df["col"].isna()
        if original_missing_mask.any():
            for v in result.loc[original_missing_mask, "col"]:
                fv = float(v)
                # The fill value must exactly equal what pandas computes as the
                # mean — we compare against the same computation, so they should
                # be bit-identical.  Use a tight relative tolerance to absorb
                # any Float64 <-> float64 conversion differences.
                assert (
                    pytest.approx(fv, rel=1e-6, abs=1e-9) == expected_mean
                ), f"mean fill {fv} differs from expected {expected_mean}"

    @given(values=_numeric_with_some_missing)
    @settings(max_examples=100)
    def test_median_fill_value_is_within_data_range(self, values: list) -> None:
        """The median fill value must equal the recomputed median from the same data."""
        df = pd.DataFrame({"col": pd.array(values, dtype="Float64")})
        non_missing = df["col"].dropna()

        raw_median = non_missing.median()
        if pd.isna(raw_median):
            return

        expected_median = float(raw_median)  # type: ignore[arg-type]
        if math.isnan(expected_median) or math.isinf(expected_median):
            return

        result = FillMissing(strategy="median").apply(df)
        original_missing_mask = df["col"].isna()
        if original_missing_mask.any():
            for v in result.loc[original_missing_mask, "col"]:
                fv = float(v)
                assert (
                    pytest.approx(fv, rel=1e-6, abs=1e-9) == expected_median
                ), f"median fill {fv} differs from expected {expected_median}"


# ---------------------------------------------------------------------------
# DropMissingRows
# ---------------------------------------------------------------------------


class TestDropMissingRows:
    def test_drops_rows_with_any_missing(self):
        df = _make_df(a=[1, None, 3], b=["x", "y", None])
        op = DropMissingRows()
        result = op.apply(df)
        assert len(result) == 1
        assert result["a"].iloc[0] == 1

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3])
        _ = DropMissingRows().apply(df)
        assert df["a"].isna().sum() == 1

    def test_subset_only_drops_based_on_subset(self):
        df = _make_df(a=[1, None, 3], b=[None, None, None])
        op = DropMissingRows(subset=["a"])
        result = op.apply(df)
        assert len(result) == 2

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropMissingRows(subset=["nonexistent"]).apply(df)

    def test_describe_returns_correct_keys(self):
        op = DropMissingRows(subset=["a"])
        desc = op.describe()
        assert desc["name"] == "drop_missing_rows"
        assert desc["subset"] == ["a"]

    def test_describe_subset_none(self):
        op = DropMissingRows()
        assert op.describe()["subset"] is None

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            DropMissingRows().apply([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DropMissingColumns
# ---------------------------------------------------------------------------


class TestDropMissingColumns:
    def test_drops_columns_with_missing_values(self):
        df = _make_df(a=[1, None, 3], b=[1, 2, 3])
        op = DropMissingColumns()
        result = op.apply(df)
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3], b=[1, 2, 3])
        _ = DropMissingColumns().apply(df)
        assert "a" in df.columns

    def test_subset_drops_specified_columns(self):
        df = _make_df(a=[1, 2, 3], b=[4, 5, 6])
        op = DropMissingColumns(subset=["a"])
        result = op.apply(df)
        assert "a" not in result.columns

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropMissingColumns(subset=["nonexistent"]).apply(df)

    def test_describe_name(self):
        assert DropMissingColumns().describe()["name"] == "drop_missing_columns"


# ---------------------------------------------------------------------------
# FillMissing
# ---------------------------------------------------------------------------


class TestFillMissing:
    def test_fills_all_missing_with_scalar(self):
        # Use type-compatible fill values: numeric for numeric, string for string.
        # Filling a str-dtype column with an int is rejected by pandas 2+ ArrowDtype.
        df_num = _make_df(a=[1.0, None, 3.0])
        result_num = FillMissing(value=0).apply(df_num)
        assert result_num["a"].isna().sum() == 0

        df_str = _make_df(b=[None, "y", None])
        result_str = FillMissing(value="MISSING").apply(df_str)
        assert result_str["b"].isna().sum() == 0

    def test_fill_string_value(self):
        df = _make_df(name=["Alice", None, "Bob"])
        result = FillMissing(value="UNKNOWN").apply(df)
        assert result["name"].iloc[1] == "UNKNOWN"

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3])
        _ = FillMissing(value=99).apply(df)
        assert df["a"].isna().sum() == 1

    def test_subset_fills_only_subset(self):
        df = _make_df(a=[1, None, 3], b=[None, None, None])
        result = FillMissing(value=0, subset=["a"]).apply(df)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 3

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            FillMissing(value=0, subset=["nonexistent"]).apply(df)

    def test_describe_includes_value_and_subset(self):
        op = FillMissing(value=42, subset=["a"])
        desc = op.describe()
        assert desc["value"] == 42
        assert desc["subset"] == ["a"]


# ---------------------------------------------------------------------------
# FillMissing – extended strategy tests (task 2.4)
# ---------------------------------------------------------------------------


class TestFillMissingMean:
    """Tests for strategy='mean'."""

    def test_mean_fills_correctly(self):
        df = _make_df(a=[1.0, None, 3.0])
        result = FillMissing(strategy="mean").apply(df)
        # mean of [1, 3] = 2.0
        assert result["a"].iloc[1] == pytest.approx(2.0)
        assert result["a"].isna().sum() == 0

    def test_mean_does_not_mutate_input(self):
        df = _make_df(a=[1.0, None, 3.0])
        _ = FillMissing(strategy="mean").apply(df)
        assert df["a"].isna().sum() == 1

    def test_mean_subset_only_fills_targeted_column(self):
        df = _make_df(a=[1.0, None, 3.0], b=[None, None, None])
        result = FillMissing(strategy="mean", subset=["a"]).apply(df)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 3

    def test_mean_non_numeric_raises_data_validation_error(self):
        df = _make_df(a=["x", None, "z"])
        with pytest.raises(DataValidationError, match="mean"):
            FillMissing(strategy="mean").apply(df)

    def test_mean_integer_column(self):
        df = pd.DataFrame({"a": pd.array([2, None, 4], dtype="Int64")})
        result = FillMissing(strategy="mean").apply(df)
        assert result["a"].isna().sum() == 0
        assert float(result["a"].iloc[1]) == pytest.approx(3.0)

    def test_mean_with_infinite_values_excluded(self):
        # inf is a valid float so it participates in the mean;
        # only NaN is treated as missing
        df = _make_df(a=[1.0, float("inf"), None])
        result = FillMissing(strategy="mean").apply(df)
        assert result["a"].isna().sum() == 0
        # mean of [1.0, inf] = inf
        assert math.isinf(float(result["a"].iloc[2]))

    def test_mean_apply_with_result_records_values_filled(self):
        df = _make_df(a=[1.0, None, None, 4.0])
        _, op_result = FillMissing(strategy="mean").apply_with_result(df)
        assert op_result.rows_affected == 2
        assert op_result.details["values_filled"] == 2

    def test_mean_already_clean_column_fills_zero(self):
        df = _make_df(a=[1.0, 2.0, 3.0])
        _, op_result = FillMissing(strategy="mean").apply_with_result(df)
        assert op_result.rows_affected == 0
        assert op_result.details["values_filled"] == 0

    def test_mean_apply_with_result_before_after_shape(self):
        df = _make_df(a=[1.0, None, 3.0])
        result_df, op_result = FillMissing(strategy="mean").apply_with_result(df)
        assert op_result.before_shape == df.shape
        assert op_result.after_shape == result_df.shape

    def test_mean_describe_exposes_strategy(self):
        op = FillMissing(strategy="mean", subset=["a"])
        desc = op.describe()
        assert desc["strategy"] == "mean"
        assert desc["subset"] == ["a"]


class TestFillMissingMedian:
    """Tests for strategy='median'."""

    def test_median_fills_correctly(self):
        df = _make_df(a=[1.0, None, 3.0, 5.0])
        result = FillMissing(strategy="median").apply(df)
        # median of [1, 3, 5] = 3.0
        assert result["a"].iloc[1] == pytest.approx(3.0)
        assert result["a"].isna().sum() == 0

    def test_median_even_length_non_missing(self):
        df = _make_df(a=[1.0, 2.0, None, 4.0])
        result = FillMissing(strategy="median").apply(df)
        # median of [1, 2, 4] = 2.0
        assert result["a"].iloc[2] == pytest.approx(2.0)

    def test_median_does_not_mutate_input(self):
        df = _make_df(a=[1.0, None, 3.0])
        _ = FillMissing(strategy="median").apply(df)
        assert df["a"].isna().sum() == 1

    def test_median_non_numeric_raises_data_validation_error(self):
        df = _make_df(a=["alpha", None, "gamma"])
        with pytest.raises(DataValidationError, match="median"):
            FillMissing(strategy="median").apply(df)

    def test_median_single_non_missing_value(self):
        df = _make_df(a=[7.0, None, None])
        result = FillMissing(strategy="median").apply(df)
        assert (result["a"] == 7.0).all()

    def test_median_apply_with_result_records_values_filled(self):
        df = _make_df(a=[None, 2.0, None, 4.0])
        _, op_result = FillMissing(strategy="median").apply_with_result(df)
        assert op_result.rows_affected == 2
        assert op_result.details["values_filled"] == 2

    def test_median_already_clean_column_fills_zero(self):
        df = _make_df(a=[10.0, 20.0])
        _, op_result = FillMissing(strategy="median").apply_with_result(df)
        assert op_result.rows_affected == 0

    def test_median_describe_exposes_strategy(self):
        desc = FillMissing(strategy="median").describe()
        assert desc["strategy"] == "median"


class TestFillMissingMode:
    """Tests for strategy='mode'."""

    def test_mode_fills_with_most_frequent_value(self):
        df = _make_df(a=[1, 2, 2, None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].iloc[3] == 2
        assert result["a"].isna().sum() == 0

    def test_mode_tie_selects_first_sorted_value(self):
        # 1 and 3 both appear twice; sorted → 1 wins
        df = _make_df(a=[1, 3, 1, 3, None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].iloc[4] == 1

    def test_mode_string_column(self):
        df = _make_df(a=["cat", "dog", "cat", None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].iloc[3] == "cat"

    def test_mode_string_tie_selects_first_sorted(self):
        # "apple" and "banana" tied; sorted → "apple" wins
        df = _make_df(a=["apple", "banana", "apple", "banana", None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].iloc[4] == "apple"

    def test_mode_does_not_mutate_input(self):
        df = _make_df(a=[1, 2, 2, None])
        _ = FillMissing(strategy="mode").apply(df)
        assert df["a"].isna().sum() == 1

    def test_mode_apply_with_result_records_values_filled(self):
        df = _make_df(a=[1, 2, 2, None, None])
        _, op_result = FillMissing(strategy="mode").apply_with_result(df)
        assert op_result.rows_affected == 2
        assert op_result.details["values_filled"] == 2

    def test_mode_already_clean_column_fills_zero(self):
        df = _make_df(a=["x", "y", "x"])
        _, op_result = FillMissing(strategy="mode").apply_with_result(df)
        assert op_result.rows_affected == 0

    def test_mode_describe_exposes_strategy(self):
        desc = FillMissing(strategy="mode").describe()
        assert desc["strategy"] == "mode"


class TestFillMissingConstant:
    """Tests for the default strategy='constant' and the extended describe()."""

    def test_constant_is_default_strategy(self):
        op = FillMissing(value=99)
        assert op.strategy == "constant"

    def test_constant_describe_exposes_strategy(self):
        desc = FillMissing(value="N/A").describe()
        assert desc["strategy"] == "constant"
        assert desc["value"] == "N/A"

    def test_constant_apply_with_result_records_values_filled(self):
        df = _make_df(a=[None, None, 3.0])
        _, op_result = FillMissing(value=0).apply_with_result(df)
        assert op_result.rows_affected == 2
        assert op_result.details["values_filled"] == 2

    def test_constant_mixed_type_column(self):
        df = pd.DataFrame({"a": [1, "text", None, 3.14]})
        result = FillMissing(value="FILL").apply(df)
        assert result["a"].iloc[2] == "FILL"
        assert result["a"].isna().sum() == 0


class TestFillMissingAllNullColumn:
    """All values in a column are missing."""

    def test_mean_all_null_leaves_column_null(self):
        # mean of no values is NaN → fillna(NaN) is a no-op
        df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Float64")})
        result = FillMissing(strategy="mean").apply(df)
        assert result["a"].isna().sum() == 3

    def test_median_all_null_leaves_column_null(self):
        df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Float64")})
        result = FillMissing(strategy="median").apply(df)
        assert result["a"].isna().sum() == 3

    def test_mode_all_null_leaves_column_null(self):
        # mode on all-null: no non-missing values → fill_value is None → no fill
        df = _make_df(a=[None, None, None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].isna().sum() == 3

    def test_constant_all_null_fills_all(self):
        df = _make_df(a=[None, None, None])
        result = FillMissing(value=0).apply(df)
        assert result["a"].isna().sum() == 0

    def test_apply_with_result_records_zero_filled_for_statistical_all_null(self):
        df = pd.DataFrame({"a": pd.array([None, None], dtype="Float64")})
        _, op_result = FillMissing(strategy="mean").apply_with_result(df)
        # mean is NaN; fillna(NaN) fills nothing
        assert op_result.details["values_filled"] == 0


class TestFillMissingSingleRow:
    """DataFrames with a single row."""

    def test_mean_single_non_missing_row(self):
        df = _make_df(a=[5.0])
        result = FillMissing(strategy="mean").apply(df)
        assert result["a"].iloc[0] == 5.0

    def test_median_single_non_missing_row(self):
        df = _make_df(a=[7.0])
        result = FillMissing(strategy="median").apply(df)
        assert result["a"].iloc[0] == 7.0

    def test_mode_single_non_missing_row(self):
        df = _make_df(a=["hello"])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].iloc[0] == "hello"

    def test_constant_single_missing_row(self):
        df = _make_df(a=[None])
        result = FillMissing(value=42).apply(df)
        assert result["a"].iloc[0] == 42


class TestFillMissingEdgeCasesEmptyAndShape:
    """Empty DataFrame and wide/tall edge cases."""

    def test_mean_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        result = FillMissing(strategy="mean").apply(df)
        assert result.shape == (0, 1)

    def test_median_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        result = FillMissing(strategy="median").apply(df)
        assert result.shape == (0, 1)

    def test_mode_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="object")})
        result = FillMissing(strategy="mode").apply(df)
        assert result.shape == (0, 1)

    def test_constant_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="object")})
        result = FillMissing(value=0).apply(df)
        assert result.shape == (0, 1)

    def test_wide_dataframe_mean_fills_all_columns(self):
        # 10 numeric columns, each with one missing value
        n_cols = 10
        data = {f"col{i}": [float(i), None, float(i) + 2] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = FillMissing(strategy="mean").apply(df)
        assert result.isna().sum().sum() == 0

    def test_tall_dataframe_mean_correct_fill(self):
        n_rows = 5_000
        values = [float(i) for i in range(n_rows)]
        values[100] = float("nan")
        df = pd.DataFrame({"a": values})
        expected_mean = pd.Series([v for v in values if not math.isnan(v)]).mean()
        result = FillMissing(strategy="mean").apply(df)
        assert result["a"].iloc[100] == pytest.approx(expected_mean, rel=1e-9)
        assert result["a"].isna().sum() == 0

    def test_wide_constant_fill(self):
        n_cols = 50
        data = {f"c{i}": [None, 1, None] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = FillMissing(value=-1).apply(df)
        assert result.isna().sum().sum() == 0
        assert (result.iloc[0] == -1).all()


class TestFillMissingIdempotency:
    """
    Applying the same fill strategy twice must leave the data unchanged
    after the first application (Requirement 1.9).
    """

    def test_idempotent_mean(self):
        df = _make_df(a=[1.0, None, 3.0])
        first = FillMissing(strategy="mean").apply(df)
        second = FillMissing(strategy="mean").apply(first)
        pd.testing.assert_frame_equal(first, second)

    def test_idempotent_median(self):
        df = _make_df(a=[1.0, None, 5.0, 9.0])
        first = FillMissing(strategy="median").apply(df)
        second = FillMissing(strategy="median").apply(first)
        pd.testing.assert_frame_equal(first, second)

    def test_idempotent_mode(self):
        df = _make_df(a=["x", None, "x", "y"])
        first = FillMissing(strategy="mode").apply(df)
        second = FillMissing(strategy="mode").apply(first)
        pd.testing.assert_frame_equal(first, second)

    def test_idempotent_constant(self):
        df = _make_df(a=[1, None, 3])
        first = FillMissing(value=99).apply(df)
        second = FillMissing(value=99).apply(first)
        pd.testing.assert_frame_equal(first, second)


class TestFillMissingInvalidStrategy:
    """Invalid strategy should be rejected at construction time."""

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="strategy"):
            FillMissing(strategy="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DropDuplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    def test_drops_duplicate_rows(self):
        df = _make_df(a=[1, 1, 2], b=["x", "x", "y"])
        result = DropDuplicates().apply(df)
        assert len(result) == 2

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, 1, 2])
        _ = DropDuplicates().apply(df)
        assert len(df) == 3

    def test_keep_first_default(self):
        df = _make_df(a=[1, 1, 2], b=["x", "x", "y"])
        result = DropDuplicates(keep="first").apply(df)
        assert result.index.tolist()[0] == 0

    def test_keep_last(self):
        df = _make_df(a=[1, 1, 2])
        result = DropDuplicates(keep="last").apply(df)
        assert 1 in result.index.tolist()
        assert 0 not in result.index.tolist()

    def test_keep_false_drops_all(self):
        df = _make_df(a=[1, 1, 2])
        result = DropDuplicates(keep=False).apply(df)
        assert len(result) == 1  # only unique row 2

    def test_invalid_keep_raises_value_error(self):
        with pytest.raises(ValueError):
            DropDuplicates(keep="invalid")  # type: ignore[arg-type]

    def test_subset_duplicates(self):
        df = _make_df(a=[1, 1, 2], b=["x", "y", "z"])
        result = DropDuplicates(subset=["a"]).apply(df)
        assert len(result) == 2

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropDuplicates(subset=["nonexistent"]).apply(df)

    def test_describe_keys(self):
        op = DropDuplicates(keep="last")
        desc = op.describe()
        assert desc["keep"] == "last"
        assert desc["name"] == "drop_duplicates"


# ---------------------------------------------------------------------------
# DropColumns
# ---------------------------------------------------------------------------


class TestDropColumns:
    def test_drops_specified_columns(self):
        df = _make_df(a=[1, 2], b=[3, 4], c=[5, 6])
        result = DropColumns(columns=["a", "b"]).apply(df)
        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, 2], b=[3, 4])
        _ = DropColumns(columns=["a"]).apply(df)
        assert "a" in df.columns

    def test_missing_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropColumns(columns=["nonexistent"]).apply(df)

    def test_empty_columns_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            DropColumns(columns=[])

    def test_describe_lists_columns(self):
        op = DropColumns(columns=["x", "y"])
        desc = op.describe()
        assert desc["columns"] == ["x", "y"]


# ---------------------------------------------------------------------------
# Property 2: Fill operations are idempotent
# Validates: Requirements 1.9
# ---------------------------------------------------------------------------

# Numeric values that represent real floats (no NaN/inf by default so we can
# produce well-defined statistics; we add explicit missing via None below).
_finite_floats = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)

# A numeric series element: either a finite float or None (missing).
_numeric_element = st.one_of(st.none(), _finite_floats)

# A text/object series element: short ASCII strings or None (missing).
_text_element = st.one_of(
    st.none(),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=10),
)


class TestFillMissingIdempotencyProperty:
    """
    Property 2: Fill operations are idempotent.

    Applying FillMissing once (which fills all missing values) and then applying
    the same operation a second time must yield an identical DataFrame.

    Validates: Requirements 1.9
    """

    @given(
        values=st.lists(st.one_of(st.none(), _finite_floats), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_mean_strategy_is_idempotent(self, values: list) -> None:
        """Property 2 (mean): apply(apply(df)) == apply(df) for strategy='mean'."""
        # Guarantee at least one non-missing value so mean is defined.
        if all(v is None for v in values):
            values = [1.0] + values
        df = pd.DataFrame({"a": pd.array(values, dtype="Float64")})
        op = FillMissing(strategy="mean")
        first = op.apply(df)
        second = op.apply(first)
        pd.testing.assert_frame_equal(first, second)

    @given(
        values=st.lists(st.one_of(st.none(), _finite_floats), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_median_strategy_is_idempotent(self, values: list) -> None:
        """Property 2 (median): apply(apply(df)) == apply(df) for strategy='median'."""
        if all(v is None for v in values):
            values = [1.0] + values
        df = pd.DataFrame({"a": pd.array(values, dtype="Float64")})
        op = FillMissing(strategy="median")
        first = op.apply(df)
        second = op.apply(first)
        pd.testing.assert_frame_equal(first, second)

    @given(
        values=st.lists(
            st.one_of(
                st.none(),
                st.text(
                    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                    min_size=1,
                    max_size=10,
                ),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_mode_strategy_is_idempotent(self, values: list) -> None:
        """Property 2 (mode): apply(apply(df)) == apply(df) for strategy='mode'."""
        if all(v is None for v in values):
            values = ["x"] + values
        df = pd.DataFrame({"a": values})
        op = FillMissing(strategy="mode")
        first = op.apply(df)
        second = op.apply(first)
        pd.testing.assert_frame_equal(first, second)

    @given(
        values=st.lists(st.one_of(st.none(), _finite_floats), min_size=1, max_size=50),
        fill_value=_finite_floats,
    )
    @settings(max_examples=100)
    def test_constant_strategy_is_idempotent(
        self, values: list, fill_value: float
    ) -> None:
        """Property 2 (constant): apply(apply(df)) == apply(df) for
        strategy='constant'."""
        df = pd.DataFrame({"a": pd.array(values, dtype="Float64")})
        op = FillMissing(value=fill_value, strategy="constant")
        first = op.apply(df)
        second = op.apply(first)
        pd.testing.assert_frame_equal(first, second)

    @given(
        values=st.lists(st.one_of(st.none(), _finite_floats), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_after_first_fill_no_missing_remain_mean(self, values: list) -> None:
        """After one mean-fill application, no missing values remain
        (precondition for idempotency)."""
        if all(v is None for v in values):
            values = [2.0] + values
        df = pd.DataFrame({"a": pd.array(values, dtype="Float64")})
        first = FillMissing(strategy="mean").apply(df)
        # The mean of at-least-one non-missing value is a real number,
        # so all cells should be filled.
        assert first["a"].isna().sum() == 0

    @given(
        values=st.lists(st.one_of(st.none(), _finite_floats), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_after_first_fill_no_missing_remain_median(self, values: list) -> None:
        """After one median-fill application, no missing values remain."""
        if all(v is None for v in values):
            values = [2.0] + values
        df = pd.DataFrame({"a": pd.array(values, dtype="Float64")})
        first = FillMissing(strategy="median").apply(df)
        assert first["a"].isna().sum() == 0

    @given(
        values=st.lists(
            st.one_of(
                st.none(),
                st.text(
                    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                    min_size=1,
                    max_size=10,
                ),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_after_first_fill_no_missing_remain_mode(self, values: list) -> None:
        """After one mode-fill application, no missing values remain."""
        if all(v is None for v in values):
            values = ["x"] + values
        df = pd.DataFrame({"a": values})
        first = FillMissing(strategy="mode").apply(df)
        assert first["a"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Task 2.4 – Additional edge-case unit tests for FillMissing
# Requirements: 1.1, 1.7, 1.8, 15.2
# ---------------------------------------------------------------------------


class TestFillMissingInfiniteValues:
    """Infinite floats are valid non-missing values; only NaN is missing."""

    def test_median_with_inf_participates_in_median(self) -> None:
        # inf is finite in pandas terms (not NaN), so it joins the non-missing set.
        # median of [1.0, inf] = inf  (middle of sorted [1.0, inf])
        df = _make_df(a=[1.0, float("inf"), None])
        result = FillMissing(strategy="median").apply(df)
        assert result["a"].isna().sum() == 0
        assert math.isinf(float(result["a"].iloc[2]))

    def test_median_with_negative_inf_participates_in_median(self) -> None:
        # median of [-inf, 2.0, 4.0] = 2.0
        df = _make_df(a=[float("-inf"), 2.0, None, 4.0])
        result = FillMissing(strategy="median").apply(df)
        assert result["a"].isna().sum() == 0
        # median of [-inf, 2.0, 4.0] → 2.0
        assert result["a"].iloc[2] == pytest.approx(2.0)

    def test_mode_with_inf_selects_inf_when_most_frequent(self) -> None:
        df = _make_df(a=[float("inf"), float("inf"), 1.0, None])
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].isna().sum() == 0
        assert math.isinf(float(result["a"].iloc[3]))

    def test_mean_inf_fills_with_inf(self) -> None:
        # mean([1.0, inf]) = inf — already covered in TestFillMissingMean,
        # included here for cross-class completeness
        df = _make_df(a=[1.0, float("inf"), None])
        result = FillMissing(strategy="mean").apply(df)
        assert math.isinf(float(result["a"].iloc[2]))


class TestFillMissingWideAndTallMedianMode:
    """Wide and tall DataFrames for median and mode strategies."""

    def test_wide_dataframe_median_fills_all_columns(self) -> None:
        n_cols = 10
        data = {f"col{i}": [float(i), None, float(i) + 2] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = FillMissing(strategy="median").apply(df)
        assert result.isna().sum().sum() == 0

    def test_tall_dataframe_median_correct_fill(self) -> None:
        n_rows = 5_000
        values = [float(i % 100) for i in range(n_rows)]
        values[500] = float("nan")
        df = pd.DataFrame({"a": values})
        expected_median = pd.Series([v for v in values if not math.isnan(v)]).median()
        result = FillMissing(strategy="median").apply(df)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[500] == pytest.approx(expected_median, rel=1e-9)

    def test_wide_dataframe_mode_fills_all_columns(self) -> None:
        n_cols = 10
        # Each column has a clear mode: value 1 appears twice, 2 once.
        data = {f"col{i}": [1, 1, 2, None] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = FillMissing(strategy="mode").apply(df)
        assert result.isna().sum().sum() == 0
        for col in df.columns:
            assert result[col].iloc[3] == 1

    def test_tall_dataframe_mode_correct_fill(self) -> None:
        n_rows = 5_000
        # All values are 7 except one missing; mode must be 7.
        values: list[object] = [7] * n_rows
        values[42] = None
        df = pd.DataFrame({"a": values})
        result = FillMissing(strategy="mode").apply(df)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[42] == 7


class TestFillMissingMultiColumnSubset:
    """Subset targeting multiple columns under statistical strategies."""

    def test_mean_subset_multiple_columns_partial_missing(self) -> None:
        df = _make_df(
            a=[1.0, None, 3.0],
            b=[10.0, None, 30.0],
            c=[None, None, None],  # not in subset → untouched
        )
        result = FillMissing(strategy="mean", subset=["a", "b"]).apply(df)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0
        assert result["c"].isna().sum() == 3  # untouched

    def test_median_subset_multiple_columns_correct_values(self) -> None:
        df = _make_df(
            a=[1.0, 3.0, None],
            b=[10.0, None, 30.0],
        )
        result = FillMissing(strategy="median", subset=["a", "b"]).apply(df)
        # median([1, 3]) = 2.0; median([10, 30]) = 20.0
        assert result["a"].iloc[2] == pytest.approx(2.0)
        assert result["b"].iloc[1] == pytest.approx(20.0)

    def test_mode_subset_multiple_columns_correct_values(self) -> None:
        df = _make_df(
            a=["x", "x", "y", None],
            b=["p", "q", "q", None],
        )
        result = FillMissing(strategy="mode", subset=["a", "b"]).apply(df)
        assert result["a"].iloc[3] == "x"
        assert result["b"].iloc[3] == "q"

    def test_apply_with_result_counts_missing_across_subset_columns(self) -> None:
        df = _make_df(
            a=[1.0, None, 3.0],
            b=[None, None, 6.0],
        )
        _, op_result = FillMissing(
            strategy="mean", subset=["a", "b"]
        ).apply_with_result(df)
        # 1 missing in 'a' + 2 missing in 'b' = 3 total
        assert op_result.details["values_filled"] == 3
        assert op_result.rows_affected == 3


class TestFillMissingOperationResultDetails:
    """OperationResult carries correct metadata for all strategies."""

    def test_mean_result_strategy_description_contains_strategy(self) -> None:
        df = _make_df(a=[1.0, None, 3.0])
        _, op_result = FillMissing(strategy="mean").apply_with_result(df)
        assert "mean" in op_result.strategy_description

    def test_median_result_strategy_description_contains_strategy(self) -> None:
        df = _make_df(a=[1.0, None, 3.0])
        _, op_result = FillMissing(strategy="median").apply_with_result(df)
        assert "median" in op_result.strategy_description

    def test_mode_result_strategy_description_contains_strategy(self) -> None:
        df = _make_df(a=["a", None, "a"])
        _, op_result = FillMissing(strategy="mode").apply_with_result(df)
        assert "mode" in op_result.strategy_description

    def test_constant_result_strategy_description_contains_strategy(self) -> None:
        df = _make_df(a=[None, 1.0])
        _, op_result = FillMissing(value=0).apply_with_result(df)
        assert "constant" in op_result.strategy_description

    def test_result_before_after_shape_unchanged_row_count(self) -> None:
        # Fill operations never drop or add rows.
        # Use type-compatible fill: float column filled with float value.
        df = _make_df(a=[1.0, None, 3.0], b=[None, 2.0, 4.0])
        _, op_result = FillMissing(value=0.0).apply_with_result(df)
        assert op_result.before_shape[0] == op_result.after_shape[0]
        assert op_result.before_shape[1] == op_result.after_shape[1]

    def test_result_operation_name(self) -> None:
        _, op_result = FillMissing(strategy="mean").apply_with_result(
            _make_df(a=[1.0, None])
        )
        assert op_result.operation_name == "fill_missing"

    def test_result_details_contains_strategy_key(self) -> None:
        for strategy in ("mean", "median", "mode", "constant"):
            _, op_result = FillMissing(strategy=strategy, value=0).apply_with_result(  # type: ignore[arg-type]
                _make_df(a=[1.0, None])
            )
            assert op_result.details.get("strategy") == strategy

    def test_dry_run_true_does_not_fill_data(self) -> None:
        df = _make_df(a=[1.0, None, 3.0])
        result_df, op_result = FillMissing(strategy="mean").apply_with_result(
            df, dry_run=True
        )
        # dry_run=True returns the original data unchanged
        assert result_df["a"].isna().sum() == 1
        assert op_result.dry_run is True

    def test_dry_run_false_fills_data(self) -> None:
        df = _make_df(a=[1.0, None, 3.0])
        result_df, op_result = FillMissing(strategy="mean").apply_with_result(
            df, dry_run=False
        )
        assert result_df["a"].isna().sum() == 0
        assert op_result.dry_run is False


class TestFillMissingMixedTypeCoverage:
    """Mixed-type column edge cases for statistical and constant strategies."""

    def test_median_rejects_mixed_type_object_column_with_data_validation_error(
        self,
    ) -> None:
        # object column with integers and strings → non-numeric → DataValidationError
        df = pd.DataFrame({"a": [1, "two", None, 4]})
        with pytest.raises(DataValidationError, match="median"):
            FillMissing(strategy="median").apply(df)

    def test_mean_rejects_mixed_type_object_column_with_data_validation_error(
        self,
    ) -> None:
        df = pd.DataFrame({"a": [1, "two", None, 4]})
        with pytest.raises(DataValidationError, match="mean"):
            FillMissing(strategy="mean").apply(df)

    def test_mode_accepts_mixed_type_object_column(self) -> None:
        # mode works on any comparable column type
        df = pd.DataFrame({"a": [1, "cat", "cat", None]})
        result = FillMissing(strategy="mode").apply(df)
        # mode of [1, "cat", "cat"] = "cat"
        assert result["a"].iloc[3] == "cat"
        assert result["a"].isna().sum() == 0

    def test_constant_fill_on_object_column_with_mixed_types(self) -> None:
        df = pd.DataFrame({"a": pd.array([1, "hello", None, 3.14], dtype=object)})
        result = FillMissing(value="FILL").apply(df)
        assert result["a"].iloc[2] == "FILL"
        assert result["a"].isna().sum() == 0


class TestFillMissingAllNullColumnExtended:
    """Extended all-null tests including apply_with_result metadata accuracy."""

    def test_mean_all_null_result_records_zero_filled(self) -> None:
        df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Float64")})
        _, op_result = FillMissing(strategy="mean").apply_with_result(df)
        assert op_result.details["values_filled"] == 0
        assert op_result.rows_affected == 0

    def test_median_all_null_result_records_zero_filled(self) -> None:
        df = pd.DataFrame({"a": pd.array([None, None, None], dtype="Float64")})
        _, op_result = FillMissing(strategy="median").apply_with_result(df)
        assert op_result.details["values_filled"] == 0
        assert op_result.rows_affected == 0

    def test_mode_all_null_result_records_zero_filled(self) -> None:
        df = _make_df(a=[None, None, None])
        _, op_result = FillMissing(strategy="mode").apply_with_result(df)
        assert op_result.details["values_filled"] == 0
        assert op_result.rows_affected == 0

    def test_constant_all_null_result_records_all_filled(self) -> None:
        df = _make_df(a=[None, None, None])
        _, op_result = FillMissing(value=42).apply_with_result(df)
        assert op_result.details["values_filled"] == 3
        assert op_result.rows_affected == 3
