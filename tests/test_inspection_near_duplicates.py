"""
Tests for sanitizepy.inspection.near_duplicates.NearDuplicateDetector

Task 13.6 – unit tests for near-duplicate detection
Requirements: 10.1, 10.3, 10.5, 10.6, 10.7

Covers:
  - exact_normalized detection groups normalization-equivalent rows (Req 10.1, 10.3)
  - column subset restricts comparison (Req 10.3)
  - duplicate_count reflects redundant rows per group (Req 10.1)
  - determinism of group ordering (Req 10.7)
  - similarity mode raises DependencyError naming sanitizepy[fuzzy] when
    rapidfuzz is absent (Req 10.5, 10.6)
  - ValueError on empty DataFrame / missing subset columns
  - non-mutation of the input DataFrame
  - edge cases: empty, single-row, all-null, mixed-type, infinite, wide, tall
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sanitizepy.exceptions import DependencyError
from sanitizepy.inspection.near_duplicates import (
    NearDuplicateDetector,
    NearDuplicateResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# 1. exact_normalized detection (Requirement 10.1, 10.3)
# ---------------------------------------------------------------------------


class TestExactNormalizedDetection:
    """exact_normalized groups rows equivalent under normalization."""

    def test_case_insensitive_grouping(self) -> None:
        df = _make_df(name=["Alice", "alice", "Bob"])
        result = NearDuplicateDetector().detect(df, method="exact_normalized")
        assert result.groups == ((0, 1),)
        assert result.duplicate_count == 1

    def test_whitespace_normalization_grouping(self) -> None:
        df = _make_df(name=["  hello   world ", "hello world", "other"])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1),)
        assert result.duplicate_count == 1

    def test_default_method_is_exact_normalized(self) -> None:
        df = _make_df(name=["A", "a"])
        result = NearDuplicateDetector().detect(df)
        assert result.method == "exact_normalized"

    def test_no_duplicates_yields_empty_groups(self) -> None:
        df = _make_df(name=["one", "two", "three"])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ()
        assert result.duplicate_count == 0

    def test_multiple_groups(self) -> None:
        df = _make_df(name=["a", "A", "b", "B", "c"])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1), (2, 3))
        assert result.duplicate_count == 2

    def test_group_of_three_counts_two_duplicates(self) -> None:
        df = _make_df(name=["x", "X", "  x  "])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1, 2),)
        assert result.duplicate_count == 2

    def test_result_is_immutable_dataclass(self) -> None:
        df = _make_df(name=["a", "a"])
        result = NearDuplicateDetector().detect(df)
        assert isinstance(result, NearDuplicateResult)
        with pytest.raises((AttributeError, TypeError)):
            result.duplicate_count = 99  # type: ignore[misc]

    def test_columns_default_to_all_columns(self) -> None:
        df = _make_df(a=["x", "x"], b=["y", "y"])
        result = NearDuplicateDetector().detect(df)
        assert result.columns == ("a", "b")

    def test_full_row_comparison_requires_all_columns_equal(self) -> None:
        df = _make_df(a=["x", "x"], b=["y", "z"])
        result = NearDuplicateDetector().detect(df)
        # Rows differ in column b, so no duplicates over the full row.
        assert result.groups == ()

    def test_preserves_non_default_index_labels(self) -> None:
        df = pd.DataFrame({"name": ["a", "A", "b"]}, index=["r1", "r2", "r3"])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == (("r1", "r2"),)


# ---------------------------------------------------------------------------
# 2. Column subset (Requirement 10.3)
# ---------------------------------------------------------------------------


class TestColumnSubset:
    """subset restricts which columns participate in comparison."""

    def test_subset_single_column(self) -> None:
        df = _make_df(key=["a", "A"], other=["1", "2"])
        result = NearDuplicateDetector().detect(df, subset=["key"])
        assert result.columns == ("key",)
        assert result.groups == ((0, 1),)

    def test_subset_multiple_columns_preserve_order(self) -> None:
        df = _make_df(a=["x", "x"], b=["y", "y"], c=["1", "2"])
        result = NearDuplicateDetector().detect(df, subset=["b", "a"])
        assert result.columns == ("b", "a")
        assert result.groups == ((0, 1),)

    def test_subset_excludes_differing_column(self) -> None:
        df = _make_df(key=["dup", "dup"], noise=["p", "q"])
        # Without subset, noise differs → no group.
        assert NearDuplicateDetector().detect(df).groups == ()
        # With subset on key only, they group.
        assert NearDuplicateDetector().detect(df, subset=["key"]).groups == ((0, 1),)

    def test_missing_subset_column_raises_value_error(self) -> None:
        df = _make_df(a=["x"])
        with pytest.raises(ValueError, match="not found"):
            NearDuplicateDetector().detect(df, subset=["missing"])

    def test_subset_tuple_accepted(self) -> None:
        df = _make_df(a=["x", "X"], b=["1", "2"])
        result = NearDuplicateDetector().detect(df, subset=("a",))
        assert result.groups == ((0, 1),)


# ---------------------------------------------------------------------------
# 3. Determinism (Requirement 10.7)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Repeated detection yields identical results."""

    def test_repeated_detection_identical(self) -> None:
        df = _make_df(name=["a", "A", "b", "B", "a"])
        detector = NearDuplicateDetector()
        first = detector.detect(df)
        second = detector.detect(df)
        assert first == second

    def test_groups_ordered_by_first_appearance(self) -> None:
        df = _make_df(name=["z", "z", "a", "a"])
        result = NearDuplicateDetector().detect(df)
        # 'z' group appears before 'a' group despite alphabetical order.
        assert result.groups == ((0, 1), (2, 3))


# ---------------------------------------------------------------------------
# 4. Similarity mode + DependencyError (Requirement 10.5, 10.6)
# ---------------------------------------------------------------------------


class TestSimilarityDependencyError:
    """similarity mode requires rapidfuzz; absence raises DependencyError."""

    def test_similarity_raises_dependency_error_when_rapidfuzz_missing(self) -> None:
        df = _make_df(name=["hello", "hallo"])
        with (
            patch.dict(sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError, match="sanitizepy\\[fuzzy\\]"),
        ):
            NearDuplicateDetector().detect(df, method="similarity")

    def test_dependency_error_names_extra(self) -> None:
        df = _make_df(name=["abc"])
        with (
            patch.dict(sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError) as exc_info,
        ):
            NearDuplicateDetector().detect(df, method="similarity")
        assert "sanitizepy[fuzzy]" in str(exc_info.value)

    def test_exact_normalized_never_needs_rapidfuzz(self) -> None:
        df = _make_df(name=["a", "A"])
        with patch.dict(sys.modules, {"rapidfuzz": None}):
            # Must not raise DependencyError.
            result = NearDuplicateDetector().detect(df, method="exact_normalized")
        assert result.groups == ((0, 1),)


# ---------------------------------------------------------------------------
# 5. Non-mutation guarantee
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_detect_does_not_mutate_input(self) -> None:
        df = _make_df(name=["a", "A", "b"])
        original = df.copy()
        NearDuplicateDetector().detect(df)
        pd.testing.assert_frame_equal(df, original)

    def test_result_does_not_reference_dataframe(self) -> None:
        df = _make_df(name=["a", "A"])
        result = NearDuplicateDetector().detect(df)
        # groups holds only index labels, not DataFrame references.
        assert result.groups == ((0, 1),)


# ---------------------------------------------------------------------------
# 6. Empty DataFrame raises ValueError
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_empty_dataframe_raises_value_error(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            NearDuplicateDetector().detect(df)

    def test_columns_but_no_rows_raises_value_error(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        with pytest.raises(ValueError, match="empty"):
            NearDuplicateDetector().detect(df)


# ---------------------------------------------------------------------------
# 7. Edge case: Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_has_no_duplicates(self) -> None:
        df = _make_df(name=["solo"])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ()
        assert result.duplicate_count == 0


# ---------------------------------------------------------------------------
# 8. Edge case: All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_rows_group_together(self) -> None:
        # Missing values normalize to empty string → identical keys.
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1, 2),)
        assert result.duplicate_count == 2

    def test_null_and_empty_string_normalize_equal(self) -> None:
        df = _make_df(col=[np.nan, "", "  "])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1, 2),)


# ---------------------------------------------------------------------------
# 9. Edge case: Mixed-type column
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_numeric_string_normalizes_to_same_key(self) -> None:
        # 1 (int) and "1" (str) both coerce to "1".
        df = pd.DataFrame({"col": pd.array([1, "1", 2], dtype="object")})
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1),)

    def test_mixed_types_no_false_grouping(self) -> None:
        df = pd.DataFrame({"col": pd.array([1, 2.5, "text"], dtype="object")})
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ()


# ---------------------------------------------------------------------------
# 10. Edge case: Infinite numeric values
# ---------------------------------------------------------------------------


class TestEdgeCaseInfinite:
    def test_infinite_values_group_by_string_repr(self) -> None:
        df = _make_df(value=[float("inf"), float("inf"), 1.0])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ((0, 1),)

    def test_positive_and_negative_inf_not_grouped(self) -> None:
        df = _make_df(value=[float("inf"), float("-inf")])
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ()


# ---------------------------------------------------------------------------
# 11. Edge case: Wide DataFrame (many columns)
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_dataframe_full_row_duplicates(self) -> None:
        n_cols = 30
        data = {f"c{i}": ["x", "X"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = NearDuplicateDetector().detect(df)
        assert result.columns == tuple(f"c{i}" for i in range(n_cols))
        assert result.groups == ((0, 1),)

    def test_wide_dataframe_one_differing_column_breaks_group(self) -> None:
        n_cols = 30
        data = {f"c{i}": ["x", "x"] for i in range(n_cols)}
        data["c15"] = ["x", "different"]
        df = pd.DataFrame(data)
        result = NearDuplicateDetector().detect(df)
        assert result.groups == ()


# ---------------------------------------------------------------------------
# 12. Edge case: Tall DataFrame (many rows)
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_dataframe_grouping(self) -> None:
        n_rows = 5_000
        # Alternate two normalization-equivalent values.
        values = ["Val" if i % 2 == 0 else "val" for i in range(n_rows)]
        df = _make_df(col=values)
        result = NearDuplicateDetector().detect(df)
        # All rows normalize to "val" → one big group.
        assert len(result.groups) == 1
        assert len(result.groups[0]) == n_rows
        assert result.duplicate_count == n_rows - 1

    def test_tall_dataframe_deterministic(self) -> None:
        n_rows = 2_000
        values = [f"item{i % 100}" for i in range(n_rows)]
        df = _make_df(col=values)
        detector = NearDuplicateDetector()
        assert detector.detect(df) == detector.detect(df)


# ===========================================================================
# Property tests (task 13.4)
# **Property 11: Near-duplicate detection is deterministic**
# **Validates: Requirements 10.2**
# ===========================================================================

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@st.composite
def _near_duplicate_cell(draw: st.DrawFn) -> object:
    """
    Draw a single object-column cell biased toward normalization collisions.

    The pool draws from a small set of base tokens and returns them in
    randomly cased / whitespace-padded variants so that many rows normalize to
    the same key and actual near-duplicate groups form. It also mixes in
    ``None`` and non-string scalars so the detector's coercion and
    missing-handling paths are exercised.
    """
    base = draw(st.sampled_from(["alice", "bob", "carol", "dave"]))

    def _casing(token: str) -> str:
        style = draw(st.sampled_from(["lower", "upper", "title", "as-is"]))
        if style == "lower":
            return token.lower()
        if style == "upper":
            return token.upper()
        if style == "title":
            return token.title()
        return token

    def _pad(token: str) -> str:
        lead = draw(st.sampled_from(["", " ", "   ", "\t"]))
        trail = draw(st.sampled_from(["", " ", "  ", "\n"]))
        inner = draw(st.sampled_from([" ", "   "]))
        # Occasionally introduce an internal whitespace run around the token.
        if draw(st.booleans()):
            return f"{lead}{token}{inner}{token}{trail}"
        return f"{lead}{token}{trail}"

    variant = st.builds(lambda t: _pad(_casing(t)), st.just(base))
    scalar = st.one_of(st.none(), st.integers(-50, 50))

    return draw(st.one_of(variant, scalar))


@st.composite
def _near_duplicate_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """
    Draw a non-empty DataFrame with one or more object/string columns.

    Cells are drawn from :func:`_near_duplicate_cell` so case/whitespace
    variants of a shared token pool land in the frame, ensuring detection
    actually produces groups (rather than always the empty result).
    """
    n_cols = draw(st.integers(min_value=1, max_value=3))
    n_rows = draw(st.integers(min_value=1, max_value=30))
    data = {
        f"col_{i}": pd.array(
            [draw(_near_duplicate_cell()) for _ in range(n_rows)], dtype="object"
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


class TestNearDuplicateDeterminismProperty:
    """
    **Property 11: Near-duplicate detection is deterministic**
    **Validates: Requirements 10.2**

    Detecting near-duplicates over the same data with the same configuration
    must yield an identical NearDuplicateResult (columns, method, groups, and
    duplicate_count) every time.
    """

    @settings(max_examples=200, deadline=None)
    @given(_near_duplicate_dataframe())
    def test_repeated_detection_is_identical(self, df: pd.DataFrame) -> None:
        detector = NearDuplicateDetector()
        first = detector.detect(df)
        second = detector.detect(df)
        assert first == second

    @settings(max_examples=200, deadline=None)
    @given(_near_duplicate_dataframe())
    def test_detection_on_copy_is_identical(self, df: pd.DataFrame) -> None:
        # A structurally identical copy must produce an identical result,
        # proving the outcome depends only on the data, not object identity.
        detector = NearDuplicateDetector()
        first = detector.detect(df)
        second = detector.detect(df.copy())
        assert first == second
        assert first.groups == second.groups
        assert first.duplicate_count == second.duplicate_count
        assert first.columns == second.columns

    @settings(max_examples=100, deadline=None)
    @given(_near_duplicate_dataframe())
    def test_detection_with_subset_is_deterministic(self, df: pd.DataFrame) -> None:
        detector = NearDuplicateDetector()
        subset = [str(df.columns[0])]
        first = detector.detect(df, subset=subset)
        second = detector.detect(df.copy(), subset=subset)
        assert first == second
