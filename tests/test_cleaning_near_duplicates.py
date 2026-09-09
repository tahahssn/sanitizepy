"""
Tests for sanitizepy.cleaning.near_duplicates.NearDuplicateRemovalOperation

Task 13.6 – unit tests for near-duplicate removal
Requirements: 10.1, 10.3, 10.5, 10.6, 10.7

Covers:
  - exact_normalized removal keeps one representative per group (Req 10.1)
  - keep="first"/"last" policy selection (Req 10.1)
  - column subset restricts comparison (Req 10.3)
  - OperationResult records removed count (rows_affected / records_removed)
  - DependencyError naming sanitizepy[fuzzy] when method="similarity" and
    rapidfuzz is absent (Req 10.5, 10.6)
  - is_chunk_safe = False, registry presence, describe() output
  - non-mutation of the input DataFrame
  - edge cases: empty, single-row, all-null, mixed-type, infinite, wide, tall
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sanitizepy.cleaning.near_duplicates import NearDuplicateRemovalOperation
from sanitizepy.exceptions import DependencyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# 1. exact_normalized removal (Requirement 10.1)
# ---------------------------------------------------------------------------


class TestExactNormalizedRemoval:
    """Near-duplicate rows collapse to a single representative."""

    def test_case_insensitive_duplicates_removed(self) -> None:
        df = _make_df(name=["Alice", "alice", "Bob"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert list(result["name"]) == ["Alice", "Bob"]

    def test_whitespace_variants_removed(self) -> None:
        df = _make_df(name=["  hello  world ", "hello world", "x"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert list(result["name"]) == ["  hello  world ", "x"]

    def test_no_duplicates_returns_all_rows(self) -> None:
        df = _make_df(name=["a", "b", "c"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 3

    def test_group_of_three_collapses_to_one(self) -> None:
        df = _make_df(name=["x", "X", "  x  ", "y"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert list(result["name"]) == ["x", "y"]

    def test_multiple_groups_each_collapse(self) -> None:
        df = _make_df(name=["a", "A", "b", "B"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert list(result["name"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# 2. keep policy (Requirement 10.1)
# ---------------------------------------------------------------------------


class TestKeepPolicy:
    def test_keep_first_default(self) -> None:
        df = _make_df(name=["First", "first", "FIRST"])
        result = NearDuplicateRemovalOperation(keep="first").apply(df)
        assert list(result["name"]) == ["First"]

    def test_keep_last(self) -> None:
        df = _make_df(name=["First", "first", "FIRST"])
        result = NearDuplicateRemovalOperation(keep="last").apply(df)
        assert list(result["name"]) == ["FIRST"]

    def test_keep_first_preserves_earliest_index(self) -> None:
        df = pd.DataFrame({"name": ["a", "A"]}, index=[10, 20])
        result = NearDuplicateRemovalOperation(keep="first").apply(df)
        assert list(result.index) == [10]

    def test_keep_last_preserves_latest_index(self) -> None:
        df = pd.DataFrame({"name": ["a", "A"]}, index=[10, 20])
        result = NearDuplicateRemovalOperation(keep="last").apply(df)
        assert list(result.index) == [20]

    def test_invalid_keep_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="keep"):
            NearDuplicateRemovalOperation(keep="middle")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Column subset (Requirement 10.3)
# ---------------------------------------------------------------------------


class TestColumnSubset:
    def test_subset_restricts_comparison(self) -> None:
        df = _make_df(key=["dup", "dup"], noise=["p", "q"])
        result = NearDuplicateRemovalOperation(subset=["key"]).apply(df)
        assert len(result) == 1

    def test_without_subset_differing_column_keeps_both(self) -> None:
        df = _make_df(key=["dup", "dup"], noise=["p", "q"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 2

    def test_subset_missing_column_raises_value_error(self) -> None:
        df = _make_df(a=["x", "x"])
        op = NearDuplicateRemovalOperation(subset=["missing"])
        with pytest.raises(ValueError, match="not found"):
            op.apply(df)


# ---------------------------------------------------------------------------
# 4. OperationResult removed count
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_records_removed_count(self) -> None:
        df = _make_df(name=["a", "A", "a", "b"])
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.rows_affected == 2
        assert res.details["records_removed"] == 2

    def test_no_duplicates_zero_removed(self) -> None:
        df = _make_df(name=["a", "b", "c"])
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.rows_affected == 0
        assert res.details["records_removed"] == 0

    def test_before_and_after_shape(self) -> None:
        df = _make_df(name=["a", "A", "b"])
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.before_shape == (3, 1)
        assert res.after_shape == (2, 1)

    def test_operation_name_in_result(self) -> None:
        df = _make_df(name=["a", "A"])
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.operation_name == "near_duplicate_removal"

    def test_affected_columns_reflect_subset(self) -> None:
        df = _make_df(a=["x", "x"], b=["1", "2"])
        _, res = NearDuplicateRemovalOperation(subset=["a"]).apply_with_result(df)
        assert res.affected_columns == ["a"]

    def test_dry_run_does_not_mutate_and_returns_copy(self) -> None:
        df = _make_df(name=["a", "A"])
        original = df.copy()
        out, res = NearDuplicateRemovalOperation().apply_with_result(df, dry_run=True)
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(out, original)
        assert res.dry_run is True
        # Count still reflects what would be removed.
        assert res.rows_affected == 1

    def test_dry_run_false_returns_transformed(self) -> None:
        df = _make_df(name=["a", "A"])
        out, res = NearDuplicateRemovalOperation().apply_with_result(df, dry_run=False)
        assert len(out) == 1
        assert res.dry_run is False


# ---------------------------------------------------------------------------
# 5. Similarity mode + DependencyError (Requirement 10.5, 10.6)
# ---------------------------------------------------------------------------


class TestSimilarityDependencyError:
    def test_similarity_raises_dependency_error_when_rapidfuzz_missing(self) -> None:
        df = _make_df(name=["hello", "hallo"])
        op = NearDuplicateRemovalOperation(method="similarity")
        with (
            patch.dict(sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError, match="sanitizepy\\[fuzzy\\]"),
        ):
            op.apply(df)

    def test_apply_with_result_raises_dependency_error(self) -> None:
        df = _make_df(name=["hello", "hallo"])
        op = NearDuplicateRemovalOperation(method="similarity")
        with (
            patch.dict(sys.modules, {"rapidfuzz": None}),
            pytest.raises(DependencyError, match="sanitizepy\\[fuzzy\\]"),
        ):
            op.apply_with_result(df)

    def test_exact_normalized_never_needs_rapidfuzz(self) -> None:
        df = _make_df(name=["a", "A"])
        op = NearDuplicateRemovalOperation(method="exact_normalized")
        with patch.dict(sys.modules, {"rapidfuzz": None}):
            result = op.apply(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 6. Classification attributes
# ---------------------------------------------------------------------------


class TestClassificationAttributes:
    def test_is_chunk_safe_is_false(self) -> None:
        assert NearDuplicateRemovalOperation.is_chunk_safe is False

    def test_is_inplace_safe_default(self) -> None:
        assert NearDuplicateRemovalOperation.is_inplace_safe is False

    def test_name_attribute(self) -> None:
        assert NearDuplicateRemovalOperation.name == "near_duplicate_removal"


# ---------------------------------------------------------------------------
# 7. Registry presence
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_module_registry(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.contains("near_duplicate_removal")

    def test_registry_returns_correct_class(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.get("near_duplicate_removal") is NearDuplicateRemovalOperation


# ---------------------------------------------------------------------------
# 8. describe() output
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_required_keys(self) -> None:
        op = NearDuplicateRemovalOperation(
            subset=["col"], method="exact_normalized", keep="last"
        )
        desc = op.describe()
        assert desc["name"] == "near_duplicate_removal"
        assert desc["subset"] == ["col"]
        assert desc["method"] == "exact_normalized"
        assert desc["keep"] == "last"

    def test_describe_subset_none_when_not_set(self) -> None:
        desc = NearDuplicateRemovalOperation().describe()
        assert desc["subset"] is None

    def test_describe_threshold_present(self) -> None:
        desc = NearDuplicateRemovalOperation(threshold=85.0).describe()
        assert desc["threshold"] == 85.0


# ---------------------------------------------------------------------------
# 9. Non-mutation guarantee
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_apply_does_not_mutate_input(self) -> None:
        df = _make_df(name=["a", "A", "b"])
        original = df.copy()
        NearDuplicateRemovalOperation().apply(df)
        pd.testing.assert_frame_equal(df, original)

    def test_apply_with_result_does_not_mutate_input(self) -> None:
        df = _make_df(name=["a", "A", "b"])
        original = df.copy()
        NearDuplicateRemovalOperation().apply_with_result(df, dry_run=False)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# 10. Edge case: Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_apply_empty_returns_empty(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        result = NearDuplicateRemovalOperation().apply(df)
        assert result.empty

    def test_apply_with_result_empty_zero_removed(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.rows_affected == 0
        assert res.details["records_removed"] == 0

    def test_completely_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = NearDuplicateRemovalOperation().apply(df)
        assert result.empty


# ---------------------------------------------------------------------------
# 11. Edge case: Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_unchanged(self) -> None:
        df = _make_df(name=["solo"])
        result = NearDuplicateRemovalOperation().apply(df)
        assert list(result["name"]) == ["solo"]

    def test_single_row_zero_removed(self) -> None:
        df = _make_df(name=["solo"])
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.rows_affected == 0


# ---------------------------------------------------------------------------
# 12. Edge case: All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_rows_collapse_to_one(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 1

    def test_null_and_empty_string_collapse(self) -> None:
        df = _make_df(col=[np.nan, "", "  "])
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 13. Edge case: Mixed-type column
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_numeric_and_string_equivalent_collapse(self) -> None:
        df = pd.DataFrame({"col": pd.array([1, "1", 2], dtype="object")})
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 2

    def test_distinct_mixed_types_kept(self) -> None:
        df = pd.DataFrame({"col": pd.array([1, 2.5, "text"], dtype="object")})
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 14. Edge case: Infinite numeric values
# ---------------------------------------------------------------------------


class TestEdgeCaseInfinite:
    def test_infinite_duplicates_collapse(self) -> None:
        df = _make_df(value=[float("inf"), float("inf"), 1.0])
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 2

    def test_positive_and_negative_inf_kept(self) -> None:
        df = _make_df(value=[float("inf"), float("-inf")])
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 15. Edge case: Wide DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_full_row_duplicate_removed(self) -> None:
        n_cols = 30
        data = {f"c{i}": ["x", "X"] for i in range(n_cols)}
        df = pd.DataFrame(data)
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 1

    def test_wide_one_differing_column_keeps_both(self) -> None:
        n_cols = 30
        data = {f"c{i}": ["x", "x"] for i in range(n_cols)}
        data["c15"] = ["x", "different"]
        df = pd.DataFrame(data)
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 16. Edge case: Tall DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_all_equivalent_collapse_to_one(self) -> None:
        n_rows = 5_000
        values = ["Val" if i % 2 == 0 else "val" for i in range(n_rows)]
        df = _make_df(col=values)
        _, res = NearDuplicateRemovalOperation().apply_with_result(df)
        assert res.after_shape[0] == 1
        assert res.rows_affected == n_rows - 1

    def test_tall_many_small_groups(self) -> None:
        n_rows = 2_000
        # 100 distinct values, each appearing 20 times.
        values = [f"item{i % 100}" for i in range(n_rows)]
        df = _make_df(col=values)
        result = NearDuplicateRemovalOperation().apply(df)
        assert len(result) == 100


# ===========================================================================
# Property tests (task 13.5)
# **Property 12: Near-duplicate removal preserves representatives**
# **Validates: Requirements 10.5**
# ===========================================================================

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from sanitizepy.inspection.near_duplicates import (  # noqa: E402
    NearDuplicateDetector,
)


def _normalize_token(value: str) -> str:
    """
    Mirror the detector's normalization so tests can reason about which base
    token a rendered value collapses to: strip, lowercase, collapse internal
    whitespace runs to a single space.
    """
    import re

    return re.sub(r"\s+", " ", value.strip().lower())


@st.composite
def _near_duplicate_column(draw: st.DrawFn) -> list[str]:
    """
    Draw a list of strings built from a small pool of base tokens, where each
    drawn value is a normalization-equivalent *variant* of its base token
    (random case flips and surrounding/internal whitespace). Values sharing a
    base token therefore collapse into the same near-duplicate group under the
    detector's normalization, while distinct normalized bases stay separate.
    """
    # Distinct base tokens whose normalized forms are guaranteed unique.
    base_pool = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                min_size=1,
                max_size=6,
            ),
            min_size=1,
            max_size=5,
            unique_by=_normalize_token,
        )
    )

    def render(token: str) -> str:
        # Random case variation.
        cased = "".join(
            ch.upper() if draw(st.booleans()) else ch.lower() for ch in token
        )
        # Surrounding whitespace that normalization will strip.
        lead = " " * draw(st.integers(min_value=0, max_value=3))
        trail = " " * draw(st.integers(min_value=0, max_value=3))
        return f"{lead}{cased}{trail}"

    n_rows = draw(st.integers(min_value=1, max_value=25))
    return [render(draw(st.sampled_from(base_pool))) for _ in range(n_rows)]


class TestRepresentativePreservationProperty:
    """
    **Property 12: Near-duplicate removal preserves representatives**
    **Validates: Requirements 10.5**

    For any DataFrame containing normalization-equivalent duplicates, after
    removal:

    * the result contains no remaining near-duplicates (running the detector on
      the result yields zero groups / duplicate_count 0), so exactly one
      representative from each group survives;
    * the number of rows removed equals the sum over groups of
      (group_size - 1);
    * keep="first" retains the earliest member of each group and keep="last"
      retains the latest.
    """

    @given(_near_duplicate_column())
    @settings(max_examples=150)
    def test_result_has_no_remaining_near_duplicates(self, values: list[str]) -> None:
        df = pd.DataFrame({"col": values})
        result = NearDuplicateRemovalOperation().apply(df)

        # Re-running the detector on the cleaned result must find no groups:
        # exactly one representative per original group survived.
        detector = NearDuplicateDetector()
        redetected = detector.detect(result)
        assert redetected.groups == ()
        assert redetected.duplicate_count == 0

    @given(_near_duplicate_column())
    @settings(max_examples=150)
    def test_removed_count_equals_sum_group_size_minus_one(
        self, values: list[str]
    ) -> None:
        df = pd.DataFrame({"col": values})
        detector = NearDuplicateDetector()
        detected = detector.detect(df)
        expected_removed = sum(len(group) - 1 for group in detected.groups)

        _, res = NearDuplicateRemovalOperation().apply_with_result(df)

        assert res.rows_affected == expected_removed
        assert res.details["records_removed"] == expected_removed
        assert res.after_shape[0] == len(df) - expected_removed

    @given(_near_duplicate_column())
    @settings(max_examples=150)
    def test_keep_first_retains_earliest_member_of_each_group(
        self, values: list[str]
    ) -> None:
        df = pd.DataFrame({"col": values})
        detector = NearDuplicateDetector()
        detected = detector.detect(df)
        # Groups preserve DataFrame row order, so the earliest member is [0].
        expected_kept = {group[0] for group in detected.groups}
        # Singleton rows (no group) are also always retained.
        grouped_labels = {label for group in detected.groups for label in group}
        singletons = set(df.index) - grouped_labels
        expected_index = sorted(expected_kept | singletons, key=list(df.index).index)

        result = NearDuplicateRemovalOperation(keep="first").apply(df)

        assert list(result.index) == expected_index

    @given(_near_duplicate_column())
    @settings(max_examples=150)
    def test_keep_last_retains_latest_member_of_each_group(
        self, values: list[str]
    ) -> None:
        df = pd.DataFrame({"col": values})
        detector = NearDuplicateDetector()
        detected = detector.detect(df)
        # The latest member of each group is its final element.
        expected_kept = {group[-1] for group in detected.groups}
        grouped_labels = {label for group in detected.groups for label in group}
        singletons = set(df.index) - grouped_labels
        expected_index = sorted(expected_kept | singletons, key=list(df.index).index)

        result = NearDuplicateRemovalOperation(keep="last").apply(df)

        assert list(result.index) == expected_index
