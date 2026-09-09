"""
Tests for sanitizepy.inspection.text_quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.exceptions import DependencyError
from sanitizepy.inspection.text_quality import TextQualityAnalyzer
from sanitizepy.models.profile import TextQualityResult


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


def _result_by_column(
    results: tuple[TextQualityResult, ...],
) -> dict[str, TextQualityResult]:
    return {result.column: result for result in results}


class TestTextQualityAnalyzerEmpty:
    def test_raises_on_empty_dataframe(self):
        analyzer = TextQualityAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze(pd.DataFrame())

    def test_raises_on_dataframe_with_columns_but_no_rows(self):
        analyzer = TextQualityAnalyzer()
        empty = pd.DataFrame({"text": pd.Series([], dtype="object")})
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze(empty)


class TestTextQualityAnalyzerAdvancedMode:
    def test_advanced_mode_raises_dependency_error(self):
        analyzer = TextQualityAnalyzer()
        df = _make_df(text=["hello world"])
        with pytest.raises(DependencyError):
            analyzer.analyze(df, mode="advanced")

    def test_advanced_mode_names_text_extra(self):
        analyzer = TextQualityAnalyzer()
        df = _make_df(text=["hello world"])
        with pytest.raises(DependencyError, match=r"sanitizepy\[text\]"):
            analyzer.analyze(df, mode="advanced")


class TestTextQualityAnalyzerColumnSelection:
    def setup_method(self):
        self.df = _make_df(
            text=["hello world", "foo bar baz"],
            number=[1, 2],
            other=["a", "b"],
        )
        self.analyzer = TextQualityAnalyzer()

    def test_only_object_columns_analyzed(self):
        results = self.analyzer.analyze(self.df)
        columns = {result.column for result in results}
        assert columns == {"text", "other"}

    def test_numeric_columns_excluded(self):
        results = self.analyzer.analyze(self.df)
        assert "number" not in {result.column for result in results}

    def test_results_returned_in_column_order(self):
        results = self.analyzer.analyze(self.df)
        assert [result.column for result in results] == ["text", "other"]

    def test_subset_restricts_analysis(self):
        results = self.analyzer.analyze(self.df, subset=["text"])
        assert [result.column for result in results] == ["text"]

    def test_subset_skips_non_text_columns(self):
        results = self.analyzer.analyze(self.df, subset=["number", "text"])
        assert [result.column for result in results] == ["text"]

    def test_subset_missing_column_raises_key_error(self):
        with pytest.raises(KeyError):
            self.analyzer.analyze(self.df, subset=["missing"])

    def test_returns_tuple(self):
        results = self.analyzer.analyze(self.df)
        assert isinstance(results, tuple)

    def test_does_not_mutate_input(self):
        before = self.df.copy(deep=True)
        self.analyzer.analyze(self.df)
        pd.testing.assert_frame_equal(self.df, before)


class TestTextQualityAnalyzerLengthAndTokenStats:
    def setup_method(self):
        # Character lengths: 1, 3, 5 -> min 1, max 5, mean 3, median 3
        # Token counts:       1, 1, 1 -> all one word
        self.df = _make_df(text=["a", "abc", "abcde"])
        self.analyzer = TextQualityAnalyzer()
        self.result = self.analyzer.analyze(self.df)[0]

    def test_non_null_count(self):
        assert self.result.non_null_count == 3

    def test_character_length_min(self):
        assert self.result.character_length_min == 1

    def test_character_length_max(self):
        assert self.result.character_length_max == 5

    def test_character_length_mean(self):
        assert self.result.character_length_mean == pytest.approx(3.0)

    def test_character_length_median(self):
        assert self.result.character_length_median == pytest.approx(3.0)

    def test_token_count_min(self):
        assert self.result.token_count_min == 1

    def test_token_count_max(self):
        assert self.result.token_count_max == 1

    def test_token_count_mean(self):
        assert self.result.token_count_mean == pytest.approx(1.0)

    def test_token_count_median(self):
        assert self.result.token_count_median == pytest.approx(1.0)


class TestTextQualityAnalyzerMultiTokenStats:
    def setup_method(self):
        # Token counts: 1, 2, 3 -> min 1, max 3, mean 2, median 2
        self.df = _make_df(text=["one", "one two", "one two three"])
        self.analyzer = TextQualityAnalyzer()
        self.result = self.analyzer.analyze(self.df)[0]

    def test_token_count_min(self):
        assert self.result.token_count_min == 1

    def test_token_count_max(self):
        assert self.result.token_count_max == 3

    def test_token_count_mean(self):
        assert self.result.token_count_mean == pytest.approx(2.0)

    def test_token_count_median(self):
        assert self.result.token_count_median == pytest.approx(2.0)


class TestTextQualityAnalyzerEmptyAfterStrip:
    def setup_method(self):
        self.df = _make_df(text=["hello", "   ", "\t\n", "world"])
        self.analyzer = TextQualityAnalyzer()
        self.result = self.analyzer.analyze(self.df)[0]

    def test_counts_whitespace_only_values(self):
        assert self.result.empty_after_strip_count == 2

    def test_non_null_count_includes_whitespace_values(self):
        assert self.result.non_null_count == 4

    def test_zero_when_no_empty_values(self):
        df = _make_df(text=["hello", "world"])
        result = self.analyzer.analyze(df)[0]
        assert result.empty_after_strip_count == 0


class TestTextQualityAnalyzerBoilerplate:
    def setup_method(self):
        self.analyzer = TextQualityAnalyzer()

    def test_flags_placeholder_tokens(self):
        df = _make_df(text=["n/a", "none", "tbd", "real content here"])
        result = self.analyzer.analyze(df)[0]
        assert result.boilerplate_count == 3

    def test_flags_lorem_ipsum(self):
        df = _make_df(text=["Lorem ipsum dolor sit amet", "genuine text"])
        result = self.analyzer.analyze(df)[0]
        assert result.boilerplate_count == 1

    def test_placeholder_detection_is_case_insensitive(self):
        df = _make_df(text=["N/A", "NONE", "TBD"])
        result = self.analyzer.analyze(df)[0]
        assert result.boilerplate_count == 3

    def test_placeholder_detection_ignores_surrounding_whitespace(self):
        df = _make_df(text=["  n/a  ", "  none "])
        result = self.analyzer.analyze(df)[0]
        assert result.boilerplate_count == 2

    def test_zero_when_no_boilerplate(self):
        df = _make_df(text=["meaningful", "content", "values"])
        result = self.analyzer.analyze(df)[0]
        assert result.boilerplate_count == 0


class TestTextQualityAnalyzerEncodingGarbage:
    def setup_method(self):
        self.analyzer = TextQualityAnalyzer()

    def test_flags_replacement_character(self):
        df = _make_df(text=["caf\ufffd", "clean"])
        result = self.analyzer.analyze(df)[0]
        assert result.encoding_garbage_count == 1

    def test_flags_control_characters(self):
        df = _make_df(text=["bad\x00value", "clean value"])
        result = self.analyzer.analyze(df)[0]
        assert result.encoding_garbage_count == 1

    def test_flags_mojibake(self):
        df = _make_df(text=["donâ\x80\x99t", "clean"])
        result = self.analyzer.analyze(df)[0]
        assert result.encoding_garbage_count == 1

    def test_zero_when_clean_text(self):
        df = _make_df(text=["clean", "text", "values"])
        result = self.analyzer.analyze(df)[0]
        assert result.encoding_garbage_count == 0

    def test_common_whitespace_not_flagged(self):
        df = _make_df(text=["line1\nline2", "tab\there", "cr\rreturn"])
        result = self.analyzer.analyze(df)[0]
        assert result.encoding_garbage_count == 0


# ---------------------------------------------------------------------------
# Edge cases: empty, single-row, all-null, mixed-type, infinite, wide, tall
# ---------------------------------------------------------------------------


class TestTextQualityAnalyzerEdgeCases:
    def setup_method(self):
        self.analyzer = TextQualityAnalyzer()

    def test_single_row(self):
        df = _make_df(text=["single value"])
        result = self.analyzer.analyze(df)[0]
        assert result.non_null_count == 1
        assert result.character_length_min == len("single value")
        assert result.character_length_max == len("single value")
        assert result.character_length_mean == pytest.approx(len("single value"))
        assert result.token_count_min == 2
        assert result.token_count_max == 2

    def test_all_null_column(self):
        df = _make_df(text=[None, None, None])
        result = self.analyzer.analyze(df)[0]
        assert result.non_null_count == 0
        assert result.empty_after_strip_count == 0
        assert result.character_length_min == 0
        assert result.character_length_max == 0
        assert result.character_length_mean == pytest.approx(0.0)
        assert result.character_length_median == pytest.approx(0.0)
        assert result.token_count_min == 0
        assert result.token_count_max == 0
        assert result.boilerplate_count == 0
        assert result.encoding_garbage_count == 0

    def test_some_null_values_skipped(self):
        df = _make_df(text=["hello", None, "world", np.nan])
        result = self.analyzer.analyze(df)[0]
        assert result.non_null_count == 2

    def test_mixed_type_object_column(self):
        # Object column holding non-string values are coerced to text.
        df = _make_df(text=["hello", 42, 3.14, True])
        result = self.analyzer.analyze(df)[0]
        assert result.non_null_count == 4
        # "42" -> 2 chars, "3.14" -> 4 chars, "True" -> 4 chars, "hello" -> 5
        assert result.character_length_min == 2
        assert result.character_length_max == 5

    def test_infinite_values_in_object_column(self):
        df = _make_df(text=["hello", float("inf"), float("-inf")])
        result = self.analyzer.analyze(df)[0]
        # inf/-inf are coerced to "inf"/"-inf" text without error.
        assert result.non_null_count == 3
        assert result.character_length_max == 5

    def test_wide_dataframe_many_text_columns(self):
        data = {f"col_{i}": ["value one", "value two"] for i in range(50)}
        df = pd.DataFrame(data)
        results = self.analyzer.analyze(df)
        assert len(results) == 50
        assert all(result.non_null_count == 2 for result in results)

    def test_tall_dataframe_many_rows(self):
        df = _make_df(text=[f"row {i}" for i in range(10_000)])
        result = self.analyzer.analyze(df)[0]
        assert result.non_null_count == 10_000
        assert result.token_count_min == 2
        assert result.token_count_max == 2

    def test_no_text_columns_returns_empty_tuple(self):
        df = _make_df(a=[1, 2, 3], b=[4.0, 5.0, 6.0])
        results = self.analyzer.analyze(df)
        assert results == ()


class TestTextQualityAnalyzerResultType:
    def test_returns_text_quality_result_instances(self):
        analyzer = TextQualityAnalyzer()
        df = _make_df(text=["hello world"])
        results = analyzer.analyze(df)
        assert all(isinstance(result, TextQualityResult) for result in results)

    def test_result_is_immutable(self):
        analyzer = TextQualityAnalyzer()
        df = _make_df(text=["hello world"])
        result = analyzer.analyze(df)[0]
        with pytest.raises((AttributeError, TypeError)):
            result.column = "changed"  # type: ignore[misc]


# ===========================================================================
# Property tests (task 14.3)
# **Property 13: Text-quality analysis is deterministic**
# **Validates: Requirements 11.3, 11.5**
# ===========================================================================


@st.composite
def _text_cell(draw: st.DrawFn) -> object:
    """
    Draw a single object-column cell exercising the analyzer's code paths.

    The pool mixes ordinary text, whitespace-heavy strings, boilerplate /
    placeholder tokens, lorem-ipsum text, encoding-garbage strings, empty
    strings, ``None``, and non-string scalars so the generated columns cover
    every deterministic branch of the analyzer.
    """
    ordinary = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
            whitelist_characters="\u00a0",
        ),
        min_size=0,
        max_size=20,
    )
    boilerplate = st.sampled_from(
        [
            "n/a",
            "  none  ",
            "TBD",
            "null",
            "Lorem ipsum dolor sit amet",
            "placeholder",
        ]
    )
    encoding_garbage = st.sampled_from(
        [
            "caf\ufffd",
            "bad\x00value",
            "don\u00e2\x80\x99t",
            "line1\nline2",
        ]
    )
    whitespace_only = st.sampled_from(["", "   ", "\t\n", "  "])
    scalar = st.one_of(st.none(), st.integers(-100, 100), st.floats(allow_nan=False))
    return draw(
        st.one_of(
            ordinary,
            boilerplate,
            encoding_garbage,
            whitespace_only,
            scalar,
        )
    )


@st.composite
def _text_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """
    Draw a non-empty DataFrame containing one or more object/string columns.

    Each column holds a mix of strings, whitespace, boilerplate, encoding
    garbage, ``None`` and non-string scalars (via :func:`_text_cell`). Every
    column is the same length so pandas accepts the mapping.
    """
    n_cols = draw(st.integers(min_value=1, max_value=4))
    n_rows = draw(st.integers(min_value=1, max_value=30))
    data = {
        f"col_{i}": pd.array(
            [draw(_text_cell()) for _ in range(n_rows)], dtype="object"
        )
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


class TestTextQualityAnalyzerDeterminism:
    """
    **Property 13: Text-quality analysis is deterministic**
    **Validates: Requirements 11.3, 11.5**

    Analyzing the same DataFrame twice must yield identical
    ``TextQualityResult`` tuples. Because ``TextQualityResult`` is a frozen
    dataclass, ``==`` compares every field; the tests additionally assert
    field-by-field equality for an explicit, readable failure signal.
    """

    @given(_text_dataframe())
    @settings(max_examples=150)
    def test_repeated_analysis_yields_identical_tuple(self, df: pd.DataFrame) -> None:
        analyzer = TextQualityAnalyzer()
        first = analyzer.analyze(df)
        second = analyzer.analyze(df)
        assert first == second

    @given(_text_dataframe())
    @settings(max_examples=150)
    def test_analysis_of_copy_matches_original(self, df: pd.DataFrame) -> None:
        analyzer = TextQualityAnalyzer()
        first = analyzer.analyze(df)
        second = analyzer.analyze(df.copy())
        assert first == second

    @given(_text_dataframe())
    @settings(max_examples=100)
    def test_every_field_is_identical_across_runs(self, df: pd.DataFrame) -> None:
        analyzer = TextQualityAnalyzer()
        first = analyzer.analyze(df)
        second = analyzer.analyze(df.copy())

        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert left.column == right.column
            assert left.non_null_count == right.non_null_count
            assert left.empty_after_strip_count == right.empty_after_strip_count
            assert left.character_length_min == right.character_length_min
            assert left.character_length_max == right.character_length_max
            assert left.character_length_mean == right.character_length_mean
            assert left.character_length_median == right.character_length_median
            assert left.token_count_min == right.token_count_min
            assert left.token_count_max == right.token_count_max
            assert left.token_count_mean == right.token_count_mean
            assert left.token_count_median == right.token_count_median
            assert left.boilerplate_count == right.boilerplate_count
            assert left.encoding_garbage_count == right.encoding_garbage_count

    @given(_text_dataframe())
    @settings(max_examples=100)
    def test_determinism_holds_with_subset(self, df: pd.DataFrame) -> None:
        analyzer = TextQualityAnalyzer()
        subset = [str(df.columns[0])]
        first = analyzer.analyze(df, subset=subset)
        second = analyzer.analyze(df.copy(), subset=subset)
        assert first == second
