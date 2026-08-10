"""
Tests for cleaner.preprocessing.operations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaner.preprocessing.operations import (
    ColumnInteraction,
    DatetimeFeatures,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# ColumnInteraction
# ---------------------------------------------------------------------------

class TestColumnInteraction:
    def setup_method(self):
        self.df = _make_df(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0])

    def test_default_output_column_name(self):
        op = ColumnInteraction("a", "b")
        assert op.output_column == "a_x_b"

    def test_custom_output_column_name(self):
        op = ColumnInteraction("a", "b", output_column="product")
        assert op.output_column == "product"

    def test_fit_transform_creates_interaction(self):
        op = ColumnInteraction("a", "b")
        result = op.fit_transform(self.df)
        expected = self.df["a"] * self.df["b"]
        pd.testing.assert_series_equal(
            result["a_x_b"], expected, check_names=False
        )

    def test_does_not_mutate_input(self):
        op = ColumnInteraction("a", "b")
        op.fit_transform(self.df)
        assert "a_x_b" not in self.df.columns

    def test_same_columns_raises_value_error(self):
        with pytest.raises(ValueError, match="different"):
            ColumnInteraction("a", "a")

    def test_empty_column_a_raises_value_error(self):
        with pytest.raises(ValueError, match="column_a"):
            ColumnInteraction("", "b")

    def test_empty_column_b_raises_value_error(self):
        with pytest.raises(ValueError, match="column_b"):
            ColumnInteraction("a", "")

    def test_missing_column_raises_key_error(self):
        op = ColumnInteraction("a", "x")
        with pytest.raises(KeyError):
            op.fit(self.df)

    def test_non_numeric_raises_type_error(self):
        df = _make_df(a=[1.0, 2.0], b=["x", "y"])
        op = ColumnInteraction("a", "b")
        with pytest.raises(TypeError):
            op.fit(df)

    def test_transform_without_fit_raises_runtime_error(self):
        op = ColumnInteraction("a", "b")
        with pytest.raises(RuntimeError, match="fitted"):
            op.transform(self.df)

    def test_get_params(self):
        op = ColumnInteraction("a", "b")
        params = op.get_params()
        assert params["column_a"] == "a"
        assert params["column_b"] == "b"
        assert "output_column" in params

    def test_validate_input_rejects_non_dataframe(self):
        op = ColumnInteraction("a", "b")
        with pytest.raises(TypeError):
            op.fit([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RatioFeature
# ---------------------------------------------------------------------------

class TestRatioFeature:
    def setup_method(self):
        self.df = _make_df(num=[4.0, 6.0, 8.0], den=[2.0, 3.0, 4.0])

    def test_default_output_column_name(self):
        op = RatioFeature("num", "den")
        assert op.output_column == "num_div_den"

    def test_fit_transform_correct_values(self):
        op = RatioFeature("num", "den")
        result = op.fit_transform(self.df)
        expected = [2.0, 2.0, 2.0]
        assert result["num_div_den"].tolist() == pytest.approx(expected)

    def test_zero_division_nan_default(self):
        df = _make_df(num=[1.0, 2.0], den=[0.0, 2.0])
        op = RatioFeature("num", "den")
        result = op.fit_transform(df)
        assert np.isnan(result["num_div_den"].iloc[0])

    def test_zero_division_raise_raises(self):
        df = _make_df(num=[1.0, 2.0], den=[0.0, 2.0])
        op = RatioFeature("num", "den", zero_division="raise")
        with pytest.raises(ZeroDivisionError):
            op.fit_transform(df)

    def test_same_columns_raises_value_error(self):
        with pytest.raises(ValueError, match="different"):
            RatioFeature("a", "a")

    def test_invalid_zero_division_raises_value_error(self):
        with pytest.raises(ValueError, match="zero_division"):
            RatioFeature("num", "den", zero_division="ignore")

    def test_missing_column_raises_key_error(self):
        op = RatioFeature("num", "missing")
        with pytest.raises(KeyError):
            op.fit(self.df)

    def test_does_not_mutate_input(self):
        op = RatioFeature("num", "den")
        op.fit_transform(self.df)
        assert "num_div_den" not in self.df.columns

    def test_get_params(self):
        op = RatioFeature("num", "den")
        params = op.get_params()
        assert params["numerator"] == "num"
        assert params["denominator"] == "den"


# ---------------------------------------------------------------------------
# PolynomialFeature
# ---------------------------------------------------------------------------

class TestPolynomialFeature:
    def setup_method(self):
        self.df = _make_df(x=[1.0, 2.0, 3.0])

    def test_degree_2_creates_x_squared(self):
        op = PolynomialFeature("x", degree=2)
        result = op.fit_transform(self.df)
        assert "x^2" in result.columns
        assert result["x^2"].tolist() == pytest.approx([1.0, 4.0, 9.0])

    def test_degree_3_creates_x_squared_and_cubed(self):
        op = PolynomialFeature("x", degree=3)
        result = op.fit_transform(self.df)
        assert "x^2" in result.columns
        assert "x^3" in result.columns

    def test_original_column_preserved(self):
        op = PolynomialFeature("x", degree=2)
        result = op.fit_transform(self.df)
        assert "x" in result.columns

    def test_include_bias_adds_bias_column(self):
        op = PolynomialFeature("x", degree=2, include_bias=True)
        result = op.fit_transform(self.df)
        assert "x_bias" in result.columns
        assert (result["x_bias"] == 1.0).all()

    def test_custom_output_prefix(self):
        op = PolynomialFeature("x", degree=2, output_prefix="feat")
        result = op.fit_transform(self.df)
        assert "feat^2" in result.columns

    def test_degree_less_than_2_raises_value_error(self):
        with pytest.raises(ValueError, match="degree must be at least 2"):
            PolynomialFeature("x", degree=1)

    def test_non_integer_degree_raises_type_error(self):
        with pytest.raises(TypeError, match="degree"):
            PolynomialFeature("x", degree=2.5)  # type: ignore[arg-type]

    def test_bool_degree_raises_type_error(self):
        with pytest.raises(TypeError, match="degree"):
            PolynomialFeature("x", degree=True)  # type: ignore[arg-type]

    def test_missing_column_raises_key_error(self):
        op = PolynomialFeature("y", degree=2)
        with pytest.raises(KeyError):
            op.fit(self.df)

    def test_non_numeric_column_raises_type_error(self):
        df = _make_df(x=["a", "b", "c"])
        op = PolynomialFeature("x", degree=2)
        with pytest.raises(TypeError):
            op.fit(df)

    def test_does_not_mutate_input(self):
        op = PolynomialFeature("x", degree=2)
        op.fit_transform(self.df)
        assert "x^2" not in self.df.columns

    def test_get_params(self):
        op = PolynomialFeature("x", degree=2, include_bias=True)
        params = op.get_params()
        assert params["column"] == "x"
        assert params["degree"] == 2
        assert params["include_bias"] is True


# ---------------------------------------------------------------------------
# LogFeature
# ---------------------------------------------------------------------------

class TestLogFeature:
    def setup_method(self):
        self.df = _make_df(x=[1.0, np.e, np.e ** 2])

    def test_default_output_column_name(self):
        op = LogFeature("x")
        assert op.output_column == "log_x"

    def test_log_values_correct(self):
        op = LogFeature("x")
        result = op.fit_transform(self.df)
        assert result["log_x"].tolist() == pytest.approx([0.0, 1.0, 2.0])

    def test_offset_allows_zero_values(self):
        df = _make_df(x=[0.0, 1.0, 2.0])
        op = LogFeature("x", offset=1.0)
        result = op.fit_transform(df)
        assert result["log_x"].tolist() == pytest.approx(
            [np.log(1.0), np.log(2.0), np.log(3.0)]
        )

    def test_non_positive_value_raises_value_error(self):
        df = _make_df(x=[0.0, 1.0, 2.0])
        op = LogFeature("x")
        with pytest.raises(ValueError):
            op.fit(df)

    def test_non_finite_offset_raises_value_error(self):
        with pytest.raises(ValueError, match="finite"):
            LogFeature("x", offset=float("inf"))

    def test_non_numeric_offset_raises_type_error(self):
        with pytest.raises(TypeError, match="offset"):
            LogFeature("x", offset="large")  # type: ignore[arg-type]

    def test_missing_column_raises_key_error(self):
        op = LogFeature("missing")
        with pytest.raises(KeyError):
            op.fit(self.df)

    def test_non_numeric_column_raises_type_error(self):
        df = _make_df(x=["a", "b", "c"])
        op = LogFeature("x")
        with pytest.raises(TypeError):
            op.fit(df)

    def test_does_not_mutate_input(self):
        op = LogFeature("x")
        op.fit_transform(self.df)
        assert "log_x" not in self.df.columns

    def test_get_params(self):
        op = LogFeature("x", offset=1.0)
        params = op.get_params()
        assert params["column"] == "x"
        assert params["offset"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DatetimeFeatures
# ---------------------------------------------------------------------------

class TestDatetimeFeatures:
    def setup_method(self):
        self.df = pd.DataFrame({
            "ts": pd.to_datetime(["2023-03-15 10:30:00", "2024-07-04 22:15:45"])
        })

    def test_extracts_year(self):
        op = DatetimeFeatures("ts", features=["year"])
        result = op.fit_transform(self.df)
        assert "ts_year" in result.columns
        assert result["ts_year"].tolist() == [2023, 2024]

    def test_extracts_month(self):
        op = DatetimeFeatures("ts", features=["month"])
        result = op.fit_transform(self.df)
        assert result["ts_month"].tolist() == [3, 7]

    def test_extracts_day(self):
        op = DatetimeFeatures("ts", features=["day"])
        result = op.fit_transform(self.df)
        assert result["ts_day"].tolist() == [15, 4]

    def test_extracts_hour(self):
        op = DatetimeFeatures("ts", features=["hour"])
        result = op.fit_transform(self.df)
        assert result["ts_hour"].tolist() == [10, 22]

    def test_extracts_minute(self):
        op = DatetimeFeatures("ts", features=["minute"])
        result = op.fit_transform(self.df)
        assert result["ts_minute"].tolist() == [30, 15]

    def test_extracts_second(self):
        op = DatetimeFeatures("ts", features=["second"])
        result = op.fit_transform(self.df)
        assert result["ts_second"].tolist() == [0, 45]

    def test_extracts_multiple_features(self):
        op = DatetimeFeatures("ts", features=["year", "month", "day"])
        result = op.fit_transform(self.df)
        assert "ts_year" in result.columns
        assert "ts_month" in result.columns
        assert "ts_day" in result.columns

    def test_custom_prefix(self):
        op = DatetimeFeatures("ts", features=["year"], prefix="date")
        result = op.fit_transform(self.df)
        assert "date_year" in result.columns

    def test_original_column_preserved(self):
        op = DatetimeFeatures("ts", features=["year"])
        result = op.fit_transform(self.df)
        assert "ts" in result.columns

    def test_empty_features_raises_value_error(self):
        with pytest.raises(ValueError, match="features"):
            DatetimeFeatures("ts", features=[])

    def test_unsupported_feature_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DatetimeFeatures("ts", features=["microsecond"])

    def test_non_datetime_column_raises_type_error(self):
        df = _make_df(ts=["2023-01-01", "2023-02-01"])
        op = DatetimeFeatures("ts", features=["year"])
        with pytest.raises(TypeError, match="datetime"):
            op.fit(df)

    def test_missing_column_raises_key_error(self):
        op = DatetimeFeatures("missing", features=["year"])
        with pytest.raises(KeyError):
            op.fit(self.df)

    def test_does_not_mutate_input(self):
        op = DatetimeFeatures("ts", features=["year"])
        op.fit_transform(self.df)
        assert "ts_year" not in self.df.columns

    def test_get_params(self):
        op = DatetimeFeatures("ts", features=["year", "month"])
        params = op.get_params()
        assert params["column"] == "ts"
        assert "year" in params["features"]

    def test_deduplicates_features(self):
        op = DatetimeFeatures("ts", features=["year", "year", "month"])
        result = op.fit_transform(self.df)
        # "ts_year" should appear only once
        assert list(result.columns).count("ts_year") == 1
