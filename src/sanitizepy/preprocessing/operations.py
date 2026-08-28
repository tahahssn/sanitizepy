"""
Built-in feature engineering operations.

The operations in this module transform pandas DataFrames while preserving
the dataframe index and existing columns unless explicitly configured
otherwise.

No operation creates synthetic rows or uses dummy data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import FeatureOperation


class ColumnInteraction(FeatureOperation):
    """
    Create a feature from the interaction of two numeric columns.

    The resulting feature is:

        output = column_a * column_b
    """

    name = "column_interaction"

    def __init__(
        self,
        column_a: str,
        column_b: str,
        output_column: str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(column_a, str) or not column_a:
            raise ValueError("column_a must be a non-empty string.")

        if not isinstance(column_b, str) or not column_b:
            raise ValueError("column_b must be a non-empty string.")

        if column_a == column_b:
            raise ValueError("column_a and column_b must be different.")

        self.column_a = column_a
        self.column_b = column_b
        self.output_column = (
            output_column if output_column is not None else f"{column_a}_x_{column_b}"
        )

        if not self.output_column:
            raise ValueError("output_column must be a non-empty string.")

    def fit(self, data: pd.DataFrame) -> ColumnInteraction:
        self._validate_input(data)
        self._validate_columns(data)

        self._validate_numeric_columns(data)

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        self._require_fitted()

        self._validate_columns(data)
        self._validate_numeric_columns(data)

        result = data.copy()
        result[self.output_column] = result[self.column_a] * result[self.column_b]

        return result

    def get_params(self) -> dict[str, str]:
        return {
            "column_a": self.column_a,
            "column_b": self.column_b,
            "output_column": self.output_column,
        }

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [
            column
            for column in (self.column_a, self.column_b)
            if column not in data.columns
        ]

        if missing:
            raise KeyError(f"Required columns are missing: {missing}.")

    def _validate_numeric_columns(self, data: pd.DataFrame) -> None:
        non_numeric = [
            column
            for column in (self.column_a, self.column_b)
            if not pd.api.types.is_numeric_dtype(data[column])
        ]

        if non_numeric:
            raise TypeError(
                "ColumnInteraction requires numeric columns. "
                f"Non-numeric columns: {non_numeric}."
            )


class RatioFeature(FeatureOperation):
    """
    Create a ratio feature from two numeric columns.

    By default, division by zero produces NaN instead of inf. This avoids
    introducing infinite values into downstream machine-learning workflows.
    """

    name = "ratio_feature"

    def __init__(
        self,
        numerator: str,
        denominator: str,
        output_column: str | None = None,
        zero_division: str = "nan",
    ) -> None:
        super().__init__()

        if not isinstance(numerator, str) or not numerator:
            raise ValueError("numerator must be a non-empty string.")

        if not isinstance(denominator, str) or not denominator:
            raise ValueError("denominator must be a non-empty string.")

        if numerator == denominator:
            raise ValueError("numerator and denominator must be different.")

        if zero_division not in {"nan", "raise"}:
            raise ValueError("zero_division must be either 'nan' or 'raise'.")

        self.numerator = numerator
        self.denominator = denominator
        self.output_column = (
            output_column
            if output_column is not None
            else f"{numerator}_div_{denominator}"
        )
        self.zero_division = zero_division

        if not self.output_column:
            raise ValueError("output_column must be a non-empty string.")

    def fit(self, data: pd.DataFrame) -> RatioFeature:
        self._validate_input(data)
        self._validate_columns(data)
        self._validate_numeric_columns(data)

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        self._require_fitted()

        self._validate_columns(data)
        self._validate_numeric_columns(data)

        denominator = data[self.denominator]

        if self.zero_division == "raise" and (denominator == 0).any():
            raise ZeroDivisionError(
                f"Column {self.denominator!r} contains zero values."
            )

        result = data.copy()

        numerator_values = result[self.numerator].astype("float64")
        denominator_values = result[self.denominator].astype("float64")

        result[self.output_column] = np.divide(
            numerator_values,
            denominator_values,
            out=np.full(
                len(result),
                np.nan,
                dtype=np.float64,
            ),
            where=denominator_values != 0,
        )

        return result

    def get_params(self) -> dict[str, str]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "output_column": self.output_column,
            "zero_division": self.zero_division,
        }

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [
            column
            for column in (self.numerator, self.denominator)
            if column not in data.columns
        ]

        if missing:
            raise KeyError(f"Required columns are missing: {missing}.")

    def _validate_numeric_columns(self, data: pd.DataFrame) -> None:
        non_numeric = [
            column
            for column in (self.numerator, self.denominator)
            if not pd.api.types.is_numeric_dtype(data[column])
        ]

        if non_numeric:
            raise TypeError(
                "RatioFeature requires numeric columns. "
                f"Non-numeric columns: {non_numeric}."
            )


class PolynomialFeature(FeatureOperation):
    """
    Generate polynomial powers for a numeric column.

    For degree=3 and column='x', generated features are:

        x^2
        x^3

    The original column is preserved.
    """

    name = "polynomial_feature"

    def __init__(
        self,
        column: str,
        degree: int,
        include_bias: bool = False,
        output_prefix: str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string.")

        if not isinstance(degree, int) or isinstance(degree, bool):
            raise TypeError("degree must be an integer.")

        if degree < 2:
            raise ValueError("degree must be at least 2.")

        if not isinstance(include_bias, bool):
            raise TypeError("include_bias must be a boolean.")

        self.column = column
        self.degree = degree
        self.include_bias = include_bias
        self.output_prefix = output_prefix if output_prefix is not None else column

        if not self.output_prefix:
            raise ValueError("output_prefix must be a non-empty string.")

    def fit(self, data: pd.DataFrame) -> PolynomialFeature:
        self._validate_input(data)

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        if not pd.api.types.is_numeric_dtype(data[self.column]):
            raise TypeError(f"Column {self.column!r} must be numeric.")

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        self._require_fitted()

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        if not pd.api.types.is_numeric_dtype(data[self.column]):
            raise TypeError(f"Column {self.column!r} must be numeric.")

        result = data.copy()
        values = result[self.column]

        if self.include_bias:
            result[f"{self.output_prefix}_bias"] = 1.0

        for power in range(2, self.degree + 1):
            result[f"{self.output_prefix}^{power}"] = values.pow(power)

        return result

    def get_params(self) -> dict[str, object]:
        return {
            "column": self.column,
            "degree": self.degree,
            "include_bias": self.include_bias,
            "output_prefix": self.output_prefix,
        }


class LogFeature(FeatureOperation):
    """
    Create a natural-log transformed feature.

    Values must be strictly positive unless ``offset`` is supplied.
    """

    name = "log_feature"

    def __init__(
        self,
        column: str,
        output_column: str | None = None,
        offset: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string.")

        if not isinstance(offset, (int, float)) or isinstance(offset, bool):
            raise TypeError("offset must be numeric.")

        if not np.isfinite(offset):
            raise ValueError("offset must be finite.")

        self.column = column
        self.output_column = (
            output_column if output_column is not None else f"log_{column}"
        )
        self.offset = float(offset)

        if not self.output_column:
            raise ValueError("output_column must be a non-empty string.")

    def fit(self, data: pd.DataFrame) -> LogFeature:
        self._validate_input(data)

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        if not pd.api.types.is_numeric_dtype(data[self.column]):
            raise TypeError(f"Column {self.column!r} must be numeric.")

        adjusted = data[self.column].astype("float64") + self.offset

        if (adjusted <= 0).any():
            raise ValueError(
                f"Column {self.column!r} contains values that are "
                "not valid for logarithmic transformation with the "
                f"configured offset ({self.offset})."
            )

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        self._require_fitted()

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        if not pd.api.types.is_numeric_dtype(data[self.column]):
            raise TypeError(f"Column {self.column!r} must be numeric.")

        adjusted = data[self.column].astype("float64") + self.offset

        if (adjusted <= 0).any():
            raise ValueError(
                f"Column {self.column!r} contains values that are "
                "not valid for logarithmic transformation with the "
                f"configured offset ({self.offset})."
            )

        result = data.copy()
        result[self.output_column] = np.log(adjusted)

        return result

    def get_params(self) -> dict[str, object]:
        return {
            "column": self.column,
            "output_column": self.output_column,
            "offset": self.offset,
        }


class DatetimeFeatures(FeatureOperation):
    """
    Extract calendar features from a datetime column.

    Supported components are:

        year
        month
        day
        day_of_week
        day_of_year
        week
        quarter
        hour
        minute
        second
    """

    name = "datetime_features"

    _SUPPORTED_FEATURES = frozenset(
        {
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_year",
            "week",
            "quarter",
            "hour",
            "minute",
            "second",
        }
    )

    def __init__(
        self,
        column: str,
        features: Sequence[str],
        prefix: str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string.")

        if not features:
            raise ValueError("features must contain at least one feature.")

        normalized_features = tuple(dict.fromkeys(features))

        invalid = set(normalized_features) - self._SUPPORTED_FEATURES

        if invalid:
            raise ValueError(
                f"Unsupported datetime features: {sorted(invalid)}. "
                f"Supported features: {sorted(self._SUPPORTED_FEATURES)}."
            )

        self.column = column
        self.features = normalized_features
        self.prefix = prefix if prefix is not None else column

        if not self.prefix:
            raise ValueError("prefix must be a non-empty string.")

    def fit(self, data: pd.DataFrame) -> DatetimeFeatures:
        self._validate_input(data)

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        self._validate_datetime_column(data)

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        self._require_fitted()

        if self.column not in data.columns:
            raise KeyError(f"Required column {self.column!r} is missing.")

        self._validate_datetime_column(data)

        result = data.copy()
        datetime_values = result[self.column]

        for feature in self.features:
            output_column = f"{self.prefix}_{feature}"

            if feature == "year":
                result[output_column] = datetime_values.dt.year

            elif feature == "month":
                result[output_column] = datetime_values.dt.month

            elif feature == "day":
                result[output_column] = datetime_values.dt.day

            elif feature == "day_of_week":
                result[output_column] = datetime_values.dt.dayofweek

            elif feature == "day_of_year":
                result[output_column] = datetime_values.dt.dayofyear

            elif feature == "week":
                result[output_column] = datetime_values.dt.isocalendar().week

            elif feature == "quarter":
                result[output_column] = datetime_values.dt.quarter

            elif feature == "hour":
                result[output_column] = datetime_values.dt.hour

            elif feature == "minute":
                result[output_column] = datetime_values.dt.minute

            elif feature == "second":
                result[output_column] = datetime_values.dt.second

        return result

    def get_params(self) -> dict[str, object]:
        return {
            "column": self.column,
            "features": self.features,
            "prefix": self.prefix,
        }

    def _validate_datetime_column(self, data: pd.DataFrame) -> None:
        if not pd.api.types.is_datetime64_any_dtype(data[self.column]):
            raise TypeError(
                f"Column {self.column!r} must have a datetime dtype. "
                "Convert it during preprocessing before applying "
                "DatetimeFeatures."
            )
