from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


@dataclass(frozen=True, slots=True)
class ColumnTypeReport:
    """
    Datatype inspection for a single column.
    """

    column: str
    dtype: str
    semantic_type: str
    nullable: bool
    missing_count: int
    unique_count: int
    memory_bytes: int
    recommended_dtype: str | None


@dataclass(frozen=True, slots=True)
class DatatypeSummary:
    """
    Dataset datatype summary.
    """

    total_columns: int
    numeric_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    float_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    string_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    object_columns: tuple[str, ...]
    mixed_object_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatatypeInspectionResult:
    """
    Complete datatype inspection.
    """

    summary: DatatypeSummary
    reports: tuple[ColumnTypeReport, ...]


class DatatypeInspector:
    """
    Production-grade datatype inspection.
    """

    __slots__: Final = ()

    CATEGORY_THRESHOLD = 0.50

    def inspect(
        self,
        dataframe: pd.DataFrame,
    ) -> DatatypeInspectionResult:

        if dataframe.empty:
            raise ValueError("Cannot inspect an empty DataFrame.")

        reports: list[ColumnTypeReport] = []

        numeric = []
        integers = []
        floats = []
        booleans = []
        categoricals = []
        strings = []
        datetimes = []
        objects = []
        mixed_objects = []

        for column in dataframe.columns:

            series = dataframe[column]

            dtype = str(series.dtype)

            semantic = self._semantic_type(series)

            if semantic == "integer":
                integers.append(column)
                numeric.append(column)

            elif semantic == "float":
                floats.append(column)
                numeric.append(column)

            elif semantic == "boolean":
                booleans.append(column)

            elif semantic == "category":
                categoricals.append(column)

            elif semantic == "string":
                strings.append(column)

            elif semantic == "datetime":
                datetimes.append(column)

            elif semantic == "object":
                objects.append(column)

                if self._is_mixed_object(series):
                    mixed_objects.append(column)

            reports.append(
                ColumnTypeReport(
                    column=column,
                    dtype=dtype,
                    semantic_type=semantic,
                    nullable=series.hasnans,
                    missing_count=int(series.isna().sum()),
                    unique_count=int(series.nunique(dropna=True)),
                    memory_bytes=int(series.memory_usage(deep=True)),
                    recommended_dtype=self._recommend_dtype(series),
                )
            )

        summary = DatatypeSummary(
            total_columns=len(dataframe.columns),
            numeric_columns=tuple(numeric),
            integer_columns=tuple(integers),
            float_columns=tuple(floats),
            boolean_columns=tuple(booleans),
            categorical_columns=tuple(categoricals),
            string_columns=tuple(strings),
            datetime_columns=tuple(datetimes),
            object_columns=tuple(objects),
            mixed_object_columns=tuple(mixed_objects),
        )

        return DatatypeInspectionResult(
            summary=summary,
            reports=tuple(reports),
        )

    def _semantic_type(
        self,
        series: pd.Series,
    ) -> str:

        if is_integer_dtype(series):
            return "integer"

        if is_float_dtype(series):
            return "float"

        if is_bool_dtype(series):
            return "boolean"

        if is_datetime64_any_dtype(series):
            return "datetime"

        if is_categorical_dtype(series):
            return "category"

        if is_string_dtype(series):
            return "string"

        if is_numeric_dtype(series):
            return "numeric"

        if is_object_dtype(series):
            return "object"

        return "unknown"

    def _is_mixed_object(
        self,
        series: pd.Series,
    ) -> bool:

        values = series.dropna()

        if values.empty:
            return False

        return values.map(type).nunique() > 1

    def _recommend_dtype(
        self,
        series: pd.Series,
    ) -> str | None:

        if is_object_dtype(series):

            ratio = (
                series.nunique(dropna=True)
                / max(len(series), 1)
            )

            if ratio < self.CATEGORY_THRESHOLD:
                return "category"

            return "string"

        if is_integer_dtype(series):
            return "Int64"

        if is_float_dtype(series):
            return "Float64"

        return None