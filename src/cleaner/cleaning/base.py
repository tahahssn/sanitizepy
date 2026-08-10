from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class CleaningOperation(ABC):
    """
    Base contract for all dataframe cleaning operations.

    A cleaning operation receives a pandas DataFrame and returns a cleaned
    DataFrame. Implementations must not mutate the input DataFrame unless
    explicitly documented as part of their contract.
    """

    name: str

    @abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the cleaning operation to a dataframe.

        Parameters
        ----------
        dataframe:
            The dataframe to transform.

        Returns
        -------
        pandas.DataFrame
            The transformed dataframe.

        Raises
        ------
        TypeError
            If the supplied object is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

    def describe(self) -> dict[str, Any]:
        """
        Return a serializable description of the operation.

        The description is intentionally lightweight so that the cleaning
        engine and reporting layer can record which operations were applied
        without coupling themselves to concrete operation implementations.
        """
        return {
            "name": self.name,
        }