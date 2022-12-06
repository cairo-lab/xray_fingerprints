# -*- coding: utf-8 -*-

"""
Created December 02, 2022
"""

import os
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd


def binarize_koos(koos_value: float, score=86.1) -> float:
    """Create binned koos score.

    Args:
        koos_value (float): Raw koos score.
        score (float, optional): Cutoff value for positive koos. Defaults to 86.1.

    Returns:
        float: 1.0 or 0.0 indicating binned koos value.
    """

    try:
        return 1.0 * (koos_value <= score)
    except:
        return np.nan


def read_data_frame(file_path: str) -> Optional[pd.DataFrame]:
    """Read pandas dataframe if correct filename."""

    if not file_path.endswith(".txt"):
        return None

    df = pd.read_csv(os.path.join(file_path), sep="|")
    df = df.rename(columns=str.lower)  # type: ignore

    if len(df.columns) != len(set(df.columns)):
        raise ValueError(f"duplicate column names found for dataframe {file_path}")

    return df


def subset_dataframe(
    df: pd.DataFrame, columns: list[str], copy: bool = True
) -> pd.DataFrame:
    """Subset dataframe based on list of column keys."""

    if copy:
        return df[columns].copy()

    return df[columns]


def are_columns_consistent(visit_number: str, columns: list[str]) -> bool:
    return all(
        [
            f"v{visit_number}" in a
            for a in columns
            if a not in ["id", "side", "readprj", "version"]
            and a[:3] not in ["p01", "p02"]
        ]
    )


def validate_column(
    column: pd.Series, expected_values: Union[list[str], list[int]] = []
) -> None:
    """Utility to validate column values against expected values."""

    if not set(column.dropna().unique()).issubset(set(expected_values)):
        raise ValueError(
            (
                f"unexpected values in column: {column.dropna().unique()} -- "
                f"expected values: {expected_values}"
            )
        )


def concatenate_from_timepoints(
    originals: dict[str, pd.DataFrame],
    dataset_substring: str,
    subset: list[str] = [],
) -> pd.DataFrame:
    """Takes all datasets in original_dataframes that contain dataset_substring,
    takes the columns in columns_to_subset_on, and adds a column called "visit"
    which denotes which visit it is.
    """

    concatenation_list: list[pd.DataFrame] = []

    for dataset_name in originals:
        if dataset_substring not in dataset_name:
            continue

        visit_number = dataset_name.replace(dataset_substring, "")

        df = originals[dataset_name].copy()

        if not are_columns_consistent(visit_number, df.columns.tolist()):
            raise ValueError(f"columns {df.columns} not consistent")

        df.columns = df.columns.str.replace(f"v{visit_number}", "")

        if len(subset):
            df = subset_dataframe(df, subset, copy=False)

        df["visit"] = visit_number
        concatenation_list.append(df)

    combined_data = pd.concat(concatenation_list, axis=0)
    combined_data.index = range(len(combined_data))  # type: ignore
    combined_data["id"] = combined_data["id"].astype("string")

    return combined_data


def create_timestring() -> str:
    """Create a string based on the current timestamp.

    Returns
    -------
    str
        Current timestamp string formatted.
    """

    return (
        str(datetime.now())
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )
