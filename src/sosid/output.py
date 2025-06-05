# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains output data related classes."""

from __future__ import annotations

import datetime
import json
import warnings
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import msgpack
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import deprecated

T = TypeVar("T")


class OutputFormat(Enum):
    """Defines the output format."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    MSGPACK = "msgpack"
    FEATHER = "feather"


class TargetKey(Enum):
    """Defines py:class:`Output` classification.

    Gives meaningful descriptions to the user inputs to access
    target_key values.
    """

    AGENTS = "agents"
    SIMULATION = "simulation"


class Output(Generic[T]):
    """Descriptor to annotate methods as output data."""

    def __init__(
        self,
        method: Callable[[Any], T] | None = None,
        target_key: TargetKey = TargetKey.SIMULATION,
    ):
        self.method = method
        self.target_key = target_key

    def __call__(self, method: Callable[[Any], T]) -> Output[T]:
        """Allows the class to be callable.

        Args:
            method (Callable[[Any], T]): The method to be annotated.

        Returns:
            Output[T]: The output descriptor instance.
        """
        self.method = method
        return self

    def __set_name__(self, owner, name: str) -> None:
        """Sets the name of the attribute.

        Args:
            name (str): The name of the attribute.
        """
        self.name = name

    @overload
    def __get__(self, obj: None, owner: Any) -> Output[T]: ...

    @overload
    def __get__(self, obj: Any, owner: Any) -> T: ...

    def __get__(self, obj: None | Any, owner: Any) -> Output[T] | T:
        """Retrieves the output value.

        Args:
            obj (Union[None, Any]): The instance object.
            owner (Any): The owner class.

        Returns:
            Union[Output[T], T]: The output value or the descriptor instance.
        """
        if obj is None:
            return self
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]
        if self.method is None:
            raise AttributeError(f"Method for {self.name} is not defined")
        try:
            return self.method(obj)
        except ZeroDivisionError:
            warnings.warn(
                f"Output method {self.name} lead to ZeroDivisionError, returning default value 0"
            )
            return 0

    def __set__(self, obj: Any, value: T) -> None:
        """Allows outputs to be settable.

        Useful for situations where the Output can be set during a simulation.

        Args:
            obj (Any): The instance object.
            value (T): The value to be set.
        """
        obj.__dict__[self.name] = value

    @staticmethod
    def save_dict(
        data: dict, output_path: Path, output_format: OutputFormat
    ) -> None:
        """Saves a dictionary in the specified output format.

        Args:
            data (dict): The data to be saved.
            output_path (Path): The path of the output file.
            output_format (OutputFormat): The format to save the data in.
        """
        save_function = FORMAT_TO_FUNCTION.get(output_format)
        if save_function:
            save_function(data, output_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    @staticmethod
    def dict_to_json(data: dict, output_path: Path) -> None:
        """Converts a dictionary to a JSON file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """
        json_data = json.dumps(data, indent=4, default=str)
        with open(output_path, "w") as file:
            file.write(json_data)

    @staticmethod
    def dict_to_csv(data: dict, output_path: Path) -> None:
        """Converts a dictionary to a CSV file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """
        df = pd.DataFrame([data])
        df.to_csv(output_path, index=False)

    @staticmethod
    def dict_to_parquet(data: dict, output_path: Path) -> None:
        """Converts a dictionary to a Parquet file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """
        df = pd.DataFrame([data])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)

    @staticmethod
    def dict_to_hdf5(data: dict, output_path: Path) -> None:
        """Converts a dictionary to an HDF5 file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """
        df = pd.DataFrame([Output._prepare_data_for_hdf5(data)])
        df.to_hdf(output_path, key="data", mode="w", format="table")

    @staticmethod
    def _prepare_data_for_hdf5(data: dict) -> dict:
        """Prepares data for HDF5 format by converting non-supported types.

        Args:
            data (dict): The data to be converted.

        Returns:
            dict: The prepared data.
        """

        def convert(value):
            if isinstance(value, list):
                return str(value)  # Convert lists to strings
            if isinstance(value, dict):
                return json.dumps(
                    {k: convert(v) for k, v in value.items()}
                )  # Convert dicts to JSON strings
            if isinstance(value, datetime.datetime):
                return (
                    value.isoformat()
                )  # Convert datetime to ISO format string
            if isinstance(value, (np.integer, np.floating)):
                return (
                    value.item()
                )  # Convert numpy types to native Python types
            if isinstance(value, np.ndarray):
                return value.tolist()  # Convert numpy arrays to lists
            return value

        return {k: convert(v) for k, v in data.items()}

    @staticmethod
    def dict_to_msgpack(data: dict, output_path: Path) -> None:
        """Converts a dictionary to a MSGPACK file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """

        def convert_value(value):
            if isinstance(value, (np.integer, np.floating)):
                return (
                    value.item()
                )  # Direct conversion to native Python int/float
            if isinstance(value, np.ndarray):
                return value.tolist()  # Convert numpy array to list
            if isinstance(value, datetime.datetime):
                return value.isoformat()  # Convert datetime to string
            if isinstance(value, list):
                # Recursively convert all items in the list
                return [convert_value(v) for v in value]
            if isinstance(value, dict):
                # Recursively convert all key-value pairs in the dictionary
                return {k: convert_value(v) for k, v in value.items()}
            return value  # Return the value as is if no conversion is needed

        # Convert the dictionary only if necessary
        converted_data = {k: convert_value(v) for k, v in data.items()}

        with open(output_path, "wb") as out:
            packed = msgpack.packb(converted_data)
            out.write(packed)

    @staticmethod
    def dict_to_feather(data: dict, output_path: Path) -> None:
        """Converts a dictionary to a Feather file.

        Args:
            data (dict): The data to be converted.
            output_path (Path): The path of the output file.
        """
        df = pd.DataFrame([data])
        df.to_feather(output_path)


@deprecated(
    "This method is deprecated because it is no longer efficient for large datasets."
)
def dict_to_json(data: dict, output_path: Path) -> None:  # noqa 103
    json_data = json.dumps(data, indent=4, default=str)
    with open(output_path, "w") as file:
        file.write(json_data)


@deprecated(
    "This method is deprecated because it is no longer efficient for large datasets."
)
def pd_to_csv(  # noqa 103
    data: pd.DataFrame,
    output_filepath: Path,
    header: bool,
    mode: str,
    sep: str,
) -> None:
    while True:
        try:
            data.to_csv(
                path_or_buf=output_filepath,
                header=header,
                sep=sep,
                mode=mode,
                na_rep="",
            )
        except PermissionError:
            print(
                f"PermissionError. Access to the output {output_filepath} "
                "file denied."
            )
            input("Try closing the file and press Enter to continue.")
        else:
            print("CSV Data exported successfully.\n")
            break


def mean_with_default(array, default: float | None = 0):
    """Compute mean and return default if `array` is empty."""
    if len(array) > 0:
        return float(np.mean(array))
    return default


def sum_with_default(array, default: float | None = 0):
    """Compute Sum and return default if `array` is empty."""
    if len(array) > 0:
        return float(sum(array))
    return default


FORMAT_TO_FUNCTION = {
    OutputFormat.JSON: Output.dict_to_json,
    OutputFormat.CSV: Output.dict_to_csv,
    OutputFormat.PARQUET: Output.dict_to_parquet,
    OutputFormat.HDF5: Output.dict_to_hdf5,
    OutputFormat.MSGPACK: Output.dict_to_msgpack,
    OutputFormat.FEATHER: Output.dict_to_feather,
}
