# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/
from __future__ import annotations

import concurrent.futures
import gc
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import msgpack
import numpy as np
import pandas as pd

from sosid.output import Output, OutputFormat, TargetKey
from sosid.util.abc import ABCMeta
from sosid.util.general_funcs import combine_parameters

if TYPE_CHECKING:
    from sosid.simulation.simulation import Simulation, SimulationParameters


class DesignOfExperiments(metaclass=ABCMeta):
    """A class for running a design of experiments (DoE).

    Args:
        simulation: A SoSID :py:class:`Simulation` class.
        parameters: A SoSID :py:class:`SimulationParameters` class.
        input_file: A .json type file containing the simulation
            parameters for each design point in the DoE.
        output_file: A .json or .csv type file to export the simulation
            results to. Can be a non-existing or an already existing
            file. Existing files can either be empty or non-empty.
        multiprocessing: A py:type:`bool` to enable multiprocessing.
            Defaults to `True`.
        n_repeats: A py:type:`int` number of times to run a design
            point, each run uses
            previous seed incremented by 1. Defaults to `1`
        seed_start: A py:type:`int` to use as the first seed for the
            simulation's random number generator. Defaults to `0`.
        output_sim_param: A py:type:`bool` specifying whether the
            simulation parameters should be included in the output file.
            Defaults to `False`.
        batch_size: An int specifying the number of simulation runs to
        process before writing to disk. If 0, it defaults to using the
        index.

    """

    def __init__(
        self,
        simulation: type[Simulation],
        parameters: type[SimulationParameters],
        default_inputs: Path,
        input_file: Path,
        output_file: Path,
        multiprocessing: bool = False,
        n_repeats: int = 1,
        seed_start: int = 0,
        output_sim_param: bool = True,
        batch_size: int = 10,
        output_format: OutputFormat = OutputFormat.JSON,
        flatten: bool = True,
    ):
        self.simulation = simulation
        self.parameters = parameters
        self.input_file = input_file
        self.defaults_file = default_inputs
        self.output_file = output_file
        self.multiprocessing = multiprocessing

        # Number of times to run point with different seed starting from
        # seed_start
        self.n_repeats = n_repeats
        self.seed_start = seed_start
        self.output_sim_param = output_sim_param
        self.batch_size = batch_size
        self.data_dict = defaultdict(dict)
        self.output_format = output_format
        self.flatten = flatten

    def start(self) -> None:
        """Starting the DoE run."""
        # Extracting simulation input parameters from the input .csv
        self.doe_dicts = combine_parameters(
            self.defaults_file, self.input_file
        )

        # Executing a DoE run and exporting results to the output .csv
        self._validate_doe()
        self._execute_doe()

    def _validate_doe(self) -> None:
        # Trigger validation of all DoE parameters before executing DoE
        for design_input in self.doe_dicts.values():
            _ = self.parameters.model_validate(design_input)

    def _execute_doe(self) -> None:
        """Executes the Design of Experiments."""
        self._log_doe_start()
        if self.batch_size == 0:
            self.batch_size = len(
                self.doe_dicts
            )  # Fit batch size to number of design points

        indices, design_points, runs, seeds = self._prepare_doe_run_arrays()

        if self.multiprocessing:
            self._run_in_parallel(indices, design_points, runs, seeds)
        else:
            self._run_sequentially(indices, design_points, runs, seeds)

        self._finalize_doe()

    def _log_doe_start(self) -> None:
        """Logs the start of the DoE run."""
        self.doe_start_time = datetime.now()
        self.total_runs = len(self.doe_dicts) * self.n_repeats
        print(
            "\nDoE execution\n"
            f"Start time: {self.doe_start_time}\n"
            f"Number of data points: {len(self.doe_dicts)}\n"
            f"Total number of simulation runs: {self.total_runs}\n"
        )

    def _run_in_parallel(self, indices, design_points, runs, seeds) -> None:
        """Runs the DoEs in parallel using multiprocessing."""
        print(
            "Multiprocessing enabled.\n"
            "Ensure no conflicting files are open during the DoE run."
        )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(
                self.run_simulation, indices, design_points, runs, seeds
            )
            self._process_results(results)

    def _run_sequentially(self, indices, design_points, runs, seeds) -> None:
        """Runs the DoEs sequentially."""
        for i, index in enumerate(indices):
            key, result = self.run_simulation(
                index=index,
                design_point=design_points[index],
                run=runs[index],
                seed=seeds[index],
            )
            self._store_result(key, result, i)

    def _process_results(self, results) -> None:
        """Processes the results of the experiments."""
        for i, (key, result) in enumerate(results):
            self._store_result(key, result, i)

    def _store_result(self, key: int, result: dict, iteration: int) -> None:
        """Stores a simulation result in memory and writes batches."""
        for k, v in result.items():
            self.data_dict[k][str(key)] = v

        if (iteration + 1) % self.batch_size == 0:
            self._write_and_clear_data()

    def _finalize_doe(self) -> None:
        """Finalizes the DoE by writing any remaining data and logging the completion."""
        self._write_and_clear_data()

        doe_end_time = datetime.now()
        print(
            "\nDoE run completed successfully!",
            f"\nEnd time: {doe_end_time}",
            f"Runtime: {doe_end_time - self.doe_start_time}",
        )

    def _prepare_doe_run_arrays(self):
        """Prepares arrays of indices, design points, runs, and seeds for the DoE."""
        indices = np.arange(self.total_runs)
        design_points = np.repeat(
            np.arange(
                start=0,
                stop=len(self.doe_dicts),
            ),
            self.n_repeats,
        )
        runs = np.concatenate(
            [np.arange(self.n_repeats) for _ in range(len(self.doe_dicts))]
        )
        seeds = runs + self.seed_start
        return indices, design_points, runs, seeds

    def _write_and_clear_data(self):
        """Writes the current batch of data to disk with batch number
        and clears the in-memory dictionary.
        """
        for data_type, result in self.data_dict.items():
            self.write_data(data_dict=result, data_type=data_type)
        self.data_dict.clear()  # Clear the data_dict to free memory

    def write_data(self, data_dict: dict, data_type: str) -> None:
        """Writes the collected data to disk in the specified output format."""
        output_path = self._get_output_file_path(data_type)

        if self.output_format == OutputFormat.JSON:
            self._write_json(data_dict, output_path)
        elif self.output_format == OutputFormat.CSV:
            self._write_csv(data_dict, output_path)
        elif self.output_format == OutputFormat.PARQUET:
            self._write_parquet(data_dict, output_path)
        elif self.output_format == OutputFormat.HDF5:
            self._write_hdf5(data_dict, output_path)
        elif self.output_format == OutputFormat.MSGPACK:
            self._write_msgpack(data_dict, output_path)
        elif self.output_format == OutputFormat.FEATHER:
            self._write_feather(data_dict, output_path)

    def _get_output_file_path(self, data_type: str) -> Path:
        """Constructs the output file path based on the data type and format."""
        return (
            self.output_file.parent
            / f"{self.output_file.stem}_{data_type}.{self.output_format.value}"
        )

    def _write_json(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in JSON array format."""
        if output_path.exists():
            with Path.open(output_path, "r+") as json_file:
                try:
                    existing_data = json.load(json_file)
                    if not isinstance(existing_data, list):
                        raise TypeError("The existing JSON is not a list.")
                except json.JSONDecodeError:
                    existing_data = []

                existing_data.append(data_dict)

                # Write back the updated array
                json_file.seek(0)
                json.dump(existing_data, json_file, indent=4, default=str)
                json_file.truncate()
        else:
            with Path.open(output_path, "w") as json_file:
                json.dump([data_dict], json_file, indent=4, default=str)

    def _write_csv(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in CSV format."""
        new_data = pd.DataFrame.from_dict(data_dict, orient="index")
        if output_path.exists():
            existing_data = pd.read_csv(output_path, index_col=0)

            # Reindex new data to match existing columns
            new_data = new_data.reindex(columns=existing_data.columns)

            combined_data = pd.concat([existing_data, new_data], axis=0)
            combined_data.to_csv(output_path, mode="w", index=True)
        else:
            new_data.to_csv(output_path, mode="w", header=True, index=True)

    def _write_parquet(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in Parquet format, ensuring all dictionary keys are strings."""
        # Convert all keys to strings
        data_dict = self._convert_keys_to_string(data_dict)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([data_dict])

        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df

        combined_df.to_parquet(output_path, engine="auto")

    def _convert_keys_to_string(self, data):
        """Recursively converts all dictionary keys to strings."""
        if isinstance(data, dict):
            return {
                str(k): self._convert_keys_to_string(v)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [self._convert_keys_to_string(item) for item in data]
        return data

    def _write_hdf5(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in HDF5 format."""
        if output_path.exists():
            existing_data = pd.read_hdf(output_path)
            new_data = pd.DataFrame([Output._prepare_data_for_hdf5(data_dict)])
            combined_data = pd.concat([existing_data, new_data])
        else:
            combined_data = pd.DataFrame(
                [Output._prepare_data_for_hdf5(data_dict)]
            )
        combined_data.to_hdf(output_path, key="data", mode="w", format="table")

    def _write_msgpack(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in MsgPack format."""
        if output_path.exists():
            with open(output_path, "rb") as out:
                unpacker = msgpack.Unpacker(
                    out, raw=False, strict_map_key=False
                )
                for unpacked in unpacker:
                    data_dict.update(unpacked)
            Output.dict_to_msgpack(data_dict, output_path)
        else:
            Output.dict_to_msgpack(data_dict, output_path)

    def _write_feather(self, data_dict: dict, output_path: Path) -> None:
        """Writes data in Feather format."""
        # Convert all keys to strings
        data_dict = self._convert_keys_to_string(data_dict)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([data_dict])

        # If the output file already exists, load existing data and concatenate
        if output_path.exists():
            existing_df = pd.read_feather(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df

        # Write the combined data to Feather
        combined_df.to_feather(output_path)

    def run_simulation(
        self, index, design_point, run, seed
    ) -> tuple[int, dict]:
        """Runs a single simulation instance and collects output data.

        This method initializes a simulation instance using the provided
        design point and seed, runs the simulation, and processes the
        output data. It includes additional indexing metadata and
        supports flattening of agent data if the `self.flatten` flag is
        enabled.

        Returns:
            A tuple containing the simulation index and a dictionary of
            processed output data. The output data includes:
            - Simulation results under the `TargetKey.SIMULATION` key.
            - Agent data under the `TargetKey.AGENTS` key, optionally
            flattened.

        Args:
            index (int): The index of the simulation run.
            design_point(int): The index of the design point from the
            input file.
            run (int): The run number for repeated simulations at the
            same design point.
            seed (int): The seed value for the simulation's random
            number generator.

        """
        # Simulation run statement
        print(f"Running simulation {index} ({index + 1}/{self.total_runs})")
        points = list(self.doe_dicts.keys())
        param = self.doe_dicts[points[design_point]]
        parameters = self.parameters(**param)
        sim = self.simulation(parameters=parameters, seed=seed)
        sim.start()
        sim.is_stopped.wait()
        sim_outputs = sim.get_output_data()
        # Adding additional indexing data
        run_data = {
            "entry_time": datetime.now(),
            "design_point": design_point,
            "run": run,
            "n_repeats": self.n_repeats,
        }
        sim_dict = {}
        sim_data = sim_outputs[TargetKey.SIMULATION.value]
        sim_dict[TargetKey.SIMULATION.value] = deepcopy(sim_data)
        agent_dict = {}
        if self.flatten:
            for agent_type, agent_data in sim_outputs[
                TargetKey.AGENTS.value
            ].items():
                flattened_agent_data = sim.flatten_dict(
                    data=agent_data, parent_key=agent_type
                )
                flattened_agent_data.update(run_data)
                flattened_agent_data = deepcopy(flattened_agent_data)
                agent_dict[agent_type] = flattened_agent_data
        else:
            agent_dict.update(sim_outputs[TargetKey.AGENTS.value])

        complete_data = {**sim_dict, **agent_dict}
        complete_data = deepcopy(complete_data)

        # Returns the number of objects it has collected and deallocated
        _ = gc.collect()
        return index, complete_data
