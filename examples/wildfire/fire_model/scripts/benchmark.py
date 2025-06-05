"""This file benchmarks various versions of the CA fire-models."""

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from timeit import default_timer as time

import numpy as np
from matplotlib import pyplot as plt
from numba import cuda
from tqdm import tqdm

from examples.wildfire.fire_model import preallocate
from examples.wildfire.fire_model.jit_funcs import cpu, gpu
from examples.wildfire.fire_model.legacy.inplace import (
    propagate as propagate_inplace,
)
from examples.wildfire.fire_model.legacy.padded import (
    propagate as propagate_padded,
)
from examples.wildfire.paths import DATA_DIR, FIGURE_DIR
from sosid.util.abc import ABCMeta, abstractmethod


class FireModelBenchmark(metaclass=ABCMeta):
    """Abstract Base Class (ABC) of a single FireModel run."""

    ambient_temperature = 15
    wind_speed = 0
    relative_humidity = 35
    combustibility = 1
    cell_size = 30
    correction_coefficient = 1
    time_step = 1 / 60

    def __init__(self, shape: tuple[int, int], n_iters: int = 1000):
        self.n_iters = n_iters
        self.shape = shape
        self.run_time = None  # SI second
        self.iter_time = None  # SI second

    @property
    def n_cells(self):
        """Returns the number of cells of the current Fire-Model."""
        return self.shape[0] * self.shape[1]

    @property
    @abstractmethod
    def data(self):
        """Responsible for returning a dictionary of data."""

    @property
    @abstractmethod
    def plot_name(self) -> str:
        """Provides a plot-friendly name for the fire-model."""

    @abstractmethod
    def warmup(self):
        """Gets complication time out of the way for JIT functions."""

    @abstractmethod
    def run(self):
        """Runs synthetic benchmark for :py:attr:`n_cells`."""


class CPUFireModelBenchmark(FireModelBenchmark):
    """Benchmarks the final CPU Fire Model."""

    @property
    def data(self):  # noqa: D102
        return preallocate.preallocate_cpu(
            shape=self.shape,
            ambient_temperature=self.ambient_temperature,
            wind_speed=self.wind_speed,
            wind_aspect=0,
            relative_humidity=self.relative_humidity,
            terrain_aspect=0,
            terrain_slope=0,
            avg_combustibility=self.combustibility,
            stochastic=False,
            dtype=np.float64,
        )

    @property
    def plot_name(self):  # noqa: D102
        return "CPU v2"

    def warmup(self):
        """Gets compilation time out of the way."""
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        cpu.step(
            time_step,
            **data,
            cell_size=cell_size,
            correction_coefficient=correction_coeff,
        )

    def run(self):  # noqa: D102
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        self.warmup()
        run_start = time()
        iter_time = 0
        for _ in range(self.n_iters):
            iter_start = time()
            cpu.step(
                time_step,
                **data,
                cell_size=cell_size,
                correction_coefficient=correction_coeff,
            )
            iter_end = time()
            iter_time += iter_end - iter_start
        run_end = time()
        self.run_time = run_end - run_start
        self.iter_time = iter_time / self.n_iters


class GPUFireModelBenchmark(FireModelBenchmark):
    """Benchmarks the final CPU Fire Model."""

    threadsperblock = (16, 16)

    @property
    def blockspergrid(self):  # noqa: D102
        height, width = self.shape
        blockspergrid_x = math.ceil(height / self.threadsperblock[0])
        blockspergrid_y = math.ceil(width / self.threadsperblock[1])
        return blockspergrid_x, blockspergrid_y

    @property
    def data(self):  # noqa: D102
        _, device_arrays = preallocate.preallocate_gpu(
            shape=self.shape,
            ambient_temperature=self.ambient_temperature,
            wind_speed=self.wind_speed,
            wind_aspect=0,
            relative_humidity=self.relative_humidity,
            terrain_aspect=0,
            terrain_slope=0,
            avg_combustibility=self.combustibility,
            stochastic=False,
            dtype=np.float64,
        )
        return device_arrays

    @property
    def plot_name(self):  # noqa: D102
        return "GPU"

    def warmup(self):
        """Gets compilation time out of the way."""
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        stream = cuda.stream()
        bpg, tpb = self.blockspergrid, self.threadsperblock
        with stream.auto_synchronize():
            gpu.compute_kernel[bpg, tpb, stream](
                correction_coeff,
                data["fire_states"],
                data["spread_rates"],
                data["temperatures"],
                data["wind_speeds"],
                data["wind_aspects"],
                data["humidities"],
                data["terrain_slopes"],
                data["terrain_aspects"],
                data["combustibilities"],
                data["prop_aspect"],
                data["can_ignite"],
                data["can_extinguish"],
            )
            gpu.step_kernel[bpg, tpb, stream](
                time_step,
                cell_size,
                data["fire_states"],
                data["spread_rates"],
                data["intermediate_states"],
                data["can_ignite"],
                data["can_extinguish"],
            )

    def run(self):  # noqa: D102
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        stream = cuda.stream()
        bpg, tpb = self.blockspergrid, self.threadsperblock
        self.warmup()
        run_start = time()
        iter_time = 0
        for _ in range(self.n_iters):
            with stream.auto_synchronize():
                iter_start = time()
                gpu.compute_kernel[bpg, tpb, stream](
                    correction_coeff,
                    data["fire_states"],
                    data["spread_rates"],
                    data["temperatures"],
                    data["wind_speeds"],
                    data["wind_aspects"],
                    data["humidities"],
                    data["terrain_slopes"],
                    data["terrain_aspects"],
                    data["combustibilities"],
                    data["prop_aspect"],
                    data["can_ignite"],
                    data["can_extinguish"],
                )
                gpu.step_kernel[bpg, tpb, stream](
                    time_step,
                    cell_size,
                    data["fire_states"],
                    data["spread_rates"],
                    data["intermediate_states"],
                    data["can_ignite"],
                    data["can_extinguish"],
                )
            iter_end = time()
            iter_time += iter_end - iter_start
        run_end = time()
        self.run_time = run_end - run_start
        self.iter_time = iter_time / self.n_iters


class CPUPaddedFireModelBenchmark(FireModelBenchmark):
    """Benchmarks the Padded CPU Fire Model."""

    @property
    def data(self):  # noqa: D102
        return propagate_padded.initialize_arrays(
            shape=self.shape,
            ambient_temperature=self.ambient_temperature,
            wind_speed=self.wind_speed,
            wind_direction=(1, 0),
            relative_humidity=self.relative_humidity,
            avg_combustibility=self.combustibility,
            stochastic=False,
        )

    @property
    def plot_name(self):  # noqa: D102
        return "CPU v1 [Padded]"

    def warmup(self):
        """Gets compilation time out of the way."""
        propagate = propagate_padded.propagate
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        propagate(
            **data,
            cell_size=cell_size,
            correction_coeff=correction_coeff,
            time_step=time_step,
        )

    def run(self):  # noqa: D102
        propagate = propagate_padded.propagate
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        self.warmup()
        run_start = time()
        iter_time = 0
        for _ in range(self.n_iters):
            iter_start = time()
            propagate(
                **data,
                cell_size=cell_size,
                correction_coeff=correction_coeff,
                time_step=time_step,
            )
            iter_end = time()
            iter_time += iter_end - iter_start
        run_end = time()
        self.run_time = run_end - run_start
        self.iter_time = iter_time / self.n_iters


class CPUInplaceFireModelBenchmark(FireModelBenchmark):
    """Benchmarks the Padded CPU Fire Model."""

    @property
    def data(self):  # noqa: D102
        return propagate_inplace.initialize_arrays(
            shape=self.shape,
            ambient_temperature=self.ambient_temperature,
            wind_speed=self.wind_speed,
            wind_direction=(1, 0),
            relative_humidity=self.relative_humidity,
            avg_combustibility=self.combustibility,
            stochastic=False,
        )

    @property
    def plot_name(self):  # noqa: D102
        return "CPU v1 [Inplace]"

    def warmup(self):
        """Gets compilation time out of the way."""
        propagate = propagate_inplace.propagate
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        propagate(
            **data,
            cell_size=cell_size,
            correction_coeff=correction_coeff,
            time_step=time_step,
        )

    def run(self):  # noqa: D102
        propagate = propagate_inplace.propagate
        data = self.data
        cell_size = self.cell_size
        correction_coeff = self.correction_coefficient
        time_step = self.time_step
        self.warmup()
        run_start = time()
        iter_time = 0
        for _ in range(self.n_iters):
            iter_start = time()
            propagate(
                **data,
                cell_size=cell_size,
                correction_coeff=correction_coeff,
                time_step=time_step,
            )
            iter_end = time()
            iter_time += iter_end - iter_start
        run_end = time()
        self.run_time = run_end - run_start
        self.iter_time = iter_time / self.n_iters


BENCHMARK_CLASSES = [
    CPUFireModelBenchmark,
    GPUFireModelBenchmark,
    CPUPaddedFireModelBenchmark,
    CPUInplaceFireModelBenchmark,
]


# TODO refractor serialization (saving) into new function
def benchmark(shapes=[(2**i, 2**i) for i in range(7, 12)], n_iters=1000):
    """Runs benchmarks defined by :py:const:`BENCHMARK_CLASSES`.

    The benchmark results dictionary is serialzied into .json and saved
    in the data directory specified by :py:const:`DATA_DIR`.
    """
    results = defaultdict(dict)

    for benchmark_cls in tqdm(BENCHMARK_CLASSES):
        run_times = []
        cell_counts = []
        iter_times = []
        for shape in tqdm(shapes):
            bench = benchmark_cls(shape=shape, n_iters=n_iters)
            bench.run()
            run_times.append(bench.run_time)
            cell_counts.append(bench.n_cells)
            iter_times.append(bench.iter_time)

        plot_name = bench.plot_name
        results[plot_name]["run_times"] = run_times
        results[plot_name]["cell_counts"] = cell_counts
        results[plot_name]["iter_times"] = iter_times

    formatted_date = "_".join(datetime.now().isoformat().split(":")[0:2])
    filename = f"benchmark_{formatted_date}.json"
    with open(os.path.join(DATA_DIR, "benchmark", filename), "w") as fp:
        json.dump(results, fp, indent=4)

    return results


def plot_benchmark(results: dict | str):
    """Creates a time complexity plot based on benchmark ``results``."""
    if isinstance(results, str) and os.path.exists(results):
        with open(results) as fp:
            results = json.load(fp)

    fig = plt.figure("FireModelBenchmark")
    plt.style.use("ggplot")
    markers = ["o", "^", "x", "D"]
    for i, (name, data) in enumerate(results.items()):
        plt.plot(
            data["cell_counts"],
            data["run_times"],
            label=name,
            marker=markers[i % len(markers)],  # Cycles markers
        )
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"Number of Cells [-]")
    plt.ylabel(r"Runtime [s]")
    plt.title("Synthetic Benchmark of Fire Models")
    plt.legend(loc="best")
    plt.show()
    fig.savefig(
        fname=os.path.join(FIGURE_DIR, f"{fig.get_label()}.pdf"),
        format="pdf",
    )
    return "Figure Plotted and Saved"


if __name__ == "__main__":
    results = benchmark(shapes=[(2**i, 2**i) for i in range(7, 12)])
    plot_benchmark(
        os.path.join(DATA_DIR, "benchmark", "benchmark_2019-11-05T18_18.json")
    )
