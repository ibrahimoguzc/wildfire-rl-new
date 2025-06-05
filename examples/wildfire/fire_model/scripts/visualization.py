"""Contains functions used for sisualization of the fire model."""

import math
import os
from timeit import default_timer as time

# import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import cuda

from examples.wildfire.fire_model.jit_funcs.cpu import step as step_cpu
from examples.wildfire.fire_model.jit_funcs.gpu import (
    compute_kernel,
    step_kernel,
)
from examples.wildfire.fire_model.states import BURNT, CMAP, FULL_BURNING
from examples.wildfire.paths import FIGURE_DIR

# TODO define neighborhood and radius parameters globally
RADIUS = 1
MAX_ITER = 1000


def animation_gpu(
    host_arrays,
    device_arrays,
    time_step: float,
    cell_size: int,
    correction_coeff: float | None = 1,
) -> FuncAnimation:
    """Run the fire model on the GPU and create an animation."""
    fig, ax = plt.subplots()
    frames = []

    threadsperblock = (32, 32)
    fire_states = device_arrays["fire_states"].copy_to_host()
    blockspergrid_x = math.ceil((fire_states.shape[0]) / threadsperblock[0])
    blockspergrid_y = math.ceil((fire_states.shape[1]) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    step_time = 0
    start = time()
    stream = cuda.stream()
    for _ in range(MAX_ITER):
        # State-Array, Transition Array, Rate-Array (s, t, r)
        step_start = time()
        with stream.auto_synchronize():
            compute_kernel[blockspergrid, threadsperblock, stream](
                correction_coeff,
                device_arrays["fire_states"],
                device_arrays["spread_rates"],
                device_arrays["temperatures"],
                device_arrays["wind_speeds"],
                device_arrays["wind_aspects"],
                device_arrays["humidities"],
                device_arrays["terrain_slopes"],
                device_arrays["terrain_aspects"],
                device_arrays["combustibilities"],
                device_arrays["prop_aspect"],
                device_arrays["can_ignite"],
                device_arrays["can_extinguish"],
            )
            step_kernel[blockspergrid, threadsperblock, stream](
                time_step,
                cell_size,
                device_arrays["fire_states"],
                device_arrays["spread_rates"],
                device_arrays["intermediate_states"],
                device_arrays["can_ignite"],
                device_arrays["can_extinguish"],
            )
        step_stop = time()
        step_time += step_stop - step_start

        # FIXME transferring data to and from the GPU is the bottleneck
        frames += [
            device_arrays["fire_states"].copy_to_host(
                host_arrays["fire_states"].copy()
            )
        ]

    stop = time()

    print(
        f"Fire-Spread simulation complete, performed {MAX_ITER} iterations in "
        f"{stop - start} s"
    )

    print(f"Avg. step-time = {step_time / MAX_ITER} ")

    ax = plt.imshow(frames[0], vmin=0, vmax=BURNT, cmap=CMAP)

    def update(frame):
        ax.set_data(frame)
        return (ax,)

    return FuncAnimation(
        fig, update, frames=frames[1:], interval=0.1, repeat=True, blit=True
    )


def animation_cpu(
    arrays,
    time_step: float,
    cell_size: int,
    n_burning: int = 1,
    correction_coeff: float | None = 1,
) -> FuncAnimation:
    fig, ax = plt.subplots()
    frames = []

    step_time = 0
    start = time()
    for _ in range(MAX_ITER):
        start_step = time()
        n_burning = step_cpu(
            time_step,
            **arrays,
            cell_size=cell_size,
            correction_coefficient=correction_coeff,
            n_burning=n_burning,
        )
        stop_step = time()
        step_time += stop_step - start_step
        frames += [arrays["fire_states"].copy()]

    stop = time()

    print(
        f"Fire-Spread simulation complete, performed {MAX_ITER} iterations in "
        f"{stop - start} s"
    )

    print(f"Avg. step-time = {step_time / MAX_ITER} ")

    ax = plt.imshow(frames[0], vmin=0, vmax=BURNT, cmap=CMAP)

    def update(frame):
        ax.set_data(frame)
        return (ax,)

    return FuncAnimation(
        fig, update, frames=frames[1:], interval=0.1, repeat=True, blit=True
    )


def plot_comparison(cpu_fire_states: np.ndarray, gpu_fire_states: np.ndarray):
    """Creates a time complexity plot based on benchmark ``results``."""
    plot_data = {
        "FireModelCPUResult": cpu_fire_states,
        "FireModelGPUResult": gpu_fire_states,
    }
    for label, data in plot_data.items():
        fig = plt.figure(label)
        plt.imshow(data, vmin=0, vmax=BURNT, cmap=CMAP)
        plt.xlabel("Index along j-axis [-]")
        plt.ylabel("Index along i-axis [-]")
        n_burnt = np.sum(data == BURNT)
        n_rows, n_cols = data.shape
        ax = plt.gca()
        ax.text(
            n_cols * 0.5,
            n_rows * 0.9,
            f"Number of Burnt Cells = {n_burnt}",
            horizontalalignment="center",
            color="w",
        )
        ax.set_title(label)
        fig.savefig(
            fname=os.path.join(FIGURE_DIR, f"{fig.get_label()}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    return "Figures Plotted and Saved"


if __name__ == "__main__":
    from examples.wildfire.fire_model.preallocate import (
        preallocate_cpu,
        preallocate_gpu,
    )

    width = 512
    height = 512

    INPUT_DATA = {
        "shape": (height, width),
        "ambient_temperature": 35,
        "wind_speed": 10,
        "wind_aspect": 135,
        "relative_humidity": 45,
        "terrain_slope": 0,
        "terrain_aspect": 0,
        "avg_combustibility": 1.8,
        "stochastic": True,
        # float32 is faster than float64
        # (Anaconda introduction to GPU programming)
        "dtype": np.float64,
    }

    # GPU Model
    np.random.seed(1)  # Ensuring output is predictable
    host_arrays, device_arrays = preallocate_gpu(**INPUT_DATA)
    device_arrays["fire_states"][height // 2, width // 2] = FULL_BURNING
    ani = animation_gpu(
        host_arrays, device_arrays, time_step=5 / 60, cell_size=10
    )
    ani.save(FIGURE_DIR / "FireModelVerificationGPU.mp4", fps=120)
    device_arrays["fire_states"].copy_to_host(host_arrays["fire_states"])
    print(np.sum(host_arrays["fire_states"] == BURNT))

    # CPU Model
    np.random.seed(1)  # Ensuring output is predictable
    arrays = preallocate_cpu(**INPUT_DATA)
    arrays["fire_states"][height // 2, width // 2] = FULL_BURNING
    arrays["fire_indices"][0] = (height // 2, width // 2)
    ani = animation_cpu(arrays, time_step=5 / 60, cell_size=10, n_burning=1)
    ani.save(FIGURE_DIR / "FireModelVerificationCPU.mp4", fps=120)
    print(np.sum(arrays["fire_states"] == BURNT))

    plot_comparison(
        cpu_fire_states=arrays["fire_states"],
        gpu_fire_states=host_arrays["fire_states"],
    )
    plt.show()
