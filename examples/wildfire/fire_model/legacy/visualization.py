import numpy as np

try:
    from examples.wildfire.fire_model.legacy.inplace import (
        propagate as propagate_inplace,
    )
    from examples.wildfire.fire_model.legacy.padded import (
        propagate as propagate_padded,
    )
    from examples.wildfire.fire_model.legacy.states import burnt, full_burning
except ModuleNotFoundError:
    import os
    import sys

    sys.path.insert(0, os.getcwd())
    from examples.wildfire.fire_model.legacy.inplace import (
        propagate as propagate_inplace,
    )
    from examples.wildfire.fire_model.legacy.padded import (
        propagate as propagate_padded,
    )
    from examples.wildfire.fire_model.legacy.states import burnt, full_burning

""" Contains the functions required for preliminary visualization of the
fire-propagation model """


def animation_padded(
    arrays, cell_size: int, correction_coeff: float | None = 1
) -> None:
    from timeit import default_timer as time

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    frames = []
    propagating, max_iter = True, 1000

    i = 0
    start = time()
    while propagating and i < max_iter:
        # State-Array, Transition Array, Rate-Array (s, t, r)
        s, t, r = propagate_padded.propagate(
            **arrays, cell_size=cell_size, correction_coeff=correction_coeff
        )
        i += 1
        if r is None:
            propagating = False
        else:
            # arrays['transition_array'] = t
            frames += [s.copy()]
    stop = time()

    print(
        f"Fire-Spread simulation complete, performed {i} iterations in "
        f"{stop - start} s"
    )

    ax = plt.imshow(frames[0], vmin=0, vmax=burnt)

    def update(frame):
        ax.set_data(frame)
        return (ax,)

    ani = FuncAnimation(
        fig, update, frames=frames[1:], interval=0.1, repeat=True, blit=True
    )
    plt.show()


def animation_inplace(
    arrays, cell_size: int, correction_coeff: float | None = 1
) -> None:
    from timeit import default_timer as time

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    frames = []
    propagating, max_iter = True, 1000

    i = 0
    start = time()
    while propagating and i < max_iter:
        # State-Array, Transition Array, Rate-Array (s, t, r)
        s, t, r = propagate_inplace.propagate(
            **arrays, cell_size=cell_size, correction_coeff=correction_coeff
        )
        i += 1
        if r is None:
            propagating = False
        else:
            # arrays['transition_array'] = t
            frames += [s.copy()]
    stop = time()

    print(
        f"Fire-Spread simulation complete, performed {i} iterations in "
        f"{stop - start} s"
    )

    ax = plt.imshow(frames[0], vmin=0, vmax=burnt)

    def update(frame):
        ax.set_data(frame)
        return (ax,)

    ani = FuncAnimation(
        fig, update, frames=frames[1:], interval=0.1, repeat=True, blit=True
    )
    plt.show()


if __name__ == "__main__":
    height = 256
    width = 256

    INPUTS = {
        "shape": (height, width),
        "ambient_temperature": 35,
        "wind_speed": 0,
        "wind_direction": (-1, -1),
        "relative_humidity": 50,
        "avg_combustibility": 0.2,
        "stochastic": False,
    }

    np.random.seed(0)

    # Inplace Fire Model
    arrays = propagate_inplace.initialize_arrays(**INPUTS)
    arrays["state_array"][height // 2, width // 2] = full_burning
    animation_inplace(arrays, cell_size=10)

    # Padded Fire Model
    arrays = propagate_padded.initialize_arrays(**INPUTS)
    arrays["state_array"][height // 2, width // 2] = full_burning
    animation_padded(arrays, cell_size=10)
