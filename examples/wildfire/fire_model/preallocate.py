# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains utility functions for pre-allocating fire-model arrays."""

from functools import partial

# import cupy as cp
import numpy as np
from numba import cuda

from examples.wildfire.fire_model.states import COMBUSTIBLE

# from util.total_size import total_size


# TODO consider making this a data-class
# TODO use pattern for dict unpacking from:
# https://stackoverflow.com/questions/52079816/how-to-make-attrs-class-with-tuple-and-dict-unpacking-but-without-extra-methods
# FIXME this will be much cleaner inside of a class
def preallocate_gpu(
    shape: tuple[int, int],
    ambient_temperature: float,
    wind_speed: float,
    wind_aspect: float,
    relative_humidity: float,
    terrain_slope: float,
    terrain_aspect: float,
    avg_combustibility: float,
    stochastic: bool | None = True,
    dtype: np.dtype | None = np.float64,
    order: str | None = "C",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-allocates fire-model arrays on the CPU (host) & GPU (device).

    Simplifies the creation of the arrays required for the forest-fire
    spread model of Rui et al. 2018. This is useful to test the
    forest-fire model with fabricated data sets.

    Args:
        shape: Number of cells in (height, width) format
        ambient_temperature: Ambient temperature in SI Celcius
        wind_speed: Average wind speed in SI meter per second
        wind_aspect: Wind aspect (bearing) in SI degree (North = 0)
        relative_humidity: Relative air humidity in SI percent (%)
        terrain_slope: Incline of terrain in SI degree
        terrain_aspect: Downhill aspect (bearing) in SI degree
        avg_combustibility: Average combustibility of the terrain
            features
        stochastic: Determines if noise is added to the combustibility.
            If stochastic is True, the combustibility will be a random
            number between 0 and 2 and the average combustibility will
            be ignored.
        dtype: Numpy data type of the float arrays
        order: Order of the Numpy arrays (Defaults to C Row-Major)

    Returns:
        Pre-allocated arrays on both the GPU (device) and CPU (host)

    """
    # f_array = float_array, used to reduce boilerplate
    f_array = partial(np.full, shape=shape, dtype=dtype, order=order)

    # Pinned arrays increase bandwidth and reduces transfer latency
    # Refer to How to optimizer data transfers in Cuda C/C++
    def pinned_array(array):
        pinned = cuda.pinned_array(array.shape, array.dtype, order=order)
        pinned[:] = array[:]
        return pinned

    # Initializing Combustibility Array
    combustibilities = f_array(fill_value=avg_combustibility)
    if stochastic:
        combustibilities = np.random.uniform(0, 2, shape).astype(dtype, order)

    bool_array = np.full(
        fill_value=False, shape=shape, dtype=bool, order=order
    )

    host_arrays = {
        "fire_states": pinned_array(
            np.full(shape, fill_value=COMBUSTIBLE, dtype=np.uint8, order=order)
        ),
        "spread_rates": pinned_array(f_array(fill_value=0)),
        # Must start from 1 due to additional nonflammable state
        "intermediate_states": pinned_array(f_array(fill_value=1)),
        "temperatures": pinned_array(f_array(fill_value=ambient_temperature)),
        "wind_speeds": pinned_array(f_array(fill_value=wind_speed)),
        "wind_aspects": pinned_array(f_array(fill_value=wind_aspect)),
        "humidities": pinned_array(f_array(fill_value=relative_humidity)),
        "terrain_slopes": pinned_array(f_array(fill_value=terrain_slope)),
        "terrain_aspects": pinned_array(f_array(fill_value=terrain_aspect)),
        "combustibilities": pinned_array(combustibilities),
        "prop_aspect": pinned_array(f_array(fill_value=np.nan)),
        "can_ignite": pinned_array(bool_array),
        "can_extinguish": pinned_array(bool_array),
    }

    device_arrays = {
        name: cuda.to_device(array) for name, array in host_arrays.items()
    }

    # array_handlers = {
    #     # cp.ndarray: lambda a: a.nbytes,
    #     np.ndarray: lambda a: a.nbytes,
    #     cuda.cudadrv.devicearray.DeviceNDArray: lambda a: a.nbytes,
    # }

    # memory_usage = total_size([device_arrays, host_arrays], array_handlers)
    # print("ARRAYS INITIALIZED, TOTAL_SIZE = {} MB".format(memory_usage / 1e6))

    return host_arrays, device_arrays


def preallocate_cpu(
    shape: tuple[int, int],
    ambient_temperature: float,
    wind_speed: float,
    wind_aspect: float,
    relative_humidity: float,
    terrain_slope: float,
    terrain_aspect: float,
    avg_combustibility: float,
    stochastic: bool | None = True,
    dtype: np.dtype | None = np.float64,
    order: str | None = "C",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-allocates fire-model arrays on the CPU (host).

    Simplifies the creation of the arrays required for the forest-fire
    spread model of Rui et al. 2018. This is useful to test the
    forest-fire model with fabricated data sets.

    Args:
        shape: Number of cells in (height, width) format
        ambient_temperature: Ambient temperature in SI Celcius
        wind_speed: Average wind speed in SI meter per second
        wind_aspect: Wind aspect (bearing) in SI degree (North = 0)
        relative_humidity: Relative air humidity in SI percent (%)
        terrain_slope: Incline of terrain in SI degree
        terrain_aspect: Downhill aspect (bearing) in SI degree
        avg_combustibility: Average combustibility of the terrain
            features
        stochastic: Determines if noise is added to the combustibility.
            If stochastic is True, the combustibility will be a random
            number between 0 and 2 and the average combustibility will
            be ignored.
        dtype: Numpy data type of the float arrays
        order: Order of the Numpy arrays (Defaults to C Row-Major)

    Returns:
        Pre-allocated arrays on the CPU

    """
    # f_array = float_array, used to reduce boilerplate
    f_array = partial(np.full, shape=shape, dtype=dtype, order=order)

    # Initializing Combustibility Array
    combustibilities = f_array(fill_value=avg_combustibility)
    if stochastic:
        combustibilities = np.random.uniform(0, 2, shape).astype(dtype, order)

    bool_array = np.full(
        fill_value=False, shape=shape, dtype=np.bool_, order=order
    )
    arrays = {
        "fire_states": np.full(
            shape, fill_value=COMBUSTIBLE, dtype=np.uint8, order=order
        ),
        "fire_indices": np.zeros((np.prod(shape), 2), dtype=np.int64),
        "spread_rates": f_array(fill_value=0),
        # Must start from 1 due to additional nonflammable state
        "intermediate_states": f_array(fill_value=1),
        "temperatures": f_array(fill_value=ambient_temperature),
        "wind_speeds": f_array(fill_value=wind_speed),
        "wind_aspects": f_array(fill_value=wind_aspect),
        "humidities": f_array(fill_value=relative_humidity),
        "terrain_slopes": f_array(fill_value=terrain_slope),
        "terrain_aspects": f_array(fill_value=terrain_aspect),
        "combustibilities": combustibilities,
        "prop_aspect": f_array(fill_value=np.nan),
        "can_ignite": bool_array,
        "can_extinguish": bool_array.copy(),
    }

    # memory_usage = total_size(arrays, {np.ndarray: lambda a: a.nbytes})
    # print("ARRAYS INITIALIZED, TOTAL_SIZE = {} MB".format(memory_usage / 1e6))

    return arrays


def add_noise(array: np.ndarray, std: float = 1.0) -> np.ndarray:
    """Adds noise to an ``array`` in-place.

    A Normal (Gaussian) distribution is used where the standard
    deviation, ``std``, can be used to adjust the amount of variance
    in the ``array``.

    Args:
        array: Something
        std: Coefficient of Variation (default = 0.01)

    Returns:
        Stochastic Array with a variance scaled by the ``mean``

    """
    array += np.random.normal(loc=0, scale=std, size=array.shape)
    return array


if __name__ == "__main__":
    host_arrays, device_arrays = preallocate_gpu(
        shape=(1000, 1000),
        ambient_temperature=35,
        wind_speed=10,
        wind_aspect=0,
        relative_humidity=20,
        terrain_slope=0,
        terrain_aspect=0,
        avg_combustibility=1.8,
        stochastic=True,
    )
