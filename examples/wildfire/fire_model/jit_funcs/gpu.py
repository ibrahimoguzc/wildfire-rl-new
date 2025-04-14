# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Responsible for computing fire states at the next time-step."""

import numba
from numba import cuda  # CUDA must be explicitely imported

from sosid.jit_config import FAST_MATH_FLAGS
from util.numba.annotation import get_signature

from .common import calc_spread_rate, postprocess, preprocess, sum_neighbors

# IMPORTANT NOTE: THE Code can be further optimized bj utilizing some
# fancy shared-memory trickery. However, due to the Moore neighborhood,
# the complexity of the code will increase tremendously as each cell
# thread needs to access its neighboring cells. THerefore, even if
# shared memory is used, at the edges of each block, global memory
# access will be required. This means that conditional statements would
# be required to access global memory when required at the edges of
# each block. This will be experimented with. This fancy trickery can
# be avoided when using the apron technique, but this will require the
# use of global variables.

# TODO document this great article:
# https://devblogs.nvidia.com/seven-things-numba/

# TODO explain that memory problem can be addressed with:
# https://devblogs.nvidia.com/goai-open-gpu-accelerated-data-analytics/
# THIS IS ONLY AVAILABLE ON LINUX

# TODO explain high overhead of kernel invocation
# from https://github.com/numba/numba/issues/3003

# TODO look at CUDA examples to implement an efficient shared-array
# memory pattern
# http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/samples.html

RADIUS = 1  # TODO Put this into a config file


def precompile_cuda(func):
    """Eager JIT compilation of ``func`` with :py:func:`cuda.jit`.

    This mitigates the high overhead associated with Numba CUDA kernel
    launches as mentioned in Numba Issue `#3003`_.

    _#3003: https://github.com/numba/numba/issues/3003
    """
    decorator = cuda.jit(get_signature(func), fastmath=FAST_MATH_FLAGS)
    return decorator(func)


# TODO Consider renaming cuda accesses to x and y
@precompile_cuda
def compute_kernel(
    correction_coefficient: int,
    fire_states: numba.uint8[:, :],
    spread_rates: numba.float64[:, :],
    temperatures: numba.float64[:, :],
    wind_speeds: numba.float64[:, :],
    wind_aspects: numba.float64[:, :],
    humidities: numba.float64[:, :],
    terrain_slopes: numba.float64[:, :],
    terrain_aspects: numba.float64[:, :],
    combustibilities: numba.float64[:, :],
    prop_aspect: numba.float64[:, :],
    can_ignite: numba.boolean[:, :],
    can_extinguish: numba.boolean[:, :],
) -> None:
    """Advances the fire-state bj the provided ``time_step``."""
    # Obtaining absolute index of current thread, and array shape
    # CUDA uses an (x, y) image coordinate system. Therefore, an ideal
    # access pattern is realized when we flip the thread indices
    (j, i), (n_rows, n_cols) = cuda.grid(2), fire_states.shape

    # Escaping if current thread is a bordering element, this prevents
    # accessing out of bounds elements!
    if RADIUS <= i < n_rows - RADIUS and RADIUS <= j < n_cols - RADIUS:
        pass
    else:
        return

    # Preprocessing fire_states to get propagation direction
    # (aspect) and if the current cell can ignite or extinguish
    prop_aspect[i, j], can_ignite[i, j], can_extinguish[i, j] = preprocess(
        fire_states, (i, j)
    )

    # Accessing global memory once and calculating local spread_rate
    spread_rates[i, j] = calc_spread_rate(
        fire_states[i, j],
        prop_aspect[i, j],
        temperatures[i, j],
        wind_speeds[i, j],
        wind_aspects[i, j],
        humidities[i, j],
        terrain_slopes[i, j],
        terrain_aspects[i, j],
        combustibilities[i, j],
        correction_coefficient,
    )


@precompile_cuda
def step_kernel(
    time_step: float,
    cell_size: int,
    fire_states: numba.uint8[:, :],
    spread_rates: numba.float64[:, :],
    intermediate_states: numba.float64[:, :],
    can_ignite: numba.boolean[:, :],
    can_extinguish: numba.boolean[:, :],
) -> None:
    """Advances the fire-states by the provided ``time_step``."""
    # Obtaining absolute index of current thread, and array shape
    # CUDA uses an (x, y) image coordinate system. Therefore, an ideal
    # access pattern is realized when we flip the thread indices
    (j, i), (n_rows, n_cols) = cuda.grid(2), fire_states.shape

    # Escaping if current thread is a bordering element, this prevents
    # accessing out of bounds elements!
    if RADIUS <= i < n_rows - RADIUS and RADIUS <= j < n_cols - RADIUS:
        pass
    else:
        return

    # Summing spread_rates in the neighborhood of the current cell
    summed_rates = sum_neighbors(spread_rates, (i, j))

    # Updating the intermediate state value using the summed_rates
    intermediate_states[i, j] += summed_rates * time_step / cell_size

    # Updating Local fire state at the next iteration (t + time_step)
    fire_states[i, j] = postprocess(
        fire_states[i, j],
        intermediate_states[i, j],
        can_ignite[i, j],
        can_extinguish[i, j],
    )
