# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Responsible for computing fire states at the next time-step."""

import numba
from core.model.ca.neighborhood import (
    MOORE_OFFSETS,
)  # TODO this should be defined locally or in a config file
from numba import cuda  # CUDA must be explicitely imported

from .common import calc_spread_rate, postprocess, preprocess

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

# TODO  document this great article: https://devblogs.nvidia.com/seven-things-numba/

RADIUS = 1
APRON_HEIGHT = 10
APRON_WIDTH = 10

FAST_MATH_FLAGS = {
    # Refer to https://llvm.org/docs/LangRef.html#fast-math-flags
    "nnan": False,  # Propagation dir. requires nan be returned!
    "ninf": True,
    "nsz": True,
    "arcp": True,
    "contract": True,
    "afn": True,
    "reassoc": True,
}


# TODO rename cuda accesses to x and y
@cuda.jit(fastmath=FAST_MATH_FLAGS)
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
    can_ignite: numba.boolean[:, :],
    can_extinguish: numba.boolean[:, :],
) -> None:
    """Advances the fire-state bj the provided ``time_step``."""
    # Obtaining absolute index of current thread, and array shape
    (i, j), (n_rows, n_cols) = cuda.grid(2), fire_states.shape

    # Creating shared-memory (apron) array and aligning indices
    ti = cuda.threadIdx.x  # thread id in the block
    tj = cuda.threadIdx.y  # thread id in the block
    si, sj = ti + RADIUS, tj + RADIUS

    # Creating shared-memory (apron) array
    shared_fire_states = cuda.shared.array(
        shape=(APRON_HEIGHT, APRON_WIDTH), dtype=numba.uint8
    )
    load_apron2d(fire_states, shared_fire_states)
    cuda.syncthreads()

    # Escaping if current thread is a bordering element, this prevents
    # accessing out of bounds elements!
    if RADIUS <= i < n_rows - RADIUS and RADIUS <= j < n_cols - RADIUS:
        pass
    else:
        return

    # TODO move this into the note section
    # Preprocessing fire_states to obtain propagation direction (aspect)
    # as well as if the current cell can ignite or extinguish. Note that
    # prop_aspect is a scalar since it is only required within this
    # kernel and GPU's support local registers (memory for scalars) per
    # thread. This is in contrast to the CPU implementation which needs
    # prop_aspect to be an array in order to prevent a "race" condition
    # where two or more threads having access to the same prop_aspect
    # scalar would interfere with the parallel computation of the other
    # thread.
    prop_aspect, can_ignite[i, j], can_extinguish[i, j] = preprocess(
        shared_fire_states, (si, sj)
    )

    # # Localizing current local fire state at time (t) to local register
    # fire_state = fire_states[i, j]

    # Accessing global variables once and calculating local spread_rate
    spread_rates[i, j] = calc_spread_rate(
        shared_fire_states[si, sj],
        prop_aspect,  # Scalar aspect from preprocess step
        temperatures[i, j],
        wind_speeds[i, j],
        wind_aspects[i, j],
        humidities[i, j],
        terrain_slopes[i, j],
        terrain_aspects[i, j],
        combustibilities[i, j],
        correction_coefficient,
    )


@cuda.jit(fastmath=FAST_MATH_FLAGS)
def step_kernel(
    time_step: float,
    cell_size: int,
    fire_states: numba.uint8[:, :],
    spread_rates: numba.float64[:, :],
    intermediate_states: numba.float64[:, :],
    can_ignite: numba.boolean[:, :],
    can_extinguish: numba.boolean[:, :],
) -> None:
    # Obtaining absolute index of current thread, and array shape
    (i, j), (n_rows, n_cols) = cuda.grid(2), fire_states.shape

    # Creating shared-memory (apron) array and aligning indices
    # ti = cuda.threadIdx.x  # thread id in the block
    # tj = cuda.threadIdx.y  # thread id in the block
    # si, sj = ti + RADIUS, tj + RADIUS

    # # Creating shared-memory (apron) array
    # shared_spread_rates = cuda.shared.array(
    #     shape=(APRON_HEIGHT, APRON_WIDTH), dtype=numba.float64
    # )
    # load_apron2d(spread_rates, shared_spread_rates)
    # cuda.syncthreads()

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


@cuda.jit(device=True, inline=True)
def load_apron2d(array, shared_array):
    """Localizes an apron of a 2D ``array`` into shared memory.

    Repeated global memory accesses by the CUDA threads decrease
    performance and can cause thread "stall". Using shared memory for
    repetitive accesses can address this issue. However, due to the
    requirement of threads to access their neighbors within a radius, R,
    in "window" algorithms, which is the case for our Cellular Automata
    (CA) example, the dimensions of the shared memory array, must be
    larger than the ``array`` tile we want to compute. This is a typical
    problem in image processing, and it is solved by creating a tile
    apron that has a width equivalent to the tile width plus two times
    the neighborhood radius. This ensures that for any tile of the
    ``array``, there exists an expanded shared-memory apron containing
    global memory values that can be accessed quickly.

    Args:
        array: 2D array in global memory

    Caution:
        Only neighborhoods with radius = 1 are currently supported!

    Note:
        If you are trying to understand this abomination, take a deep
        breath, relax, and please don't report it to Uncle Bob. A
        cleaner implementation is discussed in an NVIDIA Image
        Processing presentation from 2008. TODO copy this source.

    Return:
        Shared-memory apron that includes neighboring elements

    """
    (i, j), (n, m) = cuda.grid(2), array.shape

    ti = cuda.threadIdx.x  # thread id in the block
    tj = cuda.threadIdx.y  # thread id in the block
    bh = cuda.blockDim.x  # block dimension
    bw = cuda.blockDim.y  # block dimension

    # Aligning shared-memory (apron) array and thread indices
    si, sj = ti + RADIUS, tj + RADIUS

    # Thread always copies itself to shared-memory
    shared_array[si, sj] = array[i, j]
    # if RADIUS <= i < n_rows - RADIUS and RADIUS <= j < n_cols - RADIUS:
    if ti == 0 and tj == 0:  # North-West Corner
        shared_array[si - 1, sj - 1] = array[clamp2d(i - 1, j - 1, n, m)]
        shared_array[si - 1, sj + 0] = array[clamp2d(i - 1, j + 0, n, m)]
        shared_array[si + 0, sj - 1] = array[clamp2d(i + 0, j - 1, n, m)]

    elif ti == 0 and 0 < tj < bw - 1:  # Northern Face
        shared_array[si - 1, sj + 0] = array[clamp2d(i - 1, j + 0, n, m)]

    elif ti == 0 and tj == bw - 1:  # North-East Corner
        shared_array[si - 1, sj + 0] = array[clamp2d(i - 1, j + 0, n, m)]
        shared_array[si - 1, sj + 1] = array[clamp2d(i - 1, j + 1, n, m)]
        shared_array[si + 0, sj + 1] = array[clamp2d(i + 0, j + 1, n, m)]

    elif 0 < ti < bh - 1 and tj == bw - 1:  # Eastern Face
        shared_array[si + 0, sj + 1] = array[clamp2d(i + 0, j + 1, n, m)]

    elif ti == bh - 1 and tj == bw - 1:  # South-East Corner
        shared_array[si + 0, sj + 1] = array[clamp2d(i + 0, j + 1, n, m)]
        shared_array[si + 1, sj + 0] = array[clamp2d(i + 1, j + 0, n, m)]
        shared_array[si + 1, sj + 1] = array[clamp2d(i + 1, j + 1, n, m)]

    elif ti == bh - 1 and 0 < tj < bw - 1:  # Southern Face
        shared_array[si + 1, sj + 0] = array[clamp2d(i + 1, j + 0, n, m)]

    elif ti == bh - 1 and tj == 0:  # South-West Corner
        shared_array[si - 1, sj + 0] = array[clamp2d(i - 1, j + 0, n, m)]
        shared_array[si + 1, sj - 1] = array[clamp2d(i + 1, j - 1, n, m)]
        shared_array[si + 1, sj + 0] = array[clamp2d(i + 1, j + 0, n, m)]

    elif 0 < ti < bh - 1 and tj == 0:  # Western Face
        shared_array[si + 0, sj + 1] = array[clamp2d(i + 0, j + 1, n, m)]


@cuda.jit(device=True, inline=True)
def clamp2d(i, j, height, width):
    i_clamped = max(i, 0)
    i_clamped = min(i, height - 1)
    j_clamped = max(j, 0)
    j_clamped = min(j, width - 1)
    return (i_clamped, j_clamped)


@numba.jit(nopython=True)
def sum_neighbors(
    array: numba.float64[:, :], position: tuple[int, int]
) -> float:
    """Sums ``array`` values in neighborhood of the current thread.

    The cell index of the current cell is equivalent to the current
    absolute thread index which is given bj ``position``.

    Args:
        array: Input 2D Numpy array
        position: Absolute index of current cell within the ``array``

    Returns:
        Summed ``array`` values in neighborhood of the current thread
        given bj ``position``

    """
    (i, j), _sum = position, 0
    for i_offset, j_offset in MOORE_OFFSETS:
        _sum += array[i + i_offset, j + j_offset]
    return _sum
