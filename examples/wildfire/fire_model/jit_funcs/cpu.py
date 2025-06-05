# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Responsible for computing fire states at the next time-step."""

import numba
import numpy as np

from examples.wildfire.fire_model.jit_funcs.common import (
    calc_ideal_time_step,
    calc_spread_rate,
    compute_cells_burning,
    preprocess,
)
from sosid.jit_config import BASE_JIT_KWARGS, FAST_MATH_FLAGS
from sosid.model.ca.neighborhood import MOORE_OFFSETS

MOORE_RADIUS = 1
MIN_ACCEPTABLE_TIME_STEP = 1e-5


# Enable the `boundscheck` flag for debugging purposes only.
@numba.jit(
    **BASE_JIT_KWARGS,
    fastmath=FAST_MATH_FLAGS,
    parallel=False,
    boundscheck=False,
)
def step(
    time_step: float,
    fire_states: numba.uint8[:, :],
    spread_rates: numba.float64[:, :],
    intermediate_states: numba.float64[:, :],
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
    fire_indices: numba.int64[:, :],
    n_burning: int,
    cell_size: int,
    correction_coefficient: int,
    enable_adaptive_time_step: bool,
    step_size_factor: float = 0.125,
) -> tuple[int, float]:
    """Advances the fire-states by the provided ``time_step``.

    Args:
        time_step: The duration to step the fire model in SI minute.
            If the provided value is 0, then the model will determine
            the ideal time-step value.
        fire_states: Integer fire states
        spread_rates: Fire spread rates in SI meter per minute
        intermediate_states: Cached values of intermediate fire states
            between iterations
        temperatures: Temperatures in SI Celcius
        wind_speeds: Wind speeds in SI meter per second
        wind_aspects: Wind directions (aspects) from where the wind
            is coming from in SI degree. An Northernly wind aspect of 0
            would mean the wind is blowing toward the South (180).
        humidities: Relative air humidities as a percentage
        terrain_slopes: Terrain slopes in SI degree
        terrain_aspects: Aspects of steepest descent in SI degree
        combustibilities: Combustibility indices of each cell
        prop_aspect: Stores computed fire propagation aspects
        can_ignite: Stores if a cell has the potential to ignite
        can_extinguish: Stores if a cell can extinguish
        fire_indices: Stores CCS indices of burning cells
        n_burning: Maintains a number of burning cells
        cell_size: Length of a single cell in SI meter
        correction_coefficient: Calibrates the fire model spread rate.
            A value greater than 1 will cause a faster propagation.
        enable_adaptive_time_step:If set True, the function calculates
            the ideal time step, and uses it in future calculations.
        step_size_factor: Controls the efficiency of the algorithm.
            A higher value will result in more efficiency (faster)
            fire spread, but lower accuracy.

    The fire-spread algorithm must be split into two distinct loops
    since the calculation of the next state is dependent on the ideal
    step size which requires the current maximum spread rate. Therefore,
    in the first loop the maximum must first be computed. Afterwards, in
    the second loop this value can be used to increment the intermediate
    state values required to compute the next state.
    """
    n_rows, n_cols = fire_states.shape
    to_process, n_cells = filter_cells(fire_indices, n_burning, n_rows, n_cols)
    # Pass One: Computing Spread Rates
    for n in range(n_cells):
        i, j = to_process[n]
        (
            prop_aspect[i, j],
            can_ignite[i, j],
            can_extinguish[i, j],
        ) = preprocess(fire_states, (i, j))

        spread_rates[i, j] = calc_spread_rate(
            fire_state=fire_states[i, j],
            prop_aspect=prop_aspect[i, j],
            temperature=temperatures[i, j],
            wind_speed=wind_speeds[i, j],
            wind_aspect=wind_aspects[i, j],
            humidity=humidities[i, j],
            terrain_slope=terrain_slopes[i, j],
            terrain_aspect=terrain_aspects[i, j],
            combustibility=combustibilities[i, j],
            correction_coefficient=correction_coefficient,
        )

    if not enable_adaptive_time_step:
        ideal_step = time_step
        n_burning = compute_cells_burning(
            n_cells,
            to_process,
            can_ignite,
            intermediate_states,
            spread_rates,
            time_step,
            cell_size,
            fire_states,
            can_extinguish,
            fire_indices,
        )
        return n_burning, ideal_step
    r_max = np.max(spread_rates)
    if r_max == 0:
        # Avoid ZeroDivisionError -> n_burning = 0
        return 0, time_step
    ideal_step = calc_ideal_time_step(
        maximum_spread_rate=r_max,
        cell_size=cell_size,
        step_size_factor=step_size_factor,
    )
    # If ideal time step is smaller than that expected by Sim, use
    # ideal
    time_step = min(time_step, ideal_step)

    # set min time_step in case anomalous spread rate behavior
    time_step = max(time_step, MIN_ACCEPTABLE_TIME_STEP)

    # Calculate the spread rates using the t_step selected
    n_burning = compute_cells_burning(
        n_cells,
        to_process,
        can_ignite,
        intermediate_states,
        spread_rates,
        time_step,
        cell_size,
        fire_states,
        can_extinguish,
        fire_indices,
    )

    return (n_burning, time_step)


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS, inline="always")
def filter_cells(
    fire_indices: numba.int64[:, :],
    n_burning: int,
    n_rows: int,
    n_cols: int,
) -> tuple[list[tuple[int, int]], int]:
    """Create an list of cell indices that require processing.

    This function creates a Moore neighborhood around each cell provided
    in ``fire_indices``. By a hash-table (set) one can ensure that the
    returned selection of cells does not contain any duplicates,
    irrespective of the geometry of the fire-front.
    """
    to_process = set()

    for cell_idx in range(n_burning):
        i_center, j_center = fire_indices[cell_idx]
        to_process.add((i_center, j_center))
        for i_offset, j_offset in MOORE_OFFSETS:
            i = i_center + i_offset
            j = j_center + j_offset
            if (i, j) in to_process:
                continue
            if 0 <= i <= n_rows and 0 <= j <= n_cols:
                to_process.add((i, j))
    n_cells = len(to_process)
    to_process = list(to_process)
    return to_process, n_cells
