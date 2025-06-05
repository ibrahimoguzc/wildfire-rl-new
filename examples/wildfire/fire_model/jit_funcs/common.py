# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Common JIT-compiled functions of the wildfire model."""

import math

import numba

from examples.wildfire.fire_model.states import (
    BURNT,
    COMBUSTIBLE,
    EARLY_BURNING,
    EXTINGUISHING,
    FULL_BURNING,
    NONFLAMMABLE,
    SUPPRESSED,
)
from sosid.jit_config import BASE_JIT_KWARGS, FAST_MATH_FLAGS
from sosid.model.ca.jit_funcs.geom2d import angle_between, aspect_2d
from sosid.model.ca.neighborhood import MOORE_OFFSETS

# TODO change MOORE_OFFSETS to be defined within configuration file
# TODO add check to make sure that the current cell is not flammable,
# ONLY run for a combustible cell!


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def preprocess(
    fire_states: numba.uint8[:, :], position: tuple[int, int]
) -> tuple[float, bool, bool]:
    """Obtains fire propagation information from neighboring cells.

    Before calculating the spread-rate of the fire a necessary step is
    to obtain the local fire propagation direction in each cell. This
    depends on the value of the ``fire_states`` in neighboring cells.
    Furthermore, later on in the algorithm when it is necessary to
    compute the value of the fire state at the following time_step,
    information on whether the present cell can ignite or extinguish
    also depends on the neighboring values of the ``fire_states`` array
    in neighboring cells. Therefore, to reduce the number of global
    memory accesses by the active threads (important in CUDA
    programming), it is helpful to iterate over the neighboring cells
    only once per thread. This act of iterating over the neighboring
    cells for each thread to retrieve necessary information for the fire
    propagation-model is deemed `preprocessing`.

    #TODO link to ADR on CUDA decision to loop at once
    #TODO finish docstring

    Caution:
        No checks are performed to make sure that the access of
        neighboring cells are within the bounds of the ``fire_states``
        array. Therefore, the ``fire_states`` array must either be
        padded with fill values or the ``position`` within the array
        must be limited to non-border elements. The width of bordering
        elements is then defined by the radius of the neighborhood.

    Tip:
        The emphasis of both returned booleans are on the word **can**,
        signifying that these conditions have the potential to take
        place, but are not required to happen. As such, one must check
        additional rules before determining the fire-state at the next
        iteration. These rules are implemented by the postprocessing
        step.

    Note:
        Although this violates the Single Responsibility Principle (SRP)
        and hence makes it more difficult to test, iterating over the
        neighboring cells only once limits the global memory access to
        the ``fire_states`` array and increases performance!

    Args:
        fire_states: 2D Array containing fire states
        position: Absolute index within the ``fire_states`` array

    Returns:
        A tuple containing the output of the preprocessing step of the
        ``fire_states``. This tuple contains the following in order:

        - [0] Local propagation aspect (bearing)
        - [1] If the current interrogation cell can ignite
        - [2] If the current interrogation cell can extinguish

    """
    (i, j) = position  # Un-packing position of current thread

    # Creating summing variables for neighboring cells iteration
    i_sum, j_sum, count_extinguishable, count_neighbors = 0, 0, 0, 0

    aspect = 0
    can_ignite = False  # Initially the cell is assumed not ignitable
    can_extinguish = False
    if fire_states[i, j] in [NONFLAMMABLE, SUPPRESSED, BURNT]:
        return aspect, can_ignite, can_extinguish
    for i_offset, j_offset in MOORE_OFFSETS:
        count_neighbors += 1
        neighbor = fire_states[i + i_offset, j + j_offset]

        # If neighboring cell is on fire, it contributes to propagation
        if neighbor == FULL_BURNING:
            i_sum += i_offset
            j_sum += j_offset
            can_ignite = True  # At least one neighbor on fire = can_ignite

        # If the neighboring cell is at or past the full_burning state,
        # or it is non-flammable it contributes to extinguishing
        if (
            neighbor >= FULL_BURNING
            or neighbor == NONFLAMMABLE
            or neighbor == SUPPRESSED
        ):
            count_extinguishable += 1

    # Aspect is defined as North = 0, East = 90, South = 180, West = 270
    # Note: The offset sums are reversed to get the propagation vector
    aspect = aspect_2d(float(-i_sum), float(-j_sum))

    # A cell is extinguishable if ALL neighboring cells have either
    # FULL_BURNING, EXTINGUISHING, BURNT or NONFLAMMABLE states
    can_extinguish = count_extinguishable == count_neighbors

    return aspect, can_ignite, can_extinguish


# TODO add docstring
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calc_spread_rate(
    fire_state: int,
    prop_aspect: float,
    temperature: float,
    wind_speed: float,
    wind_aspect: float,
    humidity: float,
    terrain_slope: float,
    terrain_aspect: float,
    combustibility: float,
    correction_coefficient: float,
) -> float:
    """Compute fire spread rate."""
    # Limit computation of spread rate to cells that are full burning
    if fire_state == FULL_BURNING:
        r_0 = calc_initial_spread_rate(temperature, wind_speed, humidity)

        # If propagation direction is undefined, wind & slope have no
        # effect, this happens when a cell is fully surrounded by fire
        if math.isnan(prop_aspect):
            k_phi = k_theta = 1
        else:
            k_phi = calc_wind_coefficient(wind_speed, wind_aspect, prop_aspect)
            k_theta = calc_slope_coefficient(
                terrain_slope, terrain_aspect, prop_aspect
            )
        k_s = combustibility

        # Computing the rate_array with calculated rate at current idx
        return r_0 * k_phi * k_theta * k_s * correction_coefficient
    return 0


# TODO finish documentation add citation to Wang, 1992
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calc_initial_spread_rate(
    temperature: float, wind_speed: float, humidity: float
) -> float:
    """Calculates the initial fire spread rate in SI meter per minute.

    The initial spread rate is based on the ambient conditions given by
    ``temperature``, ``wind_speed``, and ``humidity``. The constants
    a, b, c, and d are obtained from Rui, 2018.

    Args:
        temperature: Ambient temperature in SI Celcius
        wind_speed: Wind speed in SI meter per second
        humidity: Relative air humidity in SI percent (%)

    Returns:
        Initial speed of forest fire spread in SI meter per minute

    Tip:
        To keep the algorithm fast, no checks are performed to ensure
        that the provided inputs are positive. However, it would be
        wise to perform such a check outside of the JIT-compiled
        functions during a simulation run.

    Note:
        The Wind-Force Integer is neglected (set equal to 1) as it is
        arbitrarily defined by Rui, 2018. Upon further investigation,
        into the paper of xxx, it is seen that the wind force integer
        is computed from a table and serves the purpose of scaling the
        wind-speed. However, as the model already has the potential to
        be calibrated through use of the correction coefficient, using
        the wind-force integer is redundant and prone to error.

    """
    a, b, c, d = (
        0.03,
        0.05,
        0.01,
        0.3,
    )
    return (
        a * temperature
        + b * (wind_speed / 0.836) ** (2 / 3)
        + c * (100 - humidity)
        - d
    )


# TODO convert symbol to latex
# TODO document & test functions below
# TODO check if embedded LaTeX works w/ Sphinx
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calc_wind_coefficient(
    wind_speed: float, wind_aspect: float, prop_aspect: float
) -> float:
    """Calculates the non-dimensional wind coefficient, $K_phi$.

    The utilizinthe relation

    Args:
        wind_speed: [description]
        wind_aspect: Compass direction (bearing) in which the wind is
            coming from (Meteorological convention) in SI degree
        prop_aspect: [description]

    Returns:
        Non-dimensional wind coefficient, $K_phi$.


    """
    phi = angle_between(wind_aspect, prop_aspect) * math.pi / 180
    return math.exp(0.1783 * wind_speed * -math.cos(phi))


# TODO finish docstring
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calc_slope_coefficient(
    terrain_slope: float, terrain_aspect: float, prop_aspect: float
) -> float:
    """Compute slope coefficient from fire propagation and terrain."""
    # Computing the hill direction "g" which is either -1 or 1
    if math.isnan(prop_aspect):  # Handling edge-case w/ undefined prop_dir
        hill_dir = math.nan
    else:
        hill_dir = -1 if angle_between(terrain_aspect, prop_aspect) < 90 else 1
    return math.exp(
        3.553 * hill_dir * math.tan(1.2 * terrain_slope * math.pi / 180)
    )


# TODO finish documentation
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calc_ideal_time_step(
    maximum_spread_rate: float,
    cell_size: float,
    step_size_factor: float | None = 0.125,
) -> float:
    """Calculates the ideal physical time-step in SI minutes.

    This quantity ensures that the fire is constrained to propagating
    to a single cell per iteration. Therefore, its calculation is
    based on the ``maximum_spread_rate`` and ``cell_size``.

    Note:
        According to Rui, 2018 the ``step_size_factor`` should be equal
        to the inverse of the number of neighboring cells. For a Moore
        neighborhood this is (1/8 = 0.125). From testing, with various
        values this produces the most circular propagation in uniform
        conditions.

    Caution:
        The ``cell_size`` should be an even number!

    Args:
        maximum_spread_rate: [description]
        cell_size: [description]
        step_size_factor: DESCRIPTION HERE. Defaults to 0.125
            (1/n_neighbors).

    Returns:
        Ideal physical time-step in SI minutes.

    """
    return step_size_factor * cell_size / maximum_spread_rate


# TODO finish docstring and cite Rui
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def postprocess(
    fire_state: int,
    intermediate_state: float,
    can_ignite: bool,
    can_extinguish: bool,
) -> int:
    """Advances the ``fire_state`` based on transition rules.

    This function implements the rules defined by Rui, 2018 to define
    how the discrete integer cell values, deemed ``fire_states`` are
    updated at each iteration.

    Args:
        fire_state: The state of the fire in the current cell
        intermediate_state: The cached intermediate fire state. This
            only has significance when the current cell has not yet
            fully caught fire and is transitioning to the early-burning
            or full-burning state.
        can_ignite: Defines if the current cell can potentially ignite
        can_extinguish: Defines if the current cell can extinguish

    Returns:
        Updated fire-state in the current cell for the next iteration

    """
    # If fire state is non-flammable excape out of the function
    if fire_state in [NONFLAMMABLE, SUPPRESSED, BURNT]:
        return fire_state

    # Transition to Early Burning
    if fire_state == COMBUSTIBLE and can_ignite:
        if EARLY_BURNING <= intermediate_state < FULL_BURNING:
            return EARLY_BURNING
        if intermediate_state >= FULL_BURNING:
            return FULL_BURNING

    # Transition to Full Burning
    elif fire_state == EARLY_BURNING:
        return FULL_BURNING

    # Transition to Extinguishing State
    elif fire_state == FULL_BURNING and can_extinguish:
        return EXTINGUISHING

    # Transition to Burnt State
    elif fire_state == EXTINGUISHING:
        return BURNT

    return fire_state


# TODO consider moving this into factory function from arrayops
@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def sum_neighbors(
    array: numba.float64[:, :], position: tuple[int, int]
) -> float:
    """Sums ``array`` values in neighborhood of the current thread.

    The cell index of the current cell is equivalent to the current
    absolute thread index which is given by ``position``.

    Args:
        array: Input 2D Numpy array
        position: Absolute index of current cell within the ``array``

    Returns:
        Summed ``array`` values in neighborhood of the current thread
        given by ``position``

    """
    (i, j), _sum = position, 0
    for i_offset, j_offset in MOORE_OFFSETS:
        _sum += array[i + i_offset, j + j_offset]
    return _sum


@numba.jit(
    **BASE_JIT_KWARGS,
    fastmath=FAST_MATH_FLAGS,
    parallel=False,
    boundscheck=False,
)
def compute_cells_burning(
    n_cells,
    to_process,
    can_ignite,
    intermediate_states,
    spread_rates,
    t_step,
    cell_size,
    fire_states,
    can_extinguish,
    fire_indices,
) -> int:
    """Computes the number of cells on fire using the t_step."""
    n_burning = 0
    for n in range(n_cells):
        i, j = to_process[n]
        # Updating intermediate state value using the summed_rates
        if can_ignite[i, j]:
            intermediate_states[i, j] += (
                sum_neighbors(spread_rates, (i, j)) * t_step / cell_size
            )
        # Updating fire state at next iteration (t + time_step)
        fire_state = postprocess(
            fire_states[i, j],
            intermediate_states[i, j],
            can_ignite[i, j],
            can_extinguish[i, j],
        )
        if (
            fire_state == FULL_BURNING
            or fire_state == EARLY_BURNING
            or fire_state == EXTINGUISHING
        ):
            fire_indices[n_burning, 0] = i
            fire_indices[n_burning, 1] = j
            n_burning += 1

        fire_states[i, j] = fire_state
    return n_burning
