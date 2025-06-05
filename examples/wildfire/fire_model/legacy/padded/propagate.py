import math

import numba
import numpy as np

try:
    from examples.wildfire.fire_model.legacy.array_ops import (
        any_in_neighbors,
        depad,
        pad,
        sum_neighbors,
    )
    from examples.wildfire.fire_model.legacy.fast_math import (
        dot_2d,
        normalize_2d,
    )
    from examples.wildfire.fire_model.legacy.states import (
        burnt,
        combustible,
        early_burning,
        extinguishing,
        full_burning,
        nonflammable,
    )
    from sosid.model.ca.neighborhood import MooreNeighborhood
except ModuleNotFoundError:
    import os
    import sys

    sys.path.insert(0, os.getcwd())
    from examples.wildfire.fire_model.legacy.array_ops import (
        any_in_neighbors,
        depad,
        pad,
        sum_neighbors,
    )
    from examples.wildfire.fire_model.legacy.fast_math import (
        dot_2d,
        normalize_2d,
    )
    from examples.wildfire.fire_model.legacy.states import (
        burnt,
        combustible,
        early_burning,
        extinguishing,
        full_burning,
        nonflammable,
    )
    from sosid.model.ca.neighborhood import MooreNeighborhood


MOORE = MooreNeighborhood(radius=1, include_center=False)
MOORE_OFFSETS = MOORE.as_tuple()
MOORE_RADIUS = MOORE.radius


@numba.jit(nopython=True, parallel=True)
def find_extinguishables(array: np.ndarray) -> np.ndarray:
    """Parallelizes and JIT compiles the kernel defined by
    :py:func:`_any_kernel` which applies a Moore neighborhood search on
    each cell of the ``array``, attempting to find any matching
    ``value`` within the neighboring 8 cells. Therefore, the kernel
    iteratres through (-1, 0, 1) for both the i and j indices, and
    checks if ``array[i + i_offset, j + j_offset] == value``.

    Args:
        array: Input 2D Numpy array

    Returns:
        A boolean array of the same shape as the input ``array``
    """
    _padded_array = pad(array, width=MOORE_RADIUS, pad_value=nonflammable)
    return depad(_extinguishables_kernel(_padded_array), width=MOORE_RADIUS)


@numba.stencil(neighborhood=MOORE.limits)  # Moore Neighborhood
def _extinguishables_kernel(array: np.ndarray) -> np.ndarray:
    """Defines the stencil kernel that applies a fixed pattern to
    search the Moore neighborhood of all cells in the the input
    ``array`` for a matching ``value``.

    Warning:
        The Numba stencil decorator will default to a fill-value at the
        boundaries of the ``array`` to prevent out-of-bounds errors. As
        such, the ``array`` must be padded with a fill-value to obtain
        correct results at the boundaries.

    Args:
        array: Input 2D Numpy array
        value: Compare value to search neighboring cells for a match

    Returns:
        A boolean array of the same shape as the input ``array``
    """
    count = 0  # number of True values
    for i_offset, j_offset in MOORE_OFFSETS:
        neighbor = array[i_offset, j_offset]
        if neighbor >= full_burning or neighbor == nonflammable:
            count += 1
    return count == 8  # Moore Neighborhood


# TODO try re-writing function w/ the Numba stencil decorator
@numba.jit(nopython=True, fastmath=True)
def propagation_dir(
    cell_idx: tuple[int, int],
    state_array: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[int, int]:
    """Obtains the propagation direction of the local fire-spread, for
    an interrogation cell specified by ``cell_idx`` by constructing a
    vector based on the summation of the directions to adjacent
    Moore neighborhood cells that are in the `full_burning` state.
    If only one cell to the North is at a  `full_burning` state, then
    the resulting propagation direction will simply be to the South.

    Note:
        Even though the number of rows and columns can be
        inferred from the size of the `state_array``, the typical
        iterative use of this function means that this information
        is already available in the outer loop. Therefore, to reduce the
        execution time, an additional look-up of the size of the
        `state_array` is avoided by passing these values explicitely.

    Args:
        cell_idx: (row, column) index of interrogation cell
        state_array: 2D Numpy array containing fire states
        n_rows: Number of total rows in the ``state_array``
        n_cols: Number of total columns in the ``state_array``

    Returns:
        Propagation direction (2D vector) of local fire-spread
    """
    i_sum, j_sum = 0, 0
    for (
        i_offset,
        j_offset,
    ) in MOORE_OFFSETS:  # Iterate through neighboring sides
        # Compute neighboring index
        i = cell_idx[0] + i_offset
        j = cell_idx[1] + j_offset

        # Testing to check if neighboring index is within matrix bounds
        if 0 <= i <= n_rows - 1 and 0 <= j <= n_cols - 1:
            if state_array[i, j] == full_burning:
                i_sum += i_offset
                j_sum += j_offset

    # Reversed offset vector results in the propagation direction (vector)
    return normalize_2d(-i_sum, -j_sum)


# TODO try re-writing function w/ the Numba stencil decorator
# FIXME can_ignite should not be used
@numba.jit(nopython=True, fastmath=True, parallel=True)
def spread_rate(
    state_array: np.ndarray,
    can_ignite: np.ndarray,
    temperature_array: np.ndarray,
    wind_array: np.ndarray,
    humidity_array: np.ndarray,
    slope_array: np.ndarray,
    combustibility_array: np.ndarray,
    correction_coeff: float | None = 1,
) -> tuple[np.ndarray, float]:
    # Initializing empty rate-array (m/min) of same shape as `state_array`
    rate_array = np.zeros(state_array.shape, dtype=np.float64)

    # Determining which cells require calculation (Lazifies computation)
    calc_needed = state_array == full_burning

    # Unpacking shape of the `state_array` into variables for clarity
    n_rows, n_cols = state_array.shape

    # Iterating over all cells utilizing parallel loops
    for i in numba.prange(n_rows):
        for j in numba.prange(n_cols):  # Inner-loop is serialized by Numba
            if calc_needed[i, j]:
                r_0 = initial_spread_rate(
                    temperature=temperature_array[i, j],
                    wind_speed=wind_array[i, j, 0],
                    humidity=humidity_array[i, j],
                )

                prop_dir = propagation_dir(
                    cell_idx=(i, j),
                    state_array=state_array,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )

                k_phi = wind_coefficient(
                    wind_speed=wind_array[i, j, 0],
                    wind_dir=wind_array[i, j, 1:],
                    prop_dir=prop_dir,
                )
                k_theta = slope_array[i, j]  # All ones for now
                k_s = combustibility_array[i, j]

                # Computing the rate_array with calculated rate at current idx
                rate_array[i, j] += (
                    r_0 * k_phi * k_theta * k_s * correction_coeff
                )
    return rate_array


# TODO use line-profiler without jit to see where expensive ops are
# TODO add option to provide time_step
# TODO increase useability by using a decorator to parse input to acceptable foramt
@numba.jit(
    nopython=True, parallel=True, fastmath=True
)  # 10x improvement from parallelization
def propagate(
    state_array: np.ndarray,
    transition_array: np.ndarray,
    temperature_array: np.ndarray,
    wind_array: np.ndarray,
    humidity_array: np.ndarray,
    slope_array: np.ndarray,
    combustibility_array: np.ndarray,
    cell_size: float,
    correction_coeff: float | None = 1,
    time_step: float | None = 1,
) -> tuple[np.ndarray, ...]:
    """[summary]

    Note:
        All input arrays must have the same 2D shape

    Args:
        state_array (np.ndarray): [description]
        transition_array (np.ndarray): [description]
        temperature_array (np.ndarray): [description]
        wind_array (np.ndarray): [description]
        humidity_array (np.ndarray): [description]
        slope_array (np.ndarray): [description]
        combustibility_array (np.ndarray): [description]
        cell_size (float): [description]
        correction_coeff (Optional[float], optional): [description]. Defaults to 1.

    Returns:
        Tuple[np.ndarray, ...]: [description]
    """
    # Pre-Processing Arrays # TODO better documentation
    n_rows, n_cols = state_array.shape
    can_ignite = any_in_neighbors(state_array, full_burning)
    can_extinguish = find_extinguishables(state_array)
    rate_array = spread_rate(
        state_array=state_array,
        can_ignite=can_ignite,
        temperature_array=temperature_array,
        wind_array=wind_array,
        humidity_array=humidity_array,
        slope_array=slope_array,
        combustibility_array=combustibility_array,
        correction_coeff=correction_coeff,
    )

    summed_rates = sum_neighbors(rate_array, include_center=False)
    transition_array += summed_rates * time_step / cell_size

    # UNCOMMENT BELOW FOR A DYNAMIC TIME STEP
    # r_max = np.max(rate_array)
    # if r_max > 0:
    #     t_step = time_step(maximum_spread_rate=r_max, cell_size=cell_size)
    #     summed_rates = sum_neighbors(rate_array, include_center=False)
    #     transition_array += summed_rates * t_step / cell_size
    # else:
    #     print('WARNING: NO ACTIVE FIRES REMAINING, MAX RATE OF SPREAD = 0 m/s')
    #     return state_array, transition_array, None

    state_array = update_states(
        state_array=state_array,
        transition_array=transition_array,
        can_ignite=can_ignite,
        can_extinguish=can_extinguish,
    )
    return state_array, transition_array, rate_array


@numba.jit(nopython=True, parallel=True)
def update_states(state_array, transition_array, can_ignite, can_extinguish):
    n_rows, n_cols = state_array.shape
    for i in numba.prange(n_rows):
        for j in numba.prange(n_cols):  # Inner-loop is serialized by Numba
            state = state_array[i, j]

            # Transition to Early Burning
            if state == combustible and can_ignite[i, j]:
                transition_state = transition_array[i, j]
                if early_burning <= transition_state < full_burning:
                    state_array[i, j] = early_burning
                elif transition_state >= full_burning:
                    state_array[i, j] = full_burning

            # Transition to Full Burning
            elif state == early_burning:
                state_array[i, j] = full_burning

            # Transition to Extinguishing State
            elif state == full_burning and can_extinguish[i, j]:
                state_array[i, j] = extinguishing

            # Transition to Burnt State
            elif state == extinguishing:
                state_array[i, j] = burnt
    return state_array


@numba.jit(nopython=True, parallel=False, fastmath=True)
def initial_spread_rate(
    temperature: float, wind_speed: float, humidity: float
) -> float:
    """Note:
    Wind-Force Integer is neglected (set equal to 1) as it is
    arbitrarily defined
    """
    (
        a,
        b,
        c,
        d,
    ) = 0.03, 0.05, 0.01, 0.3
    return (
        a * temperature
        + b * (wind_speed / 0.836) ** (2 / 3)
        + c * (100 - humidity)
        - d
    )


# TODO document & test functions below
@numba.jit(
    nopython=True, parallel=False, fastmath=True
)  # TODO test if parallel makes a difference!
def wind_coefficient(
    wind_speed: float, wind_dir: tuple[float, float], prop_dir: tuple[float]
) -> float:
    """Computes the non-dimensional Wind Coefficient, K_phi, utilizing
    the relation

    Args:
        wind_speed: [description]
        wind_dir: [description]
        prop_dir: [description]

    Returns:

    """
    return math.exp(0.1783 * wind_speed * dot_2d(wind_dir, prop_dir))


@numba.jit(nopython=True, parallel=False, fastmath=True)
def slope_coefficient(direction: int, slope: float):
    raise NotImplementedError("The direction is not yet implemented")
    # return math.exp(3.553 * direction * math.tan(1.2 * slope))


@numba.jit(nopython=True, parallel=False, fastmath=True)
def time_step(
    maximum_spread_rate: float,
    cell_size: float,
    step_size_factor: float | None = 0.125,
) -> float:
    """Calculates the dynamic physical time-step in SI minutes. Since,
    it is based on the maximum spread-rate and cell-size, this quantity
    ensures that the fire is constrained to propagating to a single cell
    per iteration.

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
        step_size_factor: DESCRIPTION HERE. Defaults to 0.125 (1/n_neighbors).

    Returns:
        float: [description]
    """
    return step_size_factor * cell_size / maximum_spread_rate


def initialize_arrays(
    shape: tuple[int, int],
    ambient_temperature: float,
    wind_speed: float,
    wind_direction: tuple[float],
    relative_humidity: float,
    avg_combustibility: float,
    stochastic: bool | None = True,
):
    #   Dict[str, np.ndarray]:
    """Simplifies the creation of the arrays required for the
    forest-fire spread model of Rui et al. 2018
    """
    from functools import partial

    # State-Array Initialization (All cells are ignitable)
    state_array = np.full(shape, fill_value=combustible, dtype=np.int8)

    # Partially filled-in function to ease creation of arrays
    float_array = partial(np.full, shape=shape, dtype=np.float64)

    # Transition-State array
    transition_array = float_array(fill_value=1)

    # Temperature Array Initialization
    temperature_array = float_array(fill_value=ambient_temperature)

    # Wind Array Initialization
    wind_speed = float_array(fill_value=wind_speed)
    wind_dir_i = float_array(fill_value=wind_direction[0])
    wind_dir_j = float_array(fill_value=wind_direction[1])
    wind_array = np.stack((wind_speed, wind_dir_i, wind_dir_j), axis=2)

    # Initializing Humidity Array
    humidity_array = float_array(fill_value=relative_humidity)

    # Initializing Slope Array
    # FIXME slope should not be 1 if an actual calculation is taking place
    slope_array = float_array(fill_value=1)

    # Initializing Combustibility Array
    combustibility_array = float_array(fill_value=avg_combustibility)
    if stochastic:
        combustibility_array = add_noise(combustibility_array, 2.0)

    return {
        "state_array": state_array,
        "transition_array": transition_array,
        "temperature_array": temperature_array,
        "wind_array": wind_array,
        "humidity_array": humidity_array,
        "slope_array": slope_array,
        "combustibility_array": combustibility_array,
    }


def add_noise(array: np.ndarray, std: float = 1.0) -> np.ndarray:
    """Adds noise to an ``array`` in-place utilizing a Normal
    (Gaussian) distribution. The standard deviation, ``std``, can be
    used to adjust the amount of variance in the output array.

    Args:
        array: Something
        shape: Shape of ouput array (n, m, k)
        std: Coefficient of Variation (default = 0.01)

    Returns:
        Stochastic Array with a variance scaled by the ``mean``

    """
    array += np.random.normal(loc=0, scale=std, size=array.shape)
    return array


if __name__ == "__main__":
    width = 300
    height = 300
    arrays = initialize_arrays(
        shape=(height, width),
        ambient_temperature=35,
        wind_speed=0,
        wind_direction=(1, 0),
        relative_humidity=20,
        avg_combustibility=1.8,
        stochastic=True,
    )

    arrays["state_array"][height // 2, width // 2] = full_burning

    propagate(**arrays, cell_size=30)
    # propagate.parallel_diagnostics(level=4)

    _msg = "Size of Arrays in Memory: {} MB"
    print(_msg.format(sum(a.nbytes for a in arrays.values()) / 1e6))
