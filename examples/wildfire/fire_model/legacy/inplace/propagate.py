import math

import numba
import numpy as np

try:
    from examples.wildfire.fire_model.legacy.fast_math import (
        dot_2d,
        normalize_2d,
    )
    from examples.wildfire.fire_model.legacy.inplace.moore_ops import (
        any_in_neighbors,
        sum_neighbors,
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
    from examples.wildfire.fire_model.legacy.fast_math import (
        dot_2d,
        normalize_2d,
    )
    from examples.wildfire.fire_model.legacy.inplace.moore_ops import (
        any_in_neighbors,
        sum_neighbors,
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

# CONCLUSIONS:
# SIZE OF ARRAYS MAKE THEM MEMORY BOUND (CPU Parallelization doesn't make too
# much of a difference)

MOORE = MooreNeighborhood(radius=1, include_center=False)
MOORE_OFFSETS = MOORE.as_tuple()
MOORE_RADIUS = MOORE.radius


# Solve problem of constant pad/unpad
# TODO remember that w/ integers the iteration is much faster
# TODO if there is time, generalize this function using rethrow w/ decorator (a python decorator that will interpret the input and serialize it to a Numba understandable format)
# TODO change this back to all_in_neighborhood w/ tuple argument or using Numpy binary_and
@numba.jit(nopython=True, parallel=True)
def find_extinguishables(
    array: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
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
    return (
        _extinguishables_kernel(array)
        if out is None
        else _extinguishables_kernel(array, out=out)
    )


@numba.stencil(neighborhood=MOORE.limits, cval=False)  # Moore Neighborhood
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
    rate_array: np.ndarray,
    can_ignite: np.ndarray,
    temperature_array: np.ndarray,
    wind_array: np.ndarray,
    humidity_array: np.ndarray,
    slope_array: np.ndarray,
    combustibility_array: np.ndarray,
    calc_needed: np.ndarray,
    correction_coeff: float | None = 1,
) -> tuple[np.ndarray, float]:
    # Determining which cells require calculation (Lazifies computation)
    calc_needed = state_array == full_burning
    # any_in_neighbors(can_ignite, True, out=calc_needed)

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
                rate_array[i, j] = (
                    r_0 * k_phi * k_theta * k_s * correction_coeff
                )
            else:
                rate_array[i, j] = 0
    return rate_array


# TODO use line-profiler without jit to see where expensive ops are
# TODO add option to provide time_step
# TODO increase useability by using a decorator to parse input to acceptable foramt
@numba.jit(
    nopython=True, parallel=True, fastmath=True
)  # 10x improvement from parallelization
def propagate(
    state_array: np.ndarray,
    rate_array: np.ndarray,
    transition_array: np.ndarray,
    temperature_array: np.ndarray,
    wind_array: np.ndarray,
    humidity_array: np.ndarray,
    slope_array: np.ndarray,
    combustibility_array: np.ndarray,
    can_ignite: np.ndarray,
    can_extinguish: np.ndarray,
    calc_needed: np.ndarray,
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
    any_in_neighbors(state_array, full_burning, out=can_ignite)
    find_extinguishables(state_array, out=can_extinguish)
    spread_rate(
        state_array=state_array,
        rate_array=rate_array,
        can_ignite=can_ignite,
        temperature_array=temperature_array,
        wind_array=wind_array,
        humidity_array=humidity_array,
        slope_array=slope_array,
        combustibility_array=combustibility_array,
        correction_coeff=correction_coeff,
        calc_needed=calc_needed,
    )

    # UNCOMMENT BELOW FOR DYNAMIC TIME STEP
    # r_max = np.max(rate_array)
    # # if r_max > 0:
    # t_step_ideal = ideal_time_step(
    #     maximum_spread_rate=r_max, cell_size=cell_size
    # )
    # if time_step is not None and time_step < t_step_ideal:
    #     t_step = time_step
    # else:
    #     t_step = t_step_ideal
    summed_rates = sum_neighbors(rate_array)
    transition_array += summed_rates * time_step / cell_size

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
    ) = (
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
def ideal_time_step(
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

    # Propagation rates array
    rate_array = float_array(fill_value=0)

    # Transition-State array (+1 since we have added a new state)
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
        combustibility_array = add_noise(combustibility_array, 1.8)

    # Boolean Arrays
    can_ignite = np.full(shape, fill_value=False, dtype=bool)
    can_extinguish = np.full(shape, fill_value=False, dtype=bool)
    calc_needed = np.full(shape, fill_value=False, dtype=bool)

    return {
        "state_array": state_array,
        "rate_array": rate_array,
        "transition_array": transition_array,
        "temperature_array": temperature_array,
        "wind_array": wind_array,
        "humidity_array": humidity_array,
        "slope_array": slope_array,
        "combustibility_array": combustibility_array,
        "can_ignite": can_ignite,
        "can_extinguish": can_extinguish,
        "calc_needed": calc_needed,
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
    width = 1000
    height = 1000
    arrays = initialize_arrays(
        shape=(height, width),
        ambient_temperature=35,
        wind_speed=0,
        wind_direction=(0.5, 0),
        relative_humidity=20,
        avg_combustibility=1.8,
        stochastic=True,
    )

    arrays["state_array"][height // 2, width // 2] = full_burning

    # %timeit -n 10 propagate(**arrays, cell_size=30)
    # propagate.parallel_diagnostics(level=4)

    _msg = "Size of Arrays in Memory: {} MB"
    print(_msg.format(sum(a.nbytes for a in arrays.values()) / 1e6))
