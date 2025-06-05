import numba
import numpy as np

from sosid.model.ca.neighborhood import MooreNeighborhood

""" Contains utility functions for performing operations on the Moore
neighborhood of 2D Cellular Automata (CA) spaces """

__author__ = "San Kilkis"
__all__ = ["any_in_neighbors", "depad", "pad", "sum_neighbors"]

# Moore Neighborhood = 8 neighboring cells in a square cellular space
moore_neighborhood = ((-1, 1), (-1, 1))  # ((i_limits), (j_limits))
MOORE = MooreNeighborhood(radius=1, include_center=False)
MOORE_OFFSETS = MOORE.as_tuple()


@numba.jit(nopython=True, parallel=True)
def sum_neighbors(
    array: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """Parallelizes and JIT compiles the kernel defined by
    :py:func:`_sum_kernel` which sums the values in the Moore
    neighborhood of each cell of the ``array``.

    Args:
        array: Input 2D Numpy array
        include_center: Toggles if the current cell value should be summed

    Returns:
        A boolean array of the same shape as the input ``array``
    """
    # Utilizing a pad-value of 0 does not affect the summing result
    return _sum_kernel(array) if out is None else _sum_kernel(array, out=out)


@numba.stencil(neighborhood=MOORE.limits, cval=0.0)  # Moore Neighborhood
def _sum_kernel(array: np.ndarray) -> np.ndarray:
    """Defines the stencil kernel that sums the values in the Moore
    neighborhood of all cells in the the input ``array``.

    Warning:
        The Numba stencil decorator will default to a fill-value at the
        boundaries of the ``array`` to prevent out-of-bounds errors. As
        such, the ``array`` must be padded with a fill-value to obtain
        correct results at the boundaries.

    Args:
        array: Input 2D Numpy array
        include_center: Toggles if the current cell value should be summed

    Returns:
        A boolean array of the same shape as the input ``array``
    """
    _sum = 0
    for i_offset, j_offset in MOORE_OFFSETS:
        _sum += array[i_offset, j_offset]
    return _sum


# The function below is approximitely 10x faster for a 1000x1000 array!!!
# TODO remember that w/ integers the iteration is much faster
@numba.jit(nopython=True, parallel=True)
def any_in_neighbors(
    array: np.ndarray,
    value: float | bool,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Parallelizes and JIT compiles the kernel defined by
    :py:func:`_any_kernel` which applies a Moore neighborhood search on
    each cell of the ``array``, attempting to find any matching
    ``value`` within the neighboring 8 cells. Therefore, the kernel
    iterates through (-1, 0, 1) for both the i and j indices, and
    checks if ``array[i + i_offset, j + j_offset] == value``.

    Args:
        array: Input 2D Numpy array
        value: Compare value to search neighboring cells for a match

    Returns:
        A boolean array of the same shape as the input ``array``
    """
    return (
        _any_kernel(array, value)
        if out is None
        else _any_kernel(array, value, out=out)
    )


@numba.stencil(neighborhood=MOORE.limits, cval=False)
def _any_kernel(array: np.ndarray, value: float | bool | tuple) -> np.ndarray:
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
    match = False
    for i_offset, j_offset in MOORE_OFFSETS:
        neighbor = array[i_offset, j_offset]
        if neighbor == value:
            match = True
            break  # breaks from the inner loop
    return match


# TODO generalize into N-dimensions if possible
# TODO add tests
@numba.jit(nopython=True)
def pad(array, width, pad_value: float | None = 0):
    """Pads the edges of an input ``array`` with a constant
    ``pad_value``. The width determines the number of values padded to
    the edges of each axis.

    Caution:
        No checks are performed to make sure that the ``pad_value`` does
        not appear in the ``array``. Therefore, the pad-value must be
        carefully picked such that it is guaranteed not to appear within
        the input ``array``.

    Args:
        array: The input 2D Numpy array to pad
        width: Number of values padded to the edges of each axis
        pad_value: Value to be padded to the edges of each axis. Defaults to 0.

    Returns:
        Padded array
    """
    n_rows, n_cols = array.shape
    padded_array = np.zeros(
        (n_rows + 2 * width, n_cols + 2 * width), dtype=array.dtype
    )
    if pad_value != 0:
        padded_array[:, :] = pad_value
    padded_array[width:-width, width:-width] = array
    return padded_array


# TODO generalize into N dimensions
# TODO add tests
@numba.jit(nopython=True)
def depad(array: np.ndarray, width: int):
    """Removes an N-amount of edge values defined by ``width``
    from a padded ``array``.

    Args:
        array: The input 2D Numpy array to remove a pad from (depad)
        width: Number of values padded to the edges of each axis

    Returns:
        De-padded array
    """
    return array[width:-width, width:-width]
