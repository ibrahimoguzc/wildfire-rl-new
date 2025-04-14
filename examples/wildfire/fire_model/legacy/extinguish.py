import numba
import numpy as np

from examples.wildfire.fire_model.legacy.states import (
    full_burning,
    nonflammable,
)

""" Contains all functions responsible for calculating the information
necessary to determine when cells can be extinguished """


# TODO remember that w/ integers the iteration is much faster
# TODO if there is time, generalize this function using rethrow w/ decorator
# (a python decorator that will interpret the input and serialize it to a Numba
#  understandable format)
# TODO change this back to all_in_neighborhood w/ tuple argument or using Numpy
# binary_and
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
    _padded_array = pad(array, width=1, pad_value=nonflammable)
    return depad(_extinguishables_kernel(_padded_array), width=1)


@numba.stencil(neighborhood=((-1, 1), (-1, 1)))  # Moore Neighborhood
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
    for i_offset in (-1, 0, 1):  # Row Indices
        for j_offset in (-1, 0, 1):  # Column Indices
            # Exclude caller (center) cell
            if i_offset == 0 and j_offset == 0:
                pass
            else:
                neighbor = array[i_offset, j_offset]
                if neighbor >= full_burning or neighbor == nonflammable:
                    count += 1
    return count == 8  # Moore Neighborhood
