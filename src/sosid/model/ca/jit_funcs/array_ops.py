# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains functions for performing operations on 2D arrays."""

import numba
import numpy as np

__author__ = "San Kilkis"
__all__ = ["any_in_neighbors", "depad", "pad", "sum_neighbors"]

# Moore Neighborhood = 8 neighboring cells in a square cellular space
moore_neighborhood = ((-1, 1), (-1, 1))  # ((i_limits), (j_limits))


@numba.jit(nopython=True, parallel=True)
def sum_neighbors(
    array: np.ndarray, include_center: bool = False
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
    _padded_array = pad(array, width=1, pad_value=0)
    return depad(_sum_kernel(_padded_array, include_center), width=1)


@numba.stencil(neighborhood=moore_neighborhood)  # Moore Neighborhood
def _sum_kernel(array: np.ndarray, include_center: bool = False) -> np.ndarray:
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
    for i_offset in (-1, 0, 1):  # Row Indices
        for j_offset in (-1, 0, 1):  # Column Indices
            # Optionally exclude caller (center) cell
            if i_offset == 0 and j_offset == 0 and not include_center:
                pass
            else:
                _sum += array[i_offset, j_offset]
    return _sum


# The function below is approximitely 10x faster for a 1000x1000 array!!!
# TODO remember that w/ integers the iteration is much faster
@numba.jit(nopython=True, parallel=True)
def any_in_neighbors(array: np.ndarray, value: float | bool) -> np.ndarray:
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
    # TODO implement an automatic way to determine the pad value
    _padded_array = pad(array, width=1, pad_value=0)
    return depad(_any_kernel(_padded_array, value), width=1)


@numba.stencil(neighborhood=moore_neighborhood)
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
    for i_offset in (-1, 0, 1):  # Row Indices
        for j_offset in (-1, 0, 1):  # Column Indices
            # Exclude caller (center) cell
            if i_offset == 0 and j_offset == 0:
                pass
            else:
                neighbor = array[i_offset, j_offset]
                if neighbor == value:
                    match = True
                    break  # breaks from the inner loop
        if match:  # Executed if the inner loop found a match
            break
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
        pad_value: Value to be padded to the edges of each axis.
            Defaults to 0.

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
