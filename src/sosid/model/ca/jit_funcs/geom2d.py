# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Collection of JIT-compiled 2D geometry functions.

These functions are compiled utilizing the `fast_math` option, to
decrease the execution time (micro-performance) of calculations. In
addition, although they can be written in a more generalized way, most
Cellular Automata (CA) models deal with 2D arrays. As such, they have
been written specifically for this case, once again with the objective
of decreasing the run-time of these calculations.

Note:
    Calling a :py:func:`numba.jit` decorated function from a CUDA kernel
    automatically compiles an equivalent CUDA device function according
    to the following example:

    https://github.com/numba/numba-examples/blob/master/examples/density_estimation/histogram/gpu.py

"""

import math

import numba
import numpy as np

from sosid.jit_config import BASE_JIT_KWARGS, FAST_MATH_FLAGS

__all__ = [
    "angle_between",
    "aspect_2d",
    "dot_2d",
    "gradient_2d",
    "magnitude_2d",
    "normalize_2d",
]


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def magnitude_2d(i: float, j: float) -> float:
    """Obtains the magnitude of a 2D vector from its components.

    Tip:
        Use Python unpacking to pass any Sequence of length == 2 with
        the following syntax ``magnitude_2d(*(my_tuple))``

    Args:
        i: First component of vector
        j: Second component of vector

    Returns:
        Scalar magnitude of a vector computed from its components

    """
    return math.sqrt(i**2 + j**2)


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def normalize_2d(i: float, j: float) -> tuple[float, float]:
    """Normalizes a 2D vector into a unit-vector with a magnitude of 1.

    Tip:
        Use Python unpacking to pass any Sequence of length == 2 with
        the following syntax ``magnitude_2d(*(my_tuple))``

    Args:
        i: First component of vector
        j: Second component of vector

    Returns:
        Unit-vector of magnitude = 1

    """
    mag = magnitude_2d(i, j)
    return (i / mag, j / mag) if mag > 0 else (0, 0)


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def dot_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Dot product of two 2D vectors (1D arrays), ``a`` and ``b``.

    Args:
        a: 2D vector (i, j)
        b: 2D vector (i, j)

    Returns:
        The scalar dot product of a and b.

    """
    product = 0
    for i in range(2):
        product += a[i] * b[i]
    return product


@numba.jit(**BASE_JIT_KWARGS, parallel=True)
def gradient_2d(
    array: numba.float64[:, :], cell_size: float
) -> tuple[numba.float64[:, :], numba.float64[:, :]]:
    r"""Calculates gradients of ``array`` using the `ArcGIS`_ approach.

    This approach uses a Moore neighborhood of radius = 1 to compute
    the gradients of all cells within ``array``.

    Note:
        To avoid out-of-bounds errors, the gradient algorithm only runs
        on non-border elements. Therefore, if gradients are required at
        the edge of an array, then ``array`` must be padded with zeros
        for a pad-width equal to 1.

    Args:
        array: 2D array of scalar values
        cell_size: The physical dimension of each cell within ``array``

    Returns:
        The non-dimensional gradients of the terrain
        :math:`\frac{dz}{dx}` and :math:`\frac{dz}{dy}`.
    """
    n_rows, n_cols = array.shape

    # Pre-allocating gradients 3D array
    gradients = np.zeros((n_rows, n_cols, 2), dtype=np.float64)

    # Indexing, m x n to avoid conflict with i variable
    for m in numba.prange(1, n_rows - 1):
        for n in range(1, n_cols - 1):
            # Getting neighboring values, this is the optimum access
            # pattern for row-major order arrays
            a = array[m - 1, n - 1]
            b = array[m - 1, n + 0]
            c = array[m - 1, n + 1]
            d = array[m + 0, n - 1]
            f = array[m + 0, n + 1]
            g = array[m + 1, n - 1]
            h = array[m + 1, n + 0]
            i = array[m + 1, n + 1]

            # Calculating gradient in 1-st (x) and 0-th (y) axes
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_size)
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell_size)

            # Writing result into empty gradients array
            gradients[m, n] = dz_dx, dz_dy

    return gradients[..., 0], gradients[..., 1]


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def aspect_2d(i: int, j: int) -> float:
    r"""Converts a 2D vector into a scalar aspect in SI degree.

    Desired Orientation is North = 0 deg, East = 90 deg, South = 180
    deg, West = 270 deg. However, by default the coordinate system of
    the :py:func:`math.atan2` function is counter-clockwise positive
    (CCW+) and with ``i`` = East, ``j`` = North. Therfore, to obtain the
    desired orientation with the coordinate system of 2D Cellular
    Automata (``i`` = South, ``j`` = East), the output of the
    :py:func:`atan2` function is subtracted from :math:`\pi`. Finally,
    the resultant angle is converted to degrees manually since the
    :py:func:`numba.cuda.jit` decorator does not support
    :py:func:`math.degrees`.

    Note:
        The result of the :py:func:`math.atan2` function is between
        :math:`-\pi` and :math:`\pi`.

    See Also:
        Details about the implementation, as well as how to implement a
        way to change the orientation is documented in the following
        Stack Overflow post:
        https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors

    Args:
        i: First component of vector
        j: Second component of vector

    Returns:
        Aspect of the ``vector`` between 0 and 360 in SI degree. If the
        vector components ``i`` and ``j`` are both zero, then the aspect
        is equal to :py:obj:`math.nan`.

    """
    # Handling edge-case for vector of magnitude = 0
    if i == 0 and j == 0:
        return math.nan
    return (math.pi - math.atan2(j, i)) * 180 / math.pi


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def angle_between(aspect_1: float, aspect_2: float) -> float:
    """Retrieves the minimum unsigned angle between two aspects.

    Note:
        Since the minimum angle is unsigned, the order in which the
        aspects are provided does not matter. Nor does it matter if the
        aspects are between 0 and 360. If either of the provided aspects
        is :py:data:`math.nan` then this will also translate to the
        return value. This is required to deal with the possible
        :py:data:`math.nan` return of :py:func:`aspect_2d` to
        represent the zero vector edge-case.

    Args:
        aspect_1: First aspect (bearing) in SI Degree
        aspect_2: Second aspect (bearing) in SI Degree

    Returns:
        Minimum angle between ``aspect_1`` and ``aspect_2``

    Examples:
        >>> angle_between(0, 180)
        180
        >>> angle_between(-45, 90)
        135
        >>> angle_between(350, 10)
        20

    """
    return abs((180 + (aspect_1 - aspect_2)) % 360 - 180)


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def reverse_angle(aspect: float) -> float:
    """Reverses the angle specified by ``aspect``.

    Args:
        aspect: An aspect (bearing) in SI Degree

    Returns:
        Reversed (opposite) ``aspect`` where the angle between the
        returned value and ``aspect`` is 180.

    Examples:
        >>> reverse_angle(0)
        180
        >>> reverse_angle(-45)
        135
        >>> reverse_angle(270)
        90

    """
    return (aspect + 180) % 360


@numba.jit(**BASE_JIT_KWARGS, fastmath=FAST_MATH_FLAGS)
def calculate_confidence_area(center_x, center_y, width, length, orientation):
    """Defines the nodes that delimit the confidence area (a rotated rectangle)
    for the agent.

    Args:
        center_x (float): x-coordinate of the center of the confidence area
        center_y (float): y-coordinate of the center of the confidence area
        width (float): Width og the confidence area
        length (float): Length of the confidence area
        orientation (float): Orientation of the confidence area

    Returns:
        Array (2D): [(x,y),(x,y),...], Coordinates of edge nodes of confidence
        area
    """
    half_width = width / 2
    half_distance = length / 2
    hypotenuse = math.sqrt(half_distance**2 + half_width**2)

    # The angles to be used to define top, bottom, side and diagonal points.
    angle_alpha = math.atan(half_width / half_distance)
    angle_beta = math.radians(-90 + orientation)
    angle_gamma = math.radians(orientation)

    point_1 = [
        center_x + math.cos(angle_beta) * half_distance,
        center_y + math.sin(angle_beta) * half_distance,
    ]
    point_2 = [
        center_x + math.cos(angle_alpha + angle_beta) * hypotenuse,
        center_y + math.sin(angle_alpha + angle_beta) * hypotenuse,
    ]
    point_3 = [
        center_x + math.cos(angle_gamma) * half_width,
        center_y + math.sin(angle_gamma) * half_width,
    ]
    point_4 = [
        center_x + math.cos(angle_alpha + angle_gamma) * hypotenuse,
        center_y + math.sin(angle_alpha + angle_gamma) * hypotenuse,
    ]
    point_5 = [
        center_x - math.cos(angle_beta) * half_distance,
        center_y - math.sin(angle_beta) * half_distance,
    ]
    point_6 = [
        center_x - math.cos(angle_alpha + angle_beta) * hypotenuse,
        center_y - math.sin(angle_alpha + angle_beta) * hypotenuse,
    ]
    point_7 = [
        center_x - math.cos(angle_gamma) * half_width,
        center_y - math.sin(angle_gamma) * half_width,
    ]
    point_8 = [
        center_x + math.cos(angle_beta - angle_alpha) * hypotenuse,
        center_y + math.sin(angle_beta - angle_alpha) * hypotenuse,
    ]

    return np.array(
        [
            point_1,
            point_2,
            point_3,
            point_4,
            point_5,
            point_6,
            point_7,
            point_8,
        ]
    )
