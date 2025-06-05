import math

import numba

""" Collection of JIT-compiled math functions utilizing the `fast_math`
option, to decrease the execution time (micro-performance) of some
calculations """

__author__ = "San Kilkis"


@numba.jit(nopython=True, fastmath=True)
def magnitude_2d(i: float, j: float) -> float:
    """Obtains the magnitude of a 2D vector (1D array)
    from its components ``i`` and ``j``.

    Note:
        Use Python unpacking to pass any Sequence of length == 2 with
        the following syntax ``magnitude_2d(*(my_tuple))``

    Args:
        i: First component of vector
        j: Second component of vector

    Returns:
        Scalar magnitude of a vector computed from its components
    """
    return math.sqrt(i**2 + j**2)


@numba.jit(nopython=True, fastmath=True)
def normalize_2d(i: float, j: float) -> tuple[float, float]:
    """Normalizes a 2D vector (1D array) into a unit-vector with a
    magnitude equal to 1.

    Note:
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


@numba.jit(nopython=True, fastmath=True)
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


# TODO consider moving to jit_funcs of fire_model
@numba.jit(nopython=True, fastmath=True)
def bearing_2d(
    vector: tuple[float, float], true_north: tuple[int] | None = (-1, 0)
) -> float:
    """Converts the provided ``vector`` in to a scalar bearing in SI
    degree. Default Orientation is North = 0 deg, East = 90 deg,
    South = 180 deg, West = 270 deg.

    Note:
        Refer to https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors

    Args:
        direction: 2D direction vector (i, j) to convert into a bearing
        true_north: 2D vector (i, j) specifying true-north

    Returns:
        Bearing of the vector ``vector``
    """
    angle = math.atan2(*true_north[::-1]) - math.atan2(*vector[::-1])
    if angle < 0:
        angle += 2 * math.pi
    return math.degrees(angle)


@numba.jit(nopython=True, fastmath=True)
def angle_between(angle_1: float, angle_2: float) -> float:
    return 360 - (abs(angle_1 - angle_2) % 360)
