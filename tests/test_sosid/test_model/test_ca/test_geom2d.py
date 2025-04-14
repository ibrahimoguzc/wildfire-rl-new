# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import math

import pytest

from sosid.model.ca.jit_funcs.geom2d import (
    angle_between,
    aspect_2d,
    dot_2d,
    magnitude_2d,
    normalize_2d,
    reverse_angle,
)
from tests.snippets import test_jit_compile  # noqa: F401

REL_TOL = 1e-09
"""float: Defines the relative tolerance (% Difference) between the
returned value of the test and the expected value of the test
"""

MAGNITUDE_2D_TEST_CASES = [
    ((0, 0), 0),
    ((-1, 0), 1),
    ((1, 1), math.sqrt(2)),
    ((20, 5), math.sqrt(425)),
]


@pytest.mark.parametrize("vector, expected_result", MAGNITUDE_2D_TEST_CASES)
def test_magnitude_2d(vector, expected_result):
    """Test of vector magnitude with float and int cases."""
    result, jit_result = magnitude_2d.py_func(*vector), magnitude_2d(*vector)
    assert result == pytest.approx(jit_result, rel=REL_TOL)
    assert result == pytest.approx(expected_result, rel=REL_TOL)


NORMALIZE_2D_TEST_CASES = [
    # Testing edge-case with divide by zero
    ((0, 0), (0, 0)),
    # Testing w/ integer return
    ((-2, 0), (-1, 0)),
    # Testing w/ float return
    ((-3, -4), (-0.6, -0.8)),
]


@pytest.mark.parametrize("vector, expected_result", NORMALIZE_2D_TEST_CASES)
def test_normalize_2d(vector, expected_result):
    """Test of vector normalization with float, int, and edge cases."""
    result, jit_result = normalize_2d.py_func(*vector), normalize_2d(*vector)
    assert result == pytest.approx(jit_result, rel=REL_TOL)
    assert result == pytest.approx(expected_result, rel=REL_TOL)


DOT_2D_TEST_CASES = [
    ((0, 0), (1, 0), 0),  # int test
    ((1.5, 2.5), (-3.125, 2.0), 0.3125),  # float test
]


@pytest.mark.parametrize("a, b, expected_result", DOT_2D_TEST_CASES)
def test_dot_2d(a, b, expected_result):
    """Test of dot product with int and float test cases."""
    result, jit_result = dot_2d.py_func(a, b), dot_2d(a, b)
    assert result == pytest.approx(jit_result, rel=REL_TOL)
    assert result == pytest.approx(expected_result, rel=REL_TOL)


ASPECT_TEST_CASES = {
    "argnames": "vector, expected_aspect",
    "argvalues": [
        ((0, 0), math.nan),  # Edge case w/ no direction
        ((-1, 0), 0),  # North
        ((1, 1), 135),  # South-East
        ((-1, -1), 315),  # North-West
    ],
}


@pytest.mark.parametrize(**ASPECT_TEST_CASES)
def test_aspect_2d(vector, expected_aspect):
    """Tests transformation of a 2D vector into aspect (bearing).

    Note:
        Coordinate system for 2D space in test is assumed to be
        i down j to the right, however this should not make a difference

    """
    result, jit_result = aspect_2d.py_func(*vector), aspect_2d(*vector)
    assert result == pytest.approx(jit_result, nan_ok=True)
    assert result == pytest.approx(expected_aspect, rel=REL_TOL, nan_ok=True)


ANGLE_BETWEEN_TEST_CASES = {
    "argnames": "aspect_1, aspect_2, expected_result",
    "argvalues": [
        (0, 180, 180),  # Sanity check
        (-45, 90, 135),  # Testing if sign matters
        (350, 10, 20),  # Testing if the minimum angle is returned
        (370, 10, 0),  # Testing if > 360 values work
        (math.nan, 0, math.nan),  # Testing if nan is returned
    ],
}


@pytest.mark.parametrize(**ANGLE_BETWEEN_TEST_CASES)
def test_angle_between(aspect_1, aspect_2, expected_result):
    """Tests if minimum angle diregards sign of aspects and nan."""
    result = angle_between.py_func(aspect_1, aspect_2)
    jit_result = angle_between(aspect_1, aspect_2)
    assert jit_result == pytest.approx(result, nan_ok=True)
    assert result == pytest.approx(expected_result, rel=REL_TOL, nan_ok=True)


REVERSE_TEST_CASES = {
    "argnames": "aspect, expected_result",
    "argvalues": [
        (0, 180),  # Sanity check
        (45, 225),  # Testing Northeast direction, Quadrant (I)
        (135, 315),  # Testing Southeast direction, Quadrant (II)
        (215, 35),  # Testing Slightly off Southwest direction, Quadrant (III)
        (295, 115),  # Testing Slightly off Northwest direction, Quadrant (IV)
        (math.nan, math.nan),  # Ensuring nan is returned
    ],
}


@pytest.mark.parametrize(**REVERSE_TEST_CASES)
def test_reverse_angle(aspect, expected_result):
    """Testing if reversing an angle (aspect) works as expected."""
    result = reverse_angle.py_func(aspect)
    jit_result = reverse_angle(aspect)
    assert jit_result == pytest.approx(result, nan_ok=True)
    assert result == pytest.approx(expected_result, nan_ok=True)
