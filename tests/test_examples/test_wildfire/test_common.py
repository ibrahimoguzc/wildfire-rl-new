# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from functools import partial
from math import isnan, nan

import numpy as np
import pytest

from examples.wildfire.fire_model.jit_funcs.common import (
    calc_ideal_time_step,
    calc_initial_spread_rate,
    calc_slope_coefficient,
    calc_spread_rate,
    calc_wind_coefficient,
    postprocess,
    preprocess,
    sum_neighbors,
)
from examples.wildfire.fire_model.states import (
    BURNT,
    COMBUSTIBLE,
    EARLY_BURNING,
    EXTINGUISHING,
    FULL_BURNING,
    NONFLAMMABLE,
    SUPPRESSED,
)
from tests.snippets import test_jit_compile  # noqa: F401

# Short-hand Notation for reducing verbosity of test-cases
NF, CB, EB, FB, EX, BT = (
    NONFLAMMABLE,
    COMBUSTIBLE,
    EARLY_BURNING,
    FULL_BURNING,
    EXTINGUISHING,
    BURNT,
)
# TODO these formats should be extracted from someplace such as a class
fs = partial(np.array, dtype=np.uint8)  # fs = Fire States
PREPROCESS_TEST_CASES = {
    "argnames": (
        "fire_states,"
        "expected_aspect,"
        "expected_can_ignite,"
        "expected_can_extinguish"
    ),
    "argvalues": [
        # All neighbors are full burning
        (fs([[FB, FB, FB], [FB, CB, FB], [FB, FB, FB]]), nan, True, True),
        # All neighbors are combustible and SE neighbor is full burning
        (fs([[CB, CB, CB], [CB, CB, CB], [CB, CB, FB]]), 315, True, False),
        # All neighbors are combustible and NE neighbor is full burning
        (fs([[CB, CB, FB], [CB, CB, CB], [CB, CB, CB]]), 225, True, False),
        # All neighbors are combustible and SW neighbor is full burning
        (fs([[CB, CB, CB], [CB, CB, CB], [FB, CB, CB]]), 45, True, False),
        # All neighbors are combustible and NE & SW corners are burning
        (fs([[CB, CB, FB], [CB, CB, CB], [FB, CB, CB]]), nan, True, False),
        # All neighbors burning or extinguishable states
        (fs([[FB, FB, EX], [FB, EB, EX], [NF, FB, EX]]), 117, True, True),
        # Mix of early burning, combustible, and non-flammable
        (fs([[NF, EB, NF], [CB, EB, EB], [EB, NF, NF]]), nan, False, False),
    ],
}


@pytest.mark.parametrize(**PREPROCESS_TEST_CASES)
def test_preprocess(
    fire_states, expected_aspect, expected_can_ignite, expected_can_extinguish
):
    """Tests preprocessing step with sample 3x3 arrays.

    Since the aspect (bearing) is used, it is important to test that
    the correct propagation direction is returned in all 4 quadrants
    """
    pos = (1, 1)  # Position is set to the center of the 3x3 window

    # Testing the correctness of the fast_math option!
    result = preprocess.py_func(fire_states, pos)
    jit_result = preprocess(fire_states, pos)
    assert result == pytest.approx(jit_result, nan_ok=True)

    # Unpacking results for readability
    aspect, can_ignite, can_extinguish = result
    if isnan(aspect):
        assert isnan(expected_aspect)
    else:
        # Aspect does not need to be so precise ~1 SI degree is enough
        assert aspect == pytest.approx(expected_aspect, abs=1e0)
    assert can_ignite == expected_can_ignite
    assert can_extinguish == expected_can_extinguish


SPREAD_RATE_BASE_DATA = {  # DO NOT EDIT!
    "temperature": 15,
    "wind_speed": 10,
    "wind_aspect": 180,  # Wind blowing from the South (towards North)
    "humidity": 35,
    "terrain_slope": 25,
    "terrain_aspect": 0,  # Dir. of steepest descent (North = downhill)
    "combustibility": 1.0,
    "correction_coefficient": 1.0,
}
# NAN_RESULT is for the edge-case with no propagation direction
BASE_RESULT, NAN_RESULT = 0.8116980012, 1.061516535
SPREAD_RATE_TEST_CASES = {
    "argnames": "fire_state, prop_aspect, variable, validator",
    "argvalues": [
        # Testing that early-burning cells do not return a spread-rate
        (EARLY_BURNING, 0, None, lambda r: r == 0),
        # Testing base-case w/ wind & slope coefficients neglected
        (FULL_BURNING, nan, None, lambda r: r == pytest.approx(NAN_RESULT)),
        # Testing base-case w/ wind & slope coefficients computed
        (FULL_BURNING, 0, None, lambda r: r == pytest.approx(BASE_RESULT)),
        # Testing positive correlation of increased temperature
        (FULL_BURNING, 0, {"temperature": 25}, lambda r: r > BASE_RESULT),
        # Testing positive correlation of increased wind speed
        (FULL_BURNING, 0, {"wind_speed": 15}, lambda r: r > BASE_RESULT),
        # Testing negative correlation of reversed wind aspect
        (FULL_BURNING, 0, {"wind_aspect": 0}, lambda r: r < BASE_RESULT),
        # Testing negative correlation of increased humidity
        (FULL_BURNING, 0, {"humidity": 45}, lambda r: r < BASE_RESULT),
        # Testing negative correlation of increased (downhill) slope
        (FULL_BURNING, 0, {"terrain_slope": 35}, lambda r: r < BASE_RESULT),
        # Testing positive correlation of reversed terrain aspect
        (FULL_BURNING, 0, {"terrain_aspect": 180}, lambda r: r > BASE_RESULT),
        # Testing positive correlation of increased combustibility
        (FULL_BURNING, 0, {"combustibility": 1.8}, lambda r: r > BASE_RESULT),
        # Testing positive correlation of increased correction_coeff
        (
            FULL_BURNING,
            0,
            {"correction_coefficient": 1.5},
            lambda r: r > BASE_RESULT,
        ),
    ],
}


@pytest.mark.parametrize(**SPREAD_RATE_TEST_CASES)
def test_calc_spread_rate(fire_state, prop_aspect, variable, validator):
    """Tests the spread-rate calculation for expected correlations.

    The base-result is calculated manually (old fashioned way with a
    calculator). This numeric value is used to check the exact
    spread-rate. Afterwards, this value is used to gauge if the
    correct correlation is observed when updating base-case variable(s)
    by providing updated key, value pairs to ``variable``.
    """
    data = SPREAD_RATE_BASE_DATA.copy()
    if variable is not None:
        data.update(variable)  # Updating copy of base case data
    result = calc_spread_rate.py_func(fire_state, prop_aspect, **data)
    jit_result = calc_spread_rate(fire_state, prop_aspect, **data)
    assert result == pytest.approx(jit_result, nan_ok=True)
    assert validator(result)


def test_calc_initial_spread_rate():
    """Tests the initial spread rate yields correct results.

    Note:
        The underlying python function is not tested beforehand by
        :py:func`test_calc_spread_rate`, as it calls the JIT-compiled
        version of :py:func`calc_initial_spread_rate`.

    """
    data = {
        key: SPREAD_RATE_BASE_DATA[key]
        for key in ("temperature", "humidity", "wind_speed")
    }
    result = calc_initial_spread_rate.py_func(**data)
    jit_result = calc_initial_spread_rate(**data)
    assert result == pytest.approx(jit_result)  # nan is not ok here!
    assert result == pytest.approx(NAN_RESULT)


WIND_COEFFICIENT_TEST_CASES = {
    "argnames": "wind_speed, wind_aspect, prop_aspect, expected_result",
    "argvalues": [
        # Testing if zero wind-speed returns 1.0
        (0, 180, 0, 1.0),
        # Re-testing base-case w/ ``py_func``
        (10, 180, 0, 5.947672699),
        # Testing wind blowing from quadrant I (NE)
        (10, 225, 0, 3.528142014),
        # Testing wind blowing from quadrant II (SE)
        (10, 315, 0, 0.2834353027),
        # Testing wind blowing from quadrant III (SW)
        (10, 45, 0, 0.2834353027),
        # Testing wind blowing from quadrant IV (NW)
        (10, 135, 0, 3.528142014),
        # Testing prop-aspect is nan (edge-case)
        (10, 0, nan, nan),
    ],
}


@pytest.mark.parametrize(**WIND_COEFFICIENT_TEST_CASES)
def test_calc_wind_coefficient(
    wind_speed, wind_aspect, prop_aspect, expected_result
):
    """Testing wind coefficient calculation in all quadrants."""
    result = calc_wind_coefficient.py_func(
        wind_speed, wind_aspect, prop_aspect
    )
    jit_result = calc_wind_coefficient(wind_speed, wind_aspect, prop_aspect)
    assert result == pytest.approx(jit_result, nan_ok=True)
    assert result == pytest.approx(expected_result, nan_ok=True)


SLOPE_COEFFICIENT_TEST_CASES = {
    "argnames": "terrain_slope, terrain_aspect, prop_aspect, expected_result",
    "argvalues": [
        # Testing if zero slope returns 1.0
        (0, 0, 0, 1.0),
        # Re-testing base-case w/ ``py_func``
        (25, 0, 0, 0.1285643777),
        # Testing slope facing ~East (exactly 90 degrees min. angle)
        (25, 89.9, 0, 0.1285643777),
        # Testing slope facing East (exactly 90 degrees min. angle)
        (25, 90, 0, 7.778204336),
        # Testing slope facing South (exactly 180 degrees min. angle)
        (25, 180, 0, 7.778204336),
        # Testing slope facing West (exactly 90 degrees min. angle)
        (25, 270, 0, 7.778204336),
        # Testing reversal of prop. aspect and slope aspect
        (25, 0, 180, 7.778204336),
        # Testing nan value for prop aspect (should return nan)
        (25, 0, nan, nan),
    ],
}


@pytest.mark.parametrize(**SLOPE_COEFFICIENT_TEST_CASES)
def test_calc_slope_coefficient(
    terrain_slope, terrain_aspect, prop_aspect, expected_result
):
    """Testing slope coefficient calculation in all edge-cases."""
    result = calc_slope_coefficient.py_func(
        terrain_slope, terrain_aspect, prop_aspect
    )
    jit_result = calc_slope_coefficient(
        terrain_slope, terrain_aspect, prop_aspect
    )
    assert result == pytest.approx(jit_result, nan_ok=True)
    assert result == pytest.approx(expected_result, nan_ok=True)


def test_calc_ideal_time_step():
    """Simple test of ideal time step equation."""
    with pytest.raises(ZeroDivisionError):
        calc_ideal_time_step.py_func(0, 30, 1)
    result = calc_ideal_time_step.py_func(1, 30, 1)
    jit_result = calc_ideal_time_step(1, 30, 1)
    assert result == pytest.approx(jit_result)
    assert result == 30


POSTPROCESS_TEST_CASES = {
    "argnames": (
        "fire_state,"
        "intermediate_state,"
        "can_ignite,"
        "can_extinguish,"
        "expected_result"
    ),
    "argvalues": [
        # Ensuring Nonflammable states aren't affected by other vars
        (NONFLAMMABLE, 2.5, True, True, NONFLAMMABLE),
        (SUPPRESSED, 2.5, True, True, SUPPRESSED),
        # Testing transition from combustible to early-burning
        (COMBUSTIBLE, EARLY_BURNING, True, False, EARLY_BURNING),
        # Testing transition directly from combustible to full-burning
        (COMBUSTIBLE, FULL_BURNING, True, False, FULL_BURNING),
        # Testing transition form early-burning to full-burning
        (EARLY_BURNING, FULL_BURNING, True, False, FULL_BURNING),
        # Testing transition to extinguishing
        (FULL_BURNING, FULL_BURNING, True, True, EXTINGUISHING),
        # Testing transition from extinguishing to burnt
        (EXTINGUISHING, FULL_BURNING, True, True, BURNT),
    ],
}


@pytest.mark.parametrize(**POSTPROCESS_TEST_CASES)
def test_postprocess(
    fire_state, intermediate_state, can_ignite, can_extinguish, expected_result
):
    """Test if the postprocessing step yields expected results.

    Note:
        The underlying Python function is tested at this step to ensure
        that the code-coverage report neglects the Numba decorator.
        Otherwise, a very low coverage will be returned as coverage.py
        has a difficult time tracing the path of Numba. This means, that
        another test needs to be performed per function decorated with
        :py:function:`numba.jit` to make sure that it compiles!

    """
    result = postprocess.py_func(
        fire_state, intermediate_state, can_ignite, can_extinguish
    )
    jit_result = postprocess(
        fire_state, intermediate_state, can_ignite, can_extinguish
    )
    assert result == jit_result
    assert result == expected_result


# TODO add test cases for different neighborhoods
def test_sum_neighbors():
    """Tests summation of neighbors within a Moore neighborhood."""
    test_array = np.ones((3, 3), dtype=np.float64)
    result = sum_neighbors.py_func(test_array, (1, 1))
    jit_result = sum_neighbors(test_array, (1, 1))
    assert result == jit_result
    assert result == 8
