# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import math
from functools import partial

import numpy as np
import pytest

from sosid.environment.terrain import HeightMappedElevation
from tests.snippets import ScenarioTestSuite

pad_zero = partial(np.pad, pad_width=1, mode="constant")

TERRAIN_SCENARIOS = {
    "argnames": "label, args",
    "argvalues": [
        # Testing single central peak (elevation = 1 meter at [1, 1])
        # Additionally tests if cell_size scales output properly
        ("peak", (pad_zero(np.array([[1]])), 2)),
        # Testing edge case (flat terrain with zeros elevation)
        ("zero_altitude", (np.zeros((3, 3)),)),
        # Testing with ArcGIS sample calculation
        # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-hillshade-works.htm
        ("arcgis", (np.array([[101, 92, 85], [101, 92, 85], [101, 91, 84]]),)),
    ],
}


class TestTerrain(ScenarioTestSuite):
    test_class = HeightMappedElevation
    scenarios = TERRAIN_SCENARIOS

    EXPECTED_GRADIENTS = {
        "peak": (
            np.array(
                [
                    [0.0625, 0, -0.0625],
                    [0.125, 0, -0.125],
                    [0.0625, 0, -0.0625],
                ]
            ),
            np.array(
                [
                    [0.0625, 0.125, 0.0625],
                    [0, 0, 0],
                    [-0.0625, -0.125, -0.0625],
                ]
            ),
        ),
        "zero_altitude": (np.zeros((3, 3)), np.zeros((3, 3))),
        "arcgis": (
            np.array(
                [
                    [-4.5, -8, -3.5],
                    [-4.625, -8.125, -3.5],
                    [-4.875, -8.375, -3.5],
                ]
            ),
            np.array(
                [[0, 0, 0], [-0.125, -0.375, -0.5], [-0.125, -0.375, -0.5]]
            ),
        ),
    }

    def test_gradients(self, scenario):
        """Tests gradients calculation with sample 3x3 matrices."""
        expected_results = self.EXPECTED_GRADIENTS[scenario.label]
        results = scenario.obj.gradients
        for result, expected in zip(results, expected_results):
            assert np.allclose(result, expected)

    EXPECTED_SLOPES = {
        "peak": (
            np.array(
                [
                    [5.05115253, 7.12501635, 5.05115253],
                    [7.12501635, 0.00000000, 7.12501635],
                    [5.05115253, 7.12501635, 5.05115253],
                ]
            )
        ),
        "zero_altitude": np.zeros((3, 3)),
        "arcgis": (
            np.array(
                [
                    [77.47119229, 82.87498365, 74.05460410],
                    [77.80385147, 82.99088528, 74.20683095],
                    [78.41153038, 83.19770141, 74.20683095],
                ]
            )
        ),
    }

    def test_slopes(self, scenario):
        """Tests calculation of slopes with sample 3x3 matrices."""
        result = scenario.obj.slopes
        assert np.allclose(result, self.EXPECTED_SLOPES[scenario.label])

    EXPECTED_ASPECTS = {
        "peak": np.array([[315, 0, 45], [270, math.nan, 90], [225, 180, 135]]),
        "zero_altitude": np.full(shape=(3, 3), fill_value=math.nan),
        "arcgis": np.array(
            [
                [90.00000000, 90.00000000, 90.00000000],
                [91.54815770, 92.64254529, 98.13010235],
                [91.46880071, 92.56377021, 98.13010235],
            ]
        ),
    }

    def test_aspects(self, scenario):
        """Tests calculation of aspects with sample 3x3 matrices."""
        try:
            result = scenario.obj.aspects
            expected_result = self.EXPECTED_ASPECTS[scenario.label]
            assert np.allclose(result, expected_result, equal_nan=True)
        except KeyError:
            raise KeyError(f"No expected result defined for {scenario.label}")

    TO_AZIMUTH_TEST_CASES = {
        "argnames": "aspect, expected_result",
        "argvalues": [
            (0, 90),
            (360, 90),
            (405, 45),
            (630, 180),
            (np.array([270, 225]), np.array([180, 225])),
        ],
    }

    @pytest.mark.parametrize(**TO_AZIMUTH_TEST_CASES)
    def test_to_azimuth(self, aspect, expected_result):
        """Tests the aspect to azimuth conversion with edge-cases."""
        assert np.allclose(self.test_class.to_azimuth(aspect), expected_result)

    TO_ZENITH_TEST_CASES = {
        "argnames": "altitude, expected_result",
        "argvalues": [
            (0, 90),
            (90, 0),
            (75, 15),
            pytest.param(
                180, None, marks=pytest.mark.xfail(raises=ValueError)
            ),
        ],
    }

    @pytest.mark.parametrize(**TO_ZENITH_TEST_CASES)
    def test_to_zenith(self, altitude, expected_result):
        """Tests the altitude to zenith conversion with edge-cases."""
        assert self.test_class.to_zenith(altitude) == expected_result

    def test_hillshade(self):
        """Tests hillshading algorithm with example from ArcGIS.

        This test isn't as important as it is just for visual
        representation of the terrain. Therefore, we are only testing it
        with a single case.
        """
        elevation = np.array(
            [[2450, 2461, 2483], [2452, 2461, 2483], [2447, 2455, 2477]]
        )
        terrain = self.test_class(elevation=elevation, cell_size=5)
        result = terrain.hillshade()  # Use-default value
        assert np.allclose(
            result,
            np.array(
                [
                    [228.65030288, 176.74478937, 190.68545721],
                    [177.25211486, 154.02865991, 154.34350541],
                    [156.26543819, 152.18251786, 154.34350541],
                ]
            ),
        )
