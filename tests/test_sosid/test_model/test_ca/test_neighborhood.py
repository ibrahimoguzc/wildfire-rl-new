# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import copy

import pytest

from sosid.model.ca.neighborhood import (
    MooreNeighborhood,
    Neighborhood,
    NeumannNeighborhood,
)
from tests.snippets import ScenarioTestSuite

NEIGHBORHOOD_SCENARIOS = {
    "argnames": "label, args, kwargs",
    "argvalues": [
        ("r1_nocenter", (1,), {"include_center": False}),
        ("r1_center", (1,), {"include_center": True}),
        ("r2_nocenter", (2,), {"include_center": False}),
        ("r2_center", (2,), {"include_center": True}),
    ],
}


class TestNeighborhood(ScenarioTestSuite):
    test_class = Neighborhood
    test_obj = Neighborhood(radius=1, include_center=False)
    scenarios = NEIGHBORHOOD_SCENARIOS

    EXPECTED_LIMITS = {
        "r1_nocenter": ((-1, 1), (-1, 1)),
        "r1_center": ((-1, 1), (-1, 1)),
        "r2_nocenter": ((-2, 2), (-2, 2)),
        "r2_center": ((-2, 2), (-2, 2)),
    }

    def test_limits(self, scenario):
        """Ensuring limits works for all scenarios."""
        assert scenario.obj.limits == self.EXPECTED_LIMITS[scenario.label]

    @pytest.mark.parametrize("__dim__, expected", [(1, 1), (2, 2), (3, 3)])
    def test_center(self, __dim__, expected):
        """Tests if the __dim__ attribute affects the center."""
        obj = copy.copy(self.test_obj)  # copying avoids changing class attrs
        obj.__dim__ = __dim__

        assert len(obj.center) == __dim__
        assert all(coordinate == 0 for coordinate in obj.center)

    def test_in_neighborhood(self):
        """Ensures that in_neighborhood is not defined in base class."""
        with pytest.raises(NotImplementedError):
            self.test_obj.in_neighborhood((0, 0))

    def test_cell_count(self):
        """Ensures cell_count is not defined in base class."""
        with pytest.raises(NotImplementedError):
            self.test_obj.cell_count


class NeighborhoodTester(ScenarioTestSuite):
    scenarios = NEIGHBORHOOD_SCENARIOS

    def test__dim__(self):
        """Ensuring Neighborhood is 2 Dimensional."""
        assert self.test_obj.__dim__ == 2

    def test_cell_count(self, scenario):
        """Tests if neighborhood contains the right number of cells."""
        expected_result = self.EXPECTED_CELL_COUNT[scenario.label]
        assert scenario.obj.cell_count == expected_result

    def test_as_tuple(self, scenario):
        """Tests if the offset tuple has correct length and content."""
        result = scenario.obj.as_tuple()
        assert len(result) == self.EXPECTED_CELL_COUNT[scenario.label]
        if scenario.obj.include_center:
            assert (0, 0) in result
        else:
            assert (0, 0) not in result


class TestMooreNeighborhood(NeighborhoodTester):
    test_class = MooreNeighborhood
    test_obj = MooreNeighborhood(radius=1, include_center=False)
    scenarios = NEIGHBORHOOD_SCENARIOS

    EXPECTED_CELL_COUNT = {
        "r1_nocenter": 8,
        "r1_center": 9,
        "r2_nocenter": 24,
        "r2_center": 25,
    }


class TestNeumannNeighborhood(NeighborhoodTester):
    test_class = NeumannNeighborhood
    test_obj = NeumannNeighborhood(radius=1, include_center=False)
    scenarios = NEIGHBORHOOD_SCENARIOS

    EXPECTED_CELL_COUNT = {
        "r1_nocenter": 4,
        "r1_center": 5,
        "r2_nocenter": 12,
        "r2_center": 13,
    }
