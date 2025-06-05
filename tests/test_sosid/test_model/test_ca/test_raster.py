# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import numpy as np
import pytest

from sosid.model.ca.raster import (
    Ellipse,
    LineSegment,
    RasterizedShape,
    Rectangle,
)
from tests.snippets import ScenarioTestSuite


def test_abc():
    """Ensures RasterizedShape ABC requires painter to be overridden."""

    # Painter property has not been overridden (Should fail)
    class BadShape(RasterizedShape):
        pass

    # Painter property is correctly overridden here
    class GoodShape(RasterizedShape):
        def __init__(self):
            super().__init__(idx=(0, 0), size=(1, 1), aspect=90)

        @property
        def painter(self):
            from PIL import ImageDraw

            return ImageDraw.Draw(self.canvas).ellipse

    with pytest.raises(TypeError):
        BadShape()

    ellipse = GoodShape()
    # Making sure the defined painter functioned on initialization
    assert ellipse.shape.size == (1, 1)


class RasterizedShapeTester(ScenarioTestSuite):
    def test_rotated(self, scenario):
        """Tests if the rotated shape matches the expected value."""
        expected_result = self.EXPECTED_ROTATED[scenario.label]
        result = scenario.obj.rotated
        assert np.allclose(result, expected_result)

    def test_shape(self, scenario):
        """Tests if shape returns the rotated version or not."""
        expected_result = getattr(
            scenario.obj, self.EXPECTED_SHAPE[scenario.label]
        )
        assert np.allclose(scenario.obj.shape, expected_result)

    def test_positioned_bbox(self, scenario):
        """Tests if the bounding-box has been correctly offset."""
        expected_result = self.EXPECTED_POSITIONED_BBOX[scenario.label]
        assert scenario.obj.positioned_bbox == expected_result

    def test_boolean_mask(self, scenario):
        """Tests if 5x5 boolean mask returns expected shapes."""
        expected_result = self.EXPECTED_BOOLEAN_MASK[scenario.label]
        result = scenario.obj.boolean_mask(expected_result.shape)
        assert np.allclose(result, expected_result)

    def test_nonzero(self, scenario):
        """Tests if 5x5 boolean mask returns expected shapes."""
        expected_result = self.EXPECTED_BOOLEAN_MASK[scenario.label]
        result = np.zeros((5, 5))
        indices = scenario.obj.nonzero(expected_result.shape)
        result[indices] = 1
        assert np.allclose(result, expected_result)

    def test_bounds_error(self):
        """Tests if out of bounds check functions properly."""
        obj = self.test_class((5, 5), 1, 1)
        with pytest.raises(ValueError):
            obj.boolean_mask((5, 5))

        with pytest.raises(ValueError):
            obj.nonzero((5, 5))

    def test_assymetric(self):
        """Tests if an assymetric shape is always applied at ``pos``.

        This makes sure that at least 1 cell is masked at ``pos`` even
        with an assymetric shape.
        """
        obj = self.test_class((2, 2), 2, 2)
        for aspect in range(0, 360, 5):
            obj.aspect = aspect
            mask = obj.boolean_mask((5, 5))
            assert mask[2, 2] == 1


ELLIPSE_SCENARIOS = {
    "argnames": "label, args",
    "argvalues": [
        ("1x1_aspect=0", ((2, 2), 1, 1, 0)),
        ("3x1_aspect=45", ((2, 2), 3, 1, 45)),
        ("3x1_aspect=90", ((2, 2), 3, 1, 90)),
        ("3x1_aspect=180", ((2, 2), 3, 1, 180)),
    ],
}


class TestEllipse(RasterizedShapeTester):
    scenarios = ELLIPSE_SCENARIOS
    test_class = Ellipse

    EXPECTED_ROTATED = {
        "1x1_aspect=0": np.array([1]),
        "3x1_aspect=45": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        "3x1_aspect=90": np.array([1, 1, 1]),
        "3x1_aspect=180": np.array([1, 1, 1]).T,
    }

    EXPECTED_SHAPE = {
        "1x1_aspect=0": "rotated",
        "3x1_aspect=45": "rotated",
        "3x1_aspect=90": "canvas",
        "3x1_aspect=180": "rotated",
    }

    EXPECTED_POSITIONED_BBOX = {
        "1x1_aspect=0": (2, 2, 3, 3),
        "3x1_aspect=90": (1, 2, 4, 3),
        "3x1_aspect=45": (1, 1, 4, 4),
        "3x1_aspect=180": (2, 1, 3, 4),
    }

    EXPECTED_BOOLEAN_MASK = {
        "1x1_aspect=0": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "3x1_aspect=45": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "3x1_aspect=90": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "3x1_aspect=180": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    }


RECTANGLE_SCENARIOS = {
    "argnames": "label, args",
    "argvalues": [
        # 1x1 is already covered by the Ellipse case above!
        ("2x2_aspect=90", ((2, 2), 2, 2, 90)),
        ("3x3_aspect=45", ((2, 2), 3, 3, 45)),
        # Testing an assymetric shape at cell (x=1, y=3)
        ("2x1_aspect=180", ((3, 1), 2, 1, 180)),
    ],
}


class TestRectangle(RasterizedShapeTester):
    scenarios = RECTANGLE_SCENARIOS
    test_class = Rectangle

    EXPECTED_ROTATED = {
        "2x2_aspect=90": np.array([[1, 1], [1, 1]]),
        "3x3_aspect=45": np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        "2x1_aspect=180": np.array([1, 1]).T,
    }

    EXPECTED_SHAPE = {
        "2x2_aspect=90": "canvas",
        "3x3_aspect=45": "rotated",
        "2x1_aspect=180": "rotated",
    }

    EXPECTED_POSITIONED_BBOX = {
        "2x2_aspect=90": (1, 1, 3, 3),
        "3x3_aspect=45": (0, 0, 5, 5),
        "2x1_aspect=180": (1, 2, 2, 4),
    }

    EXPECTED_BOOLEAN_MASK = {
        "2x2_aspect=90": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "3x3_aspect=45": np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        "2x1_aspect=180": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    }


LINESEGMENT_SCENARIOS = {
    "argnames": "label, args",
    "argvalues": [
        # 1x1 is already covered by the Ellipse case above!
        ("5x2_aspect=90", ((2, 2), 5, 2, 90)),
        ("4x1_aspect=135", ((1, 1), 4, 1, 135)),
    ],
}


class TestLineSegment(RasterizedShapeTester):
    scenarios = LINESEGMENT_SCENARIOS
    test_class = LineSegment

    EXPECTED_ROTATED = {
        "5x2_aspect=90": np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        "4x1_aspect=135": np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
            ]
        ),
    }

    EXPECTED_SHAPE = {"5x2_aspect=90": "canvas", "4x1_aspect=135": "rotated"}

    EXPECTED_POSITIONED_BBOX = {
        "5x2_aspect=90": (0, 1, 5, 3),
        "4x1_aspect=135": (-1, -1, 3, 4),
    }

    EXPECTED_BOOLEAN_MASK = {
        "5x2_aspect=90": np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        "4x1_aspect=135": np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    }
