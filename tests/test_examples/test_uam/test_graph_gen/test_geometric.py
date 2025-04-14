import numpy as np
import pytest
import shapely as shp

from examples.uam.graph_gen.geometric import visible_boundary_from_point


@pytest.mark.parametrize(
    "point, expected_line",
    [
        ((0.5, 1.5), [(1, 1), (0, 1)]),
        ((1.5, 1.5), [(1, 0), (1, 1), (0, 1)]),
        ((1.5, 0.5), [(1, 0), (1, 1)]),
        ((1.5, -0.5), [(0, 0), (1, 0), (1, 1)]),
        ((0.5, -0.5), [(0, 0), (1, 0)]),
        ((-0.5, -0.5), [(0, 1), (0, 0), (1, 0)]),
        ((-0.5, 0.5), [(0, 1), (0, 0)]),
        ((-0.5, 1.5), [(1, 1), (0, 1), (0, 0)]),
    ],
)
def test_visible_boundary_from_point_box(point, expected_line):
    line = visible_boundary_from_point(
        shp.box(0, 0, 1, 1),
        shp.Point(point),
    ).coords[:]
    assert expected_line in (line, line[::-1])


@pytest.mark.parametrize("reverse", [True, False])
def test_visible_boundary_from_point_fully_visible(reverse):
    """Test when the all points on the boundary are visible."""
    circle = shp.Point(0, 0).buffer(10)
    point = shp.Point(10, 100)
    line = visible_boundary_from_point(circle, point)
    assert isinstance(line, shp.LineString)
    assert line.length > 0
    if reverse:
        visible_poly = shp.Polygon(line.coords[::-1])
    else:
        visible_poly = shp.Polygon(line.coords)
    visible_boundary = visible_boundary_from_point(visible_poly, point)
    boundary_coords = shp.get_coordinates(visible_boundary)
    expected_coords = shp.get_coordinates(line)
    if reverse:
        expected_coords = expected_coords[::-1]
    np.testing.assert_allclose(boundary_coords, expected_coords)
