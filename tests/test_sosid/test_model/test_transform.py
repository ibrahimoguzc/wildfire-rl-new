# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/


import json

import numpy as np
import pytest
from pyproj import Geod

from examples.uam.paths import TERRAIN_DIR
from sosid.model.transform import (
    bearing_from_coords,
    bounding_box_coordinates,
    filter_out_of_bounds_pos,
    gps_to_pos,
    index_to_pos,
    pos_to_gps,
    pos_to_index,
)
from sosid.typedef import GridDescriptor

# TODO check negative position
POS_TO_INDEX_TEST_CASES = {
    "argnames": "pos, grid_description, expected_result",
    "argvalues": [
        # Checking if floor on a tuple position works
        (
            (5.5, 5.5),
            GridDescriptor(dimensions=(100, 100), shape=(10, 10)),
            (0, 0),
        ),
        (
            (55, 5),
            # Cell size of 10
            GridDescriptor(dimensions=(100, 50), shape=(5, 10)),
            (0, 5),
        ),
        (
            (55, 21),
            # Rectangular cell size, 20 in x-axis, 10 in y-axis
            GridDescriptor(dimensions=(100, 50), shape=(5, 5)),
            (2, 2),
        ),
        # Checking if tuple of np.ndarray works (output of nonzero(a))
        (
            np.array([[20, 10], [30, 20]]),
            GridDescriptor(dimensions=(100, 100), shape=(10, 10)),
            (np.array([1, 2]), np.array([2, 3])),
        ),
    ],
}


@pytest.mark.parametrize(**POS_TO_INDEX_TEST_CASES)
def test_pos_to_index(pos, grid_description, expected_result):
    """Testing if position to index transformation works."""
    index = pos_to_index(pos, grid_description)
    assert np.allclose(index, expected_result)


INDEX_TO_POS_TEST_CASES = {
    "argnames": "index, grid_description, expected_result",
    "argvalues": [
        # Testing a cell size less than 1
        (
            (200, 10),
            # Cell size of 0.05 in x, and 0.005 in y
            GridDescriptor(dimensions=(50, 10), shape=(2000, 1000)),
            (0.525, 1.0025),
        ),
        (
            (5, 5),
            # Cell size of 5 in x, and 0.5 in y
            GridDescriptor(dimensions=(50, 10), shape=(20, 10)),
            (27.5, 2.75),
        ),
        (
            (5, 10),
            GridDescriptor(dimensions=(50, 100), shape=(20, 10)),
            (52.5, 27.5),
        ),
        # Checking if tuple of np.ndarray works (output of nonzero(a))
        (
            (
                np.array((1, 2, 3, 4), dtype=np.uint64),
                np.array((3, 2, 1, 0), dtype=np.uint64),
            ),
            GridDescriptor(dimensions=(50, 50), shape=(10, 10)),
            np.array([[17.5, 7.5], [12.5, 12.5], [7.5, 17.5], [2.5, 22.5]]),
        ),
    ],
}


@pytest.mark.parametrize(**INDEX_TO_POS_TEST_CASES)
def test_index_to_pos(index, grid_description, expected_result):
    """Testing if index to position transformation works."""
    pos = index_to_pos(index, grid_description)
    assert np.allclose(pos, expected_result)


TEST_FILE = TERRAIN_DIR / "ZAL.meta"
with open(TEST_FILE) as file:
    data = json.load(file)
BOUNDS = np.array(data["bbox"])
((TOP, LEFT), (BOTTOM, RIGHT)) = data["extent"]
# Test location's geodetic and web mercator data retrieved online
TEST_GPS = (53.53653649510751, 9.86701159995881)
TEST_WEB_MERCATOR_POS = (1098390.7069587, 7082865.9317957)
TEST_POS = TEST_WEB_MERCATOR_POS[0] - LEFT, TOP - TEST_WEB_MERCATOR_POS[1]


@pytest.fixture(scope="function")
def expected_position(n_coords) -> np.ndarray:
    # Generate comparison point for method from different source
    return np.vstack([TEST_POS for _ in range(n_coords)])


@pytest.mark.parametrize(
    "testpoint, n_coords, top_left_bounds",
    [
        (TEST_GPS, 1, (TOP, LEFT)),
        (TEST_GPS, 2, (TOP, LEFT)),
        (TEST_GPS, 3, (TOP, LEFT)),
    ],
)
def test_gps_to_pos(testpoint, n_coords, top_left_bounds, expected_position):
    testpoint_pos = gps_to_pos(
        np.vstack([testpoint for _ in range(n_coords)]), top_left_bounds
    )
    rtol = 0.01
    atol = 0.5  # meters
    assert np.allclose(testpoint_pos, expected_position, atol=atol, rtol=rtol)


@pytest.fixture(scope="function")
def expected_coords(n_coords) -> np.ndarray:
    # Generate comparison coord for method from different source
    return np.vstack([TEST_GPS for _ in range(n_coords)])


@pytest.mark.parametrize(
    "testpoint, n_coords, top_left_bounds",
    [
        (TEST_POS, 1, (TOP, LEFT)),
        (TEST_POS, 2, (TOP, LEFT)),
        (TEST_POS, 3, (TOP, LEFT)),
    ],
)
def test_pos_to_gps(testpoint, n_coords, top_left_bounds, expected_coords):
    testpoint_gps = pos_to_gps(
        np.vstack([testpoint for _ in range(n_coords)]), top_left_bounds
    )
    rtol = 0.1
    assert np.allclose(testpoint_gps, expected_coords, rtol=rtol)


FILTER_OUT_OF_BOUNDS_POS_TEST_CASES = {
    "argnames": "pos, max_dimensions, expected_result",
    "argvalues": [
        # Testing with pos in bounds
        (np.array([[20, 10], [30, 20]]), (100, 100), ((20, 10), (30, 20))),
        # Testing with X coordinate out of bounds
        (np.array([[200, 10], [30, 20]]), (100, 100), ((30, 20),)),
        # Testing with Y coordinate out of bounds
        (np.array([[20, 101], [30, 20]]), (100, 100), ((30, 20),)),
        # Testing with coordinates on the boundary
        (np.array([[100, 100], [30, 20]]), (100, 100), ((100, 100), (30, 20))),
    ],
}


@pytest.mark.parametrize(**FILTER_OUT_OF_BOUNDS_POS_TEST_CASES)
def test_filter_out_of_bounds_pos(pos, max_dimensions, expected_result):
    """Testing if filtering of out of bounds positions works."""
    filtered_pos = filter_out_of_bounds_pos(pos, max_dimensions)
    assert filtered_pos == expected_result


COORD = (-0.5274001238560642, 73.14671761439577)
PERTURBANCE = 0.1
TOLERANCE = 1e-3
BEARING_FROM_COORDS_TEST_CASES = {
    "argnames": "origin, destination, expected_result",
    "argvalues": [
        # Testing with Eastward bearing
        ((COORD[0], COORD[1]), (COORD[0], COORD[1] + PERTURBANCE), 90),
        # Testing with Southward bearing
        ((COORD[0] + PERTURBANCE, COORD[1]), (COORD[0], COORD[1]), 180),
        # Testing with Westward bearing
        ((COORD[0], COORD[1] + PERTURBANCE), (COORD[0], COORD[1]), -90),
        # Testing with Northward bearing
        ((COORD[0], COORD[1]), (COORD[0] + PERTURBANCE, COORD[1]), 0),
    ],
}


@pytest.mark.parametrize(**BEARING_FROM_COORDS_TEST_CASES)
def test_bearing_from_coords(origin, destination, expected_result):
    bearing = bearing_from_coords(origin, destination)
    assert abs(bearing - expected_result) < TOLERANCE


GEODESIC = Geod(ellps="WGS84")
BOUNDING_BOX_COORDINATES_TEST_CASES = {
    "argnames": "ignition_center_coords, map_height_cells, \
        map_width_cells, resolution_meters",
    "argvalues": [
        ([42.8231, -0.1721], 15000, 20000, 2),
        ([38.7692, -9.4413], 6000, 5000, 2),
        ([42.8231, -0.1721], 5000, 5000, 2),
        ([42.8231, -0.1721], 1500, 2000, 3),
        ([42.8231, -0.1721], 5000, 5000, 4),
        ([44.434832, 26.097688], 5000, 5000, 2),
        ([48.085024, 11.286168], 13000, 11000, 2),
        ([39.052478, -76.764714], 3000, 7000, 3),
    ],
}
TOLERANCE_BBOX = 6e-3


def expected_height_width_distance(center_coords, bbox_coords) -> np.ndarray:
    """Calculate the expected height, width and distance of the BBOX."""
    center_coords = np.array(center_coords)
    # dist =    Distance(top-left corner, center)                   [m]
    # height =  Distance(top-left corner, bottom-left corner)       [m]
    # width =   Distance(bottom-left corner, bottom-right corner)   [m]
    _, _, dist_height_width = GEODESIC.inv(
        lons1=[bbox_coords[0][1], bbox_coords[0][1], bbox_coords[0][1]],
        lats1=[bbox_coords[0][0], bbox_coords[0][0], bbox_coords[1][0]],
        lons2=[center_coords[1], bbox_coords[0][1], bbox_coords[1][1]],
        lats2=[center_coords[0], bbox_coords[1][0], bbox_coords[1][0]],
    )
    dist_height_width = np.array(dist_height_width)

    return dist_height_width


@pytest.mark.parametrize(**BOUNDING_BOX_COORDINATES_TEST_CASES)
def test_bounding_box_coordinates(
    ignition_center_coords,
    map_height_cells,
    map_width_cells,
    resolution_meters,
):
    """Testing the height, width and distance to the center of BBOX."""
    bbox_coords = bounding_box_coordinates(
        ignition_center_coords,
        map_height_cells,
        map_width_cells,
        resolution_meters,
    )

    dist_height_width = expected_height_width_distance(
        ignition_center_coords, bbox_coords
    )
    map_width_meters = map_width_cells * resolution_meters
    map_height_meters = map_height_cells * resolution_meters
    hypothenuse = ((map_width_meters**2 + map_height_meters**2) ** 0.5) / 2

    assert (
        abs(dist_height_width[0] - hypothenuse) < TOLERANCE_BBOX * hypothenuse
    ), "Distance to the center not within limits."

    assert (
        abs(dist_height_width[1] - map_height_meters)
        < TOLERANCE_BBOX * map_height_meters
    ), "Map height not within limits."

    assert (
        abs(dist_height_width[2] - map_width_meters)
        < TOLERANCE_BBOX * map_width_meters
    ), "Map width not within limits."
