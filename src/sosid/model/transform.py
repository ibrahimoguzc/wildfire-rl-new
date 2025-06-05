# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains mappings between coordinate systems of various models."""

import math
from functools import lru_cache

import numpy as np
from haversine import inverse_haversine, inverse_haversine_vector
from scipy.interpolate import interp1d

from sosid.typedef import GridDescriptor, Index, LatLon, Position, Tuple
from sosid.util.imports import PostponedImportError

try:
    from pyproj import CRS, Geod, Proj, Transformer
except ImportError:
    CRS = PostponedImportError("pyproj")
    Geod = PostponedImportError("pyproj")
    Proj = PostponedImportError("pyproj")
    Transformer = PostponedImportError("pyproj")


# TODO add test for ndim=1 vector as ndarray
def pos_to_index(
    pos: Position,
    grid_description: GridDescriptor,
    origin: Tuple[float, float] = (0, 0),
) -> Index:
    # TODO update the below text
    """GCS positions (x, y) to CCS (i, j) indices.

    There are two use intended use-cases for this function. First, one
    can simply convert a 2D tuple position into an index::

        >>> pos = (20, 10)
        >>> pos_to_index(pos, cell_size=10)
        (1, 2)

    Or it can be used on a position matrix as used by the Mesa library
    which is a vertically stacked group of position row-vectors::

        >>> pos_matrix = np.array([[20, 10], [30, 20]])
        >>> pos_to_index(pos_matrix, cell_size=10)
        (
            np.array([1, 2], dtype=np.uint64),
            np.array([2, 3], dtype=np.uint64)
        )

    The output of the above can then be used to look up the values of
    cells located at (x=20, y=10) and (x=30, y=20) in one operation::

        cell_data[pos_to_index(pos_matrix, cell_size=10)]

    Note: indices are assumed to start from origin with origin at
    top-left corner

    """
    cell_size_x = (gd := grid_description).dimensions[0] / gd.shape[1]
    cell_size_y = gd.dimensions[1] / gd.shape[0]
    cell_size = (cell_size_x, cell_size_y)
    # Checking if position is np.ndarray of row-vectors (Mesa)
    if isinstance(pos, np.ndarray) and pos.ndim == 2:
        # Get position relative to origin to convert to indices
        pos = pos - np.array(origin)
        # Casting to integer floors the
        indices = np.fliplr(pos / cell_size).astype(np.uint64)

        # Returning tuple of row-vectors corresponding to i, j indices
        return (indices[:, 0], indices[:, 1])

    # Tuple[int, int]
    j, i = (pos[0] - origin[0], pos[1] - origin[1])
    return (int(i / cell_size[1]), int(j / cell_size[0]))


def index_to_pos(
    index: Index,
    grid_description: GridDescriptor,
    origin: Tuple[float, float] = (0, 0),
) -> Position:
    """CCS indices (i, j) to GCS positions (x, y).

    Note:
        The origin of a single cell is at its upper-left hand corner.
        Therefore, half of its ``cell_size`` must be added to return
        a position in the GCS that corresponds to the cell-center.
        Therefore, for a cell_size of 1.0, an index of (0, 0) would
        correspond to a GCS position of (0.5, 0.5). This definition
        is also used by the :std:doc:`PIL` package.

    Note: indices are assumed to start from origin

    """
    y, x = index
    # TODO replace with walrus operator when migrating to Python 3.8
    is_array = isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    if is_array:  # Converting dtype to float to represent decimal value
        x, y = x.astype(np.float64), y.astype(np.float64)

    cell_size_x = (gd := grid_description).dimensions[0] / gd.shape[1]
    cell_size_y = gd.dimensions[1] / gd.shape[0]

    # Scaling index by cell_size
    x *= cell_size_x
    y *= cell_size_y

    # Translating position from top-left cell-corner to cell-center
    x += cell_size_x / 2
    y += cell_size_y / 2

    # Translating position from origin-relative position to global
    # position
    x += origin[0]
    y += origin[1]

    return np.vstack((x, y)).T if is_array else (x, y)


# EPSG: 4326 uses a coordinate system on the surface of a sphere or
# ellipsoid of reference.
# EPSG: 3857 uses a coordinate system PROJECTED from the surface of the
# sphere or ellipsoid to a flat surface.
CRS_4326 = CRS("EPSG:4326")
CRS_3857 = CRS("EPSG:3857")
CRS_4326_TO_3857 = Transformer.from_crs(CRS_4326, CRS_3857)
CRS_3857_TO_4326 = Transformer.from_crs(CRS_3857, CRS_4326)
# define global mercator top left pos [top = 90 deg North--> positive,
# left = -180 deg West --> negative]
GLOBAL_TOP_LEFT_POS = np.array(CRS_4326_TO_3857.transform(90, -180)).T

# WGS84 == EPSG:4326
GEODESIC = Geod(ellps="WGS84")

# Define coordinate systems for GPS to ECEF and ECEF to GPS conversions
# ECEF is a cartesian coordinate system with origin at the center of the
# earth. LLA is a spherical coordinate system with origin at the center
# of the earth.
ECEF = Proj(proj="geocent", ellps="WGS84", datum="WGS84")
LLA = Proj(proj="latlong", ellps="WGS84", datum="WGS84")
LLA_TO_ECEF = Transformer.from_proj(LLA, ECEF)
ECEF_TO_LLA = Transformer.from_proj(ECEF, LLA)


def gps_to_mercator(coords: LatLon):
    """Converts gps coordinates into mercator projection (x,y).

    Conversion is based on a defined on having coordinates [0,0] be the
    same on x,y frame [0,0]. Positive GPS values translate to positive
    x,y values (North -> +, South -> -).
    """
    coords = np.array(coords)
    pos_array = np.array(
        CRS_4326_TO_3857.transform(coords[..., 0], coords[..., 1])
    ).T
    return pos_array


def mercator_to_gps(extent: Tuple[float, ...]):
    """Converts the Mercator coordinates to geographic coordinates.

    Parameters:
    extent (tuple): A tuple of four values (left, right, bottom, top ).

    Returns: list: A list of tuples, each containing latitude and
    longitude of the top-left and bottom-right corners.
    """
    # Update bbox of actually retrieved tile
    left, right, bottom, top = extent
    north, west = CRS_3857_TO_4326.transform(left, top)
    south, east = CRS_3857_TO_4326.transform(right, bottom)

    return [[north, west], [south, east]]


@lru_cache(maxsize=100)
def gps_to_pos_calc(
    coords: LatLon, top_left_bounds: Tuple[float, float]
) -> Position:
    """Global Coordinate System (lat, lon) to Position (x,y).

    Considers edge case scenario where bounding box spans across +/- 180
    degrees longitude and +/- 90 degreeslatitude
    """
    top, left = top_left_bounds

    # Transform latitude longitude to (x, y) coordinates
    coord_array = np.array(coords, dtype=np.float64)
    pos_array = np.array(
        CRS_4326_TO_3857.transform(coord_array[..., 0], coord_array[..., 1])
    ).T

    # Transform position to Global Coordinate System in-place
    if pos_array.ndim == 1:
        iter_array = np.array([pos_array])
    else:
        iter_array = pos_array
    for pos in iter_array:
        if left > 0 and pos[0] < 0:
            # this means the position is in West hemisphere while left
            # bound is in East (map spans across pacific)
            # eg map_top_left = [179.5 , 0.0] --> East
            # point = [-179.5 , 0.0] --> West
            diff_left = np.abs(GLOBAL_TOP_LEFT_POS[0]) - left
            diff_x = pos[0] - GLOBAL_TOP_LEFT_POS[0]
            x_from_left = diff_left + diff_x
            pos[0] = x_from_left
        else:
            pos[0] = pos[0] - left
        pos[1] = top - pos[1]

    if pos_array.ndim == 1:
        pos_array = iter_array[0]

    return pos_array


def gps_to_pos(
    coords: LatLon, top_left_bounds: Tuple[float, float]
) -> Position:
    """Processes gps to output cached `gps_to_pos_calc`."""
    coords = tuple(coords)
    if any(isinstance(item, np.ndarray) for item in coords):
        coords = tuple(map(tuple, coords))
    return gps_to_pos_calc(coords, tuple(top_left_bounds))


@lru_cache(maxsize=100)
def pos_to_gps_calc(
    pos: Position, top_left_bounds: Tuple[float, float]
) -> LatLon:
    """Position (x,y)  to Global Coordinate System (lat, lon).

    Considers edge case scenario where bounding box spans across +/- 180
    degrees longitude and +/- 90 degreeslatitude
    """
    pos_array = pos_to_global_xy(pos, top_left_bounds)

    # Transform Global Position (x,y) into Latitude Longitude
    return np.array(
        CRS_3857_TO_4326.transform(pos_array[..., 0], pos_array[..., 1])
    ).T


def pos_to_gps(
    pos: Position | Tuple[float, float],
    top_left_bounds: Tuple[float, float],
) -> LatLon:
    """Processes pos to output cached `pos_to_gps_calc`."""
    pos = tuple(pos)
    if any(isinstance(item, np.ndarray) for item in pos):
        pos = tuple(map(tuple, pos))
    return pos_to_gps_calc(pos, tuple(top_left_bounds))


def pos_to_global_xy(
    pos: Position, top_left_bounds: tuple[float, float]
) -> Position:
    """Converts position to global xy coordinate system (EPSG:3857)."""
    top, left = top_left_bounds
    pos_array = np.array(pos, dtype=np.float64)
    pos_array[..., 0] = left + pos_array[..., 0]
    pos_array[..., 1] = top - pos_array[..., 1]
    pos_past_boundary = pos_array[
        pos_array[..., 0] > np.abs(GLOBAL_TOP_LEFT_POS[0])
    ]
    pos_past_boundary[..., 0] = (
        GLOBAL_TOP_LEFT_POS[0] + pos_past_boundary[..., 0]
    )
    return pos_array


def filter_out_of_bounds_pos(
    pos: Position, max_dimensions: Position
) -> Tuple[Position, ...]:
    """Filters out elements of ``pos`` that are out of bounds."""
    return tuple(
        (x, y)
        for x, y in pos
        if (x <= max_dimensions[0]) and (y <= max_dimensions[1])
    )


def bearing_from_coords(origin: LatLon, destination: LatLon) -> float:
    """Compute bearing (azimuth) from two gps coordinates."""
    origin_lat, origin_long = origin
    dest_lat, dest_long = destination
    forward_azimuth, _, _ = GEODESIC.inv(
        origin_long,
        origin_lat,
        dest_long,
        dest_lat,
        radians=False,
        return_back_azimuth=False,
    )
    return forward_azimuth


def bounding_box_coordinates(
    coords: LatLon, map_height_cells, map_width_cells, resolution_meters
) -> LatLon:
    """Compute the coordinates of the bounding box for GPS map.

    Outputs BBOX coordinates: [top-left, bottom right].
    """
    coords_array = np.array(coords)
    if coords_array.ndim != 1:
        raise IndexError("Input coordinates must be of size 1.")

    actual_height_km = map_height_cells * resolution_meters / 1000.0
    actual_width_km = map_width_cells * resolution_meters / 1000.0
    hypothenuse = (
        (actual_height_km / 2) ** 2 + (actual_width_km / 2) ** 2
    ) ** 0.5

    # Location of the corners (°) if width == height:
    # top-left = 315, bottom-right=135, bottom-left = 225, top-right=45.
    location_left = math.radians(
        270 + math.degrees(math.acos(actual_width_km / (hypothenuse * 2)))
    )
    location_right = math.radians(
        90 + math.degrees(math.acos(actual_width_km / (hypothenuse * 2)))
    )

    bbox_coords_ihaversine = inverse_haversine_vector(
        [coords_array] * 2, [hypothenuse] * 2, [location_left, location_right]
    )
    latitude_coords = bbox_coords_ihaversine[0]
    longitude_coords = bbox_coords_ihaversine[1]
    bbox_coords = np.array(
        [
            val
            for pair in zip(latitude_coords, longitude_coords)
            for val in pair
        ]
    )
    bbox_coords = bbox_coords.reshape(-1, 2)

    return bbox_coords


# TODO Account for non-concentric grid maps by using relative position.
def pos_small_to_large_grid(
    pos: Position,
    small_grid_dimensions: Tuple[float, float],
    large_grid_dimensions: Tuple[float, float],
) -> Tuple[float, float]:
    """Converts positions on smaller grids to larger grid frames.

    Args:
        pos (ndarray): The position (x, y) in the smaller map coordinate
        system.
        small_grid_dimensions (tuple): Dimensions of the smaller map
        (width, height).
        large_grid_dimensions (tuple): Dimensions of the
        larger map (width, height).

    Returns:
        tuple: The transformed position (x', y') in the larger map
        coordinate system.
    """
    pos_x, pos_y = pos
    small_grid_width, small_grid_height = small_grid_dimensions
    large_grid_width, large_grid_height = large_grid_dimensions

    small_grid_center_x = int(small_grid_width // 2)
    small_grid_center_y = int(small_grid_height // 2)
    large_grid_center_x = int(large_grid_width // 2)
    large_grid_center_y = int(large_grid_height // 2)

    small_grid_origin_x = large_grid_center_x - small_grid_center_x
    small_grid_origin_y = large_grid_center_y - small_grid_center_y

    operational_pos_x = small_grid_origin_x + pos_x
    operational_pos_y = small_grid_origin_y + pos_y

    return operational_pos_x, operational_pos_y


def compute_larger_grid_coordinates(
    smaller_grid_coords: list[LatLon],
    larger_grid_shape: Tuple[int, int],
    larger_grid_resolution: int,
) -> list:
    """Defines larger bbox coordinates from embedded bbox.

    Smaller grid is within larger grid and gps coordinates of smaller
    grid top left and bottom right are inputted to define larger grid
    coordinates based on distance between grid bounds.
    """
    larger_grid_y_km = larger_grid_shape[0] * larger_grid_resolution / 1000
    larger_grid_x_km = larger_grid_shape[1] * larger_grid_resolution / 1000

    smaller_grid_coords = np.array(smaller_grid_coords)
    center_gps = (
        np.mean(smaller_grid_coords[:, 0]),
        np.mean(smaller_grid_coords[:, 1]),
    )

    left_mid = inverse_haversine(
        center_gps, larger_grid_x_km / 2, 1.5 * math.pi
    )
    right_mid = inverse_haversine(
        center_gps, larger_grid_x_km / 2, 0.5 * math.pi
    )
    top_mid = inverse_haversine(center_gps, larger_grid_y_km / 2, 0)
    bottom_mid = inverse_haversine(center_gps, larger_grid_y_km / 2, math.pi)

    top_left = top_mid[0], left_mid[1]
    bottom_right = bottom_mid[0], right_mid[1]

    larger_grid_coords = [top_left, bottom_right]
    return larger_grid_coords


def scale_from_point(
    point: Index, point_ref: Index = (0, 0), scale_factor: float = 1.0
) -> Position:
    """Scales a point with respect to point_ref by a scale factor."""
    pX, pY = point

    is_array = isinstance(pX, np.ndarray) and isinstance(pY, np.ndarray)

    cX, cY = point_ref

    x = (pX - cX) * scale_factor + cX
    y = (pY - cY) * scale_factor + cY

    return np.vstack((x, y)).T if is_array else (x, y)


def interpolate_gps(
    x: np.ndarray[np.float64],
    x_ref: np.ndarray[np.float64],
    gps_ref: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    """Linearly interpolate GPS coordinates based on ECEF.

    The function interpolates GPS coordinates based on the reference
    GPS coordinates. It first transforms the GPS coordinates to ECEF
    coordinates (xyz) and then interpolates the ECEF coordinates. The
    interpolated ECEF coordinates are then transformed back to GPS
    coordinates and returned.

    Args:
        x: The x values to interpolate.
        x_ref: The reference x values.
        gps_ref: The reference GPS coordinates [(lat, lon), ...].

    Returns:
        The interpolated GPS coordinates.
    """
    xyz = LLA_TO_ECEF.transform(
        gps_ref[:, 1], gps_ref[:, 0], np.zeros(len(gps_ref))
    )
    return np.array(ECEF_TO_LLA.transform(*interp1d(x_ref, xyz)(x))[:2])[
        ::-1
    ].T
