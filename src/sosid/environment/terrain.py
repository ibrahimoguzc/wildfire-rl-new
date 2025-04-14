# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains functions for computing terrain attributes.

These functions have been implemented following a tutorial authored
by `ArcGIS`_.
In the future when more advanced features are required to load and
modify Digital Elevation Models (DEM), it is highly recommended to
check out `RichDEM`_.
.. _ArcGIS:
http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox
/how-hillshade-works.htm # noqa: E501, W505
.. _RichDEM: https://richdem.readthedocs.io/en/latest/index.html.
TODO: Re-assign flammabilities based on some future research.
"""

import json
import math
from datetime import datetime
from enum import IntEnum
from functools import cached_property
from pathlib import Path, WindowsPath

import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtGui
from scipy.ndimage import gaussian_filter

from sosid.abstract import Viewable
from sosid.gui.items import ImageItem
from sosid.jit_config import BASE_JIT_KWARGS
from sosid.model.ca.jit_funcs import gradient_2d
from sosid.model.transform import CRS_3857_TO_4326, gps_to_mercator, gps_to_pos
from sosid.typedef import GridDescriptor, Position
from sosid.util.abc import abstractattribute
from sosid.util.imports import PostponedImportError

try:
    import contextily as ctx
except ImportError:
    ctx = PostponedImportError("contextily")

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_cmap",
    [
        (0.0, (0.15, 0.3, 0.15)),
        (0.25, (0.3, 0.45, 0.3)),
        (0.5, (0.5, 0.5, 0.35)),
        (0.8, (0.4, 0.36, 0.33)),
        (1.0, (1.0, 1.0, 1.0)),
    ],
)


class TerrainTypes(IntEnum):
    """Matches terrain types to their index.

    Used in (Imported)Features class to map to color indices and
    combustibility values.
    """

    # Values in this Enum class should be consecutive and start from 0
    # Values sorted in ascending combustibility with residential at
    # highest to prioritize assignment

    NON_COMBUSTIBLE: int = 0
    WATER: int = 1
    FIELD: int = 2
    NEEDLE_LITTER: int = 3
    PINUS: int = 4
    FALLEN_LEAVES: int = 5
    GRASSES_WEEDS: int = 6
    CAREX_FORBS: int = 7
    PASTURE: int = 8
    RESIDENTIAL: int = 9


# Each feature in the color table must have a unique color index. This
# is due to the color table being used for future feature distinction
# and filtering
FEATURES_COLOR_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: (242, 239, 233, 255),
    TerrainTypes.WATER: (66, 161, 244, 255),
    TerrainTypes.NEEDLE_LITTER: (205, 235, 176, 255),
    TerrainTypes.FALLEN_LEAVES: (51, 102, 0, 255),
    TerrainTypes.GRASSES_WEEDS: (0, 102, 34, 255),
    TerrainTypes.CAREX_FORBS: (163, 186, 15, 255),
    TerrainTypes.PASTURE: (191, 255, 179, 255),
    TerrainTypes.PINUS: (85, 128, 0, 255),
    TerrainTypes.FIELD: (230, 249, 230, 255),
    TerrainTypes.RESIDENTIAL: (232, 100, 232, 255),
}

# Duplicate combustibility values are undesirable as it will affect burnt
# area output tags
COMBUSTIBILITY_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: 0,
    TerrainTypes.WATER: 0,
    TerrainTypes.NEEDLE_LITTER: 0.8,
    TerrainTypes.FALLEN_LEAVES: 1.2,
    TerrainTypes.GRASSES_WEEDS: 1.6,
    TerrainTypes.CAREX_FORBS: 1.8,
    TerrainTypes.PASTURE: 2,
    TerrainTypes.PINUS: 1,
    TerrainTypes.FIELD: 0.8,
    TerrainTypes.RESIDENTIAL: 0.2,
}
PRIORITY_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: 0.0,
    TerrainTypes.WATER: 0.0,
    TerrainTypes.NEEDLE_LITTER: 0.0,
    TerrainTypes.FALLEN_LEAVES: 0.0,
    TerrainTypes.GRASSES_WEEDS: 0.0,
    TerrainTypes.CAREX_FORBS: 0.0,
    TerrainTypes.PASTURE: 0.0,
    TerrainTypes.PINUS: 0.0,
    TerrainTypes.FIELD: 0.0,
    TerrainTypes.RESIDENTIAL: 1.0,
}


# TODO this values should later be revised
# Cost in Euros per hectar
COSTS_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: 0,
    TerrainTypes.WATER: 0,
    # Landscape: Forest; Density: Low
    TerrainTypes.NEEDLE_LITTER: 4e3,
    # Landscape: Forest; Density: Low
    TerrainTypes.FALLEN_LEAVES: 4e3,
    # Landscape: Farmland; Density: Low
    TerrainTypes.GRASSES_WEEDS: 1120,
    # Landscape: Farmland; Density: Low
    TerrainTypes.CAREX_FORBS: 1120,
    # Landscape: Farmland; Density: Low
    TerrainTypes.PASTURE: 1120,
    # Landscape: Forest; Density: Medium
    TerrainTypes.PINUS: 12e3,
    # Landscape: Farmlan; Density: High
    TerrainTypes.FIELD: 2350,
    # Landscape: Urban; Density: High
    TerrainTypes.RESIDENTIAL: 100e6,
}

# TODO this values should later be revised
# Emissoins in tones per hectar
EMISSIONS_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: 0,
    TerrainTypes.WATER: 0,
    # Landscape: Forest; Density: Low
    TerrainTypes.NEEDLE_LITTER: 180,
    # Landscape: Forest; Density: Low
    TerrainTypes.FALLEN_LEAVES: 180,
    # Landscape: Farmland; Density: Low
    TerrainTypes.GRASSES_WEEDS: 2,
    # Landscape: Farmland; Density: Low
    TerrainTypes.CAREX_FORBS: 2,
    # Landscape: Farmland; Density: Low
    TerrainTypes.PASTURE: 2,
    # Landscape: Forest; Density: Medium
    TerrainTypes.PINUS: 576,
    # Landscape: Farmland; Density: High
    TerrainTypes.FIELD: 6,
    # Landscape: Urban; Density: High
    TerrainTypes.RESIDENTIAL: 600,
}

# TODO this values should later be revised
# Casualties per hectar
CASUALTIES_TABLE = {
    TerrainTypes.NON_COMBUSTIBLE: 0,
    TerrainTypes.WATER: 0,
    # Landscape: Forest; Density: Low
    TerrainTypes.NEEDLE_LITTER: 0,
    # Landscape: Forest; Density: Low
    TerrainTypes.FALLEN_LEAVES: 0,
    # Landscape: Farmland; Density: Low
    TerrainTypes.GRASSES_WEEDS: 2,
    # Landscape: Farmland; Density: Low
    TerrainTypes.CAREX_FORBS: 2,
    # Landscape: Farmland; Density: Low
    TerrainTypes.PASTURE: 2,
    # Landscape: Forest; Density: Medium
    TerrainTypes.PINUS: 0,
    # Landscape: Farmland; Density: High
    TerrainTypes.FIELD: 10,
    # Landscape: Urban; Density: High
    TerrainTypes.RESIDENTIAL: 500,
}


class TerrainElement(Viewable):
    """Define the base description of an element of `Terrain` class."""

    @abstractattribute
    def height(self) -> float:
        """Height of Terrain in SI meters."""

    @abstractattribute
    def width(self) -> float:
        """Width of Terrain in SI meters."""

    @abstractattribute
    def origin(self) -> tuple[float, float]:
        """Position of top left corner of terrain."""

    @abstractattribute
    def visual_scale_factor(self) -> float:
        """Factor with which to scale GUI representation of Terrain."""

    @property
    def dimensions(self) -> tuple[float, float]:
        """Dimensions of terrain grid."""
        return self.width, self.height


class GridBasedElement(TerrainElement):
    """Characterizes terrain into a grid."""

    @abstractattribute
    def grid_shape(self) -> tuple[int, int]:
        """Shape of the grid in format (n_rows, n_cols)."""

    @abstractattribute
    def grid_dimensions(self) -> tuple[float, float]:
        """Dimensions of grid in format (width, height)."""

    @cached_property
    def grid_description(self) -> GridDescriptor:
        """Standard description of a grid."""
        return GridDescriptor(
            shape=self.grid_shape, dimensions=self.grid_dimensions
        )

    @abstractattribute
    def cell_size(self) -> None | int:
        """Grid cell size."""

    @property
    def height_in_cells(self) -> int:
        """Height of terrain in cells."""
        return self.grid_shape[0]

    @property
    def width_in_cells(self) -> int:
        """Width of terrain in cells."""
        return self.grid_shape[1]

    @cached_property
    def height(self) -> int:
        """Height of Terrain in SI meters."""
        return self.height_in_cells * self.cell_size

    @cached_property
    def width(self) -> int:
        """Width of Terrain in SI meters."""
        return self.width_in_cells * self.cell_size


class HeightMappedElevation(GridBasedElement):
    """Adds useful properties to ``elevation`` data.

    Note: The provided ``cell_size`` must be in the same units as the
    ``elevation`` array. Args: elevation: A 2D Numpy array containing
    scalar height values. cell_size: The dimension of each cell within
    the elevation array. For most publically available Digital Elevation
    Models (DEM), this is around 30 m. Defaults to 1 m.
    """

    def __init__(
        self,
        elevation: np.ndarray,
        cell_size: float = 1,
        origin: tuple[float, float] = (0, 0),
    ):
        self.elevation_data = elevation
        self.cell_size = cell_size
        self.origin = origin
        self.visual_scale_factor = 1

    @property
    def grid_shape(self):
        """Shape of the grid in format (n_rows, n_cols)."""
        return self.elevation_data.shape

    @property
    def grid_dimensions(self) -> tuple[int, int]:
        """width, height of grid."""
        return (self.width, self.height)

    @cached_property
    def gradients(self) -> tuple[(np.ndarray, np.ndarray)]:
        r"""Calculates gradients of :py:attr:`elevation_data`.

        This utilizes the same Moore neighborhood algorithm as
        `ArcGIS`_, implemented by :py:func:`gradient`, to return the
        gradient in x and y direction corresponding to the 1-st (j) and
        0-th (i) axis of the :py:attr:`elevation_data` array.

        Returns:
            The non-dimensional gradients of the terrain
            :math:`\frac{dz}{dx}` and :math:`\frac{dz}{dy}`.

        Note:
            A symmetric pad is used in order to minimize the errors
            at the boundary. This is preferred over padding with a zero
            value as that could artifically introduce a large downslope
            at the boundary if the average-ground level is non-zero.
        """
        padded_elevation = np.pad(
            self.elevation_data, pad_width=1, mode="symmetric"
        )
        dz_dx, dz_dy = gradient_2d(padded_elevation, self.cell_size)

        # Removing padding from gradient arrays and returning
        return (dz_dx[1:-1, 1:-1], dz_dy[1:-1, 1:-1])

    @cached_property
    def slopes(self) -> np.ndarray:
        """Calculates the slope magnitude of each terrain cell.

        Returns:
            Slope magnitude of each terrain cell in SI degree.
        """
        dz_dx, dz_dy = self.gradients
        return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    @cached_property
    def aspects(self) -> np.ndarray:
        r"""Calculates the aspect of the steepest dowslope direction.

        Aspect is the clockwise positive angle angle measured from
        North (Aspect = 0). Therefore, North = 0, East = 90, South =
        180 and West = 270 degrees.

        Returns:
            Aspect of steepest downhill slope in SI degree. If the
            gradients of a cell within the array are both zero, then the
            aspect cannot be compute for that cell and
            :py:data:`math.nan` will be returned.

        Note:
            The gradient vector is reversed since aspect points toward
            the steepest downslope direction. Furthermore,
            :math:`\frac{dz}{dx}` and :math:`\frac{dz}{dy}` are passed
            to :py:data:`arctan2` in reverse order to obtain the correct
            compass direction.
        """
        dz_dx, dz_dy = self.gradients
        aspects = (np.pi - np.arctan2(-dz_dx, -dz_dy)) * 180 / np.pi
        aspects = aspects % 360
        is_flat = np.logical_and(dz_dx == 0, dz_dy == 0)
        if np.any(is_flat):
            aspects[np.nonzero(is_flat)] = math.nan
        return aspects

    def hillshade(
        self, altitude: float = 45, aspect: float = 315
    ) -> np.ndarray:
        """Implements the Hillshade algorith of `ArcGIS`.

        Args:
            altitude: Anglular direction of light-source (sun) above
                the horizon in SI degree. 0 degrees is on the horizon
                and 90 degrees corresponds to the position of the sun
                at high-noon. Defaults to 45.
            aspect: Angular direction of the light-source (sun) in
                SI degree. An aspect of 90 degrees is East, while an
                aspect of 270 is West. Defaults to 315.

        Returns:
            A 2D array between 0-255 that provides a shaded
            representation of the :py:attr:`elevation_data`. This
            is useful for visualization purposes.
        """
        zenith_rad = math.radians(self.to_zenith(altitude))
        azimuth_rad = math.radians(self.to_azimuth(aspect))
        aspect_rad = self._hillshade_aspect(self.gradients)
        slope_rad = np.radians(self.slopes)
        return 255 * (
            np.cos(zenith_rad) * np.cos(slope_rad)
            + np.sin(zenith_rad)
            * np.sin(slope_rad)
            * np.cos(azimuth_rad - aspect_rad)
        )

    def hillshade_matplotlib(
        self, altitude: float = 45, aspect: float = 315
    ) -> np.ndarray:
        """Implements the Hillshade algorith of `matplotlib`.

        Args:
            altitude: Anglular direction of light-source (sun) above
                the horizon in SI degree. 0 degrees is on the horizon
                and 90 degrees corresponds to the position of the sun
                at high-noon. Defaults to 45.
            aspect: Angular direction of the light-source (sun) in
                SI degree. An aspect of 90 degrees is East, while an
                aspect of 270 is West. Defaults to 315.

        Note:
            This version produces more accentuated and detailed images
            than the `ArcGIS`_ hillshade algorithm implemented
            by :py:meth:`hillshade`.

        Returns:
            A 2D array between 0-1 that provides a shaded
            representation of the :py:attr:`elevation` data. This
            is useful for visualization purposes.
        """
        ls = LightSource(azdeg=aspect, altdeg=altitude)
        return ls.hillshade(
            (self.elevation_data), dx=(self.cell_size), dy=(self.cell_size)
        )

    @staticmethod
    @numba.jit(**BASE_JIT_KWARGS)
    def _hillshade_aspect(
        gradients: tuple[(numba.float32[:, :], numba.float32[:, :])],
    ) -> numba.float32[:, :]:
        r"""Naive implementation of `ArcGIS` hillshade aspect.

        Args:
            gradients: The non-dimensional gradients of the terrain
                :math:`\frac{dz}{dx}` and :math:`\frac{dz}{dy}`.

        Returns:
            `ArcGIS`_ hillshade formula compatible aspect where
            0 = East, and 90 = North. Counter-clockwise positive (CCW+).
        """
        dz_dx, dz_dy = gradients
        n_rows, n_cols = dz_dx.shape
        aspects = np.empty((n_rows, n_cols), dtype=(np.float32))
        for i in numba.prange(n_rows):
            for j in range(n_cols):
                dx, dy = dz_dx[(i, j)], dz_dy[(i, j)]
                if dx != 0:
                    aspect_rad = math.atan2(dy, -dx)
                    if aspect_rad < 0:
                        aspect_rad = 2 * math.pi + aspect_rad
                elif dx == 0:
                    if dy > 0:
                        aspect_rad = math.pi / 2
                    elif dy < 0:
                        aspect_rad = 2 * math.pi - math.pi / 2
                    else:
                        aspect_rad = 0
                aspects[(i, j)] = aspect_rad

        return aspects

    @staticmethod
    def to_azimuth(
        aspect: float | np.ndarray,
    ) -> float | np.ndarray:
        """Converts an aspect (compass units) to `ArcGIS`_ azimuth.

        Args:
            aspect: Compass units (0 - 360) in SI degree

        Returns:
             Counter-clockwise positive (CCW+) `ArcGIS` azimuth angle
             (0 = East, 90 = North) in SI degree.
        """
        return (360 - aspect % 360 + 90) % 360

    @staticmethod
    def to_zenith(altitude: float) -> float:
        """Converts an altitude (0 = horizon) to `ArcGIS`_ zenith angle.

        Args:
            altitude: Altitude of light source above horizon
                (0 = horizon, 90 = sun at high-noon)

        Returns:
            `ArcGIS` zenith angle that is (0 = sun at high noon, 90 =
            horizon)
        """
        if 0 <= altitude <= 90:  # noqa: PLR2004
            return 90 - altitude
        raise ValueError("Light source altitude must be between 0 and 90")

    def plot_3d(self) -> None:
        """Creates a 3D plot of terrain with elevation_data."""
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y = (np.meshgrid)(
            *[range(length) for length in self.elevation_data.shape]
        )
        ax.plot_surface(x, y, self.elevation_data)
        plt.show()
        return ax

    def plot(self, ls_altitude: float = 45, ls_aspect: float = 315) -> None:
        """Creates a 2D plot of the terrain using :py:meth:`hillshade`.

        See Also:
            :py:meth:`hillshade` for definitions of ``ls_altitude`` and
            ``ls_aspect``
        """
        plt.imshow(self.hillshade(altitude=ls_altitude, aspect=ls_aspect))
        plt.show()

    def __gui_repr__(self) -> dict:
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": ImageItem,
            "on_init": lambda p, _: p(
                (self.hillshade()),
                opacity=0.75,
                compositionMode=(QtGui.QPainter.CompositionMode_Multiply),
                rect=(
                    self.origin[0],
                    self.origin[1],
                    self.grid_dimensions[0],
                    self.grid_dimensions[1],
                ),
            ),
            "on_update": None,
            "z_order": 3,
            "scale": self.visual_scale_factor,
            # TODO Test without using a scale factor
        }


class RealHeightMappedElevation(HeightMappedElevation):
    """A HeightMappedTerrain modelling an actual geographical location.

    Note: Adopts a scaled Mercator representation for "pos".

    """

    def __init__(
        self,
        elevation: np.ndarray,
        cell_size: int,
        coordinates: tuple[float, float],
        origin: tuple[float, float] = (0, 0),
    ) -> None:
        super().__init__(elevation, cell_size, origin=origin)
        self.coordinates = coordinates

    @property
    def grid_dimensions(
        self,
    ) -> tuple[float, float]:
        """Overwrite with Mercator dimensions.

        The dimensions of the map represent unstretched distances as
        needed for the fire model. However for GPS to POS to Index
        transformations to work accurately, the POS coordinate system is
        considered to be Mercator coordinate system (stretched) when
        representng real terrain. Therefore, the grid_dimensions are
        overwritten with Mercator dimensions.

        This value is used for coordinate system transformations and
        visual representation.
        """
        return self.mercator_dimensions

    @cached_property
    def mercator_dimensions(self) -> tuple[float, float]:
        """Dimensions of map in mercator position (not meters)."""
        pos = gps_to_mercator(self.coordinates)
        width = pos[1, 0] - pos[0, 0]
        height = pos[0, 1] - pos[1, 1]
        return (width, height)


class OperationalTerrainElevation(RealHeightMappedElevation):
    """Extended terrain for operational use without fire model."""

    def __init__(
        self,
        elevation: np.ndarray,
        cell_size: float,
        coordinates: tuple[float, float],
        origin: tuple[float, float] = (0, 0),
    ):
        super().__init__(elevation, cell_size, coordinates, origin)

    def __gui_repr__(self) -> dict:
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": ImageItem,
            "on_init": lambda p, _: p(
                (self.hillshade()),
                opacity=1,
                compositionMode=(QtGui.QPainter.CompositionMode_Multiply),
                rect=(
                    self.origin[0],
                    self.origin[1],
                    self.grid_dimensions[0],
                    self.grid_dimensions[1],
                ),
            ),
            "on_update": None,
            "z_order": 1,
            "scale": self.visual_scale_factor,
        }


class CityTerrain(TerrainElement):
    """Terrain class for implementing GUI repr protocol of a city."""

    def __init__(self, location_data: dict, data_directory: str) -> None:
        super().__init__()
        self.origin = (0, 0)
        self.directory = data_directory
        self.location_data = location_data
        self.map_view = np.load(self.retrieve_map())

    def retrieve_map(self) -> WindowsPath:
        """Retrieve desired map based on simulation inputs."""
        if grid_map := self.map_in_storage():
            self.read_metadata()
            print(
                f"Map retrieved from storage: {self.filename} generated on "
                f"{self.timestamp}"
            )
        else:
            grid_map = self.fetch_map()
            if self.bbox_is_defined:
                print(
                    f"Map generated from inputs: {self.filename} created from "
                    f"bbox inputs of {self.location_data['location_bbox']}"
                )
            else:
                print(
                    f"Map generated from inputs: {self.filename} created "
                    f"from location name inputs of "
                    f"{self.location_data['location_type']} "
                    f"and {self.location_data['location_name']}"
                )
        return grid_map

    def map_in_storage(self) -> WindowsPath | bool:
        """Check if desired map is in local storage."""
        if self.filename.exists():
            return self.filename
        return False

    def read_metadata(self) -> None:
        """Read metadata associated with desired map."""
        with Path(self.metadata_filename).open() as metadata_file:
            metadata = json.load(metadata_file)
        self.bbox = metadata["bbox"]
        self.extent = metadata["extent"]
        (top, left), (bottom, right) = self.extent
        self.height = top - bottom
        self.width = right - left
        self.timestamp = metadata["time_stamp"]

    def fetch_map(self) -> WindowsPath:
        """Trigger download of map based on simulation inputs."""
        if self.bbox_is_defined():
            map_file = self.fetch_by_bbox()
        else:
            map_file = self.fetch_by_name()
        return map_file

    def bbox_is_defined(self) -> bool:
        """Check if `location_bbox` is defined in `location_data`."""
        return bool(self.location_data["location_bbox"])

    def fetch_by_bbox(self) -> WindowsPath:
        """Download map by bounding box definition."""
        bbox = self.location_data["location_bbox"]
        zoom = self.location_data["zoom_level"]
        (north, west), (south, east) = bbox
        img, extent = ctx.bounds2img(
            west,
            south,
            east,
            north,
            ll=True,
            source=(ctx.providers.OpenStreetMap.Mapnik),
            zoom=(zoom if zoom else "auto"),
        )

        # Update bbox of actually retrieved tile
        left, right, bottom, top = extent
        north, west = CRS_3857_TO_4326.transform(left, top)
        south, east = CRS_3857_TO_4326.transform(right, bottom)
        bbox = ((north, west), (south, east))
        self.save_file_and_metadata(img, bbox, extent)
        return self.filename

    def fetch_by_name(self) -> WindowsPath:
        """Download map by location type and name."""
        location_type = self.location_data["location_type"]
        location_name = self.location_data["location_name"]
        zoom = self.location_data["zoom_level"]
        place = ctx.Place(
            {location_type: location_name},
            source=(ctx.providers.OpenStreetMap.Mapnik),
            zoom=(zoom if zoom else "auto"),
        )
        img, bbox, extent = place.im, place.bbox, place.bbox_map
        west, south, east, north = bbox
        bbox = ((north, west), (south, east))
        self.save_file_and_metadata(img, bbox, extent)
        return self.filename

    def save_file_and_metadata(
        self,
        img: np.ndarray,
        bbox: tuple[float, float],
        extent: tuple[float, float, float, float],
    ) -> None:
        """Save retrieved map and metadata."""
        np.save(self.filename, img)
        self.save_bbox(bbox, extent)
        self.save_metadata()

    def save_bbox(
        self,
        bbox: tuple[float, float],
        extent: tuple[float, float, float, float],
    ) -> None:
        """Store bounding box and extent data."""
        self.bbox = bbox
        left, right, bottom, top = extent
        self.height = top - bottom
        self.width = right - left
        self.extent = ((top, left), (bottom, right))

    def save_metadata(self) -> None:
        """Save map metadata to file."""
        metadata = {
            "time_stamp": str(datetime.now()),
            "location_type": self.location_data["location_type"],
            "location_name": self.location_data["location_name"],
            "bbox": self.bbox,
            "extent": self.extent,
            "zoom_level": self.location_data["zoom_level"],
        }
        with Path(self.metadata_filename).open("w") as metadata_file:
            json.dump(metadata, metadata_file)
            metadata_file.close()

    @property
    def metadata_filename(self) -> str:
        """Return filename for metadata."""
        return str(self.filename).replace("npy", "meta")

    @property
    def filename(self) -> str:
        """Return filename for map."""
        return self.directory / str(
            self.location_data["location_name"] + ".npy"
        )

    @property
    def top_left_bounds(self) -> tuple[float, float]:
        """Top left position coordiantes."""
        return self.extent[0]

    @cached_property
    def visual_scale_factor(self) -> float:
        """Return visual scaling factor for map."""
        n_rows, n_cols, _ = self.map_view.shape
        vertical_scaling = self.height / n_rows
        horizontal_scaling = self.width / n_cols
        return np.mean([horizontal_scaling, vertical_scaling])

    def __gui_repr__(self) -> dict:
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": ImageItem,
            "on_init": lambda p, _: p(
                (self.map_view),
                opacity=1,
                compositionMode=(QtGui.QPainter.CompositionMode_Source),
            ),
            "on_update": None,
            "z_order": 1,
            "scale": self.visual_scale_factor,
        }


class BaseFeatures(TerrainElement):
    """Base class to process satellite imagery into feature indices."""

    def __init__(self, origin: tuple[float, float] = (0, 0)) -> None:
        self.origin = origin

    @abstractattribute
    def colormap(self) -> np.ndarray | None:
        """2D array of RGB values for visual representation."""


class GridBasedFeatures(BaseFeatures, GridBasedElement):
    """Expands grid element to have features data."""

    def __init__(self, cell_size: int, origin: tuple[float, float]) -> None:
        super().__init__(origin)
        self.cell_size = cell_size

    def __gui_repr__(self) -> dict:
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": ImageItem,
            "on_init": lambda p, obj: p(
                obj.colormap,
                compositionMode=(QtGui.QPainter.CompositionMode_SourceOver),
                rect=(
                    self.origin[0],
                    self.origin[1],
                    self.grid_dimensions[0],
                    self.grid_dimensions[1],
                ),
            ),
            "on_update": None,
            "z_order": 0,
            "scale": self.visual_scale_factor,
        }


class GenericFeatures(GridBasedFeatures):
    """Turns satellite imagery into features with default settings."""

    def __init__(
        self,
        cell_size: int,
        grid_shape: np.ndarray,
        grid_dimensions: tuple[float, float],
        origin: tuple[float, float] = (0, 0),
    ) -> None:
        super().__init__(cell_size, origin)
        self.grid_shape = grid_shape
        self.grid_dimensions = grid_dimensions
        self.cell_size = cell_size
        self.visual_scale_factor = 1
        self.height = self.grid_shape[0] * self.cell_size
        self.width = self.grid_shape[1] * self.cell_size
        self.combustibilities = np.ones(grid_shape).astype(
            np.float32, order="C"
        )
        color_value = [(25, 94, 44, 255)]  # default vegetation color
        colors = [color_value * grid_shape[1]] * grid_shape[0]
        self.colormap = np.asarray(colors)
        self.prioritymap = np.zeros(grid_shape).astype(np.float32, order="C")


class ImportedFeatures(GridBasedFeatures):
    """Creates color and combustibility map from terrain data."""

    def __init__(
        self,
        cell_size: int,
        source: str,
        coordinates: tuple[float, float],
        parameters: dict,
        origin: tuple[float, float] = (0, 0),
    ) -> None:
        super().__init__(cell_size, origin)
        self.origin = origin
        self.coordinates = coordinates
        self.parameters = parameters
        self.visual_scale_factor = 1

        feature_index_map = np.load(str(source), mmap_mode="r").astype(
            np.int32
        )

        combustibility_file = Path(
            str(source).replace("features", "combustibilities")
        )
        colormap_file = Path(str(source).replace("features", "colormap"))
        prioritymap_file = Path(str(source).replace("features", "prioritymap"))

        if not combustibility_file.is_file():
            combustibility_values = np.array(
                self.sort_by_key(COMBUSTIBILITY_TABLE), dtype=np.float32
            )
            combustibilities = combustibility_values[feature_index_map]
            np.save(combustibility_file, combustibilities)

        if not colormap_file.is_file():
            color_values = np.array(
                self.sort_by_key(FEATURES_COLOR_TABLE), dtype=np.int32
            )
            colormap = color_values[feature_index_map]
            np.save(colormap_file, colormap)

        if not prioritymap_file.is_file():
            priority_values = np.array(
                self.sort_by_key(PRIORITY_TABLE), dtype=np.float32
            )
            priority_map = priority_values[feature_index_map]
            priority_map = gaussian_filter(
                priority_map, sigma=self.parameters.priority_map_sigma
            )
            np.save(prioritymap_file, priority_map)

        self.combustibilities = np.load(
            combustibility_file, mmap_mode="r+"
        ).astype(np.float32)
        self.colormap = np.load(colormap_file, mmap_mode="r+").astype(np.int32)
        self.priority_map = np.load(prioritymap_file, mmap_mode="r+").astype(
            np.float32
        )
        self.grid_shape = feature_index_map.shape

    @cached_property
    def mercator_dimensions(self) -> tuple[float, float]:
        """Dimensions of map in mercator position (not meters)."""
        pos = gps_to_mercator(self.coordinates)
        width = pos[1, 0] - pos[0, 0]
        height = pos[0, 1] - pos[1, 1]
        return (width, height)

    @staticmethod
    def sort_by_key(input_dict: dict) -> list[dict]:
        """Returns list of dictionary values sorted by key."""
        return [input_dict[key] for key in sorted(input_dict.keys())]

    @property
    def grid_dimensions(self) -> tuple[float, float]:
        """Overwrite with Mercator dimensions.

        The dimensions of the map represent unstretched distances as
        needed for the fire model. However for GPS to POS to Index
        transformations to work accurately, the POS coordinate system is
        considered to be Mercator coordinate system (stretched) when
        representng real terrain. Therefore, the grid_dimensions are
        overwritten with Mercator dimensions.

        This value is used for coordinate system transformations and
        visual representation.
        """
        return self.mercator_dimensions


class BaseTerrain:
    """Defines basic skeleton of a Terrain description."""

    @abstractattribute
    def elevation(self) -> np.ndarray | None:
        """Elevation grid data."""

    @abstractattribute
    def simulation(self) -> None:
        """Simulation for self referencing."""

    @abstractattribute
    def features(self) -> np.ndarray | None:
        """Features grid data."""

    @abstractattribute
    def grid_description(self) -> GridDescriptor:
        """Grid description for `index` and `pos` CS transforms."""

    @abstractattribute
    def dimensions(self) -> tuple[float, float] | None:
        """Max dimensions of the `Terrain`."""

    @property
    def grid_shape(self) -> tuple[float, float]:
        """Grid shape of grid description."""
        return self.grid_description.shape

    @property
    def grid_dimensions(self) -> tuple[float, float]:
        """Grid dimensions of grid description."""
        return self.grid_description.dimensions


class ForestTerrain(BaseTerrain):
    """Class to combine the elevation and features data.

    Composes `Environment` class together with `Atmosphere`.
    """

    def __init__(
        self,
        simulation,  # noqa: ANN001
        terrain_parameters: dict,
    ) -> None:
        self.simulation = simulation
        self.parameters = terrain_parameters
        self.coordinates = self.parameters.coordinates
        self.elevation = self.load_elevation(
            self.parameters.import_features_osm
        )
        self.features = self.load_features(self.parameters.import_features_osm)

    @property
    def origin(self) -> tuple[float, float]:
        """Origin/ center of terrain features grid."""
        return self.elevation.origin

    @cached_property
    def bounding_positions(self) -> tuple[Position, Position]:
        """Top-left and bottom-right corners."""
        coords = map(tuple, self.coordinates)
        return tuple(gps_to_pos(coords, self.top_left_bounds))

    @cached_property
    def fire_map_bounding_positions(self) -> tuple[Position, Position]:
        """Top-left and bottom-right corners of fire map."""
        coords = map(tuple, self.parameters.fire_map_coordinates)
        return tuple(gps_to_pos(coords, self.top_left_bounds))

    # TODO: Consider reading in fire and water map coordinates
    @cached_property
    def fire_map_top_left_bounds(self) -> tuple[float, float]:
        """Top left position of fire map grid."""
        pos = gps_to_mercator(self.parameters.fire_map_coordinates[0])
        return (pos[1], pos[0])

    @cached_property
    def water_map_top_left_bounds(self) -> tuple[float, float]:
        """Water map top left position of larger grid."""
        pos = gps_to_mercator(self.parameters.water_map_coordinates[0])
        return (pos[1], pos[0])

    @cached_property
    def top_left_bounds(self) -> tuple[float, float]:
        """Top left position of larger grid."""
        pos = gps_to_mercator(self.coordinates[0])
        return (pos[1], pos[0])

    @property
    def grid_description(self) -> GridDescriptor:
        """Grid description for `index` and `pos` CS transforms."""
        return self.elevation.grid_description

    @property
    def slopes(self) -> np.ndarray:
        """Elevation slope raster data."""
        return self.elevation.slopes

    @property
    def aspects(self) -> np.ndarray:
        """Elevation aspects raster data."""
        return self.elevation.aspects

    @property
    def combustibilities(self) -> np.ndarray:
        """Combusbility raster data."""
        return self.features.combustibilities

    @property
    def width_in_cells(self) -> int:
        """Elevation map width in cells."""
        return self.elevation.width_in_cells

    @property
    def height_in_cells(self) -> int:
        """Elevation map height in cells."""
        return self.elevation.height_in_cells

    @property
    def dimensions(self) -> tuple[float, float]:
        """Max dimensions of the `Terrain`."""
        return self.mercator_dimensions

    @cached_property
    def urban_areas(self) -> np.ndarray | None:
        """All the urban areas of the features data."""
        return np.all(
            self.features.colormap
            == FEATURES_COLOR_TABLE[TerrainTypes.RESIDENTIAL],
            axis=2,
        )

    @cached_property
    def urban_indices(self) -> np.ndarray[np.int64]:
        """Set of urban locations on fire map grid."""
        return np.argwhere(self.urban_areas)

    @cached_property
    def fire_map_mercator_dimensions(self) -> tuple[float, float]:
        """Dimensions of fire map in mercator position (not meters)."""
        pos = gps_to_mercator(self.parameters.fire_map_coordinates)
        width = pos[1, 0] - pos[0, 0]
        height = pos[0, 1] - pos[1, 1]
        return (width, height)

    @cached_property
    def mercator_dimensions(self) -> tuple[float, float]:
        """Dimensions of map in mercator position (not meters)."""
        pos = gps_to_mercator(self.coordinates)
        width = pos[1, 0] - pos[0, 0]
        height = pos[0, 1] - pos[1, 1]
        return (width, height)

    def load_elevation(
        self, import_real_elevation: bool
    ) -> HeightMappedElevation | RealHeightMappedElevation:
        """Outputs elevation data array multiplied by scaling factor.

        Returns:
            ndarray: scaled elevation data
        """
        if not import_real_elevation:
            elevation_data = (
                np.load(self.parameters.elevation_file, mmap_mode="r")
                * self.parameters.height_scale_factor
            )
            elevation = HeightMappedElevation(
                elevation=elevation_data,
                cell_size=self.parameters.cell_size,
            )

        else:
            elevation_data = (
                np.load(self.parameters.elevation_file, mmap_mode="r") * 1
            )
            elevation = RealHeightMappedElevation(
                elevation=elevation_data,
                cell_size=self.parameters.cell_size,
                coordinates=self.parameters.fire_map_coordinates,
            )
        return elevation

    def load_features(
        self, import_features_osm: bool
    ) -> ImportedFeatures | GenericFeatures:
        """Use imported OSM features or not.

        Args: import_features_osm (bool), specifies the behaviour
        """
        if import_features_osm:
            features = ImportedFeatures(
                source=self.parameters.features_file,
                parameters=self.parameters,
                coordinates=self.elevation.coordinates,
                origin=self.elevation.origin,
                cell_size=self.elevation.cell_size,
            )
        else:
            features = GenericFeatures(
                grid_shape=self.elevation.grid_shape,
                grid_dimensions=self.elevation.grid_dimensions,
                origin=self.elevation.origin,
                cell_size=self.elevation.cell_size,
            )
        return features

    def override_map_features(
        self, indices: np.ndarray, feature_value: TerrainTypes
    ) -> None:
        """Overrides map raster data for given indices and feature.

        This does not override water regions.

        Args:
            indices (ndarray): List of indices to be overriden
            feature_value (TerrainTypes): Feature to use for override
        """
        colors = self.features.colormap[indices[:, 0], indices[:, 1], :]
        # Assumes we do not have different terrain with the same color
        # as water
        mask = ~np.all(
            colors == np.array(FEATURES_COLOR_TABLE[TerrainTypes.WATER]),
            axis=1,
        )
        self.features.colormap[indices[mask, 0], indices[mask, 1]] = (
            FEATURES_COLOR_TABLE[feature_value]
        )
        self.features.colormap.flush()
        self.features.combustibilities[indices[mask, 0], indices[mask, 1]] = (
            COMBUSTIBILITY_TABLE[feature_value]
        )
        self.features.combustibilities.flush()
        new_priorities = np.zeros_like(self.features.priority_map)

        # Set the values in 'new_priorities' at the specified indices
        # where mask is True
        new_priorities[indices[mask, 0], indices[mask, 1]] = PRIORITY_TABLE[
            feature_value
        ]
        new_priorities = gaussian_filter(
            new_priorities, sigma=self.parameters.priority_map_sigma
        )
        self.features.priority_map = np.where(
            self.features.priority_map < new_priorities,
            new_priorities,
            self.features.priority_map,
        )


class OperationalTerrain(ForestTerrain):
    """Expands the terrain to the operational/ larger area."""

    def __init__(
        self,
        simulation,  # noqa: ANN001
        terrain_parameters: dict,
    ) -> None:
        super().__init__(simulation, terrain_parameters)
        self.operational_elevation = self.load_operational_elevation(
            self.parameters.operational_elevation_file
        )
        self.operational_features = GenericFeatures(
            cell_size=self.operational_elevation.cell_size,
            grid_shape=self.operational_elevation.grid_shape,
            grid_dimensions=self.operational_elevation.grid_dimensions,
        )
        # TODO: Store and Retrieve origins for fire and water maps
        y_origin = self.top_left_bounds[0] - self.fire_map_top_left_bounds[0]
        x_origin = self.fire_map_top_left_bounds[1] - self.top_left_bounds[1]
        self.elevation.origin = (x_origin, y_origin)
        self.features.origin = (x_origin, y_origin)

        x_origin_water = (
            self.water_map_top_left_bounds[1] - self.top_left_bounds[1]
        )
        y_origin_water = (
            self.top_left_bounds[0] - self.water_map_top_left_bounds[0]
        )
        self.water_map_origin = (x_origin_water, y_origin_water)
        self.file_path = self.parameters.operational_image_file
        self.map_image = np.load(self.file_path)
        self.visual_scale_factor = 1

    def load_operational_elevation(
        self, file: WindowsPath
    ) -> OperationalTerrainElevation:
        """Loads the operational terrain elevation."""
        elevation = (
            np.load(file, mmap_mode="r")
            * self.simulation.parameters.height_scale_factor
        )
        return OperationalTerrainElevation(
            elevation=elevation,
            cell_size=self.parameters.operational_terrain_cell_size,
            coordinates=self.coordinates,
        )

    def __gui_repr__(self) -> dict:
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": ImageItem,
            "on_init": lambda p, obj: p(
                obj.map_image,
                compositionMode=(QtGui.QPainter.CompositionMode_SourceOver),
                rect=(
                    0,
                    0,
                    self.mercator_dimensions[0],
                    self.mercator_dimensions[1],
                ),
            ),
            "on_update": None,
            "z_order": 0,
            "scale": self.visual_scale_factor,
        }
