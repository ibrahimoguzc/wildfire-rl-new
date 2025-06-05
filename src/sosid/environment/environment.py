# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains classes for defining SOSID simulation environments."""

import inspect
from functools import cached_property
from pathlib import Path

import numpy as np

from sosid.abstract import Viewable
from sosid.environment.terrain import (
    BaseTerrain,
    CityTerrain,
    ForestTerrain,
    OperationalTerrain,
)
from sosid.model.transform import pos_to_index
from sosid.typedef import Position
from sosid.util.abc import abstractattribute, abstractmethod


# TODO: Implement bearing function from
# https://tools.ietf.org/html/rfc7946
# TODO: Add tests for Environment
class BaseEnvironment(Viewable):
    """Defines the operational environment of a simulation."""

    def __init__(self, simulation):
        self.simulation = simulation

    @abstractattribute
    def terrain(self) -> BaseTerrain:
        """Adds useful geographic parameters to environment."""

    @property
    def origin(self) -> tuple[float, float]:
        """Defines the origin coordinates for the terrain."""
        return self.terrain.origin

    @abstractattribute
    def active_area(self) -> tuple[float, float]:
        """Active area to highlight with a `BorderItem`."""

    @abstractattribute
    def dimensions(self) -> tuple[float, float]:
        """Max dimension of the environment."""

    def __gui_repr__(self):
        return self.terrain.__gui_repr__()


class BaseWildfireEnvironment(BaseEnvironment):
    def __init__(self, simulation, terrain_parameters, atmosphere):
        super().__init__(simulation)
        self.parameters = terrain_parameters
        self.atmosphere = atmosphere

    @abstractattribute
    def max_elevation(self) -> float:
        """Max elevation in the environment."""

    @abstractmethod
    def get_elevation(self, pos: Position) -> float:
        """Get the elevation at a given position."""


class WildfireEnvironment(BaseWildfireEnvironment):
    """Defines the operational environment of a wildfire simulation.

    Attributes:
        terrain: comprised of elevation and features
        atmosphere: defines the atmospheric parameters over time

    """

    def __init__(self, simulation, terrain_parameters, atmosphere):
        super().__init__(simulation, terrain_parameters, atmosphere)
        self.terrain = ForestTerrain(simulation, terrain_parameters)

    @property
    def active_area(self) -> tuple[float, float]:
        """Active area to highlight with a `BorderItem`."""
        return self.terrain.grid_dimensions

    @property
    def dimensions(self) -> tuple[float, float]:
        """Max dimension of the environment."""
        return self.terrain.elevation.dimensions

    @cached_property
    def max_elevation(self):
        elevation_data = self.terrain.elevation.elevation_data
        return np.max(elevation_data) - np.min(elevation_data)

    def get_elevation(self, pos: Position) -> float:
        """Get the elevation at a given position."""
        idx = pos_to_index(pos, self.terrain.grid_description)
        return self.terrain.elevation.get_elevation[idx]

    def __gui_repr__(self):
        return [
            self.terrain.elevation.__gui_repr__(),
            self.terrain.features.__gui_repr__(),
        ]


class ExtendedWildfireEnvironment(BaseWildfireEnvironment):
    """Creates the environment for the operational map.

    The operational map is the extended environment which does not
    directly contain the fire map grid. Since no terrain features and
    elevation data are obtained from resources, the elevation is set to
    be equal to the mean of the underlying fire map- to reduce elevation
    inconsistencies.
    """

    def __init__(self, simulation, terrain_parameters, atmosphere):
        super().__init__(simulation, terrain_parameters, atmosphere)
        self.terrain = OperationalTerrain(simulation, terrain_parameters)
        # Set the elevation to the mean fire map elevation.
        self.terrain.operational_elevation.elevation_data[:] = np.mean(
            self.terrain.elevation.elevation_data
        )

    @property
    def dimensions(self) -> tuple[float, float]:
        """Max dimensions of environment."""
        return self.terrain.mercator_dimensions

    @property
    def active_area(self) -> tuple[float, float]:
        """Active area to highlight with a `BorderItem`."""
        return self.terrain.grid_dimensions

    @cached_property
    def max_elevation(self):
        elevation_data = self.terrain.elevation.elevation_data
        return np.max(elevation_data) - np.min(elevation_data)

    def get_elevation(self, pos: Position) -> float:
        """Get the elevation at a given position."""
        x, y = pos
        x_rng, y_rng = zip(*self.terrain.fire_map_bounding_positions)
        if x_rng[0] <= x <= x_rng[1] and y_rng[0] <= y <= y_rng[1]:
            # Position in fire map.
            idx = pos_to_index(
                pos,
                self.terrain.elevation.grid_description,
                origin=self.terrain.elevation.origin,
            )
            return self.terrain.elevation.elevation_data[idx]
        x_rng, y_rng = zip(*self.terrain.bounding_positions)
        if not (x_rng[0] <= x <= x_rng[1] and y_rng[0] <= y <= y_rng[1]):
            raise ValueError("Position is not in the environment.")
        idx = pos_to_index(
            pos, self.terrain.operational_elevation.grid_description
        )
        return self.terrain.operational_elevation.elevation_data[idx]

    def __gui_repr__(self):
        return [
            self.terrain.__gui_repr__(),
            self.terrain.elevation.__gui_repr__(),
            self.terrain.features.__gui_repr__(),
        ]


class UAMEnvironment(BaseEnvironment):
    """Defines the operational environment of a UAM simulation.

    Attributes:
        location_data: dict holding location related data in following
        format:

        location_data = {
            'location_name': 'Hamburg',
            'location_type': 'city', Optional, default: City
            'location_bbox': ((i_tl,j_tl), (i_br, j_br)) where tl: top
                left and br : bottom right, Optional, default: None)
            'zoom_level': 10, Optional, default: 10
        }
    """

    def __init__(self, simulation):  # noqa D102
        self.simulation = simulation
        self.terrain = CityTerrain(
            location_data=self.simulation.parameters.location_data,
            data_directory=self.get_data_directory(),
        )  # type: ignore

    @property
    def active_area(self) -> tuple[float, float]:
        """Active area to highlight with a `BorderItem`."""
        return self.terrain.dimensions

    @property
    def dimensions(self) -> tuple[float, float]:
        """Max dimension of the environment."""
        return self.terrain.dimensions

    def get_data_directory(self):  # noqa D102
        data_path = Path(inspect.getfile(self.simulation.__class__)).parent
        if Path.exists(data_path / "data/terrain") or Path.exists(
            data_path := data_path.parent / "data/terrain"
        ):
            return data_path
