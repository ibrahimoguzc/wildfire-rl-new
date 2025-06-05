# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains useful specializations of :py:class:`Agent` from MESA."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from functools import lru_cache
from random import Random

import numba
import numpy as np
from mesa import Agent as MesaAgent

from sosid.abstract import Model, Viewable, Writable
from sosid.gui.items import MarkerItem
from sosid.model.abm.trajectory import IN_AIR_FLIGHT_STATES
from sosid.model.transform import (
    GEODESIC,
    bearing_from_coords,
    gps_to_pos,
    pos_to_gps,
)
from sosid.paths import ICONS_PATH
from sosid.typedef import LatLon, Position
from sosid.util.abc import ABCMeta, abstractattribute, abstractmethod

from .task import Task, TaskPriority, TaskScheduler, TaskStatus

__all__ = ["Agent", "Task", "TaskPriority", "TaskStatus"]


# TODO Consider defining slots for base classes -> May make demand
# agents more efficient
class Agent(MesaAgent, Viewable, Writable):
    """Adds task-system functionality to a :py:class:`mesa.Agent`."""

    __icon__ = ICONS_PATH / "marker.svg"

    def __init__(
        self,
        unique_id: int,
        model: Model,
        autopopulate: bool,
        recursion_check: bool = False,
    ):
        self.unique_id = unique_id
        self.model = model
        self.tasks = TaskScheduler(
            agent=self,
            autopopulate=autopopulate,
            recursion_check=recursion_check,
        )

    @abstractmethod
    def step(self) -> None:
        """A single step of the agent, must be overriden by subclass."""

    @property
    def random(self) -> Random:
        """Returns the random number generator of :py:attr:`model`."""
        return self.model.random

    @property
    def pos(self):
        """Returns a view of the model position matrix."""
        positions = self.model.positions
        if positions.ndim > 1:
            return positions[self.unique_id, :]
        return positions[:]

    @property
    def scale(self) -> float:
        """Returns scale factor"""
        return self.model.simulation.parameters.zoom_scale_factor

    @pos.setter
    def pos(self, value: Position):
        """Updates the agent's position within the position matrix."""
        positions = self.model.positions
        value = np.array(value)
        if positions.ndim > 1:
            positions[self.unique_id, :] = value
        else:
            positions[:] = value

    def run_on_other_agents(
        self,
        expression: Callable[[Agent], None],
        agent_type: Agent | tuple[Agent] | None = None,
    ) -> None:
        """Runs ``expression``on all other agents of ``agent_type``.

        The intended usage with a lambda function is as follows::

            @mock_task.on_complete
            def instruct_agents_to_rtb(self):
                run_on_other_agents(
                    expression: lambda agent: agent.tasks.set_active(
                        agent.return_to_base
                    )
                )

        Args:
            expression: An unbound method or lambda that accepts a
                :py:class:`Agent` as its only argument.
            agent_type: Optional type of :py:class:`Agent` to
                run ``expression`` on. Defaults to :py:data:`None` which
                will run the expression on all agents.
        """
        agent_type = Agent if agent_type is None else agent_type
        for agent in self.get_other_agents(agent_type):
            expression(agent)

    def run_on_specific_agents(
        self,
        expression: Callable[[Agent], None],
        specific_agents: Agent | tuple[Agent] | list[Agent],
    ) -> None:
        """Runs ``expression``on all agents in ``specific_agents``.

        The intended usage with a lambda function is as described in
        :py:method:`run_on_other_agents`

        Args:
            expression: An unbound method or lambda that accepts a
                :py:class:`Agent` as its only argument.
            specific_agents: Agent or Agents to run ``expression`` on.
        """
        # Convert to iterable, if single agent
        if type(specific_agents) not in [list, tuple]:
            specific_agents = [specific_agents]  # type:ignore
        for agent in specific_agents:  # type:ignore
            expression(agent)

    def get_other_agents(
        self, agent_type: Agent | tuple[Agent] | None = None
    ) -> Generator[Agent, None, None]:
        """Gets all other agents in a :py:class:`AgentBasedModel`.

        Args:
            agent_type: Optional type of :py:class:`Agent` to get.
                Defaults to :py:data:`None` which will retrieve all
                agent types.
        """
        agent_type = Agent if agent_type is None else agent_type
        for agent in self.model.agents:
            if isinstance(agent, agent_type) and agent is not self:
                yield agent

    def __gui_repr__(self):
        """Implements the GUI representation protocol."""
        return {
            "object": self,
            "painter": MarkerItem,
            "on_init": lambda p, obj: p(
                obj.__icon__, size=(10, 10), pos=obj.pos
            ),
            "on_update": self.update_pos_and_scale,
            "z_order": 1000,
            "scale": 1,
        }

    @staticmethod
    def update_pos_and_scale(item, obj):
        """Update parameters used for agent representation."""
        item.setPos(*obj.pos)
        item.setScale(obj.scale)

    def nearest_agent(self, other_agents: list, agent: object | None = None):
        """Finds the closest agent from list to the specified ``agent``.

        Returns the nearest agent from the given array and
        the coresponding distance from the ``agent`` object.

        Args:
            agent: Instance of :py:class:`Agent`. Defaults to
                :py:data:`None` which will default to ``self``.
            other_agents: A :py:type:`list` of :py:class:`Agent`
                instances.
        """
        agent = agent if agent else self
        agent_locations = np.array([obj.pos for obj in other_agents])
        distances = self.distance(agent.pos, agent_locations)
        min_idx = np.argmin(distances)
        closest_agent = other_agents[min_idx]
        distance = distances[min_idx]
        return closest_agent, distance

    @staticmethod
    @numba.njit
    def distance(pos_1: np.ndarray, pos_2: np.ndarray) -> float | np.ndarray:
        """Calculates distance between two positions in SI meter."""
        if pos_2.ndim == 1:
            x_1, y_1 = pos_1
            x_2, y_2 = pos_2
            d2 = (x_2 - x_1) ** 2 + (y_2 - y_1) ** 2
            return np.sqrt(d2)
        distances = []
        for index in numba.prange(len(pos_2)):
            x_1, y_1 = pos_1
            x_2, y_2 = pos_2[index]
            d2 = (x_2 - x_1) ** 2 + (y_2 - y_1) ** 2
            distances.append(np.sqrt(d2))
        return np.array(distances)

    @staticmethod
    def distance_gps(
        coord: np.ndarray, other_coordinates: np.ndarray
    ) -> float | np.ndarray:
        """Calculates distance between one and multiple GPS Coordinates.

        Computes distance in SI meter.

        Args:
            coord: Single GPS Coordinate.
            other_coordinates: An array with one or multiple GPS
                Coordinates in the shape of (X,2)
        """
        coord = np.array(coord)
        other_coordinates = np.array(other_coordinates)

        if other_coordinates.ndim != 1:
            coord = np.tile(coord, (other_coordinates.shape[0], 1))

        origin_lat = coord[..., 0]
        origin_long = coord[..., 1]
        dest_lat = other_coordinates[..., 0]
        dest_long = other_coordinates[..., 1]

        _, _, distance = GEODESIC.inv(
            origin_long,
            origin_lat,
            dest_long,
            dest_lat,
            radians=False,
            return_back_azimuth=False,
        )
        return distance

    def nearest_position(
        self,
        object_locations: np.ndarray,
        pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finds the closest position in array to the specified ``pos``.

        Returns the closest location to the ``pos`` and the
        coresponding distnce between the two in SI meter.

        Args:
            object_locations: A :py:type:`ndarray` with locations of the
                objects.
            pos: Optional type of :py:type:`ndarray` to calculate
                the distances from. Defaults to :py:data:`None` which
                calculates the distances from :py:attr:`pos`.
        """
        position = self.pos if pos is None else pos
        distances = self.distance(position, object_locations)
        min_idx = np.argmin(distances)
        closest_location = np.array(object_locations)[min_idx, :]
        distance = distances[min_idx]
        return np.array(closest_location), distance

    @property
    def current_mission_time(self) -> datetime:
        return self.model.simulation.timer.mission_time


class AgentWithGPSMixin(metaclass=ABCMeta):
    """Mixin for agents requiring use of GPS coordinates.

    To give additional Agent types the GPS Coordinate methods, simply
    inheriting from the desired Agent and this class would suffice.

    When inheriting from multiple classes, it is important to pay
    attention to the order in which the inheritance is defined.
    Generally this class should be inherited from first, so that the
    methods within are given prioririty over the others as is intended.

    It is advised to only deviate from this recommendation when certain
    of its necessity.
    """

    @abstractattribute
    def top_left_bounds(self):
        """Returns the position of the top left corner of the Map.

        Can be obtained from the "extent" of the Map in SI meters
        and uses the Web Mercator Projection `CRS 3857`
        """

    @property
    def gps_coords(self):
        """Returns `gps_coords`` of the Agent."""
        return self._gps_coords(tuple(self.pos))

    @lru_cache(maxsize=1)
    def _gps_coords(self, pos):
        """Caches gps_coords of Agent for current `pos`.

        The cache is updated when the method is called with a different `pos`.
        """
        return pos_to_gps(pos, self.top_left_bounds)

    @gps_coords.setter
    def gps_coords(self, value: Position):
        """If `gps_coords` is set, update `pos`.

        `pos` is treated as master coordinate system, by updating pos,
        the `gps_coords` would also reflect this change.
        """
        self.pos = gps_to_pos(value, self.top_left_bounds)

    def distance(self, pos_1: float, pos_2: float):
        """Override `distance` method to use GPS Coordinates.

        This is done to circumvent the warping of the Mercator
        Projection.
        """
        gps_1 = pos_to_gps(pos_1, self.top_left_bounds)
        gps_2 = pos_to_gps(pos_2, self.top_left_bounds)
        return self.distance_gps(gps_1, gps_2)


class TrackFlightDurationMixin(metaclass=ABCMeta):
    """Mixin for tracking flight time of aircraft agents."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._cumulative_flight_time = 0.0

    def step(self) -> None:
        """Step in the simulation."""
        super().step()
        if self.flight_state in IN_AIR_FLIGHT_STATES:
            self._cumulative_flight_time += self.time_step.total_seconds()


# TODO Rename to DynamicAgent as opposed to StaticAgent
# TODO make velocity an abstract attribute
class MovingAgent(Agent):
    """Base class for agents requiring position update.

    Add methods to update agent position and its graphical
    representation to :py:class:`Agent`
    """

    def __init__(
        self,
        unique_id,
        model,
        pos: Position,
        autopopulate: bool,
        recursion_check: bool = True,
    ):
        """Create a new agent."""
        super().__init__(
            unique_id=unique_id, model=model, autopopulate=autopopulate
        )
        self.total_distance_covered = 0
        self.pos_prev = pos
        self.heading = (0, -1)

    # TODO make the models more easily accesible
    # TODO make the position ufunc castable
    def step(self):
        """A single step of the agent."""
        # Total distance travelled update
        pos = self.pos.copy()
        self.total_distance_covered += self.distance(pos, self.pos_prev)
        self.pos_prev = pos
        # Run active task
        self.tasks.run_active()

    def navigate_to(
        self, destination: Position, velocity: float
    ) -> TaskStatus:
        pos = self.pos
        direction_vector = destination - pos
        distance = self.distance(destination, pos)
        if (
            distance
            <= velocity * self.model.simulation.time_step.total_seconds()
        ):
            self.pos = destination
            return TaskStatus.COMPLETE
        self.heading = direction_vector / distance

        self.pos = (
            self.heading
            * velocity
            * self.model.simulation.time_step.total_seconds()
        ) + pos
        return TaskStatus.IN_PROGRESS

    @property
    def aspect(self):
        return math.degrees(math.atan2(*self.heading[::-1])) + 90

    __idle_timer__ = None

    def idle(self, duration: float) -> TaskStatus:
        """Keep agent inactive for ``duration`` in SI second."""
        if not self.__idle_timer__:
            self.__idle_timer__ = timedelta(seconds=duration)
        self.__idle_timer__ -= self.model.simulation.time_step
        if self.__idle_timer__.total_seconds() <= 0:
            self.__idle_timer__ = None  # Reset idle timer
            return TaskStatus.COMPLETE
        return TaskStatus.IN_PROGRESS

    @property
    def remaining_idle_time(self):
        """Return time left on __idle_timer__ when idling."""
        if self.__idle_timer__:
            return self.__idle_timer__.total_seconds()
        raise Exception(
            "Incorrect use: Method `remaining_idle_time` is intended for",
            "use only when agent is idling",
        )

    def __gui_repr__(self):
        """Returns GUI representation."""
        gui_repr = super().__gui_repr__()
        gui_repr.update(
            {
                "on_init": lambda p, obj: p(
                    obj.__icon__, size=obj.icon_size, pos=obj.pos
                ),
                "z_order": 1000,
                "on_update": self.update_pos_and_aspect,
            }
        )
        return gui_repr

    @abstractattribute
    def icon_size(self):
        pass

    @staticmethod
    def update_pos_and_aspect(item, obj):
        item.setPos(*obj.pos)
        item.setRotation(obj.aspect)
        item.setScale(obj.scale)


class MovingAgentWithGPS(AgentWithGPSMixin, MovingAgent):
    """Base class for MovingAgent with GPS coordinates.

    Adds navigate to GPS functionality, and also serves as an example
    for how to inherit the GPS coordinates by any desired class.
    """

    def __init__(self, unique_id, model, pos: Position, autopopulate: bool):
        super().__init__(
            unique_id=unique_id,
            model=model,
            pos=pos,
            autopopulate=autopopulate,
        )
        self._aspect = 0

    def navigate_to_gps(
        self, destination_gps: LatLon, velocity: float
    ) -> TaskStatus:
        """Navigate to a gps coordinate with ``velocity``."""
        distance = velocity * self.model.simulation.time_step.total_seconds()
        if self.distance_gps(self.gps_coords, destination_gps) <= distance:
            self.gps_coords = destination_gps
            return TaskStatus.COMPLETE
        # Get initial bearing (azimuth)
        bearing = bearing_from_coords(self.gps_coords, destination_gps)
        new_lng, new_lat, _ = GEODESIC.fwd(
            self.gps_coords[1],
            self.gps_coords[0],
            bearing,
            distance,
            radians=False,
        )
        self.gps_coords = new_lat, new_lng
        # Convert from north relative to east relative heading
        self.aspect = bearing
        return TaskStatus.IN_PROGRESS

    def navigate_to(self, pos: Position, velocity: float) -> TaskStatus:
        """Override `navigate_to` method to use GPS coordinates."""
        destination_gps = pos_to_gps(pos, self.top_left_bounds)
        return self.navigate_to_gps(destination_gps, velocity)

    def move_forward(self, distance: float) -> None:
        """Move agent forward by a given distance along its heading."""
        new_lng, new_lat, _ = GEODESIC.fwd(
            self.gps_coords[1],
            self.gps_coords[0],
            self.aspect,
            distance,
            radians=False,
        )
        self.gps_coords = new_lat, new_lng

    @property
    def aspect(self):
        """Override aspect method to not rely on `heading`."""
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        """Allow `aspect` to be set."""
        self._aspect = value


class StaticAgent(Agent):
    """Base class for static agents.

    Extends :py:class:`Agent` to add graphical representation
    """

    def __init__(
        self,
        unique_id,
        model,
        autopopulate=False,
    ):
        super().__init__(unique_id, model, autopopulate=autopopulate)

    def __gui_repr__(self):
        """Implements GUI representation protocol."""
        gui_repr = super().__gui_repr__()
        gui_repr.update(
            {
                "on_init": lambda p, obj: p(
                    obj.__icon__, size=self.icon_size, pos=obj.pos
                ),
                "z_order": 1001,
            }
        )
        return gui_repr

    @property
    def icon_size(self):
        """Returns adaptive icon size."""
        size = self.model.simulation.environment.dimensions[0] / 70
        return (size, size)


class StaticAgentWithGPS(AgentWithGPSMixin, StaticAgent):
    """Base class for a StaticAgent with GPS coordinates."""
