# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import Enum, auto, unique

import numpy as np

from sosid.abstract import ComponentWritable
from sosid.model.abm.agent import MovingAgentWithGPS
from sosid.model.abm.task import TaskStatus
from sosid.model.abm.trajectory import (
    AircraftProfileParameters,
    FlightState,
    StraightTrajectory,
)
from sosid.model.transform import GEODESIC, Position, pos_to_gps
from sosid.util.abc import abstractinterface


@unique
class PayloadStatus(Enum):
    """Payload states."""

    NONE = auto()
    ONBOARD = auto()


@unique
class DestinationType(Enum):
    """Destination types."""

    BASE = auto()
    FIRE = auto()
    WATER = auto()


class StraightTrajectoryFollower(ComponentWritable):
    """Controller for following a straight trajectory.

    This class is an controller for an agent to follow a straight
    trajectory. It does so prioritizing the altitude. This implies
    modeling circular climb and descent manoeuvres if required to reach
    a certain altitude in a short distance.
    """

    __slots__ = (
        "_destination_pos",
        "_distances_to_end",
        "_navigating_to_idx",
        "_trajectory",
    )
    _supported_states: frozenset[FlightState] = frozenset(
        {
            FlightState.CRUISE,
            FlightState.CRUISE_CLIMB,
            FlightState.CRUISE_DESCENT,
        }
    )

    def __init__(self, agent: MovingAgentWithGPS) -> None:
        super().__init__(agent)
        self._trajectory: StraightTrajectory | None = None
        self._navigating_to_idx: int = 0
        self._distances_to_end: np.ndarray[float] | None = None
        self._destination_pos: Position | None = None
        self._agent = agent

    @property
    @abstractinterface
    def current_mission_time(self) -> datetime:
        """The current simulation mission time."""
        return self._agent.current_mission_time

    @property
    @abstractinterface
    def profile_parameters(self) -> AircraftProfileParameters:
        """Aircraft profile parameters."""

    @property
    @abstractinterface
    def time_step(self) -> timedelta:
        """Time step of the simulation."""

    @property
    @abstractinterface
    def gps_coords(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the agent."""

    @gps_coords.setter
    def gps_coords(self, gps: np.ndarray[np.float64]) -> None:
        """Set GPS coordinates of the agent."""

    @property
    @abstractinterface
    def altitude(self) -> float:
        """Altitude of the agent."""

    @altitude.setter
    def altitude(self, altitude: float) -> None:
        """Set altitude of the agent."""

    @property
    @abstractinterface
    def aspect(self) -> float:
        """Aspect of the agent (azimuth)."""

    @aspect.setter
    def aspect(self, aspect: float) -> None:
        """Set aspect of the agent."""

    @property
    @abstractinterface
    def flight_state(self) -> FlightState:
        """Flight state of the agent."""

    @flight_state.setter
    def flight_state(self, state: FlightState) -> None:
        """Set flight state of the agent."""

    @abstractinterface
    def distance_gps(
        self, gps1: np.ndarray[np.float64], gps2: np.ndarray[np.float64]
    ) -> float:
        """Calculate distance between two GPS coordinates."""

    @abstractinterface
    def move_forward(self, distance: float) -> None:
        """Move agent forward by a given distance allong its heading."""

    @property
    @abstractinterface
    def top_left_bounds(self) -> Position:
        """Top left bounds of the simulation area."""

    @property
    def trajectory(self) -> StraightTrajectory | None:
        """Get the trajectory to follow."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory: StraightTrajectory | None) -> None:
        """Set the trajectory to follow."""
        if trajectory is None:
            self.abort()
            return
        if not self._supported_states.issuperset(
            np.unique(trajectory.flight_states)
        ):
            msg = (
                "Only trajectories with CRUISE, CRUISE_CLIMB, CRUISE_DESCENT"
                "are allowed."
            )
            raise ValueError(msg)
        self._trajectory = trajectory
        self._distances_to_end = np.insert(
            np.cumsum(trajectory.horizontal_distances[::-1]), 0, 0.0
        )[::-1]
        self._destination_pos = pos_to_gps(
            trajectory.gps_end, self.top_left_bounds
        )

    @property
    def destination(self) -> Position | None:
        """Destination of the trajectory."""
        return self._destination_pos

    @property
    def destination_gps(self) -> np.ndarray[np.float64] | None:
        """GPS coordinates of the destination."""
        if self._destination_pos is None:
            return None
        return self._trajectory.gps_end

    @property
    def destination_altitude(self) -> float | None:
        """Altitude of the destination."""
        if self._trajectory is None:
            return None
        return self._trajectory.altitudes[-1]

    def navigate(self) -> TaskStatus:
        """Navigate along set trajectory."""
        gps, alt, state = self.gps_coords, self.altitude, self.flight_state
        specs = self.profile_parameters
        dists = self._distances_to_end
        alts = self.trajectory.altitudes
        n_points = len(self.trajectory)
        self.aspect, _, dist_to_go = GEODESIC.inv(
            *gps[::-1], *self.trajectory.gps_end[::-1]
        )
        idx = dists.size - np.searchsorted(dists[::-1], dist_to_go)
        if idx == 0:
            self.trajectory.start_datetime = self.current_mission_time
        time_left = self.time_step.total_seconds()
        while idx < n_points:
            next_alt = alts[idx]
            dist_to_next = dist_to_go - dists[idx]
            if np.isclose(dist_to_next, 0.0):
                idx += 1
                continue
            if np.isclose(alt, next_alt):
                state = FlightState.CRUISE
                ground_speed = specs.cruise_speed
                alt = next_alt
            elif alt < next_alt:
                state = FlightState.CRUISE_CLIMB
                desired_slope = (next_alt - alt) / dist_to_next
                if desired_slope > specs.cruise_climb_slope:
                    # Circular climb.
                    ground_speed = (
                        specs.cruise_climb_ground_speed
                        * specs.cruise_climb_slope
                        / desired_slope
                    )
                else:
                    ground_speed = specs.cruise_climb_ground_speed
                alt = min(
                    next_alt,
                    alt + specs.cruise_climb_rate * time_left,
                )
            else:  # alt > next_alt
                state = FlightState.CRUISE_DESCENT
                desired_slope = (alt - next_alt) / dist_to_next
                if desired_slope > specs.cruise_descent_slope:
                    # Circular descent.
                    ground_speed = (
                        specs.cruise_descent_ground_speed
                        * specs.cruise_descent_slope
                        / desired_slope
                    )
                else:
                    ground_speed = specs.cruise_descent_ground_speed
                alt = max(
                    next_alt,
                    alt - specs.cruise_descent_rate * time_left,
                )
            time_to_next = dist_to_next / ground_speed
            if time_to_next > time_left:
                break
            time_left -= time_to_next
            dist_to_go = dists[idx]
            idx += 1

        self.flight_state = state
        self.altitude = alt
        if idx == n_points:
            self.gps_coords = self.trajectory.gps_end
            self.trajectory.end_datetime = self.current_mission_time
            return TaskStatus.COMPLETE
        self.move_forward(ground_speed * time_left)
        return TaskStatus.IN_PROGRESS

    def segment_etas(self) -> Iterable[tuple[FlightState, float]]:
        """Estimate time left for navigating along the path."""
        specs, traj = self.profile_parameters, self.trajectory
        dist_to_go = self.distance_gps(self.gps_coords, traj.gps_end)
        i = np.searchsorted(self._distances_to_end, dist_to_go)
        dist_to_next = self._distances_to_end[i] - dist_to_go
        if np.isclose(self.altitude, traj.altitudes[i]):
            state = FlightState.CRUISE
            initial_duration = dist_to_next / specs.cruise_speed
        elif self.altitude < traj.altitudes[i]:
            state = FlightState.CRUISE_CLIMB
            initial_duration = max(
                dist_to_next / specs.cruise_climb_ground_speed,
                (traj.altitudes[i] - self.altitude) / specs.cruise_climb_rate,
            )
        else:  # self.altitude > traj.altitudes[idx]
            state = FlightState.CRUISE_DESCENT
            initial_duration = max(
                dist_to_next / specs.cruise_descent_ground_speed,
                (self.altitude - traj.altitudes[i])
                / specs.cruise_descent_rate,
            )
        idxs = traj.state_start_indices
        idxs = idxs[idxs <= i]
        durations = np.diff(
            np.concatenate((traj.timestamps[idxs], [traj.timestamps[-1]]))
        )
        return zip(
            (state, *traj.flight_states[idxs]),
            (initial_duration, *durations),
            strict=True,
        )

    def abort(self) -> None:
        """Abort the current task."""
        self._trajectory = None
        self._navigating_to_idx = 0
        self._distances_to_end = None
        self._destination_pos = None
