# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from enum import IntEnum, auto, unique
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from scipy.interpolate import make_interp_spline
from typing_extensions import Self

from sosid.model.transform import GEODESIC, bearing_from_coords
from sosid.typedef import LatLon
from sosid.util.validation import to_frozen_array

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "AircraftProfileParameters",
    "BaseTrajectory",
    "FlightState",
    "StraightTrajectory",
    "WaypointTrajectory",
    "generate_straight_trajectory",
]


@unique
class FlightState(IntEnum):
    """Possible flight states of an aircraft."""

    IDLE = auto()
    TAXI_OUT = auto()
    TAKEOFF = auto()
    TRANSITION = auto()
    CRUISE_CLIMB = auto()
    CRUISE = auto()
    CRUISE_DESCENT = auto()
    RETRANSITION = auto()
    LANDING = auto()
    TAXI_IN = auto()
    ENERGIZE = auto()
    LOITER = auto()
    HOVER = auto()


ON_GROUND_STATES = frozenset(
    [
        FlightState.IDLE,
        FlightState.TAXI_IN,
        FlightState.TAXI_OUT,
        FlightState.ENERGIZE,
    ]
)

IN_AIR_FLIGHT_STATES = frozenset(
    [state for state in FlightState if state not in ON_GROUND_STATES]
)


def _concatenate_trajectories(
    start_times: Sequence[datetime | None],
    timestamp_arrays: Sequence[np.ndarray[np.float64]],
    intermediate_arrays: Sequence[Sequence[np.ndarray[object]]],
    waypoint_arrays: Sequence[Sequence[np.ndarray[object]]],
) -> tuple[
    np.ndarray[np.float64],
    Sequence[np.ndarray[object]],
    Sequence[np.ndarray[object]],
]:
    """Helper function to concatenate trajectories.

    Concatenates multiple trajectories into a single trajectory. If all
    trajectories don't have a start time, then the start of each
    trajectory is assumed to be the end of the previous trajectory. If
    all trajectories have a start time, then trajectories are offsetted
    w.r.t. each other based on the start time. The flight state in
    between two trajectories is assumed to be the final state of the
    previous trajectory. If the start time matches the previous
    trajectory's end time, then the checkpoint of the previous
    trajectory is used.

    Args:
        start_times: Start times of the trajectories. If a start time is
            not defined, provide `None`.
        timestamp_arrays: Timestamps of the trajectories.
        intermediate_arrays: Sequence of arrays of intermediate states
            to concatenate.
            E.g. ``[[vel_arr1, vel_arr2], [acc_arr1, acc_arr2]]``
        waypoint_arrays: Sequence of arrays of waypoints to concatenate.
            E.g. ``[[gps_arr1, gps_arr2], [alt_arr1, alt_arr2]]``
    """
    if len(start_times) == 0:
        raise ValueError(
            "No trajectories to concatenate. Provide at least one trajectory."
        )
    n_ints = len(intermediate_arrays)
    n_wps = len(waypoint_arrays)

    nones = [i for i in range(len(start_times)) if start_times[i] is None]
    if len(nones) == 0:  # Start times are all defined.
        start_datetime = start_times[0]
        timestamps_to_concat = [timestamp_arrays[0]]
        int_arrs_to_concat = [
            [intermediate_arrays[i][0]] for i in range(n_ints)
        ]
        wp_arrs_to_concat = [[waypoint_arrays[i][0]] for i in range(n_wps)]
        for p, c in pairwise(range(len(start_times))):
            offset = (start_times[c] - start_datetime).total_seconds()
            previous_end_time = start_times[p] + timedelta(
                seconds=timestamp_arrays[p][-1]
            )
            if previous_end_time == start_times[c]:
                # Touching paths.
                timestamps_to_concat.append(timestamp_arrays[c][1:] + offset)
                for i, int_trajs in enumerate(intermediate_arrays):
                    int_arrs_to_concat[i].append(int_trajs[c])
                for i, wp_trajs in enumerate(waypoint_arrays):
                    wp_arrs_to_concat[i].append(wp_trajs[c][1:])
            elif start_times[c] < start_times[p]:
                raise ValueError("Trajectories are not in order.")
            elif previous_end_time > start_times[c]:
                raise ValueError("Trajectories overlap.")
            else:  # Disjoint paths.
                timestamps_to_concat.append(timestamp_arrays[c] + offset)
                for i, int_trajs in enumerate(intermediate_arrays):
                    int_arrs_to_concat[i].append(  # State in between paths.
                        [int_arrs_to_concat[i][-1][-1]]
                    )
                    int_arrs_to_concat[i].append(int_trajs[c])
                for i, wp_trajs in enumerate(waypoint_arrays):
                    wp_arrs_to_concat[i].append(wp_trajs[c])
    elif len(nones) == len(start_times):
        # No start times are defined.
        # All paths are assumed to be touching.
        start_datetime = None
        offsets = np.cumsum([tmps[-1] for tmps in timestamp_arrays[:-1]])
        timestamps_to_concat = [timestamp_arrays[0]] + [
            offset + tmps[1:]
            for tmps, offset in zip(timestamp_arrays[1:], offsets)
        ]
        int_arrs_to_concat = intermediate_arrays
        wp_arrs_to_concat = [
            [wp_traj[:-1] for wp_traj in wp_trajs[:-1]] + [wp_trajs[-1]]
            for wp_trajs in waypoint_arrays
        ]
    else:
        raise ValueError(
            "Some trajectories have start times while others do not."
        )
    return (
        start_datetime,
        np.concatenate(timestamps_to_concat),
        tuple(
            np.concatenate(int_to_concat, axis=0)
            for int_to_concat in int_arrs_to_concat
        ),
        tuple(
            np.concatenate(other_to_concat, axis=0)
            for other_to_concat in wp_arrs_to_concat
        ),
    )


class BaseTrajectory(metaclass=ABCMeta):
    """Base class for aircraft trajectories.

    Generally, a trajectory is defined from departure to arrival at the
    gate unless specified differently.

    Attributes:
        start_datetime: Start datetime of the trajectory.
        timestamps: Timestamps of the trajectory in seconds starting
            from 0.
        flight_states: Flight states of the aircraft between timestamps.
    """

    # Slots is not compatible with cached_property.

    def __init__(
        self,
        *,
        start_datetime: datetime | None,
        timestamps: Sequence[float],
        flight_states: Sequence[FlightState],
    ) -> None:
        self.start_datetime = start_datetime
        self._timestamps = to_frozen_array(np.float64)(timestamps)
        self._flight_states = to_frozen_array(FlightState)(flight_states)
        n = self._timestamps.shape[0]
        msgs: list[str] = []
        if self._timestamps.shape != (n,):
            msgs.append("Timestamps must be a 1D array.")
        if n < 2:
            msgs.append("Trajectory must have at least two timestamps.")
        if self._flight_states.shape != (n - 1,):
            msgs.append(
                f"Flight states must be of shape {(n - 1,)} but is"
                f" of shape {self._flight_states.shape}."
            )
        if n != 0 and self._timestamps[0] != 0:
            msgs.append(
                f"First timestamp is {self._timestamps[0]!r} must be 0."
            )
        if msgs:
            raise ValueError("\n".join(msgs))

    @property
    @abstractmethod
    def gps_start(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the starting point."""

    @property
    @abstractmethod
    def gps_end(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the ending point."""

    @property
    def timestamps(self) -> np.ndarray[np.float64]:
        """Timestamps of the trajectory in seconds."""
        return self._timestamps

    @property
    def flight_states(self) -> np.ndarray[FlightState]:
        """Flight states of the aircraft between timestamps."""
        return self._flight_states

    @cached_property
    def state_start_indices(self) -> np.ndarray[np.int64]:
        """Start indices of each flight state.

        E.g. ``[0, 1, 1, 2, 2, 2, 3, 3, 4, 4]`` -> ``[0, 1, 3, 6, 8]``
        """
        if not self.flight_states.size:
            return to_frozen_array(np.int64)([])
        return to_frozen_array(np.int64)(
            np.insert(
                np.where(self.flight_states[1:] != self.flight_states[:-1])[0]
                + 1,
                0,
                0,
            )
        )

    @cached_property
    def state_end_indices(self) -> np.ndarray[np.int64]:
        """End indices of each flight state.

        E.g. ``[0, 1, 1, 2, 2, 2, 3, 3, 4, 4]`` -> ``[0, 2, 5, 7, 9]``
        """
        if not self.flight_states.size:
            return to_frozen_array(np.int64)([])
        starts = self.state_start_indices
        return to_frozen_array(np.int64)(
            np.append(
                starts[:-1] + np.diff(starts) - 1, self.flight_states.size - 1
            )
        )

    @cached_property
    def total_duration(self) -> float:
        """Total duration of the trajectory."""
        if self.timestamps.size == 0:
            return 0.0
        return self.timestamps[-1]

    @cached_property
    def end_datetime(self) -> datetime | None:
        """End datetime of the trajectory."""
        if self.start_datetime is None:
            return None
        return self.start_datetime + timedelta(seconds=self.total_duration)

    def first_idx_of_state(
        self, *flight_states: tuple[FlightState, ...]
    ) -> int | None:
        """Get the first index of given flight states."""
        start_indices = np.where(
            np.isin(
                self.flight_states[self.state_start_indices],
                flight_states,
                assume_unique=True,
            )
        )[0]
        if start_indices.size == 0:
            return None
        return self.state_start_indices[start_indices[0]]

    def last_idx_of_state(
        self, *flight_states: tuple[FlightState, ...]
    ) -> int | None:
        """Get the last index of given flight states."""
        end_indices = np.where(
            np.isin(
                self.flight_states[self.state_end_indices],
                flight_states,
                assume_unique=True,
            )
        )[0]
        if end_indices.size == 0:
            return None
        return self.state_end_indices[end_indices[-1]]

    def flight_state_at_datetime(self, time: datetime) -> FlightState:
        """Get flight state at a given time."""
        if self.start_datetime is None:
            raise ValueError("Trajectory has no start time.")
        return self.flight_states[
            np.searchsorted(
                self.timestamps,
                (time - self.start_datetime).total_seconds(),
                side="left",
            )
        ]

    @classmethod
    def concatenate(cls, trajectories: Sequence[Self]) -> Self:
        """Concatenate multiple trajectories."""
        start_datetime, timestamps, (flight_states,), _ = (
            _concatenate_trajectories(
                [path.start_datetime for path in trajectories],
                [traj.timestamps for traj in trajectories],
                ([traj.flight_states for traj in trajectories],),
                (),
            )
        )
        return cls(
            start_datetime=start_datetime,
            timestamps=timestamps,
            flight_states=flight_states,
        )

    def datetime_at_idx(self, idx: int) -> datetime | None:
        """Get the datetime at a given index."""
        if self.start_datetime is None:
            return None
        return self.start_datetime + timedelta(seconds=self.timestamps[idx])

    def slice_by_idx(self, start_idx: int, end_ind: int) -> Self:
        """Slice the trajectory by indices."""
        return self.__class__(
            start_datetime=self.datetime_at_idx(start_idx),
            timestamps=(
                self.timestamps[start_idx:end_ind] - self.timestamps[start_idx]
            ),
            flight_states=self.flight_states[start_idx : end_ind - 1],
        )

    def get_state_duration_segements(
        self, *, start_idx: int | None = None, end_idx: int | None = None
    ) -> Iterable[tuple[FlightState, float]]:
        """Get segments of flight states and their durations.

        Materialised rather than a lazy ``zip``: callers estimate
        propellant from these segments, and a one-shot iterator silently
        yields nothing (i.e. zero propellant) on a second pass.
        """
        idxs = self.state_start_indices
        if start_idx is not None:
            idxs = idxs[idxs >= start_idx]
            if idxs.size == 0:
                return ()
            if idxs[0] != start_idx:
                idxs = np.insert(idxs, 0, start_idx)
        if end_idx is not None:
            idxs = idxs[idxs <= end_idx]
            if idxs.size == 0:
                return ()
            if idxs[-1] != end_idx:
                idxs = np.append(idxs, end_idx)
        timestamps = np.concatenate(
            (self.timestamps[idxs], [self.timestamps[-1]])
        )
        return list(zip(self.flight_states[idxs], np.diff(timestamps)))

    def __len__(self) -> int:
        """Number of points in the trajectory."""
        return self.timestamps.size


class StraightTrajectory(BaseTrajectory):
    """Point-to-point straight trajectory for an aircraft.

    Generally, a trajectory is defined from departure to arrival at the
    gate unless specified differently. This trajectory is a straight
    line between two positions denoting the timestamps, distances,
    altitudes, and flight states.

    Attributes:
        start_datetime: Start datetime of the trajectory.
        timestamps: Timestamps of the trajectory in seconds starting
            from 0.
        flight_states: Flight states of the aircraft between timestamps.
        horizontal_distances: Horizontal distance between timestamps.
        altitudes: Altitudes of the aircraft between timestamps.
        gps_start: GPS coordinates of the starting point.
        gps_end: GPS coordinates of the ending point.
    """

    def __init__(
        self,
        *,
        start_datetime: datetime | None,
        timestamps: Sequence[float],
        flight_states: Sequence[FlightState],
        horizontal_distances: Sequence[np.float64],
        altitudes: Sequence[float],
        gps_start: Sequence[float],
        gps_end: Sequence[float],
    ) -> None:
        super().__init__(
            start_datetime=start_datetime,
            timestamps=timestamps,
            flight_states=flight_states,
        )
        self._horizontal_distances = to_frozen_array(np.float64)(
            horizontal_distances
        )
        self._altitudes = to_frozen_array(np.float64)(altitudes)
        self._gps_start = to_frozen_array(np.float64)(gps_start)
        self._gps_end = to_frozen_array(np.float64)(gps_end)
        n = self._timestamps.shape[0]
        msgs: list[str] = []
        for name, array, expected_shape in (
            ("Horizontal distances", self._horizontal_distances, (n - 1,)),
            ("Altitudes", self._altitudes, (n,)),
            ("GPS start", self._gps_start, (2,)),
            ("GPS end", self._gps_end, (2,)),
        ):
            if array.shape != expected_shape:
                msgs.append(
                    f"{name} must be of shape {expected_shape} but is of shape"
                    f" {array.shape}."
                )
        if msgs:
            raise ValueError("\n".join(msgs))

    @property
    def horizontal_distances(self) -> np.ndarray[np.float64]:
        """Horizontal distance traveled between timestamps."""
        return self._horizontal_distances

    @property
    def altitudes(self) -> np.ndarray[np.float64]:
        """Altitudes of the aircraft at each timestamp."""
        return self._altitudes

    @property
    def gps_start(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the starting point."""
        return self._gps_start

    @property
    def gps_end(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the ending point."""
        return self._gps_end

    @cached_property
    def _altitude_at_time(
        self,
    ) -> Callable[[float | np.ndarray[np.float64]], np.ndarray[np.float64]]:
        return make_interp_spline(self._timestamps, self._altitudes, k=1)

    @classmethod
    def concatenate(cls, trajectories: Sequence[Self]) -> Self:
        """Concatenate multiple trajectories."""
        for prev_traj, traj in pairwise(trajectories):
            if not np.allclose(prev_traj.gps_end, traj.gps_start):
                raise ValueError(
                    "Trajectories are not connected. GPS end of previous"
                    " trajectory must match GPS start of next trajectory."
                )
            if prev_traj.end_datetime != traj.start_datetime:
                raise ValueError(
                    "Trajectories are not connected. End datetime of previous"
                    " trajectory must match start datetime of next trajectory."
                )
        (
            start_datetime,
            timestamps,
            (flight_states, horizontal_distances),
            (altitudes,),
        ) = _concatenate_trajectories(
            [path.start_datetime for path in trajectories],
            [traj.timestamps for traj in trajectories],
            (
                [traj.flight_states for traj in trajectories],
                [traj.horizontal_distances for traj in trajectories],
            ),
            ([traj.altitudes for traj in trajectories],),
        )
        return cls(
            start_datetime=start_datetime,
            timestamps=timestamps,
            flight_states=flight_states,
            horizontal_distances=horizontal_distances,
            altitudes=altitudes,
            gps_start=trajectories[0].gps_start,
            gps_end=trajectories[-1].gps_end,
        )

    def slice_by_idx(self, start_idx: int, end_ind: int) -> Self:
        """Slice the trajectory by indices."""

        def move_gps(
            gps_start: tuple[float, float], bearing: float, distance: float
        ) -> tuple[float, float]:
            return GEODESIC.fwd(*gps_start[::-1], bearing, distance)[:2][::-1]

        dist_start = self.horizontal_distances[:start_idx].sum()
        dist_end = (
            dist_start + self.horizontal_distances[start_idx:end_ind].sum()
        )
        bearing = bearing_from_coords(self.gps_start, self.gps_end)
        gps_start = move_gps(self.gps_start, bearing, dist_start)
        gps_end = move_gps(self.gps_start, bearing, dist_end)
        return self.__class__(
            start_datetime=self.datetime_at_idx(start_idx),
            timestamps=(
                self.timestamps[start_idx:end_ind] - self.timestamps[start_idx]
            ),
            flight_states=self.flight_states[start_idx : end_ind - 1],
            horizontal_distances=(
                self.horizontal_distances[start_idx : end_ind - 1]
            ),
            altitudes=self.altitudes[start_idx:end_ind],
            gps_start=gps_start,
            gps_end=gps_end,
        )

    def altitude_at_datetime(self, time: datetime) -> np.ndarray[np.float64]:
        """Get the altitude of the aircraft at a given time."""
        if self.start_datetime is None:
            raise ValueError("Trajectory has no start time.")
        return self._altitude_at_time(
            (time - self.start_datetime).total_seconds()
        )


class WaypointTrajectory(BaseTrajectory):
    """Waypoint trajectory for an aircraft.

    Generally, a trajectory is defined from departure to arrival at the
    gate unless specified differently. A waypoint trajectory describes
    the 4D trajectory of an aircraft using GPS coordinates and
    altitudes.

    Attributes:
        start_datetime: Start datetime of the trajectory.
        timestamps: Timestamps of the trajectory in seconds starting
            from 0.
        gps: Latitude, longitude coordinates at every timestamp.
        altitudes: Altitude at every timestamp.
        flight_states: Flight states of the aircraft between timestamps.
    """

    def __init__(
        self,
        *,
        start_datetime: datetime | None,
        timestamps: Sequence[float],
        gps: Sequence[np.ndarray[np.float64]],
        altitudes: Sequence[float],
        flight_states: Sequence[FlightState],
    ) -> None:
        super().__init__(
            start_datetime=start_datetime,
            timestamps=timestamps,
            flight_states=flight_states,
        )
        self._gps = to_frozen_array(np.float64, zero_shape=(0, 2))(gps)
        self._altitudes = to_frozen_array(np.float64)(altitudes)
        n = self._timestamps.shape[0]
        msgs: list[str] = []
        for name, array, expected_shape in (
            ("GPS", self._gps, (n, 2)),
            ("Altitudes", self._altitudes, (n,)),
        ):
            if array.shape != expected_shape:
                msgs.append(
                    f"{name} must be of shape {expected_shape} but is of shape"
                    f" {array.shape}."
                )
        if msgs:
            raise ValueError("\n".join(msgs))

    @property
    def gps(self) -> np.ndarray[np.float64]:
        """Latitude, longitude coordinates at every timestamp."""
        return self._gps

    @property
    def altitudes(self) -> np.ndarray[np.float64]:
        """Altitude at every timestamp."""
        return self._altitudes

    @property
    def gps_start(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the starting point."""
        return self._gps[0]

    @property
    def gps_end(self) -> np.ndarray[np.float64]:
        """GPS coordinates of the ending point."""
        return self._gps[-1]

    @cached_property
    def horizontal_distances(self) -> float:
        """Total horizontal distance."""
        return GEODESIC.line_lengths(*self._gps.T[::-1])

    @cached_property
    def _altitude_at_time(
        self,
    ) -> Callable[[float | np.ndarray[np.float64]], np.ndarray[np.float64]]:
        return make_interp_spline(self._timestamps, self._altitudes, k=1)

    @classmethod
    def concatenate(cls, trajectories: Sequence[Self]) -> Self:
        """Concatenate multiple trajectories."""
        (
            start_datetime,
            timestamps,
            (flight_states,),
            (gps, altitudes),
        ) = _concatenate_trajectories(
            [path.start_datetime for path in trajectories],
            [traj.timestamps for traj in trajectories],
            ([traj.flight_states for traj in trajectories],),
            (
                [traj.gps for traj in trajectories],
                [traj.altitudes for traj in trajectories],
            ),
        )
        return cls(
            start_datetime=start_datetime,
            timestamps=timestamps,
            gps=gps,
            altitudes=altitudes,
            flight_states=flight_states,
        )

    def slice_by_idx(self, start_idx: int, end_ind: int) -> Self:
        """Slice the trajectory by indices."""
        return self.__class__(
            start_datetime=self.datetime_at_idx(start_idx),
            timestamps=(
                self.timestamps[start_idx:end_ind] - self.timestamps[start_idx]
            ),
            gps=self._gps[start_idx:end_ind],
            altitudes=self.altitudes[start_idx:end_ind],
            flight_states=self.flight_states[start_idx : end_ind - 1],
        )

    def altitude_at_datetime(self, time: datetime) -> np.ndarray[np.float64]:
        """Get the altitude of the aircraft at a given time."""
        if self.start_datetime is None:
            raise ValueError("Trajectory has no start time.")
        return self._altitude_at_time(
            (time - self.start_datetime).total_seconds()
        )

    def __len__(self) -> int:
        """Number of points in the trajectory."""
        return self.timestamps.size


class AircraftProfileParameters(BaseModel):
    r"""Properties of an aircraft regarding its trajectory.

    This class contains the definition of the generic profile properties
    for aircraft modelling in the toolkit. It is to be used for CTOL,
    VTOL and STOL aircraft.

    A typical trajectory of an aircraft can be divided into the
    following phases:
    ```
                     cruise
             climb  ________ descent
              -----          -----
     takeoff /                    \ landing
       _____/                      \_____
    taxi out                         taxi in
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    taxi_out_duration: NonNegativeFloat
    taxi_in_duration: NonNegativeFloat

    transition_duration: NonNegativeFloat
    retransition_duration: NonNegativeFloat

    takeoff_altitude: NonNegativeFloat
    takeoff_climb_rate: NonNegativeFloat
    takeoff_ground_speed: NonNegativeFloat

    cruise_climb_rate: NonNegativeFloat
    cruise_climb_ground_speed: NonNegativeFloat

    cruise_altitude: NonNegativeFloat
    cruise_speed: NonNegativeFloat

    cruise_descent_rate: NonNegativeFloat
    cruise_descent_ground_speed: NonNegativeFloat

    landing_altitude: NonNegativeFloat
    landing_descent_rate: NonNegativeFloat
    landing_ground_speed: NonNegativeFloat

    loiter_speed: NonNegativeFloat

    @cached_property
    def can_transition(self) -> bool:
        """Whether the aircraft can transition."""
        return self.transition_duration > 0 or self.retransition_duration > 0

    @cached_property
    def takeoff_duration(self) -> float:
        """Duration of the takeoff phase."""
        return self.takeoff_altitude / self.takeoff_climb_rate

    @cached_property
    def takeoff_ground_distance(self) -> float:
        """Horizontal distance covered during takeoff phase."""
        return self.takeoff_ground_speed * self.takeoff_duration

    @cached_property
    def takeoff_slope(self) -> float:
        """Slope of the takeoff phase."""
        return self.takeoff_climb_rate / self.takeoff_ground_speed

    @cached_property
    def cruise_climb_duration(self) -> float:
        """Duration of the cruise climb phase."""
        return (
            self.cruise_altitude - self.takeoff_altitude
        ) / self.cruise_climb_rate

    @cached_property
    def cruise_climb_distance(self) -> float:
        """Horizontal distance covered during cruise climb phase."""
        return self.cruise_climb_ground_speed * self.cruise_climb_duration

    @cached_property
    def cruise_climb_slope(self) -> float:
        """Slope of the cruise climb phase."""
        return self.cruise_climb_rate / self.cruise_climb_ground_speed

    @cached_property
    def cruise_descent_duration(self) -> float:
        """Duration of the cruise descent phase."""
        return (
            self.cruise_altitude - self.landing_altitude
        ) / self.cruise_descent_rate

    @cached_property
    def cruise_descent_distance(self) -> float:
        """Horizontal distance covered during cruise descent phase."""
        return self.cruise_descent_ground_speed * self.cruise_descent_duration

    @cached_property
    def cruise_descent_slope(self) -> float:
        """Slope of the cruise descent phase."""
        return self.cruise_descent_rate / self.cruise_descent_ground_speed

    @cached_property
    def landing_duration(self) -> float:
        """Duration of the landing phase."""
        return self.landing_altitude / self.landing_descent_rate

    @cached_property
    def landing_ground_distance(self) -> float:
        """Horizontal distance covered during landing phase."""
        return self.landing_ground_speed * self.landing_duration

    @cached_property
    def landing_slope(self) -> float:
        """Slope of the landing phase."""
        return self.landing_descent_rate / self.landing_ground_speed

    @cached_property
    def _horizontal_speed_map(self) -> dict[FlightState, float]:
        return {
            FlightState.IDLE: 0.0,
            FlightState.ENERGIZE: 0.0,
            FlightState.TAXI_IN: 0.0,
            FlightState.TAXI_OUT: 0.0,
            FlightState.TRANSITION: 0.0,
            FlightState.RETRANSITION: 0.0,
            FlightState.HOVER: 0.0,
            FlightState.TAKEOFF: self.takeoff_ground_speed,
            FlightState.CRUISE_CLIMB: self.cruise_climb_ground_speed,
            FlightState.CRUISE: self.cruise_speed,
            FlightState.CRUISE_DESCENT: self.cruise_descent_ground_speed,
            FlightState.LANDING: self.landing_ground_speed,
            FlightState.LOITER: self.loiter_speed,
        }

    @cached_property
    def _vertical_speed_map(self) -> dict[FlightState, float]:
        return {
            FlightState.IDLE: 0.0,
            FlightState.ENERGIZE: 0.0,
            FlightState.TAXI_IN: 0.0,
            FlightState.TAXI_OUT: 0.0,
            FlightState.TRANSITION: 0.0,
            FlightState.RETRANSITION: 0.0,
            FlightState.HOVER: 0.0,
            FlightState.TAKEOFF: self.takeoff_climb_rate,
            FlightState.CRUISE_CLIMB: self.cruise_climb_rate,
            FlightState.CRUISE: 0.0,
            FlightState.CRUISE_DESCENT: -self.cruise_descent_rate,
            FlightState.LANDING: -self.landing_descent_rate,
            FlightState.LOITER: 0.0,
        }

    def segment_horizontal_speed(self, flight_state: FlightState) -> float:
        """Get the horizontal speed for a given flight state."""
        return self._horizontal_speed_map[flight_state]

    def segment_vertical_speed(self, flight_state: FlightState) -> float:
        """Get the vertical speed for a given flight state."""
        return self._vertical_speed_map[flight_state]


def generate_straight_trajectory(
    profile: AircraftProfileParameters,
    *,
    start_datetime: datetime | None = None,
    gps_start: LatLon,
    gps_end: LatLon,
    altitude_start: float,
    altitude_end: float,
    elevation_start: float = 0.0,
    elevation_end: float = 0.0,
    include_takeoff: bool = False,
    include_landing: bool = False,
    include_taxi_out: bool = False,
    include_taxi_in: bool = False,
    loiter_time: int = 0,
) -> StraightTrajectory:
    """Create a straight trajectory between two points.

    When generating the trajectory it is possible to set different
    flags. It can happen that include takeoff is given, but the starting
    altitude of the agent is already higher than the altitude normally
    reached with takeoff. In this case the takeoff phase will be
    skipped, though the transition phase will still be included. Similar
    logic applies to the landing phase.

    Args:
        profile: Aircraft profile parameters.
        start_datetime: Start datetime of the trajectory.
        gps_start: GPS coordinates of the starting point.
        gps_end: GPS coordinates of the ending point.
        altitude_start: Altitude the agent starts at.
        altitude_end: Altitude the agent ends at.
        elevation_start: Elevation of the starting point.
        elevation_end: Elevation of the ending point.
        include_takeoff: Include takeoff phase.
        include_landing: Include landing phase.
        include_taxi_out: Include taxi out phase.
        include_taxi_in: Include taxi in phase.
    """
    atol = 1e-9  # Minimum distance of a segment.
    cruise_start_altitude = altitude_start
    cruise_end_altitude = altitude_end
    # Create new profile if a higher cruise altitude is required.
    min_cruise_altitude = max(
        elevation_start + include_takeoff * profile.takeoff_altitude,
        elevation_end + include_landing * profile.landing_altitude,
        altitude_start,
        altitude_end,
    )
    if profile.cruise_altitude < min_cruise_altitude:
        profile = profile.model_copy(
            update={"cruise_altitude": min_cruise_altitude}
        )

    distance_left = GEODESIC.line_length(
        (gps_start[1], gps_end[1]), (gps_start[0], gps_end[0])
    )
    durations: list[float] = []
    distances: list[float] = []
    altitudes: list[float] = [altitude_start]
    flight_states: list[FlightState] = []

    # Add taxi out phase.
    if include_taxi_out:
        durations.append(profile.taxi_out_duration)
        distances.append(0.0)
        altitudes.append(altitude_start)
        flight_states.append(FlightState.TAXI_OUT)

    # Account for takeoff and landing distances.
    if include_takeoff:
        desired_takeoff_altitude = elevation_start + profile.takeoff_altitude
        altitude_to_takeoff = desired_takeoff_altitude - altitude_start
        if altitude_to_takeoff > atol:
            duration = altitude_to_takeoff / profile.takeoff_climb_rate
            durations.append(duration)
            distances.append(duration * profile.takeoff_ground_speed)
            altitudes.append(elevation_start + profile.takeoff_altitude)
            flight_states.append(FlightState.TAKEOFF)
            cruise_start_altitude = desired_takeoff_altitude
            distance_left -= profile.takeoff_ground_distance
        if profile.can_transition:
            durations.append(profile.transition_duration)
            distances.append(0.0)
            altitudes.append(cruise_start_altitude)
            flight_states.append(FlightState.TRANSITION)
    add_landing = include_landing
    if include_landing:
        start_landing_altitude = elevation_end + profile.landing_altitude
        altitude_to_descent = start_landing_altitude - altitude_end
        if altitude_to_descent > atol:
            landing_duration = (
                altitude_to_descent / profile.landing_descent_rate
            )
            landing_distance = landing_duration * profile.landing_ground_speed
            cruise_end_altitude = start_landing_altitude
            distance_left -= landing_distance
        else:
            add_landing = False
    if distance_left < -atol:
        raise NotImplementedError(
            "Takeoff and landing distances exceed total distance."
        )

    # Compute cruise climb, cruise, and cruise descent phases.
    h_climb_distance = (
        profile.cruise_altitude - cruise_start_altitude
    ) / profile.cruise_climb_slope
    h_descent_distance = (
        profile.cruise_altitude - cruise_end_altitude
    ) / profile.cruise_descent_slope
    if distance_left - (h_climb_distance + h_descent_distance) < atol:
        # Cruise phase is not reached.
        #               /\
        # cruise climb /  \ cruise descent
        #             /    \
        daltitude = cruise_end_altitude - cruise_start_altitude
        temp = 1 / (profile.cruise_climb_slope + profile.cruise_descent_slope)
        climb_distance = temp * (
            profile.cruise_descent_slope * distance_left + daltitude
        )
        peak_altitude = (
            cruise_start_altitude + profile.cruise_climb_slope * climb_distance
        )
        if peak_altitude <= cruise_end_altitude:
            # Climbing in a straight line does not reach the desired
            # altitude. Circular climb is probably required.
            durations.append(daltitude / profile.cruise_climb_rate)
            distances.append(distance_left)
            altitudes.append(cruise_end_altitude)
            flight_states.append(FlightState.CRUISE_CLIMB)
        elif climb_distance <= atol:
            # Descending in a straight line does not reach the desired
            # altitude. Circular descent is probably required.
            durations.append(-daltitude / profile.cruise_descent_rate)
            distances.append(distance_left)
            altitudes.append(cruise_end_altitude)
            flight_states.append(FlightState.CRUISE_DESCENT)
        else:
            # Climbing and descending in a straight line reaches the
            # desired altitude.
            climb_duration = climb_distance / profile.cruise_climb_ground_speed
            descent_duration = (
                temp
                * (profile.cruise_climb_slope * distance_left - daltitude)
                / profile.cruise_descent_ground_speed
            )
            descent_distance = distance_left - climb_distance
            durations.extend((climb_duration, descent_duration))
            distances.extend((climb_distance, descent_distance))
            altitudes.extend((peak_altitude, cruise_end_altitude))
            flight_states.extend(
                (FlightState.CRUISE_CLIMB, FlightState.CRUISE_DESCENT)
            )
    else:
        # Cruise phase is reached.
        #               ________
        # cruise climb / cruise \ cruise descent
        #             /          \
        cruise_distance = distance_left - h_climb_distance - h_descent_distance
        if h_climb_distance > atol:
            durations.append(
                h_climb_distance / profile.cruise_climb_ground_speed
            )
            distances.append(h_climb_distance)
            altitudes.append(profile.cruise_altitude)
            flight_states.append(FlightState.CRUISE_CLIMB)
        durations.append(cruise_distance / profile.cruise_speed)
        distances.append(cruise_distance)
        altitudes.append(profile.cruise_altitude)
        flight_states.append(FlightState.CRUISE)
        if h_descent_distance > atol:
            durations.append(
                h_descent_distance / profile.cruise_descent_ground_speed
            )
            distances.append(h_descent_distance)
            altitudes.append(cruise_end_altitude)
            flight_states.append(FlightState.CRUISE_DESCENT)

    # Add landing phase.
    if include_landing:
        if profile.can_transition:
            durations.append(profile.retransition_duration)
            distances.append(0.0)
            altitudes.append(cruise_end_altitude)
            flight_states.append(FlightState.RETRANSITION)
        if add_landing:
            durations.append(landing_duration)
            distances.append(landing_distance)
            altitudes.append(altitude_end)
            flight_states.append(FlightState.LANDING)
    # Add taxi in phase.
    if include_taxi_in:
        durations.append(profile.taxi_in_duration)
        distances.append(0.0)
        altitudes.append(altitude_end)
        flight_states.append(FlightState.TAXI_IN)
    if loiter_time:
        durations.append(loiter_time)
        distances.append(0.0)
        altitudes.append(altitude_end)
        flight_states.append(FlightState.LOITER)
    return StraightTrajectory(
        start_datetime=start_datetime,
        timestamps=np.insert(np.cumsum(durations), 0, 0.0),
        flight_states=flight_states,
        horizontal_distances=distances,
        altitudes=altitudes,
        gps_start=gps_start,
        gps_end=gps_end,
    )
