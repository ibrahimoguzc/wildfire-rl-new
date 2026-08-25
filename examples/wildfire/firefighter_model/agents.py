# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta
from functools import cached_property

import numpy as np
import shapely.geometry as geom

from examples.wildfire.fire_model.states import COMBUSTIBLE, SUPPRESSED
from examples.wildfire.firefighter_model.follower import (
    DestinationType,
    PayloadStatus,
    StraightTrajectoryFollower,
)
from examples.wildfire.firefighter_model.suppression_tactics import (
    SuppresionTactic,
)
from examples.wildfire.firefighter_model.tactic_pieces.change_tactic import (
    CHANGE_TABLE,
    ChangeType,
)
from examples.wildfire.paths import STATIC_DIR
from sosid.model.abm.agent import (
    StaticAgent,
    StaticAgentWithGPS,
    TrackFlightDurationMixin,
)
from sosid.model.abm.model import AgentBasedModel
from sosid.model.abm.propulsion import BasePropulsionInput
from sosid.model.abm.special_agents import (
    BaseAircraftAgent,
    BaseAirTrafficManager,
    TakeoffLandingType,
)
from sosid.model.abm.trajectory import (
    AircraftProfileParameters,
    FlightState,
    StraightTrajectory,
    generate_straight_trajectory,
)
from sosid.model.transform import gps_to_pos, pos_to_gps
from sosid.output import Output, TargetKey
from sosid.typedef import Position


class SuppressionUAV(TrackFlightDurationMixin, BaseAircraftAgent):
    """Models logic for an aerial firefighting agent.

    The mission profile of the agent is considered by default to be that
    of a eVTOL. The flight states are defined as in `FlightState` in the
    same order.
    """

    def __init__(
        self,
        unique_id,
        pos,
        model,
        output_id_name: str,
        home_base: AirTrafficManager,
        takeoff_landing_type: TakeoffLandingType,
        payload_capacity: float,
        suppressant_flow_rate: float,
        can_scoop: bool,
        icon_type,
        propulsion_input: BasePropulsionInput,
        profile_parameters: AircraftProfileParameters,
        autonomous,
        empty_mass,
        mtom,
        scooping_distance: float,
        span: float,
        ac_type_id: int,
        suppression_tactic,
        change_condition: ChangeType,
        alternative_tactic,
        threshold: int,
    ):
        super().__init__(
            unique_id=unique_id,
            model=model,
            pos=pos,
            output_id_name=output_id_name,
            propulsion_input=propulsion_input,
            takeoff_landing_type=takeoff_landing_type,
        )

        self.init_states = locals().copy()
        # Exclude 'self' and '__class__' from init params
        self.init_states.pop("self")
        self.init_states.pop("__class__")

        self.ac_type_id = ac_type_id
        self.home_base = home_base
        self.payload = payload_capacity
        self.suppressant_flow_rate = suppressant_flow_rate
        self.can_scoop = can_scoop
        self.__icon__ = STATIC_DIR / icon_type
        self.total_suppressions = 0
        self.current_base = home_base
        assert self.current_base.is_compatible_with(self)
        self.current_base.register_at_base(self)

        self._profile_parameters = profile_parameters
        self._needs_water_retransition_accounting = (
            propulsion_input.architecture == "electric"
            and takeoff_landing_type == TakeoffLandingType.VERTIPAD
            and profile_parameters.retransition_duration > 0
        )
        self.follower = StraightTrajectoryFollower(self)
        self.full_trajectory = None

        self.autonomous = autonomous
        self._mtom = mtom
        self._empty_mass = empty_mass
        self.scooping_distance = scooping_distance
        self.scooping_width = span

        self.suppression_tactic = suppression_tactic

        change = CHANGE_TABLE[
            change_condition if change_condition else ChangeType.NO_CHANGE
        ](threshold, suppression_tactic, alternative_tactic)

        self.tactic = SuppresionTactic(
            suppression_tactic,
            change,
            needs_evtol_firefront_guard=(
                self._needs_water_retransition_accounting
            ),
        )

        self.altitude = self.model.simulation.environment.get_elevation(pos)
        self.payload_status = PayloadStatus.ONBOARD
        self.force_tactic_swap = False
        self.set_active_tactic()

    def step(self) -> None:
        """Step the aircraft, diverting eVTOLs before return energy is lost."""
        self._divert_evtol_to_base_if_return_energy_is_low()
        self._trace_return_commit()
        try:
            super().step()
        except ValueError as exc:
            if "propellant" in str(exc):
                self._log_propellant_depletion()
            raise

    def _trace_return_commit(self) -> None:
        """Record a rolling task/energy history for eVTOLs.

        Kept only so that :py:meth:`_log_propellant_depletion` can print the
        whole sortie leading up to a propellant exhaustion instead of just
        the final snapshot. Writes nothing unless the sim actually runs the
        battery flat, so it is silent in normal operation.
        """
        if not self._needs_water_retransition_accounting:
            return
        try:
            cur = self.tasks.active_task
            cur_name = (
                cur.task_method.__name__ if cur is not None else "None(idle)"
            )
            if cur_name == getattr(self, "_hist_last_task", None):
                return
            hist = getattr(self, "_hist", None)
            if hist is None:
                hist = self._hist = deque(maxlen=45)
            _, dist = self.get_nearest_airport()
            hist.append(
                "t=%.0f %s fs=%d usable=%.0f dist=%.0f %s m=%.0f"
                % (
                    self.model.simulation.timer.mission_runtime.total_seconds(),
                    cur_name,
                    int(getattr(self, "flight_state", 0)),
                    self.propulsion.mission_usable_propellant,
                    dist,
                    str(getattr(self, "payload_status", "?")).split(".")[-1],
                    self.current_mass,
                )
            )
            self._hist_last_task = cur_name
        except Exception:  # noqa: BLE001,S110 - best-effort history only
            # This buffer exists purely to enrich a crash report; never let
            # it raise into, or print over, a live simulation.
            pass

    def _log_propellant_depletion(self) -> None:
        """Dump eVTOL state when propellant validation fails (diagnostic).

        Wrapped in its own guard so the diagnostic can never mask the
        original ValueError. Grep the slurm .err for the prefix below.
        """
        import sys

        try:
            active_task = self.tasks.active_task
            task_name = (
                active_task.task_method.__name__
                if active_task is not None
                else "None(idle)"
            )
            nearest_airport, distance_to_base = self.get_nearest_airport()
            return_estimate = self.estimate_propellant_for_journey(
                self.pos, nearest_airport.pos, DestinationType.BASE
            )
            print(
                "EVTOL_PROPELLANT_DEPLETION "
                f"id={self.unique_id} "
                f"evtol_guard={self._needs_water_retransition_accounting} "
                f"track_poi={type(self.tactic.track_poi).__name__} "
                f"active_task={task_name} "
                f"flight_state={getattr(self, 'flight_state', '?')} "
                f"payload_status={getattr(self, 'payload_status', '?')} "
                f"mass={getattr(self, 'current_mass', float('nan')):.1f} "
                f"usable={self.propulsion.mission_usable_propellant:.1f} "
                f"reserve={self.propulsion.specs.reserve_propellant:.1f} "
                f"dist_to_base_m={distance_to_base:.1f} "
                f"return_estimate={return_estimate:.1f}",
                file=sys.stderr,
                flush=True,
            )
            for row in getattr(self, "_hist", []):
                print(f"EVTOL_HIST id={self.unique_id} {row}", file=sys.stderr)
            sys.stderr.flush()
        except Exception as diag_exc:  # noqa: BLE001 - never mask real error
            print(
                f"EVTOL_PROPELLANT_DEPLETION diag_failed={diag_exc!r}",
                file=sys.stderr,
                flush=True,
            )

    def _divert_evtol_to_base_if_return_energy_is_low(self) -> None:
        """Divert eVTOLs to base when the current task risks return energy."""
        if not self._needs_water_retransition_accounting:
            return

        # Compare by identity: ``Task.__eq__`` compares ``priority``, and
        # tactic-piece tasks (select/track/suppress) never set ``priority``,
        # so ``active_task in (...)`` raises AttributeError when one of them
        # is active. ``active_task`` is also None while the agent is idle.
        active_task = self.tasks.active_task
        excluded_tasks = (
            self.tactic.await_operational_clearance,
            self.tactic.await_takeoff_clearance,
            self.tactic.takeoff,
            self.tactic.transition_segment,
            self.tactic.return_to_base,
            self.tactic.retransition_before_land_to_base,
            self.tactic.land_to_base,
            self.tactic.refuel,
        )
        if active_task is None or any(
            active_task is task for task in excluded_tasks
        ):
            return

        nearest_airport, _ = self.get_nearest_airport()
        required_propellant = self.estimate_propellant_for_journey(
            self.pos,
            nearest_airport.pos,
            DestinationType.BASE,
        )
        if self.propulsion.is_propellant_available(required_propellant):
            return

        self.set_destination(nearest_airport.pos, DestinationType.BASE)
        self.tasks.set_active(self.tactic.return_to_base)

    def __gui_repr__(self):
        """Implements GUI representation protocol."""
        gui_repr = super().__gui_repr__()
        gui_repr.update(
            {
                "on_init": lambda p, obj: p(
                    obj.__icon__,
                    size=obj.icon_size,
                    pos=obj.pos,
                ),
                "z_order": 2000,
                "on_update": self.update_pos_aspect_size,
            }
        )
        return gui_repr

    @staticmethod
    def update_pos_aspect_size(item, obj):
        """Update parameters used for agent representation."""
        item.setPos(*obj.pos)
        item.setRotation(obj.aspect)
        item.setSize(*obj.icon_size)
        item.setScale(obj.scale)

    @property
    def icon_size(self) -> tuple[float, float]:
        """Returns adaptive icon size."""
        scaling_factor = 0.5
        size = (
            self.altitude * scaling_factor
            + self.model.simulation.environment.dimensions[0] / 1000
        )
        return (size, size)

    @property
    def top_left_bounds(self) -> Position:
        """Top left position of the simulation area."""
        return self.model.simulation.environment.terrain.top_left_bounds

    @property
    def parameters(self):
        """Returns `SimulationParameters` class."""
        return self.model.simulation.parameters

    @property
    def terrain(self):
        return self.model.simulation.environment.terrain

    @property
    def is_night(self) -> bool:
        """Checks whether it is night time.

        Compares time of next sunrise and sunset to find out which is
        closer.
        """
        atmosphere = self.model.simulation.environment.atmosphere
        next_sunset, next_sunrise = (
            atmosphere.next_sunset,
            atmosphere.next_sunrise,
        )
        if next_sunset < next_sunrise:
            return False
        return True

    @property
    def can_operate_at_night(self) -> bool:
        """Determines if the aircraft can do nighttime operations."""
        return self.parameters.enable_nighttime_operations

    @property
    def operational_clearance(self) -> bool:
        """Indicates whether agent can operate at current time."""
        if self.can_operate_at_night:
            return True
        if self.is_night:
            return False
        return True

    @property
    def effective_mission_time(self) -> float:
        """Returns the actively operating time during a mission."""
        return (
            self.model.simulation.timer.mission_runtime.total_seconds()
            - self.parameters.response_time
        )

    @property
    def n_agents(self) -> int:
        """Returns the fleet size."""
        return self.model.simulation.n_agents

    @property
    def scoop_time(self) -> int:
        """Returns the scoop time."""
        return self.parameters.scoop_time

    @cached_property
    def airports(self):
        """Returns agents of class `AirTrafficManager`."""
        return tuple(
            atm
            for atm in self.model.agents_by_type[AirTrafficManager]
            if atm.is_compatible_with(self)
        )

    @cached_property
    def fire_terrain_origin(self):
        """Returns origin of fire terrain.

        Useful when operational terrain is used alongside fire terrain.
        """
        return self.model.simulation.environment.terrain.origin

    @property
    def profile_parameters(self) -> AircraftProfileParameters:
        """Aircraft profile parameters."""
        return self._profile_parameters

    @property
    def empty_mass(self) -> float:
        """Empty mass of the aircraft."""
        return self._empty_mass

    @property
    def payload_mass(self) -> float:
        """Mass of the current payload."""
        match self.payload_status:
            case PayloadStatus.NONE:
                return 0.0
            case PayloadStatus.ONBOARD:
                return self.payload

    @property
    def mtom(self) -> float:
        """Maximum takeoff mass of the aircraft."""
        return self._mtom

    @property
    def feasible_water_locations(self) -> dict:
        """Returns the feasible water sources for each aircraft type."""
        return self.model.feasible_water_sources[self.ac_type_id]

    def set_destination(
        self, pos: Position, destination_type: DestinationType
    ) -> None:
        """Sets the destination of the agent."""
        # Remembered so the resupply hold can tell an airport stop from an
        # off-airport water stop: only the former can re-energize the agent.
        self.destination_type = destination_type
        self.full_trajectory = self.generate_trajectory(
            starting_pos=self.pos,
            ending_pos=pos,
            destination_type=destination_type,
        )
        self.full_trajectory.start_datetime = self.current_mission_time

    @property
    def full_trajectory(self) -> StraightTrajectory | None:
        """Full trajectory of the agent."""
        return self._full_trajectory

    @full_trajectory.setter
    def full_trajectory(self, trajectory: StraightTrajectory | None) -> None:
        if trajectory is None:
            self._full_trajectory = None
            self.follower.trajectory = None
            self._destination_pos = None
            return
        self._full_trajectory = trajectory
        cruise_states = (
            FlightState.CRUISE_CLIMB,
            FlightState.CRUISE,
            FlightState.CRUISE_DESCENT,
        )
        self.follower.trajectory = trajectory.slice_by_idx(
            self.full_trajectory.first_idx_of_state(*cruise_states),
            # +1 to include the end point of the state since last idx of
            # state returns the last index where the state is still
            # active 'till the next state starts.
            # +1 as the last index is exclusive in slicing.
            self.full_trajectory.last_idx_of_state(*cruise_states) + 2,
        )
        self._destination_pos = gps_to_pos(
            trajectory.gps_end, self.top_left_bounds
        )

    @property
    def destination(self) -> Position | None:
        """Destination of the trajectory."""
        return self._destination_pos

    @property
    def destination_gps(self) -> np.ndarray[np.float64] | None:
        """GPS coordinates of the destination."""
        if self._full_trajectory is None:
            return None
        return self._full_trajectory.gps_end

    @property
    def destination_altitude(self) -> float | None:
        """Altitude of the destination."""
        if self._full_trajectory is None:
            return None
        return self._full_trajectory.altitudes[-1]

    def get_nearest_airport(
        self, *, pos: Position | None = None
    ) -> tuple[AirTrafficManager, float]:
        """Get nearest ``AirTrafficManager`` to agent and distance."""
        if pos is None:
            pos = self.pos
        airport_pos = np.array([obj.pos for obj in self.airports])
        distances = self.distance(pos, airport_pos)
        nearest_idx = np.argmin(distances)
        return self.airports[nearest_idx], distances[nearest_idx]

    def generate_trajectory(
        self,
        starting_pos: Position,
        ending_pos: Position,
        destination_type: DestinationType,
        *,
        reverse: bool = False,
    ) -> StraightTrajectory:
        """Generate a straight trajectory to the destination.

        Generates a straight trajectory from the current position to the
        destination. It automatically determines the altitude based on
        the destination type and whether the trajectory includes takeoff
        and/or landing.

        Args:
            starting_pos: Position of starting point
            ending_pos: Position of the desination
            destination_type: Type of destination.
            reverse: If True, the trajectory is generated from the
                destination to the current position.

        Returns:
            The generated trajectory.
        """
        elev_start = self.model.simulation.environment.get_elevation(
            starting_pos
        )
        elev_end = self.model.simulation.environment.get_elevation(ending_pos)
        if np.all(starting_pos == self.pos):
            is_after_takeoff = (
                FlightState.TAKEOFF > self.flight_state > FlightState.LANDING
            )
            alt_start = self.altitude
        else:
            is_after_takeoff = True
            alt_start = self.get_cruise_descent_altitude(
                starting_pos, destination_type
            )
        scooping_time = 0
        if destination_type == DestinationType.WATER:
            scooping_time = self.scoop_time
        if reverse:
            return generate_straight_trajectory(
                profile=self.profile_parameters,
                gps_start=pos_to_gps(ending_pos, self.top_left_bounds),
                gps_end=pos_to_gps(starting_pos, self.top_left_bounds),
                altitude_start=self.get_cruise_descent_altitude(
                    ending_pos, destination_type
                ),
                altitude_end=alt_start,
                elevation_start=elev_end,
                elevation_end=elev_start,
                include_landing=is_after_takeoff,
                include_takeoff=destination_type == DestinationType.BASE,
            )
        return generate_straight_trajectory(
            profile=self.profile_parameters,
            gps_start=pos_to_gps(starting_pos, self.top_left_bounds),
            gps_end=pos_to_gps(ending_pos, self.top_left_bounds),
            altitude_start=alt_start,
            altitude_end=self.get_cruise_descent_altitude(
                ending_pos, destination_type
            ),
            elevation_start=elev_start,
            elevation_end=elev_end,
            include_landing=destination_type == DestinationType.BASE,
            include_takeoff=is_after_takeoff,
            loiter_time=scooping_time,
        )

    def get_cruise_descent_altitude(
        self, pos: Position, destination_type: DestinationType
    ) -> float:
        """Returns cruise descent altitude based on destiantion type."""
        elev = self.model.simulation.environment.get_elevation(pos)
        match destination_type:
            case DestinationType.BASE:
                return elev + self.profile_parameters.landing_altitude
            case DestinationType.WATER:
                return elev + self.parameters.resupply_altitude
            case DestinationType.FIRE:
                return elev + self.parameters.suppression_altitude
            case _:
                msg = f"Unknown destination type: {destination_type}."
                raise ValueError(msg)

    @staticmethod
    def exponential_cone_func(pos, vip, map_diagonal):
        """3D exponential cone functions."""
        c1 = c2 = map_diagonal * 2

        exponents = []
        for point in vip:
            exp = c2 / (
                (pos[:, 0] - point[0]) ** 2 + (pos[:, 1] - point[1]) ** 2
            )
            exponents.append(exp)

        return c1 ** sum(exponents)

    def generate_priority_cost(self, burning_indices, priority_map):
        """Return array with priority values at burning indices"""
        priority_cost = priority_map[
            burning_indices[:, 0], burning_indices[:, 1]
        ]
        return priority_cost

    def is_combustible_and_higher_combustibility(
        self, i: int, j: int, current_combustibility: float
    ) -> bool:
        """Check if the index is combustible and has higher combustibility."""
        combustibilities_map = (
            self.model.simulation.environment.terrain.features.combustibilities
        )
        if combustibilities_map[i][j] > current_combustibility:
            fire_state = self.model.wildfire.retrieve_fire_states([i, j])
            if fire_state == COMBUSTIBLE:
                return True

        return False

    def calculate_priority_vegetation(
        self, i: int, j: int, neighbor_radius: int = 1
    ) -> float:
        """Calculate priority for a single position.

        Args:
            i (int): vertical index of the cell on fire.
            j (int): horizontal index of the cell on fire.
            neighbor_radius (int): the radius of the neighborhood which
            determines the number of neighbors considered; default is 1
            <==> 8 neighbors; Nr. neighbors = (2*neighbor_radius+1)^2-1.

        Returns:
            Maximum priority of the surrounding cells on fire.
        """
        current_combustibility = self.model.simulation.environment.terrain.features.combustibilities[
            i
        ][j]
        width = self.terrain.width_in_cells
        height = self.terrain.height_in_cells

        # Getting a Moore Neighborhood of the spread_rates
        neighbor_range = np.arange(-neighbor_radius, neighbor_radius + 1)
        i_range, j_range = np.meshgrid(
            np.clip(i + neighbor_range, 0, height - 1),
            np.clip(j + neighbor_range, 0, width - 1),
            indexing="ij",
        )
        neighbors = np.vstack((i_range.flatten(), j_range.flatten())).T

        # Exclude the original (i, j) indices from the neighbors array
        neighbors = neighbors[~np.all(neighbors == [i, j], axis=1)]

        max_priority = max(
            (
                self.model.simulation.environment.terrain.features.combustibilities[
                    ni
                ][nj]
                for ni, nj in neighbors
                if self.is_combustible_and_higher_combustibility(
                    ni, nj, current_combustibility
                )
            ),
            default=0.0,
        )
        return max_priority

    def normalize_priorities(self, priorities: list[float]) -> list[float]:
        """Normalize the priority values."""
        min_priority, max_priority = min(priorities), max(priorities)
        if min_priority != max_priority:
            return [
                (p - min_priority) / (max_priority - min_priority)
                for p in priorities
            ]
        return [1 if min_priority >= 1 else 0 for _ in priorities]

    def priority_cost_vegetation(
        self, burning_indices: list[tuple[int, int]]
    ) -> np.ndarray:
        """Return priority values at burning indices based on vegetation.

        Calculates and normalizes priorities for all fire indices
        depending on the vegetation type (combustibility).
        """
        priority_positions = [
            self.calculate_priority_vegetation(i, j)
            for i, j in burning_indices
        ]
        normalized_priorities = self.normalize_priorities(priority_positions)

        return np.array(normalized_priorities)

    def is_valid_and_combustible(
        self, new_i, new_j, current_elevation, elevation_data
    ):
        """Check if the neighboring cell is higher and combustible."""
        if elevation_data[new_i][new_j] > current_elevation:
            fire_state = self.model.wildfire.retrieve_fire_states(
                (new_i, new_j)
            )
            if fire_state == COMBUSTIBLE:
                return True
        return False

    def calculate_prioritization_factor(
        self, i, j, new_i, new_j, wind_direction
    ):
        """Calculate the absolute angular difference between
        wind_direction and the direction to the neighboring cell and
        return the normalized prioritization_factor
        """
        angle_difference = abs(
            180
            - math.degrees(math.atan2(i - new_i, j - new_j))
            - wind_direction
        )
        return 1 - min(angle_difference, 360 - angle_difference) / 180

    def calculate_priority_topography(
        self, i, j, elevation_data, wind_direction, neighbor_radius=1
    ):
        """Calculate topographical priority for a single cell.

        Args:
            i (int): vertical index of the cell on fire.
            j (int): horizontal index of the cell on fire.
            elevation_data (np.ndarray): topography data of array.
            wind_direction (float): wind aspect/ direction.
            neighbor_radius (int): the radius of the neighborhood which
            determines the number of neighbors considered; default is 1
            <==> 8 neighbors; Nr. neighbors = (2*neighbor_radius+1)^2-1.

        Returns:
            Maximum priority of the surrounding cells on fire.
        """
        current_elevation = elevation_data[i][j]

        width = self.terrain.width_in_cells
        height = self.terrain.height_in_cells

        # Getting a Moore Neighborhood of the spread_rates
        neighbor_range = np.arange(-neighbor_radius, neighbor_radius + 1)
        i_range, j_range = np.meshgrid(
            np.clip(i + neighbor_range, 0, height - 1),
            np.clip(j + neighbor_range, 0, width - 1),
            indexing="ij",
        )
        neighbors = np.vstack((i_range.flatten(), j_range.flatten())).T

        # Exclude the original (i, j) indices from the neighbors array
        neighbors = neighbors[~np.all(neighbors == [i, j], axis=1)]

        priority = max(
            (
                self.calculate_prioritization_factor(
                    i, j, new_i, new_j, wind_direction
                )
                * (elevation_data[new_i][new_j] - current_elevation)
                for new_i, new_j in neighbors
                if self.is_valid_and_combustible(
                    new_i, new_j, current_elevation, elevation_data
                )
            ),
            default=0.0,
        )
        return priority

    def priority_cost_topography(self, burning_indices):
        """Return array with priority values at burning indices based on
        topography.
        """
        elevation_data = self.terrain.elevation.elevation_data
        wind_direction = (
            self.model.simulation.environment.atmosphere.wind_aspect
        )
        priority_positions = [
            self.calculate_priority_topography(
                i, j, elevation_data, wind_direction
            )
            for i, j in burning_indices
        ]
        normalized_priorities = self.normalize_priorities(priority_positions)
        return np.array(normalized_priorities)

    def max_suppression_aspect(self, suppression_area, front_indices) -> float:
        """Contains logic for optimising the suppression aspect angle.

        This optimisation aims to achieve the largest fire-front area
        suppression during one suppression event with respect to
        front_indices (this could be fire indices, block indices,
        or other) whilst maximizing the amount of combustible area being
        suppressed (based on already suppressed areas).
        """
        # Specifying available aspect angles for suppressant drop
        available_angles = np.linspace(
            start=0, stop=180, num=8, endpoint=False
        )

        # Setting intial suppression points and looping over available
        # aspect angles
        suppression_points = -math.inf
        aspect = 0
        for angle in available_angles:
            # Setting the suppression area aspect angle
            suppression_area.aspect = angle

            # Getting the indices of the area that would become
            # suppressed
            suppression_idx = suppression_area.nonzero(
                self.model.wildfire.shape
            )
            suppression_idx = np.stack(
                (suppression_idx[0], suppression_idx[1]), axis=1
            )

            # Getting the intersections between suppressed area and front
            # positions
            suppression_set = set(map(tuple, suppression_idx))
            already_suppressed = np.where(
                self.model.wildfire.fire_states == SUPPRESSED
            )
            already_suppressed = np.stack(
                (already_suppressed[0], already_suppressed[1]), axis=1
            )
            already_suppressed_set = set(map(tuple, already_suppressed))
            front_set = set(map(tuple, front_indices))
            intersections = len(suppression_set & front_set) - len(
                suppression_set & already_suppressed_set
            )

            # Updating the best aspect angle by overwriting the previous
            # best result
            if suppression_points < intersections:
                suppression_points = intersections
                aspect = angle

        return aspect

    def indirect_suppression_aspect(
        self, suppression_area, block_indices
    ) -> float:
        """Optimizes suppression angle for indirect attack.

        This optimisation aims to achieve the largest fire-block area
        suppression during one suppression event with respect to
        block indices whilst ensuring the block is maintained.
        """
        # Specifying available aspect angles for suppressant drop
        available_angles = np.linspace(
            start=0, stop=180, num=8, endpoint=False
        )

        # Setting intial suppression points and looping over available
        # aspect angles
        suppression_points = 0
        aspect = 0
        for angle in available_angles:
            # Setting the suppression area aspect angle
            suppression_area.aspect = angle

            # Getting the indices of the area that would become
            # suppressed
            suppression_idx = suppression_area.nonzero(
                self.model.wildfire.shape
            )
            suppression_idx = np.stack(
                (suppression_idx[0], suppression_idx[1]), axis=1
            )

            # Getting the intersections between suppressed area and front
            # positions
            suppression_set = set(map(tuple, suppression_idx))
            already_suppressed = np.where(
                self.model.wildfire.fire_states == SUPPRESSED
            )
            already_suppressed = np.stack(
                (already_suppressed[0], already_suppressed[1]), axis=1
            )
            already_suppressed_set = set(map(tuple, already_suppressed))
            front_set = set(map(tuple, block_indices))
            intersections = len(suppression_set & front_set)
            # Ensure at least 2 blocks of the suppression are maintained
            # (1 block sometimes lets the fire pass)
            if (
                len(already_suppressed_set) > 0
                and len(suppression_set & already_suppressed_set) <= 1
            ):
                continue

            # Updating the best aspect angle by overwriting the previous
            # best result
            if suppression_points < intersections:
                suppression_points = intersections
                aspect = angle
            if suppression_points == 0:
                return math.nan
        return aspect

    def estimate_propellant_for_journey(
        self,
        starting_pos: Position,
        ending_pos: Position,
        destination_type: DestinationType,
        *,
        retour: bool = False,
        resupply_payload: bool | None = None,
        refill_propellant: bool | None = None,
        start_mass: float | None = None,
    ) -> float:
        """Estimate propellant required to reach destination.

        Args:
            starting_pos: Position of starting point
            ending_pos: Position of the destination.
            destination_type: Type of destination.
            retour: If True, the propellant required to return
                to the current position and altitude is included.
                Default is False.
            start_mass: Mass to fly the outbound leg at. Defaults to the
                agent's current mass. Pass explicitly when the leg is
                flown at a mass the agent does not have yet, e.g. a
                bail-out from a water source that is flown with the
                payload the agent will have scooped on arrival.
            resupply_payload: If True, the mass is updated to include
                the payload capacity on the return journey. Default is
                None. If None, the value is set to True if the
                destination type is WATER or BASE, otherwise False.
            refill_propellant: If True, the mass is updated to include
                the propellant capacity on the return journey. Default
                is None. If None, the value is set to True if the
                destination type is BASE, otherwise False.

        Returns:
            The estimated propellant required for the journey.
        """
        trajectory = self.generate_trajectory(
            starting_pos=starting_pos,
            ending_pos=ending_pos,
            destination_type=destination_type,
        )
        propellant, mass = self.propulsion.estimate_propellant_for_trajectory(
            trajectory,
            start_mass=(
                self.current_mass if start_mass is None else start_mass
            ),
        )
        if (
            self._needs_water_retransition_accounting
            and destination_type == DestinationType.WATER
        ):
            retransition_propellant, _ = (
                self.propulsion.estimate_propellant_consumption(
                    [
                        (
                            FlightState.RETRANSITION,
                            self.profile_parameters.retransition_duration,
                        )
                    ],
                    start_mass=mass,
                )
            )
            propellant += retransition_propellant
        if retour:
            if resupply_payload is None:
                resupply_payload = destination_type in (
                    DestinationType.WATER,
                    DestinationType.BASE,
                )
            if refill_propellant is None:
                refill_propellant = destination_type == DestinationType.BASE
            trajectory = self.generate_trajectory(
                starting_pos=starting_pos,
                ending_pos=ending_pos,
                destination_type=destination_type,
                reverse=True,
            )
            if resupply_payload:
                mass += self.payload_capacity - self.payload_mass
            if refill_propellant:
                mass += (
                    self.propulsion.max_propellant_mass
                    - self.propulsion.propellant_mass
                )
            propellant += self.propulsion.estimate_propellant_for_trajectory(
                trajectory, start_mass=min(mass, self.mtom)
            )[0]
        return propellant

    def has_propellant_for_firefront_and_return(
        self,
        destination: Position,
    ) -> tuple[bool, StraightTrajectory]:
        """Return whether the eVTOL can chase a firefront and return."""
        trajectory = self.generate_trajectory(
            starting_pos=self.pos,
            ending_pos=destination,
            destination_type=DestinationType.FIRE,
        )
        propellant, mass = self.propulsion.estimate_propellant_for_trajectory(
            trajectory,
            start_mass=self.current_mass,
        )
        mass -= self.payload

        nearest_airport, _ = self.get_nearest_airport(pos=destination)
        elev_poi = self.model.simulation.environment.get_elevation(destination)
        elev_base = self.model.simulation.environment.get_elevation(
            nearest_airport.pos
        )
        to_base_trajectory = generate_straight_trajectory(
            profile=self.profile_parameters,
            gps_start=trajectory.gps_end,
            gps_end=nearest_airport.gps_coords,
            altitude_start=trajectory.altitudes[-1],
            altitude_end=elev_base + self.profile_parameters.landing_altitude,
            elevation_start=elev_poi,
            elevation_end=elev_base,
            include_landing=True,
            include_takeoff=True,
        )
        propellant += self.propulsion.estimate_propellant_for_trajectory(
            to_base_trajectory,
            start_mass=mass,
        )[0]
        return self.propulsion.is_propellant_available(propellant), trajectory

    def suppression_patch(
        self, payload: float, suppressant_flow_rate: float
    ) -> tuple[int, int]:
        """Calculates the dimensions of the suppressant drop area.

        Returns a py:type:`tuple` with `length` and `width` of the
        elliptical suppressant patch.

        Args:
            payload: Suppressant mass [kg]
            suppressant_flow_rate: Average suppressant flow rate [m^3/s]

        Relations between suppression area, patch width, suppressant
        payload and suppressant flow rate is based on data collected from
        the research paper:

        'Legendre, Dominique & Becker, Ryan & Alméras, Elise &
        Chassagne, Amélie. (2014). Air tanker drop patterns.
        International Journal of Wildland Fire.
        23. 272. 10.1071/WF13029.'

        The relation indicates that the ratio between a kilogram of
        payload and a square meter of suppressant drop ground area is
        roughly 1.028 (R^2 = 0.49).

        The width [w] of the suppressant area is related to the
        suppressant flow rate [Q] with:
        w = 21.07 * Q ** 0.428 (R^2 = 0.90)

        """
        area_idx = 1.028 * payload / (self.parameters.cell_size**2)
        width = (
            21.07 * suppressant_flow_rate**0.428
        ) / self.parameters.cell_size
        length = 4 * area_idx / (width * math.pi)

        # The decimal values of length and width float determine the
        # probability of rounding the values up or down

        prob = self.random.random()
        length = math.ceil(length) if length % 1 > prob else math.floor(length)

        prob = self.random.random()
        width = math.ceil(width) if width % 1 > prob else math.floor(width)

        return length, width

    def set_active_tactic(self):
        """Sets the active tactic based on simulation parameters"""
        self.tasks.set_active(self.tactic.hold)

    @Output(target_key=TargetKey.AGENTS)
    def agent_class(self):  # noqa D102
        return str(self.__class__.__name__)

    @Output(target_key=TargetKey.AGENTS)
    def unique_id(self):  # noqa D102
        return int(self.unique_id)

    @Output(target_key=TargetKey.AGENTS)
    def payload_capacity(self):  # noqa D102
        return float(self.payload)

    @Output(target_key=TargetKey.AGENTS)
    def can_scoop(self):  # noqa D102
        return bool(self.can_scoop)

    @Output(target_key=TargetKey.AGENTS)
    def suppressant_flow_rate(self):  # noqa D102
        return float(self.suppressant_flow_rate)

    @Output(target_key=TargetKey.AGENTS)
    def distance_flown(self):  # noqa D102
        distance = self.total_distance_covered + self.distance(
            self.pos, self.home_base.pos
        )
        return round((distance / 1000), 3)

    @Output(target_key=TargetKey.AGENTS)
    def cumulative_flight_time(self) -> float:
        """Cumulative flight duration of the aircraft."""
        return self._cumulative_flight_time

    @Output(target_key=TargetKey.AGENTS)
    def suppressions(self):  # noqa D102
        return self.total_suppressions

    @Output(target_key=TargetKey.AGENTS)
    def n_propellant_refills(self):  # noqa D102
        return self.propulsion.n_propellant_refills

    @Output(target_key=TargetKey.AGENTS)
    def suppressions_to_reenergizations(self) -> float:  # noqa D102
        if self.n_propellant_refills != 0:
            return self.total_suppressions / self.n_propellant_refills
        return math.inf

    @Output(target_key=TargetKey.AGENTS)
    def total_energy_consumed(self) -> float:
        """Total energy consumed by the agent."""
        nearest_airport, _ = self.get_nearest_airport()
        propellant_required_to_return = self.estimate_propellant_for_journey(
            self.pos, nearest_airport.pos, DestinationType.BASE
        )
        return (
            self.propulsion.total_energy_consumed
            + self.propulsion.propellant_to_energy(
                propellant_required_to_return
            )
        )

    @Output(target_key=TargetKey.AGENTS)
    def total_electric_energy_consumed(self) -> float:
        """Total energy consumed by the agent."""
        nearest_airport, _ = self.get_nearest_airport()
        propellant_required_to_return = self.estimate_propellant_for_journey(
            self.pos, nearest_airport.pos, DestinationType.BASE
        )
        return (
            self.propulsion.total_electric_energy_consumed
            + self.propulsion.electric_propellant_to_energy(
                propellant_required_to_return
            )
        )

    @Output(target_key=TargetKey.AGENTS)
    def total_propellant_mass_consumed(self) -> float:
        """Total propellant mass consumed by the agent."""
        nearest_airport, _ = self.get_nearest_airport()
        propellant_required_to_return = self.estimate_propellant_for_journey(
            self.pos, nearest_airport.pos, DestinationType.BASE
        )
        return (
            self.propulsion.total_mass_consumed
            + self.propulsion.propellant_to_mass(propellant_required_to_return)
        )

    @Output(target_key=TargetKey.AGENTS)
    def home_base_id(self):  # noqa D102
        return self.home_base.unique_id

    @Output(target_key=TargetKey.AGENTS)
    def average_distance_flown(self):  # noqa D102
        return (
            float(self.distance_flown)
            / self.model.simulation.timer.runtime.total_seconds()
        )


class WaterSourceManager(StaticAgent):
    """Defines and manages the water sources."""

    def __init__(
        self,
        unique_id,
        model,
        water_shell,
        water_holes,
        always_feasible: bool = False,
        autopopulate=False,
    ):
        super().__init__(unique_id, model, autopopulate)
        self.edges = water_shell
        self.holes = water_holes
        self.always_feasible = always_feasible

    __icon__ = STATIC_DIR / "water.svg"
    # Icons made by "https://www.flaticon.com/authors/freepik"

    @property
    def icon_size(self):
        """Returns adaptive icon size."""
        size = self.model.simulation.environment.dimensions[0] / 2000
        return (size, size)

    @property
    def polygon(self):
        """Returns the polygon shape of the water source manager."""
        polygon = self._create_water_polygon()

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon

    @property
    def representative_point(self):
        """Returns a point guaranteed to be within the water body."""
        return self.polygon.representative_point()

    def step(self):
        """."""

    def _create_water_polygon(self):
        """Creates the water polygon using the edges and holes."""
        return geom.Polygon(shell=self.edges, holes=self.holes)


class AirTrafficManager(BaseAirTrafficManager):
    """Agent to manage takeoff and landing requests."""

    def __init__(
        self,
        unique_id: int,
        model: AgentBasedModel,
        *,
        icon: str,
        takeoff_interval: float,
        takeoff_landing_types: (
            TakeoffLandingType | Iterable[TakeoffLandingType]
        ),
    ):
        super().__init__(
            unique_id=unique_id,
            model=model,
            takeoff_landing_types=takeoff_landing_types,
        )
        self.top_left_bounds = (
            model.simulation.environment.terrain.top_left_bounds
        )
        self.__icon__ = STATIC_DIR / icon
        self.turnaround_interval = timedelta(seconds=takeoff_interval)
        self._last_takeoff = datetime.min
        self._agents_at_base: set[BaseAircraftAgent] = set()
        self._queue = []

    def request_takeoff(self, agent: BaseAircraftAgent) -> None:
        """File request to takeoff."""
        self._queue.append(agent)

    def request_landing(self, agent: BaseAircraftAgent) -> None:
        """File request to land."""
        # Landing is always granted.

    def is_cleared_for_takeoff(self, agent: BaseAircraftAgent) -> bool:
        """Check if agent is cleared for takeoff and lock if True."""
        if agent not in self._queue:
            raise ValueError("Agent has not requested takeoff.")
        if (
            self._queue[0] == agent
            and self.current_mission_time - self._last_takeoff
            >= self.turnaround_interval
        ):
            assert agent == self._queue.pop(0)
            self._last_takeoff = self.current_mission_time
            return True
        return False

    def is_cleared_for_landing(self, agent: BaseAircraftAgent) -> bool:
        """Check if agent is cleared for landing and lock if True."""
        return True

    def register_at_base(self, agent: BaseAircraftAgent) -> None:
        """Register agent at the base."""
        self._agents_at_base.add(agent)

    def deregister_from_base(self, agent: BaseAircraftAgent) -> None:
        """Deregister agent from the base."""
        self._agents_at_base.remove(agent)

    def step(self):
        pass

    @property
    def icon_size(self):
        """Returns adaptive icon size."""
        size = self.model.simulation.environment.dimensions[0] / 100
        return (size, size)


class IgnitionCenter(StaticAgentWithGPS):
    """Defines ignition center agents with dual purpose positions."""

    __icon__ = STATIC_DIR / "fire.svg"

    # Icons made by "https://www.flaticon.com/authors/freepik"
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.top_left_bounds = (
            model.simulation.environment.terrain.top_left_bounds
        )

    @cached_property
    def fire_map_pos(self) -> Position:
        return gps_to_pos(
            self.gps_coords,
            self.model.simulation.environment.terrain.fire_map_top_left_bounds,
        )

    def step(self):
        pass

    @property
    def icon_size(self):
        """Returns adaptive icon size."""
        size = self.model.simulation.environment.dimensions[0] / 2500
        return (size, size)


class ProtectionLocation(StaticAgent):
    """Defines protection locations."""

    __icon__ = STATIC_DIR / "lock.svg"
    # Icons made by "https://www.flaticon.com/authors/freepik"

    def step(self):
        pass

    @property
    def icon_size(self) -> tuple[int, int]:
        """Adaptive icon size."""
        size = self.model.simulation.environment.dimensions[0] / 1000
        return (size, size)
