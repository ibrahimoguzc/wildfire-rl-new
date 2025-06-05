# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from examples.wildfire.firefighter_model.follower import (
    DestinationType,
    FlightState,
    PayloadStatus,
)
from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SELECT_POI_TABLE,
)
from examples.wildfire.firefighter_model.tactic_pieces.suppress import (
    SUPPRESS_TABLE,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    TRACK_POI_TABLE,
)
from sosid.model.abm.task import Task, TaskStatus
from sosid.model.transform import bearing_from_coords

if TYPE_CHECKING:
    from examples.wildfire.firefighter_model.agents import SuppressionUAV


class SuppresionTactic:
    """Suppresion Tactic that extinguishes fire in a direct attack manner
    based on protection area (residential area), distance to fire
    positions and vip cost.

    VIPs are Very Important Points that are defined in Wildfire
    Simulation Parameters, can be different from residential areas, and
    need to be protected against fire.
    """

    def __init__(self, suppression_tactic, change_task):
        self.select_poi = SELECT_POI_TABLE[suppression_tactic.select_poi]()
        self.track_poi = TRACK_POI_TABLE[suppression_tactic.track_poi]()
        self.suppress = SUPPRESS_TABLE[suppression_tactic.suppress]()
        self.change = change_task

    @Task
    def hold(self) -> TaskStatus:
        """Keeps the agent innactive for the response time."""
        return self.idle(self.parameters.response_time)

    @hold.on_complete
    def deploy(self) -> None:
        """Trigger start of task sequence."""
        self.tasks.set_active(self.tactic.await_operational_clearance)

    @Task
    def await_operational_clearance(self):
        """Hold until conditions suit operation."""
        if not self.operational_clearance:
            return TaskStatus.IN_PROGRESS
        return TaskStatus.COMPLETE

    @await_operational_clearance.on_complete
    def start_task_sequence(self):
        self.current_base.request_takeoff(self)
        self.tasks.set_active(self.tactic.await_takeoff_clearance)

    @Task
    def await_takeoff_clearance(self):  # noqa D102
        """Hold until take off clearance."""
        if self.current_base.is_cleared_for_takeoff(self):
            return TaskStatus.COMPLETE
        return TaskStatus.IN_PROGRESS

    @await_takeoff_clearance.on_complete
    def trigger_takeoff_from_base(self) -> None:
        """Trigger takeoff from base."""
        self.tactic.initiate_takeoff(self)
        self.current_base.deregister_from_base(self)
        self.current_base = None

    @staticmethod
    def initiate_takeoff(self) -> None:
        """Trigger takeoff."""
        self.flight_state = FlightState.TAKEOFF
        self.tasks.set_active(self.tactic.takeoff)
        ground_speed = self.profile_parameters.takeoff_ground_speed
        if self.follower.trajectory is not None:
            self._takeoff_altitude = self.follower.trajectory.altitudes[0]
            if ground_speed != 0:
                self.aspect = bearing_from_coords(
                    self.gps_coords, self.follower.trajectory.gps_start
                )
        else:
            self._takeoff_altitude = self.get_cruise_descent_altitude(
                self.pos, DestinationType.BASE
            )

    @Task
    def takeoff(self) -> TaskStatus:
        """Takeoff to the desired altitude."""
        reachable_altitude = (
            self.altitude
            + self.profile_parameters.takeoff_climb_rate
            * self.time_step.total_seconds()
        )
        ground_speed = self.profile_parameters.takeoff_ground_speed
        if ground_speed != 0:
            self.move_forward(ground_speed * self.time_step.total_seconds())
        if reachable_altitude < self._takeoff_altitude:
            self.altitude = reachable_altitude
            return TaskStatus.IN_PROGRESS
        if self.altitude < self._takeoff_altitude:
            # Only set the altitude to the desired altitude if the
            # desired altitude was reached during this time step.
            self.altitude = self._takeoff_altitude
        return TaskStatus.COMPLETE

    @takeoff.on_complete
    def trigger_transition(self):
        """Trigger transition phase."""
        self.flight_state = FlightState.TRANSITION
        self.tasks.set_active(self.tactic.transition_segment)

    @Task
    def transition_segment(self):
        """Hold for transition time."""
        return self.idle(self.profile_parameters.transition_duration)

    @transition_segment.on_complete
    def trigger_select_firefront(self):
        """Trigger `select_firefront` Task."""
        self.flight_state = FlightState.CRUISE_CLIMB
        self.tasks.set_active(self.tactic.select_poi)

    @Task
    def select_suppressant_source(self):
        """Contains logic for selecting a water re-supply location.

        This optimisation aims to obtain the most propellant efficient
        payload re-supply option and apply holding time needed to refill
        the payload, according to the selected re-supply location.
        """
        # Assumes nearest airport as best solution
        nearest_airport, _ = self.get_nearest_airport()
        destination = nearest_airport.pos
        destination_type = DestinationType.BASE

        # If available, retrieves the nearest water source
        if self.feasible_water_locations and self.can_scoop:
            nearest_water_pos, _ = self.nearest_position(
                self.feasible_water_locations
            )
            # When off-airport water sourcing is more efficient the
            # destination is changed to the off-airport water source
            if self.estimate_propellant_for_journey(
                self.pos, nearest_water_pos, DestinationType.WATER, retour=True
            ) < self.estimate_propellant_for_journey(
                self.pos,
                nearest_airport.pos,
                DestinationType.BASE,
                retour=True,
            ):
                destination = nearest_water_pos
                destination_type = DestinationType.WATER
        self.set_destination(destination, destination_type)
        return TaskStatus.COMPLETE

    @select_suppressant_source.on_complete
    def check_energy(self):
        """Contains logic for checking agent's usable energy.

        This method estimates whether the agent's usable energy is
        sufficient to perfrom another suppression flight, or the agent
        should return to base.

        Note: no energy requirement or hold is implemented for
        suppression.
        """
        nearest_airport, _ = self.get_nearest_airport()
        destination_type = (
            DestinationType.BASE
            if np.all(self.destination == nearest_airport.pos)
            else DestinationType.WATER
        )
        propellant_attack = self.estimate_propellant_for_journey(
            self.pos, self.destination, destination_type, retour=False
        )

        propellant_to_base = self.estimate_propellant_for_journey(
            self.destination,
            nearest_airport.pos,
            DestinationType.BASE,
            retour=False,
        )
        required_propellant = propellant_attack + propellant_to_base

        if self.propulsion.is_propellant_available(required_propellant):
            self.tasks.set_active(self.tactic.to_suppressant)
        else:
            self.set_destination(nearest_airport.pos, DestinationType.BASE)
            self.tasks.set_active(self.tactic.return_to_base)

    @Task
    def to_suppressant(self):
        """Guides agent to the selected water re-supply location."""
        return self.follower.navigate()

    @to_suppressant.on_complete
    def start_descent_to_suppressant(self) -> None:
        """Activates hold to resupply payload."""
        self.tasks.set_active(self.tactic.retransition_before_resupply)

    @Task
    def retransition_before_resupply(self) -> TaskStatus:
        """Hold for `retransition_time`."""
        return self.idle(self.profile_parameters.retransition_duration)

    @retransition_before_resupply.on_complete
    def trigger_land_to_resupply(self) -> None:
        """Trigger landing phase for resupply."""
        self.flight_state = FlightState.LANDING
        self.tasks.set_active(self.tactic.land_to_resupply)

    @Task
    def land_to_resupply(self) -> TaskStatus:
        """Descent using landing to the desired resupply altitude."""
        return self.tactic.land(self)

    @land_to_resupply.on_complete
    def initiate_resupply(self) -> None:
        """Agent refuelling and payload resupply."""
        self.flight_state = FlightState.LOITER
        self.tasks.set_active(self.tactic.resupply_payload)

    @Task
    def resupply_payload(self) -> TaskStatus:
        """Hold for `scoop_time`."""
        return self.idle(self.parameters.scoop_time)

    @resupply_payload.on_complete
    def finish_resupply(self) -> None:
        """Trigger next mission."""
        self.payload_status = PayloadStatus.ONBOARD
        self.aspect = bearing_from_coords(
            self.gps_coords, self.full_trajectory.gps_start
        )
        self.follower.abort()
        self.tactic.initiate_takeoff(self)

    @Task
    def return_to_base(self):
        """Guides agent to the selected base location."""
        return self.follower.navigate()

    @return_to_base.on_complete
    def trigger_retransition_before_land_to_base(self) -> None:
        """Trigger retransition before landing to base."""
        self.flight_state = FlightState.RETRANSITION
        self.tasks.set_active(self.tactic.retransition_before_land_to_base)

    @Task
    def retransition_before_land_to_base(self) -> TaskStatus:
        """Hold for `retransition_time`."""
        return self.idle(self.profile_parameters.retransition_duration)

    @retransition_before_land_to_base.on_complete
    def trigger_landing(self):
        """Trigger certical descent phase."""
        self.flight_state = FlightState.LANDING
        self.tasks.set_active(self.tactic.land_to_base)

    @Task
    def land_to_base(self) -> TaskStatus:
        """Land on the ground."""
        return self.tactic.land(self)

    @staticmethod
    def land(agent: SuppressionUAV) -> TaskStatus:
        """Land to the desired altitude."""
        reachable_altitude = (
            agent.altitude
            - agent.profile_parameters.landing_descent_rate
            * agent.time_step.total_seconds()
        )
        desired_altitude = agent.full_trajectory.altitudes[-1]
        ground_speed = agent.profile_parameters.landing_ground_speed
        if ground_speed != 0:
            agent.move_forward(ground_speed * agent.time_step.total_seconds())
        if reachable_altitude > desired_altitude:
            agent.altitude = reachable_altitude
            return TaskStatus.IN_PROGRESS
        # Only set the altitude to the desired altitude if the desired
        # altitude was reached during this time step.
        agent.altitude = min(agent.altitude, desired_altitude)
        if agent.full_trajectory is not None:
            agent.gps_coords = agent.full_trajectory.gps_end
            agent.full_trajectory.end_datetime = agent.current_mission_time
        return TaskStatus.COMPLETE

    @land_to_base.on_complete
    def resupply(self):
        """Agent refuelling and payload resupply."""
        self.flight_state = FlightState.ENERGIZE
        self.payload_status = PayloadStatus.ONBOARD
        airport, _ = self.get_nearest_airport()
        airport.register_at_base(self)
        self.current_base = airport
        self.tasks.set_active(self.tactic.refuel)

    @Task
    def refuel(self):
        """Trigger reenergization of vehicle."""
        if self.propulsion.is_filled:
            return TaskStatus.COMPLETE
        return TaskStatus.IN_PROGRESS

    @refuel.on_complete
    def trigger_next_mission(self):
        """Restart task sequence."""
        self.flight_state = FlightState.IDLE
        self.tasks.set_active(self.tactic.await_operational_clearance)
