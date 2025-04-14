# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from collections.abc import Iterable
from datetime import timedelta
from enum import Enum

from sosid.model.abm.agent import Agent, MovingAgentWithGPS, StaticAgentWithGPS
from sosid.model.abm.model import AgentBasedModel
from sosid.model.abm.propulsion import (
    BasePropulsionInput,
    PropulsionArchitecture,
)
from sosid.model.abm.trajectory import FlightState
from sosid.typedef import Position
from sosid.util.abc import abstractmethod

__all__ = [
    "BaseAirTrafficManager",
    "BaseAircraftAgent",
    "BaseDispatcherAgent",
    "TakeoffLandingType",
]


class TakeoffLandingType(Enum):
    """Types of takeoff and landing areas."""

    RUNWAY = "runway"
    VERTIPAD = "vertipad"
    WATER = "water"


class BaseAircraftAgent(MovingAgentWithGPS):
    """Base class for aircraft agents."""

    def __init__(
        self,
        unique_id: int,
        model: AgentBasedModel,
        pos: Position,
        *,
        output_id_name: str,
        autopopulate: bool = False,
        propulsion_input: BasePropulsionInput,
        takeoff_landing_type: TakeoffLandingType,
    ) -> None:
        super().__init__(unique_id, model, pos, autopopulate)
        self.output_id_name = output_id_name
        self.flight_state = FlightState.IDLE
        self.altitude = 0.0
        self.propulsion = PropulsionArchitecture(
            propulsion_input.architecture
        ).propulsion_cls(propulsion_input, self)
        self.takeoff_landing_type = takeoff_landing_type

    @property
    @abstractmethod
    def empty_mass(self) -> float:
        """Empty mass of the aircraft."""

    @property
    @abstractmethod
    def payload_mass(self) -> float:
        """Mass of the current payload."""

    @property
    @abstractmethod
    def mtom(self) -> float:
        """Maximum takeoff mass of the aircraft."""

    @property
    def current_mass(self) -> float:
        """Current mass of the aircraft."""
        return (
            self.empty_mass
            + self.payload_mass
            + self.propulsion.propellant_mass
        )

    @property
    def time_step(self) -> timedelta:
        """Time step size of the simulation."""
        return self.model.simulation.time_step

    @property
    def flight_state(self) -> FlightState:
        """Current flight state of the aircraft."""
        return self._flight_state

    @flight_state.setter
    def flight_state(self, value: FlightState) -> None:
        self._flight_state = value

    @property
    def altitude(self) -> float:
        """Current altitude of the aircraft."""
        return self._altitude

    @altitude.setter
    def altitude(self, value: float) -> None:
        self._altitude = value

    def step(self) -> None:
        """Step in the simulation."""
        super().step()
        self.propulsion.step()


class BaseDispatcherAgent(Agent):
    """Agent managing assignment of requests."""

    def __init__(
        self,
        unique_id: int,
        model: AgentBasedModel,
        autopopulate: bool = False,
    ):
        super().__init__(
            unique_id=unique_id, model=model, autopopulate=autopopulate
        )

    @property
    @abstractmethod
    def next_request(self) -> Agent | None:
        """Peek next request awaiting assignment."""

    @abstractmethod
    def dispatch(self, request: Agent) -> None:
        """Dispatch a request to an agent."""


class BaseAirTrafficManager(StaticAgentWithGPS):
    """Agent to manage takeoff and landing requests."""

    def __init__(
        self,
        unique_id: int,
        model: AgentBasedModel,
        *,
        takeoff_landing_types: (
            TakeoffLandingType | Iterable[TakeoffLandingType]
        ),
    ):
        super().__init__(unique_id=unique_id, model=model)
        if isinstance(takeoff_landing_types, TakeoffLandingType):
            takeoff_landing_types = (takeoff_landing_types,)
        self.takeoff_landing_types = frozenset(takeoff_landing_types)

    def is_compatible_with(self, agent: BaseAircraftAgent) -> bool:
        """Check agent's compatibility with operational area of base."""
        return agent.takeoff_landing_type in self.takeoff_landing_types

    @abstractmethod
    def request_takeoff(self, agent: BaseAircraftAgent) -> None:
        """File request to takeoff."""

    @abstractmethod
    def request_landing(self, agent: BaseAircraftAgent) -> None:
        """File request to land."""

    @abstractmethod
    def is_cleared_for_takeoff(self, agent: BaseAircraftAgent) -> bool:
        """Check if agent is cleared for takeoff and lock if True."""

    @abstractmethod
    def is_cleared_for_landing(self, agent: BaseAircraftAgent) -> bool:
        """Check if agent is cleared for landing and lock if True."""

    @abstractmethod
    def register_at_base(self, agent: BaseAircraftAgent) -> None:
        """Register agent at the base.

        This method should be called by the agent when it finished
        landing.
        """

    @abstractmethod
    def deregister_from_base(self, agent: BaseAircraftAgent) -> None:
        """Deregister agent from the base.

        This method should be called by the agent when initiates
        takeoff.
        """
