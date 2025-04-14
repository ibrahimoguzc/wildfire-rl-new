# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

import numpy as np
from annotated_types import Annotated, Len
from pydantic import (
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)
from recordclass import dataobject
from scipy.interpolate import make_interp_spline
from typing_extensions import Self

from sosid.abstract import ComponentWritable
from sosid.model.abm.trajectory import BaseTrajectory, FlightState
from sosid.output import Output
from sosid.util.abc import abstractinterface
from sosid.util.validation import PolymorphicBaseModel, check_required_fields

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime, timedelta

    from sosid.model.abm.agent import Agent


class Propellant(Protocol):
    """Protocol for propellant."""


PropellantT = TypeVar("PropellantT", bound=Propellant)
FcInputType = Annotated[dict[int, PositiveFloat], Len(2, 2)] | NonNegativeFloat


class BasePropulsionInput(PolymorphicBaseModel, polymorphic=True):
    # class BasePropulsionInput(BaseModel):
    """Base inputs for all propulsion specifications."""

    model_config = ConfigDict(
        frozen=True,
    )
    __identification_field__: ClassVar[str] = "architecture"

    architecture: str
    safety_factor: PositiveFloat = 1.0
    total_propellant: PositiveFloat
    total_mission_usable_propellant: PositiveFloat
    propellant_unit: str

    @model_validator(mode="before")
    @classmethod
    def parse_reserve_propellant(cls, values: dict) -> dict:
        """Parse the reserve propellant."""
        if "reserve_propellant" not in values:
            return values
        reserve = values.pop("reserve_propellant")
        total = values.get("total_propellant")
        usable = values.get("total_mission_usable_propellant")
        if total is not None and usable is not None:
            assert values["reserve_propellant"] == total - usable
        elif total is not None:
            values["total_mission_usable_propellant"] = total - reserve
        elif usable is not None:
            values["total_propellant"] = usable + reserve
        return values

    @model_validator(mode="after")
    def validate_propellant_values(self) -> Self:
        """Validate the propellant values."""
        if self.total_propellant < self.total_mission_usable_propellant:
            msg = (
                f"Total mission usable propellant"
                f" ({self.total_mission_usable_propellant}) must be smaller or"
                f" equal to total propellant ({self.total_propellant})."
            )
            raise ValueError(msg)
        return self

    @property
    def reserve_propellant(self) -> PositiveFloat:
        """Reserve propellant for safety."""
        return self.total_propellant - self.total_mission_usable_propellant


class BatteryElectricPropulsionInput(BasePropulsionInput):
    """Electric vehicle specifications."""

    architecture: str = "electric"
    propellant_unit: str = "kWh"
    taxi_out_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    transition_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    retransition_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    cruise_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    cruise_climb_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    takeoff_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    landing_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    cruise_descent_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    taxi_in_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    loiter_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    hover_power: dict[int, NonNegativeFloat] | NonNegativeFloat
    charging_power: NonNegativeFloat
    battery_swap_enabled: bool
    battery_swap_time: NonNegativeFloat


class ConventionalPropulsionInput(BasePropulsionInput):
    """Conventional/fuel-burning vehicle specifications."""

    architecture: str = "conventional"
    propellant_unit: str = "kg"
    taxi_out_fc: FcInputType
    taxi_in_fc: FcInputType
    takeoff_fc: FcInputType
    cruise_fc: FcInputType
    cruise_climb_fc: FcInputType
    cruise_descent_fc: FcInputType
    landing_fc: FcInputType
    transition_fc: FcInputType
    retransition_fc: FcInputType
    loiter_fc: FcInputType
    hover_fc: FcInputType = 0.0
    refueling_rate: PositiveFloat
    fuel_specific_energy: PositiveFloat = 44000  # kJ/kg


class HybridElectricPropulsionInput(BasePropulsionInput):
    """Hybrid vehicle specifications."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    architecture: str = "hybrid"
    conventional: ConventionalPropulsionInput
    electric: BatteryElectricPropulsionInput

    total_propellant: HybridElectricFuelState
    total_mission_usable_propellant: HybridElectricFuelState
    hybridization_ratio: PositiveFloat

    @model_validator(mode="before")
    @classmethod
    def default_total_propellant_values(cls, values: dict) -> dict:
        """Set default values for total propellant."""
        check_required_fields(("conventional", "electric"), values)
        conv = values["conventional"]
        elec = values["electric"]
        if not isinstance(conv, ConventionalPropulsionInput):
            conv = ConventionalPropulsionInput.model_validate(conv)
            values["conventional"] = conv
        if not isinstance(elec, BatteryElectricPropulsionInput):
            elec = BatteryElectricPropulsionInput.model_validate(elec)
            values["electric"] = elec
        values.update(
            {
                "total_propellant": HybridElectricFuelState(
                    battery_energy=elec.total_propellant,
                    fuel=conv.total_propellant,
                ),
                "total_mission_usable_propellant": HybridElectricFuelState(
                    battery_energy=elec.total_mission_usable_propellant,
                    fuel=conv.total_mission_usable_propellant,
                ),
                "propellant_unit": (
                    f"{elec.propellant_unit} x {conv.propellant_unit}"
                ),
            }
        )
        return values


class HybridElectricFuelState(dataobject):
    """Propellant state with regards to energy and fuel.

    The class also implements magic methods for mathematical operations.
    These operations are carried out with the ``battery_energy`` and
    ``fuel`` values respectively. When two states are added, the
    ``battery_energy`` values of the two states are summed, and
    similarly the ``fuel`` values. When a comparison is made, the
    comparison is applied to the ``battery_energy`` and the ``fuel``
    values respectively.

    All mathematical operations except for multiplication and vision are
    permitted between instances of the class type only. Multiplication
    and Division are permitted with integers or floats only.
    """

    battery_energy: float
    fuel: float

    def __add__(self, other: Self) -> Self:
        if self.__class__ is other.__class__:
            return self.__class__(
                battery_energy=self.battery_energy + other.battery_energy,
                fuel=self.fuel + other.fuel,
            )
        msg = (
            f"Unsupported operand type(s) for +:"
            f" {self.__class__!r} and {other.__class__!r}"
        )
        raise TypeError(msg)

    def __sub__(self, other: Self) -> Self:
        if self.__class__ is other.__class__:
            return self.__class__(
                battery_energy=self.battery_energy - other.battery_energy,
                fuel=self.fuel - other.fuel,
            )
        msg = (
            f"Unsupported operand type(s) for -:"
            f" {self.__class__!r} and {other.__class__!r}"
        )
        raise TypeError(msg)

    def __mul__(self, other: float) -> Self:
        if isinstance(other, int | float):
            return self.__class__(
                battery_energy=self.battery_energy * other,
                fuel=self.fuel * other,
            )
        msg = (
            f"Unsupported operand type(s) for *:"
            f" {self.__class__!r} and {other.__class__!r}"
        )
        raise TypeError(msg)

    def __rmul__(self, other: float) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Self:
        if isinstance(other, int | float):
            return self.__class__(
                battery_energy=self.battery_energy / other,
                fuel=self.fuel / other,
            )
        msg = (
            f"Unsupported operand type(s) for /:"
            f" {self.__class__!r} and {other.__class__!r}"
        )
        raise TypeError(msg)

    def __neg__(self) -> Self:
        return self.__class__(
            battery_energy=-self.battery_energy, fuel=-self.fuel
        )

    def __bool__(self) -> bool:
        return bool(self.battery_energy) or bool(self.fuel)

    def __eq__(self, other: Self) -> bool:
        return (
            self.__class__ is other.__class__
            and self.battery_energy == other.battery_energy
            and self.fuel == other.fuel
        )


class BasePropulsion(Generic[PropellantT], ComponentWritable):
    """Propulsion of an agent.

    The propulsion is responsible for managing the energy/fuel.
    """

    __slots__ = (
        "_is_already_refilling",
        "_n_propellant_refills",
        "_propellant_at_refill_start",
        "_refill_notifiers",
        "mission_usable_propellant",
        "specs",
    )

    def __init__(
        self,
        specs: BasePropulsionInput,
        agent: Agent,
    ):
        super().__init__(agent)
        self.specs = specs
        self.mission_usable_propellant = specs.total_mission_usable_propellant
        self._n_propellant_refills = 0
        self._total_consumption = 0.0
        self._is_already_refilling = False
        self._propellant_at_refill_start = self.mission_usable_propellant
        self._refill_notifiers: list[Callable[PropellantT], None] = []

    @property
    @abstractinterface
    def flight_state(self) -> FlightState:
        """Current flight state of the agent."""

    @property
    @abstractinterface
    def current_mass(self) -> float:
        """Current mass of the agent."""

    @property
    @abstractinterface
    def mtom(self) -> float:
        """Maximum takeoff mass of the agent."""

    @property
    @abstractinterface
    def current_mission_time(self) -> datetime:
        """Current time in the simulation."""

    @property
    @abstractinterface
    def time_step(self) -> timedelta:
        """Time step size of the simulation."""

    @property
    def propellant_mass(self) -> float:
        """Current propellant mass of the agent."""
        return self.propellant_to_mass(
            self.mission_usable_propellant + self.specs.reserve_propellant
        )

    @property
    def max_propellant_mass(self) -> float:
        """Maximum propellant mass of the agent."""
        return self.propellant_to_mass(
            self.specs.total_mission_usable_propellant
            + self.specs.reserve_propellant
        )

    @abstractmethod
    def propellant_to_mass(self, propellant: PropellantT) -> float:
        """Convert propellant to mass in kilograms."""

    @abstractmethod
    def propellant_to_energy(self, propellant: PropellantT) -> float:
        """Convert propellant to energy in Joules."""

    @abstractmethod
    def electric_propellant_to_energy(self, propellant: PropellantT) -> float:
        """Convert electric propellant to energy in Joules."""

    @abstractmethod
    def _validate_propellant(self) -> None:
        """Validate the vehicle has still propellant."""

    @abstractmethod
    def _initialize_refill_propellant(self) -> None:
        """Initialize the refill of propellant."""
        self._propellant_at_refill_start = self.mission_usable_propellant

    @abstractmethod
    def _refill_propellant(self) -> None:
        """Refill the propellant during one timestep."""

    @abstractmethod
    def _consume_propellant(self) -> None:
        """Consume the propellant during one timestep."""

    @property
    def is_filled(self) -> bool:
        """Check if the propellant is filled."""
        return (
            self.mission_usable_propellant
            == self.specs.total_mission_usable_propellant
        )

    def propellant_required_for_safety(
        self, propellant_consumption: PropellantT
    ) -> PropellantT:
        """Calculate the propellant required for safety."""
        return propellant_consumption * self.specs.safety_factor

    @abstractmethod
    def is_propellant_available(
        self,
        propellant_required: PropellantT,
        propellant_available: PropellantT | None = None,
    ) -> bool:
        """Check if enough propellant is available for the mission."""

    @abstractmethod
    def propellant_after_refill_duration(
        self, duration: float, initial_propellant: PropellantT | None = None
    ) -> PropellantT:
        """Estimate propellant after refill for a duration."""

    @abstractmethod
    def required_refill_duration(
        self,
        propellant_required: PropellantT,
        initial_propellant: PropellantT | None = None,
    ) -> tuple[float, PropellantT]:
        """Estimate duration to refill to required propellant.

        Attributes:
            initial_propellant: Propellant at the start of refill.
            propellant_required: Minimal propellant after refill. If
                None, the current propellant state and refill progress
                is used.

        Returns:
            Tuple of refill time and propellant after refill.
        """

    @abstractmethod
    def max_duration_for_propellant(
        self,
        flight_state: FlightState,
        initial_propellant: PropellantT,
        start_mass: float | None = None,
    ) -> float:
        """Estimate maximum duration for a flight state.

        Args:
            flight_state: Flight state for which to estimate the
                duration.
            initial_propellant: Initial propellant for the flight state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Maximum duration for the flight state.
        """

    @abstractmethod
    def estimate_propellant_consumption(
        self,
        state_duration_segments: list[tuple[FlightState, float]],
        start_mass: float | None = None,
    ) -> tuple[PropellantT, float]:
        """Estimate propellant consumption for a trajectory.

        Args:
            state_duration_segments: List of tuples of flight state and
                duration of that state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Tuple of propellant consumption and the mass at the end of
            the trajectory.
        """

    def estimate_propellant_for_trajectory(
        self, trajectory: BaseTrajectory, start_mass: float | None = None
    ) -> tuple[PropellantT, float]:
        """Estimate propellant consumption for a trajectory.

        Args:
            trajectory: Trajectory for which to estimate the propellant.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Tuple of propellant consumption and the mass at the end of
            the trajectory.
        """
        return self.estimate_propellant_consumption(
            trajectory.get_state_duration_segements(), start_mass
        )

    def add_refill_notifier(
        self, notifier: Callable[[PropellantT], None]
    ) -> None:
        """Add a notifier for refills."""
        self._refill_notifiers.append(notifier)

    def step(self) -> None:
        """Run the propulsion for a single iteration."""
        if self.flight_state == FlightState.ENERGIZE:
            if not self._is_already_refilling:
                self._initialize_refill_propellant()
                self._is_already_refilling = True
            self._refill_propellant()
        elif self._is_already_refilling:
            self._is_already_refilling = False
            if (
                self.mission_usable_propellant
                != self._propellant_at_refill_start
            ):
                self._log_propellant_refilled()
        if not self._is_already_refilling:
            initial = self.mission_usable_propellant
            self._consume_propellant()
            self._total_consumption += initial - self.mission_usable_propellant
        self._validate_propellant()

    def _log_propellant_refilled(self) -> None:
        """Log the propellant refilled."""
        self._n_propellant_refills += 1
        refilled_propellant = (
            self.mission_usable_propellant - self._propellant_at_refill_start
        )
        for notifier in self._refill_notifiers:
            notifier(refilled_propellant)

    @Output
    def n_propellant_refills(self) -> int:
        """Number of times the propellant has been refilled."""
        return self._n_propellant_refills

    @Output
    def total_propellant_consumed(self) -> PropellantT:
        """Total propellant consumed by the agent."""
        return self._total_consumption

    @Output
    def total_energy_consumed(self) -> float:
        """Total energy consumed by the agent in Joules."""
        return self.propellant_to_energy(self._total_consumption)

    @Output
    def total_electric_energy_consumed(self) -> float:
        """Total energy consumed from the electric propulsion [J]."""
        return self.electric_propellant_to_energy(self._total_consumption)

    @Output
    def total_mass_consumed(self) -> float:
        """Total propellant mass consumed by the agent in kilograms."""
        return self.propellant_to_mass(self._total_consumption)


class BatteryElectricPropulsion(BasePropulsion[float]):
    """Propulsion for battery electric vehicles."""

    specs: BatteryElectricPropulsionInput

    __slots__ = (
        "_consumption_functions",
        "_refill_start_time",
    )

    def __init__(self, specs: BatteryElectricPropulsionInput, agent: Agent):
        super().__init__(specs, agent)
        self._refill_start_time = self.current_mission_time
        self._consumption_functions: dict[
            FlightState, Callable[[float], float]
        ] = {}
        for flight_state, power in (
            (FlightState.ENERGIZE, 0.0),
            (FlightState.IDLE, 0.0),
            (FlightState.TAXI_OUT, specs.taxi_out_power),
            (FlightState.TAKEOFF, specs.takeoff_power),
            (FlightState.TRANSITION, specs.transition_power),
            (FlightState.CRUISE_CLIMB, specs.cruise_climb_power),
            (FlightState.CRUISE, specs.cruise_power),
            (FlightState.CRUISE_DESCENT, specs.cruise_descent_power),
            (FlightState.RETRANSITION, specs.retransition_power),
            (FlightState.LANDING, specs.landing_power),
            (FlightState.TAXI_IN, specs.taxi_in_power),
            (FlightState.LOITER, specs.loiter_power),
            (FlightState.HOVER, specs.hover_power),
        ):
            if isinstance(power, dict):
                power_func = make_interp_spline(
                    list(power.keys()), list(power.values()), k=1
                )
            else:
                power_func = lambda _, power=power: power  # noqa: E731
            self._consumption_functions[flight_state] = power_func

    def propellant_to_mass(self, _: float) -> float:
        """Convert propellant to mass in kilograms."""
        return 0.0  # Battery mass is assumed to be part of empty mass.

    def propellant_to_energy(self, propellant: float) -> float:
        """Convert propellant to energy in Joules."""
        match self.specs.propellant_unit:
            case "kWh":
                return propellant * 3.6e6
            case "kJ":
                return propellant * 1e3
            case "J":
                return propellant
            case _:
                msg = (
                    f"Unknown propellant unit: {self.specs.propellant_unit!r}"
                )
                raise ValueError(msg)

    def electric_propellant_to_energy(self, propellant: float) -> float:
        """Convert electric propellant to energy in Joules."""
        return self.propellant_to_energy(propellant)

    def _validate_propellant(self) -> None:
        """Validate the vehicle has still propellant."""
        if self.mission_usable_propellant + self.specs.reserve_propellant < 0:
            raise ValueError("Agent ran out of electric propellant.")

    def _initialize_refill_propellant(self) -> None:
        """Initialize the refill of propellant."""
        super()._initialize_refill_propellant()
        self._refill_start_time = self.current_mission_time

    def _refill_propellant(self) -> None:
        """Refill the propellant during one timestep."""
        if self.specs.battery_swap_enabled:
            if (
                self.current_mission_time - self._refill_start_time
            ).total_seconds() >= self.specs.battery_swap_time:
                self.mission_usable_propellant = (
                    self.specs.total_mission_usable_propellant
                )
        else:
            self.mission_usable_propellant = min(
                self.mission_usable_propellant
                + self.specs.charging_power * self.time_step.total_seconds(),
                self.specs.total_mission_usable_propellant,
            )

    def _consume_propellant(self) -> None:
        """Consume the propellant during one timestep."""
        self.mission_usable_propellant -= (
            self._consumption_functions[self.flight_state](self.current_mass)
            * self.time_step.total_seconds()
        )

    def is_propellant_available(
        self,
        propellant_required: float,
        propellant_available: float | None = None,
    ) -> bool:
        """Check if enough propellant is available for the mission."""
        if propellant_available is None:
            propellant_available = self.mission_usable_propellant
        return propellant_required <= propellant_available

    def propellant_after_refill_duration(
        self, duration: float, initial_propellant: float | None = None
    ) -> float:
        """Estimate propellant after refill for a duration."""
        if initial_propellant is None:
            initial_propellant = self.mission_usable_propellant
            is_already_refilling = self._is_already_refilling
        else:
            is_already_refilling = False
        if self.specs.battery_swap_enabled:
            if is_already_refilling:
                if (
                    duration
                    + (
                        self.current_mission_time - self._refill_start_time
                    ).total_seconds()
                    < self.specs.battery_swap_time
                ):
                    return initial_propellant
            elif duration < self.specs.battery_swap_time:
                return initial_propellant
            return self.specs.total_mission_usable_propellant
        initial_propellant += self.specs.charging_power * duration
        return min(
            initial_propellant, self.specs.total_mission_usable_propellant
        )

    def required_refill_duration(
        self,
        propellant_required: float,
        initial_propellant: float | None = None,
    ) -> tuple[float, float]:
        """Estimate duration to refill to required propellant.

        Attributes:
            initial_propellant: Propellant at the start of refill.
            propellant_required: Minimal propellant after refill. If
                None, the current propellant state and refill progress
                is used.

        Returns:
            Tuple of refill time and propellant after refill.
        """
        if initial_propellant is None:
            initial_propellant = self.mission_usable_propellant
            is_already_refilling = self._is_already_refilling
        else:
            is_already_refilling = False
        if initial_propellant >= propellant_required:
            return 0.0, initial_propellant
        if self.specs.battery_swap_enabled:
            if is_already_refilling:
                return (
                    self.specs.battery_swap_time
                    - (
                        self.current_mission_time - self._refill_start_time
                    ).total_seconds(),
                    self.specs.total_mission_usable_propellant,
                )
            return (
                self.specs.battery_swap_time,
                self.specs.total_mission_usable_propellant,
            )
        charge_time = (
            propellant_required - initial_propellant
        ) / self.specs.charging_power
        return charge_time, propellant_required

    def max_duration_for_propellant(
        self,
        flight_state: FlightState,
        initial_propellant: float,
        start_mass: float | None = None,
    ) -> float:
        """Estimate maximum duration for a flight state.

        Args:
            flight_state: Flight state for which to estimate the
                duration.
            initial_propellant: Initial propellant for the flight state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Maximum duration for the flight state.
        """
        if start_mass is None:
            start_mass = self.mtom
        return initial_propellant / self._consumption_functions[flight_state](
            start_mass
        )

    def estimate_propellant_consumption(
        self,
        state_duration_segments: list[tuple[FlightState, float]],
        start_mass: float | None = None,
    ) -> tuple[float, float]:
        """Estimate propellant consumption for a trajectory.

        Args:
            state_duration_segments: List of tuples of flight state and
                duration of that state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Tuple of propellant consumption and the mass at the end of
            the trajectory.
        """
        if start_mass is None:
            start_mass = self.mtom
        return sum(
            self._consumption_functions[state](start_mass) * duration
            for state, duration in state_duration_segments
        ), start_mass


def _create_exact_consumption_function(
    fc: FcInputType,
) -> tuple[Callable[[float, float], float], Callable[[float, float], float]]:
    r"""Create a function for exact fuel consumption.

    This function solves the following differential equation to obtain
    the exact fuel consumption function:

    .. math::
        f_{c}(t) = -\dot{m}(t) = \frac{f_{c,1} - f_{c,0}}{m_1 - m_0}
         \left(m(t) - m_0\right) + f_{c,0}

    where :math:`f_{c,0}` and :math:`f_{c,1}` are the fuel consumptions
    at the masses :math:`m_0` and :math:`m_1` respectively. Through
    these two points a linear function is defined as shown above. The
    slope of this function is:

    .. math::
        \text{slope} = s = \frac{f_{c,1} - f_{c,0}}{m_1 - m_0}

    With the initial condition :math:`m(0) = m_0` (current mass) the
    solution to the differential equation is:

    .. math::
        m(t) = m_i + \left(m(0) - m_i\right) e^{-s t}

    where:

    .. math::
         m_i = \frac{f_{c,1} m_0 - f_{c,0} m_1}{f_{c,1} - f_{c,0}}

    Args:
        fc: Fuel consumption at different masses. The fuel consumption
            can either be a single value if the fuel consumption is
            independent of the mass or a two-element dictionary with
            the masses as keys and the fuel consumptions as values.

    Returns:
        A tuple with two callables. The first callable is the exact
        fuel consumption for a given duration and starting mass. The
        second callable computes the duration for a given propellant
        and starting mass.

    """
    if isinstance(fc, dict):
        (m0, f0), (m1, f1) = fc.items()
        mi = (f1 * m0 - f0 * m1) / (f1 - f0)
        slope = (f1 - f0) / (m1 - m0)

        def fc_func(duration: float, current_mass: float) -> float:
            return current_mass - (
                mi + (current_mass - mi) * np.exp(-slope * duration)
            )

        def duration_func(propellant: float, current_mass: float) -> float:
            return (
                np.log((mi - current_mass) / (propellant + mi - current_mass))
                / slope
            )
    else:

        def fc_func(duration: float, current_mass: float) -> float:
            return fc * duration

        def duration_func(propellant: float, current_mass: float) -> float:
            return propellant / fc

    return fc_func, duration_func


class ConventionalPropulsion(BasePropulsion[float]):
    """Propulsion for fuel-based vehicles."""

    __slots__ = ("_exact_consumption_functions", "_exact_duration_functions")

    specs: ConventionalPropulsionInput

    def __init__(self, specs: ConventionalPropulsionInput, agent: Agent):
        super().__init__(specs, agent)
        functions = {
            flight_state: _create_exact_consumption_function(fc)
            for flight_state, fc in (
                (FlightState.ENERGIZE, 0.0),
                (FlightState.IDLE, 0.0),
                (FlightState.TAXI_OUT, specs.taxi_out_fc),
                (FlightState.TAKEOFF, specs.takeoff_fc),
                (FlightState.TRANSITION, specs.transition_fc),
                (FlightState.CRUISE_CLIMB, specs.cruise_climb_fc),
                (FlightState.CRUISE, specs.cruise_fc),
                (FlightState.CRUISE_DESCENT, specs.cruise_descent_fc),
                (FlightState.RETRANSITION, specs.retransition_fc),
                (FlightState.LANDING, specs.landing_fc),
                (FlightState.TAXI_IN, specs.taxi_in_fc),
                (FlightState.LOITER, specs.loiter_fc),
                (FlightState.HOVER, specs.hover_fc),
            )
        }
        self._exact_consumption_functions: dict[
            FlightState, Callable[[float], float]
        ] = {k: v[0] for k, v in functions.items()}
        self._exact_duration_functions: dict[
            FlightState, Callable[[float], float]
        ] = {k: v[1] for k, v in functions.items()}
        self._current_segment_state = self.flight_state

    def propellant_to_mass(self, propellant: float) -> float:
        """Convert propellant to mass in kilograms."""
        return propellant

    def propellant_to_energy(self, propellant: float) -> float:
        """Convert propellant to energy in Joules."""
        if self.specs.propellant_unit != "kg":
            msg = (
                f"Unsupported propellant unit: {self.specs.propellant_unit!r}"
            )
            raise ValueError(msg)
        return propellant * self.specs.fuel_specific_energy * 1e3

    def electric_propellant_to_energy(self, propellant: float) -> float:  # noqa: ARG002
        """Convert electric propellant to energy in Joules.

        Currently conventional assumes no electric propellant is used in
        its consumption.
        """
        return 0

    def _validate_propellant(self) -> None:
        """Validate the vehicle has still propellant."""
        if self.mission_usable_propellant + self.specs.reserve_propellant < 0:
            raise ValueError("Agent ran out of fuel propellant.")

    def _initialize_refill_propellant(self) -> None:
        """Initialize the refill of propellant."""
        super()._initialize_refill_propellant()

    def _refill_propellant(self) -> None:
        """Refill the propellant during one timestep."""
        self.mission_usable_propellant = min(
            self.mission_usable_propellant
            + self.specs.refueling_rate * self.time_step.total_seconds(),
            self.specs.total_mission_usable_propellant,
        )

    def _consume_propellant(self) -> None:
        """Consume the propellant during one timestep."""
        self.mission_usable_propellant -= self._exact_consumption_functions[
            self.flight_state
        ](self.time_step.total_seconds(), self.current_mass)

    def step(self) -> None:
        """Run the propulsion for a single iteration."""
        if self.flight_state != self._current_segment_state:
            self._current_segment_state = self.flight_state
            self._current_segment_start_mass = self.current_mass
        super().step()

    def is_propellant_available(
        self,
        propellant_required: float,
        propellant_available: float | None = None,
    ) -> bool:
        """Check if enough propellant is available for the mission."""
        if propellant_available is None:
            propellant_available = self.mission_usable_propellant
        return propellant_required <= propellant_available

    def propellant_after_refill_duration(
        self, duration: float, initial_propellant: float | None = None
    ) -> float:
        """Estimate propellant after refill for a duration."""
        if initial_propellant is None:
            initial_propellant = self.mission_usable_propellant
        return min(
            initial_propellant + self.specs.refueling_rate * duration,
            self.specs.total_mission_usable_propellant,
        )

    def required_refill_duration(
        self,
        propellant_required: float,
        initial_propellant: float | None = None,
    ) -> tuple[float, float]:
        """Estimate duration to refill to required propellant.

        Attributes:
            propellant_required: Minimal propellant after refill.
            initial_propellant: Propellant at the start of refill. If
                None, the current propellant state and refill progress
                is used.

        Returns:
            Tuple of refill time and propellant after refill.
        """
        if initial_propellant is None:
            initial_propellant = self.mission_usable_propellant
        if initial_propellant >= propellant_required:
            return 0.0, initial_propellant
        return (
            propellant_required - initial_propellant
        ) / self.specs.refueling_rate, propellant_required

    def max_duration_for_propellant(
        self,
        flight_state: FlightState,
        initial_propellant: float,
        start_mass: float | None = None,
    ) -> float:
        """Estimate maximum duration for a flight state.

        Args:
            flight_state: Flight state for which to estimate the
                duration.
            initial_propellant: Initial propellant for the flight state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Maximum duration for the flight state.
        """
        if start_mass is None:
            start_mass = self.mtom
        return self._exact_duration_functions[flight_state](
            initial_propellant, start_mass
        )

    def estimate_propellant_consumption(
        self,
        state_duration_segments: Iterable[tuple[FlightState, float]],
        start_mass: float | None = None,
    ) -> tuple[float, float]:
        """Estimate propellant consumption for a trajectory.

        Args:
            state_duration_segments: Iterable of tuples of flight state
                and duration of that state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Tuple of propellant consumption and the mass at the end of
            the trajectory.
        """
        if start_mass is None:
            start_mass = self.mtom
        mass = start_mass
        for state, duration in state_duration_segments:
            mass -= self._exact_consumption_functions[state](duration, mass)
        return start_mass - mass, mass


class HybridElectricPropulsion(BasePropulsion[HybridElectricFuelState]):
    """Propulsion for hybrid electric vehicles."""

    __slots__ = (
        "_conventional",
        "_electric",
    )

    specs: HybridElectricPropulsionInput

    def __init__(self, specs: HybridElectricPropulsionInput, agent: Agent):
        self._conventional = ConventionalPropulsion(specs.conventional, agent)
        self._electric = BatteryElectricPropulsion(specs.electric, agent)
        super().__init__(specs, agent)
        self._total_consumption = HybridElectricFuelState(
            battery_energy=0.0, fuel=0.0
        )

    @property
    def mission_usable_propellant(self) -> HybridElectricFuelState:
        """Current mission usable propellant of the agent."""
        return HybridElectricFuelState(
            battery_energy=self._electric.mission_usable_propellant,
            fuel=self._conventional.mission_usable_propellant,
        )

    @mission_usable_propellant.setter
    def mission_usable_propellant(self, v: HybridElectricFuelState) -> None:
        """Set the current mission usable propellant of the agent."""
        self._electric.mission_usable_propellant = v.battery_energy
        self._conventional.mission_usable_propellant = v.fuel

    def propellant_to_mass(self, propellant: HybridElectricFuelState) -> float:
        """Convert propellant to mass in kilograms."""
        return self._electric.propellant_to_mass(
            propellant.battery_energy
        ) + self._conventional.propellant_to_mass(propellant.fuel)

    def propellant_to_energy(
        self, propellant: HybridElectricFuelState
    ) -> float:
        """Convert propellant to energy in Joules."""
        return self._electric.propellant_to_energy(
            propellant.battery_energy
        ) + self._conventional.propellant_to_energy(propellant.fuel)

    def electric_propellant_to_energy(
        self, propellant: HybridElectricFuelState
    ) -> float:
        """Convert electric propellant to energy in Joules."""
        return self._electric.electric_propellant_to_energy(
            propellant.battery_energy
        ) + self._conventional.electric_propellant_to_energy(propellant.fuel)

    def _validate_propellant(self) -> None:
        """Validate the vehicle has still propellant."""
        self._electric._validate_propellant()
        self._conventional._validate_propellant()

    def _initialize_refill_propellant(self) -> None:
        """Initialize the refill of propellant."""
        super()._initialize_refill_propellant()
        self._electric._initialize_refill_propellant()
        self._conventional._initialize_refill_propellant()

    def _refill_propellant(self) -> None:
        """Refill the propellant during one timestep."""
        self._electric._refill_propellant()
        self._conventional._refill_propellant()

    def _consume_propellant(self) -> None:
        """Consume the propellant during one timestep."""
        self._electric._consume_propellant()
        self._conventional._consume_propellant()

    def is_propellant_available(
        self,
        propellant_required: HybridElectricFuelState,
        propellant_available: HybridElectricFuelState | None = None,
    ) -> bool:
        """Check if enough propellant is available for the mission."""
        if propellant_available is None:
            propellant_available = self.mission_usable_propellant
        return self._electric.is_propellant_available(
            propellant_required.battery_energy,
            propellant_available.battery_energy,
        ) and self._conventional.is_propellant_available(
            propellant_required.fuel,
            propellant_available.fuel,
        )

    def propellant_after_refill_duration(
        self,
        duration: float,
        initial_propellant: HybridElectricFuelState | None = None,
    ) -> HybridElectricFuelState:
        """Estimate propellant after refill for a duration."""
        if initial_propellant is None:
            initial_energy = None
            initial_fuel = None
        else:
            initial_energy = initial_propellant.battery_energy
            initial_fuel = initial_propellant.fuel
        return HybridElectricFuelState(
            battery_energy=self._electric.propellant_after_refill_duration(
                duration, initial_energy
            ),
            fuel=self._conventional.propellant_after_refill_duration(
                duration, initial_fuel
            ),
        )

    def required_refill_duration(
        self,
        propellant_required: HybridElectricFuelState,
        initial_propellant: HybridElectricFuelState | None = None,
    ) -> tuple[float, HybridElectricFuelState]:
        """Estimate duration to refill to required propellant.

        Attributes:
            propellant_required: Minimal propellant after refill.
            initial_propellant: Propellant at the start of refill. If
                None, the current propellant state and refill progress
                is used.

        Returns:
            Tuple of refill time and propellant after refill.
        """
        if initial_propellant is None:
            initial_energy = None
            initial_fuel = None
        else:
            initial_energy = initial_propellant.battery_energy
            initial_fuel = initial_propellant.fuel
        t_elec, energy = self._electric.required_refill_duration(
            propellant_required.battery_energy,
            initial_energy,
        )
        t_fuel, fuel = self._conventional.required_refill_duration(
            propellant_required.fuel, initial_fuel
        )
        if t_elec > t_fuel:
            t_max = t_elec
            fuel = self._conventional.propellant_after_refill_duration(
                t_max, initial_fuel
            )
        else:
            t_max = t_fuel
            energy = self._electric.propellant_after_refill_duration(
                t_max, initial_energy
            )
        return (
            t_max,
            HybridElectricFuelState(battery_energy=energy, fuel=fuel),
        )

    def max_duration_for_propellant(
        self,
        flight_state: FlightState,
        initial_propellant: HybridElectricFuelState,
        start_mass: float | None = None,
    ) -> float:
        """Estimate maximum duration for a flight state.

        Args:
            flight_state: Flight state for which to estimate the
                duration.
            initial_propellant: Initial propellant for the flight state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Maximum duration for the flight state.
        """
        if start_mass is None:
            start_mass = self.mtom
        return min(
            self._electric.max_duration_for_propellant(
                flight_state, initial_propellant.battery_energy, start_mass
            ),
            self._conventional.max_duration_for_propellant(
                flight_state, initial_propellant.fuel, start_mass
            ),
        )

    def estimate_propellant_consumption(
        self,
        state_duration_segments: list[tuple[FlightState, float]],
        start_mass: float | None = None,
    ) -> tuple[HybridElectricFuelState, float]:
        """Estimate propellant consumption for a trajectory.

        Args:
            state_duration_segments: List of tuples of flight state and
                duration of that state.
            start_mass: Initial mass of the agent. If None, the maximum
                takeoff mass is used.

        Returns:
            Tuple of propellant consumption and the mass at the end of
            the trajectory.
        """
        if start_mass is None:
            start_mass = self.mtom
        energy, _ = self._electric.estimate_propellant_consumption(
            state_duration_segments, start_mass
        )
        fc, mass = self._conventional.estimate_propellant_consumption(
            state_duration_segments, start_mass
        )
        return HybridElectricFuelState(battery_energy=energy, fuel=fc), mass


class PropulsionArchitecture(Enum):
    """Propulsion architectures."""

    ELECTRIC = "electric"
    CONVENTIONAL = "conventional"
    HYBRID = "hybrid"

    @property
    def propulsion_cls(self) -> type[BasePropulsion]:
        """Return the propulsion class for the architecture."""
        return {
            PropulsionArchitecture.ELECTRIC: BatteryElectricPropulsion,
            PropulsionArchitecture.CONVENTIONAL: ConventionalPropulsion,
            PropulsionArchitecture.HYBRID: HybridElectricPropulsion,
        }[self]
