# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from datetime import timedelta
from operator import add, mul, sub, truediv
from typing import ClassVar

import numpy as np
import pytest

from sosid.model.abm.agent import Agent
from sosid.model.abm.propulsion import (
    BasePropulsion,
    BasePropulsionInput,
    BatteryElectricPropulsion,
    ConventionalPropulsion,
    FlightState,
    HybridElectricFuelState,
    HybridElectricPropulsion,
    PropulsionArchitecture,
)
from tests.test_sosid.test_model.test_abm.test_agent import MockModel


class TestHybridFuelBatteryState:
    """Test the mathematical operations of the class."""

    @pytest.mark.parametrize(
        ("operation", "e_0", "f_0", "e_1", "f_1", "e_exp", "f_exp", "check_r"),
        [
            (add, 10, 6, 1, 2, 11, 8, True),
            (add, 2, 5, 3, 1, 5, 6, True),
            (sub, 10, 6, 1, 2, 9, 4, False),
            (sub, 4, 5, 3, 1, 1, 4, False),
        ],
    )
    def test_operation_state_state(
        self, operation, e_0, f_0, e_1, f_1, e_exp, f_exp, check_r
    ):
        """Test operations between class instances."""
        state_0 = HybridElectricFuelState(battery_energy=e_0, fuel=f_0)
        state_1 = HybridElectricFuelState(battery_energy=e_1, fuel=f_1)
        expected = HybridElectricFuelState(battery_energy=e_exp, fuel=f_exp)
        assert operation(state_0, state_1) == expected
        if check_r:
            assert operation(state_1, state_0) == expected
        elif check_r is None:
            with pytest.raises(TypeError):
                operation(state_1, state_0)

    @pytest.mark.parametrize(
        ("operation", "e_0", "f_0", "number", "e_exp", "f_exp", "check_r"),
        [
            (mul, 10, 6, 2, 20, 12, True),
            (mul, 4, 5, 1.5, 6, 7.5, True),
            (truediv, 10, 6, 2, 5, 3, None),
            (truediv, 4.5, 9.3, 1.5, 3, 6.2, None),
        ],
    )
    def test_operation_state_number(
        self, operation, e_0, f_0, number, e_exp, f_exp, check_r
    ):
        """Test operations between class instances and numbers."""
        state = HybridElectricFuelState(battery_energy=e_0, fuel=f_0)
        expected = HybridElectricFuelState(battery_energy=e_exp, fuel=f_exp)
        assert operation(state, number) == expected
        if check_r:
            assert operation(number, state) == expected
        elif check_r is None:
            with pytest.raises(TypeError):
                operation(number, state)

    @pytest.mark.parametrize("other", [2, 2.0])
    @pytest.mark.parametrize("operation", [add, sub])
    def test_operation_state_number_invalid(self, other, operation):
        """Test invalid operations states and numbers."""
        state = HybridElectricFuelState(battery_energy=10, fuel=6)
        with pytest.raises(TypeError):
            operation(state, other)
        with pytest.raises(TypeError):
            operation(other, state)

    @pytest.mark.parametrize("operation", [mul, truediv])
    def test_operation_state_state_invalid(self, operation):
        """Test invalid operations between class instances."""
        state_0 = HybridElectricFuelState(battery_energy=10, fuel=6)
        state_1 = HybridElectricFuelState(battery_energy=1, fuel=2)
        with pytest.raises(TypeError):
            operation(state_0, state_1)
        with pytest.raises(TypeError):
            operation(state_1, state_0)

    @pytest.mark.parametrize(
        ("state_0", "state_1", "expected"),
        [
            (
                HybridElectricFuelState(battery_energy=10, fuel=6),
                HybridElectricFuelState(battery_energy=10, fuel=6),
                True,
            ),
            (
                HybridElectricFuelState(battery_energy=10, fuel=6),
                HybridElectricFuelState(battery_energy=1, fuel=2),
                False,
            ),
            (
                HybridElectricFuelState(battery_energy=10, fuel=6),
                (10, 2),
                False,
            ),
        ],
    )
    def test_eq(self, state_0, state_1, expected):
        """Test the equality operator."""
        assert (state_0 == state_1) == expected


class MockAgentWithPropulsion(Agent):
    def __init__(
        self,
        propulsion_input: BasePropulsionInput,
        propulsion_cls: type[BasePropulsion],
    ):
        super().__init__(
            unique_id=1,
            model=MockModel([1]),
            autopopulate=False,
        )
        self.model.simulation.time_step = timedelta(seconds=2.0)
        self._flight_state = FlightState.IDLE
        self._max_mass = 5000.0
        self.propulsion = propulsion_cls(propulsion_input, self)

    @property
    def max_mass(self):
        return self._max_mass

    @property
    def current_mass(self):
        return 3500.0 + self.propulsion.propellant_mass

    @property
    def flight_state(self):
        return self._flight_state

    @flight_state.setter
    def flight_state(self, value):
        self._flight_state = value

    @property
    def time_step(self):
        return self.model.simulation.time_step

    def step(self):
        self.propulsion.step()
        self.model.simulation.step_timer()


ELECTRIC_PARAMETERS = {
    "architecture": "electric",
    "total_propellant": 550_000.0,
    "total_mission_usable_propellant": 500_000.0,
    "taxi_out_power": 40.0,
    "transition_power": 300.0,
    "retransition_power": 300.0,
    "cruise_power": 280.0,
    "cruise_climb_power": 370.0,
    "takeoff_power": 400.0,
    "landing_power": 400.0,
    "cruise_descent_power": 200.0,
    "taxi_in_power": 40.0,
    "loiter_power": 200.0,
    "hover_power": 300.0,
    "charging_power": 360.0,
    "battery_swap_enabled": False,
    "battery_swap_time": 300.0,
}
ELECTRIC_PROPELLANT_REDUCTION_STEP = {
    state: ELECTRIC_PARAMETERS.get(f"{state.name.lower()}_power", 0.0) * 2
    for state in FlightState
}
CONVENTIONAL_PARAMETERS = {
    "architecture": "conventional",
    "total_propellant": 1500.0,
    "total_mission_usable_propellant": 1100.0,
    "taxi_out_fc": 0.015,
    "transition_fc": 0.0,
    "retransition_fc": 0.0,
    "cruise_climb_fc": {3800: 0.05, 3900: 0.06},
    "cruise_fc": 0.03,
    "cruise_descent_fc": 0.04,
    "taxi_in_fc": 0.015,
    "loiter_fc": 0.04,
    "takeoff_fc": 0.015,
    "landing_fc": 0.02,
    "refueling_rate": 15.0,
}
CONVENTIONAL_PROPELLANT_REDUCTION_STEP = {
    state: fc * 2
    for state in FlightState
    if not isinstance(
        (fc := CONVENTIONAL_PARAMETERS.get(f"{state.name.lower()}_fc", 0.0)),
        dict,
    )
}
CONVENTIONAL_PROPELLANT_REDUCTION_STEP[FlightState.CRUISE_CLIMB] = 0.33996600


class PropulsionTester:
    parameters: ClassVar[dict[str, object]]
    propellant_reduction_step: ClassVar[dict[FlightState, float]]
    propellant_after_one_refill_step: ClassVar[float]
    n_steps_to_refill: ClassVar[int]

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        specs = BasePropulsionInput.model_validate(self.parameters)
        arch = PropulsionArchitecture(specs.architecture)
        self.agent = MockAgentWithPropulsion(specs, arch.propulsion_cls)

    @staticmethod
    def compare_propellant(propellant, expected):
        np.testing.assert_almost_equal(propellant, expected)

    @staticmethod
    def zero_mission_usable_propellant(propulsion: BasePropulsion):
        propulsion.mission_usable_propellant = 0.0

    @pytest.mark.parametrize("flight_state", FlightState)
    def test_consume_propellant(self, flight_state):
        self.agent.flight_state = flight_state
        prev = self.agent.propulsion.mission_usable_propellant
        self.agent.step()
        if flight_state == FlightState.ENERGIZE:
            # Should maintain full propellant.
            self.compare_propellant(
                prev, self.agent.propulsion.mission_usable_propellant
            )
        else:
            self.compare_propellant(
                prev - self.agent.propulsion.mission_usable_propellant,
                self.propellant_reduction_step[flight_state],
            )
            self.compare_propellant(
                self.agent.propulsion.total_propellant_consumed,
                self.propellant_reduction_step[flight_state],
            )

    @pytest.mark.parametrize("flight_state", FlightState)
    def test_runout_of_propellant(self, flight_state):
        self.agent.flight_state = flight_state
        self.agent.propulsion.mission_usable_propellant = (
            -self.agent.propulsion.specs.reserve_propellant
        )
        if self.propellant_reduction_step.get(flight_state, 0.0):
            with pytest.raises(ValueError, match="propellant"):
                self.agent.step()
        else:  # No propellant reduction expected.
            self.agent.step()

    def test_refill_propellant_single_step(self):
        self.zero_mission_usable_propellant(self.agent.propulsion)
        self.agent.flight_state = FlightState.ENERGIZE
        self.agent.step()
        self.compare_propellant(
            self.agent.propulsion.mission_usable_propellant,
            self.propellant_after_one_refill_step,
        )

    def test_refill_propellant_to_full(self):
        assert self.agent.propulsion.n_propellant_refills == 0
        self.zero_mission_usable_propellant(self.agent.propulsion)
        self.agent.flight_state = FlightState.ENERGIZE
        for _ in range(self.n_steps_to_refill):
            self.agent.step()
        self.compare_propellant(
            self.agent.propulsion.mission_usable_propellant,
            self.agent.propulsion.specs.total_mission_usable_propellant,
        )
        self.agent.flight_state = FlightState.IDLE
        self.agent.step()
        assert self.agent.propulsion.n_propellant_refills == 1


class TestBatteryElectricPropulsionCharging(PropulsionTester):
    parameters = ELECTRIC_PARAMETERS | {"battery_swap_enabled": False}
    propellant_reduction_step = ELECTRIC_PROPELLANT_REDUCTION_STEP
    propellant_after_one_refill_step = parameters["charging_power"] * 2
    n_steps_to_refill = int(
        parameters["total_mission_usable_propellant"]
        // (parameters["charging_power"] * 2)
        + 1
    )

    def test_init(self):
        assert isinstance(self.agent.propulsion, BatteryElectricPropulsion)
        assert self.agent.propulsion.propellant_mass == 0.0
        self.compare_propellant(
            self.agent.propulsion.mission_usable_propellant,
            self.agent.propulsion.specs.total_mission_usable_propellant,
        )

    @pytest.mark.parametrize(
        ("required", "available", "expected"),
        [
            (10_000.0, 20_000.0, True),
            (10_000.0, None, True),
            (16_000.0, None, False),
            (10_000.0, 5_000.0, False),
        ],
    )
    def test_is_propellant_available(self, required, available, expected):
        self.agent.propulsion.mission_usable_propellant = 15_000.0
        assert (
            self.agent.propulsion.is_propellant_available(required, available)
            == expected
        )

    @pytest.mark.parametrize(
        ("intial", "required", "expected_duration", "expected_propellant"),
        [
            (5000.0, 8780.0, 10.5, 8780.0),  # Refill required.
            (3000.0, 2000.0, 0.0, 3000.0),  # No refill required.
        ],
    )
    def test_required_refill_duration(
        self, intial, required, expected_duration, expected_propellant
    ):
        duration, propellant = self.agent.propulsion.required_refill_duration(
            required, intial
        )
        np.testing.assert_almost_equal(duration, expected_duration)
        self.compare_propellant(propellant, expected_propellant)

    @pytest.mark.parametrize(
        ("initial", "duration", "expected_propellant"),
        [
            (5000.0, 10.5, 8780.0),
            (3000.0, 1e5, 500_000),
        ],
    )
    def test_propellant_after_refill_duration(
        self, initial, duration, expected_propellant
    ):
        propellant = self.agent.propulsion.propellant_after_refill_duration(
            duration, initial
        )
        self.compare_propellant(propellant, expected_propellant)


class TestBatteryElectricPropulsionBatterySwap(PropulsionTester):
    parameters = ELECTRIC_PARAMETERS | {"battery_swap_enabled": True}
    propellant_reduction_step = ELECTRIC_PROPELLANT_REDUCTION_STEP
    propellant_after_one_refill_step = 0.0
    n_steps_to_refill = int(parameters["battery_swap_time"] // 2) + 1

    @pytest.mark.parametrize(
        ("intial", "required", "expected_duration", "expected_propellant"),
        [
            (5000.0, 8780.0, 300.0, 500_000.0),  # Refill required.
            (3000.0, 2000.0, 0.0, 3000.0),  # No refill required.
        ],
    )
    def test_required_refill_duration(
        self, intial, required, expected_duration, expected_propellant
    ):
        duration, propellant = self.agent.propulsion.required_refill_duration(
            required, intial
        )
        np.testing.assert_almost_equal(duration, expected_duration)
        self.compare_propellant(propellant, expected_propellant)

    @pytest.mark.parametrize(
        ("initial", "duration", "expected_propellant"),
        [
            (5000.0, 301.0, 500_000.0),
            (3000.0, 50.0, 3000.0),
        ],
    )
    def test_propellant_after_refill_duration(
        self, initial, duration, expected_propellant
    ):
        propellant = self.agent.propulsion.propellant_after_refill_duration(
            duration, initial
        )
        self.compare_propellant(propellant, expected_propellant)

    def test_not_refilled_when_energizing_too_short(self):
        assert self.agent.propulsion.n_propellant_refills == 0
        self.agent.flight_state = FlightState.ENERGIZE
        for _ in range(10):
            self.agent.step()
        self.agent.flight_state = FlightState.IDLE
        self.agent.step()
        assert self.agent.propulsion.n_propellant_refills == 0


class TestConventionalPropulsion(PropulsionTester):
    parameters = CONVENTIONAL_PARAMETERS
    propellant_reduction_step = CONVENTIONAL_PROPELLANT_REDUCTION_STEP
    propellant_after_one_refill_step = (
        CONVENTIONAL_PARAMETERS["refueling_rate"] * 2
    )
    n_steps_to_refill = int(
        CONVENTIONAL_PARAMETERS["total_mission_usable_propellant"]
        // (CONVENTIONAL_PARAMETERS["refueling_rate"] * 2)
        + 1
    )

    def test_init(self):
        assert isinstance(self.agent.propulsion, ConventionalPropulsion)
        assert self.agent.propulsion.propellant_mass == 1500.0
        self.compare_propellant(
            self.agent.propulsion.mission_usable_propellant,
            self.agent.propulsion.specs.total_mission_usable_propellant,
        )

    @pytest.mark.parametrize(
        ("required", "available", "expected"),
        [
            (300.0, 400.0, True),
            (100.0, None, True),
            (160.0, None, False),
            (500.0, 400.0, False),
        ],
    )
    def test_is_propellant_available(self, required, available, expected):
        self.agent.propulsion.mission_usable_propellant = 150.0
        assert (
            self.agent.propulsion.is_propellant_available(required, available)
            == expected
        )

    @pytest.mark.parametrize(
        ("intial", "required", "expected_duration", "expected_propellant"),
        [
            (500.0, 657.5, 10.5, 657.5),  # Refill required.
            (300.0, 200.0, 0.0, 300.0),  # No refill required.
        ],
    )
    def test_required_refill_duration(
        self, intial, required, expected_duration, expected_propellant
    ):
        duration, propellant = self.agent.propulsion.required_refill_duration(
            required, intial
        )
        np.testing.assert_almost_equal(duration, expected_duration)
        self.compare_propellant(propellant, expected_propellant)

    @pytest.mark.parametrize(
        ("initial", "duration", "expected_propellant"),
        [
            (500.0, 10.5, 657.5),
            (300.0, 1e5, 1100.0),
        ],
    )
    def test_propellant_after_refill_duration(
        self, initial, duration, expected_propellant
    ):
        propellant = self.agent.propulsion.propellant_after_refill_duration(
            duration, initial
        )
        self.compare_propellant(propellant, expected_propellant)


HYBRID_CONVENTIONAL_PARAMETERS = {}
for key, value in CONVENTIONAL_PARAMETERS.items():
    if key.endswith("_fc"):
        if isinstance(value, dict):
            HYBRID_CONVENTIONAL_PARAMETERS[key] = {
                mass: value[mass] * 0.75 for mass in value
            }
        else:
            HYBRID_CONVENTIONAL_PARAMETERS[key] = value * 0.75
    else:
        HYBRID_CONVENTIONAL_PARAMETERS[key] = value
HYBRID_ELECTRIC_PARAMETERS = {}
for key, value in ELECTRIC_PARAMETERS.items():
    if key.endswith("_power") and key != "charging_power":
        HYBRID_ELECTRIC_PARAMETERS[key] = value * 0.25
    else:
        HYBRID_ELECTRIC_PARAMETERS[key] = value
HYBRID_PARAMETERS = {
    "architecture": "hybrid",
    "electric": HYBRID_ELECTRIC_PARAMETERS,
    "conventional": HYBRID_CONVENTIONAL_PARAMETERS,
    "hybridization_ratio": 0.25,
}
HYBRID_PROPELLANT_REDUCTION_STEP = {
    state: HybridElectricFuelState(
        ELECTRIC_PROPELLANT_REDUCTION_STEP.get(state, 0.0) * 0.25,
        CONVENTIONAL_PROPELLANT_REDUCTION_STEP.get(state, 0.0) * 0.75,
    )
    for state in FlightState
}
HYBRID_PROPELLANT_REDUCTION_STEP[FlightState.CRUISE_CLIMB] = (
    HybridElectricFuelState(
        ELECTRIC_PROPELLANT_REDUCTION_STEP[FlightState.CRUISE_CLIMB] * 0.25,
        0.25498088,
    )
)


class TestHybridElectricPropulsion(PropulsionTester):
    parameters = HYBRID_PARAMETERS
    propellant_reduction_step = HYBRID_PROPELLANT_REDUCTION_STEP
    propellant_after_one_refill_step = HybridElectricFuelState(
        ELECTRIC_PARAMETERS["charging_power"] * 2,
        CONVENTIONAL_PARAMETERS["refueling_rate"] * 2,
    )
    n_steps_to_refill = int(
        ELECTRIC_PARAMETERS["total_mission_usable_propellant"]
        // (ELECTRIC_PARAMETERS["charging_power"] * 2)
        + 1
    )

    @staticmethod
    def compare_propellant(propellant, expected):
        np.testing.assert_allclose(
            [propellant.battery_energy, propellant.fuel],
            [expected.battery_energy, expected.fuel],
        )

    @staticmethod
    def zero_mission_usable_propellant(propulsion: BasePropulsion):
        propulsion.mission_usable_propellant = HybridElectricFuelState(
            0.0, 0.0
        )

    def test_init(self):
        assert isinstance(self.agent.propulsion, HybridElectricPropulsion)
        assert self.agent.propulsion.propellant_mass == 1500.0
        assert not self.agent.propulsion.specs.electric.battery_swap_enabled
        self.compare_propellant(
            self.agent.propulsion.mission_usable_propellant,
            self.agent.propulsion.specs.total_mission_usable_propellant,
        )

    @pytest.mark.parametrize(
        ("required", "available", "expected"),
        [
            (HybridElectricFuelState(10_000.0, 100.0), None, True),
            (HybridElectricFuelState(10_000.0, 200.0), None, False),
            (HybridElectricFuelState(30_000.0, 100.0), None, False),
            (
                HybridElectricFuelState(10_000.0, 100.0),
                HybridElectricFuelState(10_100.0, 200.0),
                True,
            ),
            (
                HybridElectricFuelState(10_000.0, 100.0),
                HybridElectricFuelState(40_000.0, 50.0),
                False,
            ),
            (
                HybridElectricFuelState(10_000.0, 100.0),
                HybridElectricFuelState(9000.0, 300.0),
                False,
            ),
        ],
    )
    def test_is_propellant_available(self, required, available, expected):
        self.agent.propulsion.mission_usable_propellant = (
            HybridElectricFuelState(20_000.0, 150.0)
        )
        assert (
            self.agent.propulsion.is_propellant_available(required, available)
            == expected
        )

    @pytest.mark.parametrize(
        ("intial", "required", "expected_duration", "expected_propellant"),
        [
            (
                HybridElectricFuelState(5000.0, 200.0),
                HybridElectricFuelState(8780.0, 100.0),
                10.5,
                HybridElectricFuelState(8780.0, 357.5),
            ),  # Electric refill required.
            (
                HybridElectricFuelState(5000.0, 100.0),
                HybridElectricFuelState(3000.0, 415.0),
                21.0,
                HybridElectricFuelState(12560.0, 415.0),
            ),  # Fuel refill required.
            (
                HybridElectricFuelState(3000.0, 300.0),
                HybridElectricFuelState(2000.0, 200.0),
                0.0,
                HybridElectricFuelState(3000.0, 300.0),
            ),  # No refill required.
        ],
    )
    def test_required_refill_duration(
        self, intial, required, expected_duration, expected_propellant
    ):
        duration, propellant = self.agent.propulsion.required_refill_duration(
            required, intial
        )
        np.testing.assert_almost_equal(duration, expected_duration)
        self.compare_propellant(propellant, expected_propellant)

    @pytest.mark.parametrize(
        ("initial", "duration", "expected_propellant"),
        [
            (
                HybridElectricFuelState(5000.0, 200.0),
                10.5,
                HybridElectricFuelState(8780.0, 357.5),
            ),
            (
                HybridElectricFuelState(3000.0, 300.0),
                1e5,
                HybridElectricFuelState(500_000.0, 1100.0),
            ),
        ],
    )
    def test_propellant_after_refill_duration(
        self, initial, duration, expected_propellant
    ):
        propellant = self.agent.propulsion.propellant_after_refill_duration(
            duration, initial
        )
        self.compare_propellant(propellant, expected_propellant)
