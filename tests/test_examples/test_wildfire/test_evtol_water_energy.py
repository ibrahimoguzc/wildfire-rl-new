"""Regression tests for EVTOL water-stop energy accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.wildfire.firefighter_model.follower import DestinationType
from examples.wildfire.firefighter_model.suppression_tactics import (
    SuppresionTactic,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    EvtolFollowFirefrontTrackPOI,
    FollowFirefrontTrackPOI,
)
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation
from sosid.model.abm.trajectory import FlightState


SCENARIO = Path("examples/wildfire/data/scenarios/inputs/Palisades.json")


@pytest.fixture(scope="module")
def simulation():
    """Build a small mixed conventional/eVTOL Palisades scenario."""
    data = json.loads(SCENARIO.read_text())
    data["run_headless"] = True
    data["agents"][0]["agents_per_base"] = [1, 0]
    data["agents"][1]["file_name"] = "example_aircraft_1.json"
    data["agents"][1]["agents_per_base"] = [0, 1]
    parameters = WildfireParameters.model_validate(data)
    return WildfireSimulation(parameters=parameters, seed=0)


def _water_position(agent):
    """Return one feasible water position for the agent."""
    return np.asarray(agent.feasible_water_locations[0])


def _retransition_propellant(agent, start_mass):
    """Estimate one water-arrival retransition segment."""
    return agent.propulsion.estimate_propellant_consumption(
        [
            (
                FlightState.RETRANSITION,
                agent.profile_parameters.retransition_duration,
            )
        ],
        start_mass=start_mass,
    )[0]


def test_evtol_water_estimate_adds_retransition_without_trajectory_change(
    simulation,
):
    """eVTOL WATER estimates include retransition outside the trajectory."""
    agents = simulation.firefighters.firefighters
    evtol = next(
        agent
        for agent in agents
        if agent._needs_water_retransition_accounting
    )
    water_position = _water_position(evtol)

    evtol_trajectory = evtol.generate_trajectory(
        evtol.pos,
        water_position,
        DestinationType.WATER,
    )

    assert FlightState.RETRANSITION not in evtol_trajectory.flight_states

    trajectory_propellant, mass = (
        evtol.propulsion.estimate_propellant_for_trajectory(
            evtol_trajectory,
            start_mass=evtol.current_mass,
        )
    )
    expected_propellant = trajectory_propellant + _retransition_propellant(
        evtol,
        mass,
    )

    assert evtol.estimate_propellant_for_journey(
        evtol.pos,
        water_position,
        DestinationType.WATER,
    ) == pytest.approx(expected_propellant)


def test_non_evtol_water_estimate_matches_trajectory(simulation):
    """Non-eVTOL WATER estimates keep the previous trajectory-only cost."""
    agents = simulation.firefighters.firefighters
    non_evtol = next(
        agent
        for agent in agents
        if not agent._needs_water_retransition_accounting
    )
    water_position = _water_position(non_evtol)

    trajectory = non_evtol.generate_trajectory(
        non_evtol.pos,
        water_position,
        DestinationType.WATER,
    )
    trajectory_propellant, _ = (
        non_evtol.propulsion.estimate_propellant_for_trajectory(
            trajectory,
            start_mass=non_evtol.current_mass,
        )
    )

    assert FlightState.RETRANSITION not in trajectory.flight_states
    assert non_evtol.estimate_propellant_for_journey(
        non_evtol.pos,
        water_position,
        DestinationType.WATER,
    ) == pytest.approx(trajectory_propellant)


def test_only_evtol_agents_get_water_retransition_accounting(simulation):
    """The narrow eVTOL discriminator leaves non-eVTOL agents untouched."""
    agents = simulation.firefighters.firefighters

    assert sum(
        agent._needs_water_retransition_accounting for agent in agents
    ) == 1


def test_only_evtol_firefront_tracking_gets_return_energy_guard(simulation):
    """Non-eVTOL firefront tracking keeps the original task implementation."""
    agents = simulation.firefighters.firefighters
    evtol = next(
        agent
        for agent in agents
        if agent._needs_water_retransition_accounting
    )
    non_evtol = next(
        agent
        for agent in agents
        if not agent._needs_water_retransition_accounting
    )

    assert isinstance(evtol.tactic.track_poi, EvtolFollowFirefrontTrackPOI)
    assert isinstance(non_evtol.tactic.track_poi, FollowFirefrontTrackPOI)
    assert not isinstance(
        non_evtol.tactic.track_poi,
        EvtolFollowFirefrontTrackPOI,
    )


@pytest.mark.parametrize(
    ("needs_water_retransition_accounting", "expected_state"),
    [
        (True, FlightState.RETRANSITION),
        (False, FlightState.CRUISE_DESCENT),
    ],
)
def test_water_arrival_sets_retransition_state_only_for_evtol_flag(
    needs_water_retransition_accounting,
    expected_state,
):
    """The water hold uses retransition power only for the eVTOL flag."""
    selected_task = object()

    class Tasks:
        def set_active(self, task):
            self.selected = task

    agent = SimpleNamespace(
        _needs_water_retransition_accounting=(
            needs_water_retransition_accounting
        ),
        flight_state=FlightState.CRUISE_DESCENT,
        tasks=Tasks(),
        tactic=SimpleNamespace(retransition_before_resupply=selected_task),
    )

    SuppresionTactic.start_descent_to_suppressant(agent)

    assert agent.flight_state == expected_state
    assert agent.tasks.selected is selected_task
