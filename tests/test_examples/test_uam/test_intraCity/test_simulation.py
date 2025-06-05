import pytest

from examples.uam.intraCity.demand_model.agents import RevenueDemand
from examples.uam.intraCity.simulation import (
    IntraCitySimParametersPD,
    IntraCitySimulationPD,
)
from examples.uam.mission import TransportMode
from examples.uam.paths import SCENARIOS_DIR
from sosid.util.general_funcs import combine_parameters

DEFAULTS_FILE = SCENARIOS_DIR / "inputs" / "sample_parametric_input.json"


@pytest.mark.parametrize("settings", [{}])
def test_runs_without_errors_pd(settings):
    """Tests whether the simulation runs without errors.

    Args:
        settings (dict[str, any]): The settings to use in the
            simulation.
    """
    settings = combine_parameters(DEFAULTS_FILE, settings)
    settings["max_runtime"] = 3 * settings["time_step"]
    parameters = IntraCitySimParametersPD.model_validate(settings)
    sim = IntraCitySimulationPD(parameters=parameters, seed=0)
    sim.start()
    sim.join()
    assert sim.iterations == 3


@pytest.mark.slow
class TestIntraCitySimulationOutputPD:
    """Serie of validation tests on the final state of a simulation.

    As the simulation object created in the fixture is shared among all
    tests, it is important to ensure that the simulation object is not
    modified in any way. Otherwise, the tests may not be independent of
    each other.
    """

    @pytest.fixture(scope="class")
    def simulation(self):
        """Run a simulation to be reused in multiple tests."""
        settings = combine_parameters(DEFAULTS_FILE, {"hours_to_simulate": 2})
        parameters = IntraCitySimParametersPD.model_validate(settings)
        sim = IntraCitySimulationPD(parameters=parameters, seed=42)
        sim.start()
        sim.join()
        return sim

    def test_simulation_includes_events(self, simulation):
        """Verify that the simulation includes certain events."""
        sim: IntraCitySimulationPD = simulation
        assert sim.transporters.uam_fleet_operator.n_total_flights > 0
        assert sim.transporters.uam_fleet_operator.n_deadhead_flights > 0
        assert sim.transporters.uam_fleet_operator.n_revenue_flights > 0

    @pytest.mark.parametrize("sim_fixture", ["simulation"])
    def test_simulation_valid(self, sim_fixture, request):
        sim: IntraCitySimulationPD = request.getfixturevalue(sim_fixture)
        operator = sim.transporters.uam_fleet_operator
        all_missions = [
            mission
            for queue in operator.all_scheduled_missions.values()
            for mission in queue
        ]
        all_passengers = sim.demands.agents_by_type[RevenueDemand]
        all_uam_passengers = list(
            filter(
                lambda p: p.mode_choice == TransportMode.UAM, all_passengers
            )
        )
        assert isinstance(sim, IntraCitySimulationPD)
        assert sim.is_stopped
        assert len(all_missions) > 0, "No missions are scheduled."
        assert len(all_missions) == len(set(all_missions)), (
            "There are duplicate missions."
        )
        for mission in all_missions:
            assert len(mission.passengers) == len(set(mission.passengers)), (
                "Duplicate passengers found in a single mission."
            )
        assert all(len(p.itineraries) > 0 for p in all_uam_passengers), (
            "There are passengers that chose UAM without itineraries."
        )

    def test_passenger_counters(self, simulation):
        sim: IntraCitySimulationPD = simulation
        all_passengers = set(sim.demands.agents_by_type[RevenueDemand]) - set(
            sim.demands.__demand_queue__
        )
        all_uam_passengers = list(
            filter(
                lambda p: p.mode_choice == TransportMode.UAM, all_passengers
            )
        )
        n_legs = sum(len(p.itineraries) for p in all_uam_passengers)
        assert sim.dispatcher.mode_choice_counter[TransportMode.UAM] == len(
            all_uam_passengers
        ), "Mismatch in UAM passenger count."
        assert sim.dispatcher.mode_choice_counter[TransportMode.CAR] == len(
            all_passengers
        ) - len(all_uam_passengers), "Mismatch in car passenger count."
        assert (
            sim.transporters.uam_fleet_operator.n_passengers_in_schedule
            + sim.transporters.uam_fleet_operator.n_passengers_flown
            == n_legs
        )
        assert (
            sum(
                aircraft.total_missions_flown
                for aircraft in sim.transporters.transport_aircraft
            )
            == sim.transporters.uam_fleet_operator.n_total_flights
        )
        assert (
            sum(
                aircraft.total_passengers_flown
                for aircraft in sim.transporters.transport_aircraft
            )
            == sim.transporters.uam_fleet_operator.n_passengers_flown
        )
