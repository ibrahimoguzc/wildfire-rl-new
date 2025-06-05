# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import math
import threading
import time
from datetime import timedelta
from functools import cached_property
from random import Random

import pytest
from pydantic import ValidationError

from sosid.abstract import Model
from sosid.simulation import Distribution, Simulation, SimulationParameters

STEP_DURATION = 1e-6  # Amount of time an iteration takes


class MockModel(Model):
    """Mock model used to test the :py:class:`Simulation`."""

    def step(self):
        """Dummy step method used to spy on."""
        time.sleep(STEP_DURATION / 2)  # Allow predictable behavior in thread

    def reset(self):
        pass

    def __gui_repr__(self):  # noqa: D105
        pass


class MockSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        self.foo = MockModel()
        self.bar = MockModel()
        # Allow waiting until a step occurs in the tests
        self.has_stepped = threading.Event()
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        """Override step to make sure tests can await it."""
        self.has_stepped.clear()
        super().step(*args, **kwargs)
        self.has_stepped.set()

    def environment(self):
        pass

    @cached_property
    def models(self) -> list[Model]:
        """List of mock models."""
        return [self.foo, self.bar]

    def output_collector(self):  # noqa: ANN201
        """Dummy output method used to spy on."""


@pytest.fixture
def max_runtime() -> float:
    """Default value for max_runtime."""
    return 10


@pytest.fixture
def time_step() -> float:
    """Default value for time_step."""
    return 1


@pytest.fixture
def name() -> str:
    """Default value for name."""
    return "MockSim"


@pytest.fixture(scope="function")
def simulation(max_runtime, time_step, name) -> Simulation:
    """Create a mock :py:class:`Simulation` for tests."""
    param = SimulationParameters(
        time_step=time_step, max_runtime=max_runtime, name=name
    )

    sim = MockSimulation(param)

    yield sim  # Provide the simulation object

    # Ensure the simulation is stopped on test exit
    if sim.is_alive():
        sim.stop()
        sim.join(timeout=1)


@pytest.fixture(scope="function")
def started_simulation(simulation: MockSimulation) -> MockSimulation:
    """Creates a mock :py:class:`Simulation` that has started."""
    simulation.start()
    simulation.has_stepped.wait()
    assert simulation.timer.runtime > timedelta(seconds=0)
    assert simulation.timer.mission_runtime > timedelta(seconds=0)
    assert simulation.iterations > 0
    return simulation


class MockParameters(SimulationParameters):
    spam: Distribution = Distribution(
        random_fn=Random.uniform, kwargs={"a": 0, "b": 10}
    )
    ham: Distribution = Distribution(
        random_fn=Random.randint, kwargs={"a": 0, "b": 1}
    )
    response_time: Distribution = Distribution(
        random_fn=Random.uniform, kwargs={"a": 600, "b": 600}
    )


class TestSimulationParameters:
    """Test initialization of Simulation Parameters."""

    def test_evaluate_distributions(self, time_step, max_runtime):
        """Tests if distributions are correctly evaluated."""
        param = MockParameters(time_step=time_step, max_runtime=max_runtime)
        rng = Random(1)
        param.evaluate_distributions(rng=rng)

        assert param.spam == pytest.approx(1.34362, rel=1e-3)
        assert param.ham == 0


class TestSimulation:
    test_class = MockSimulation

    @pytest.mark.parametrize(
        argnames="max_runtime",
        argvalues=[
            pytest.param(5, id="max_runtime=5"),
            pytest.param(
                math.inf,
                marks=pytest.mark.xfail(raises=ValidationError),
                id="infinity",
            ),
            pytest.param(
                3.1415,
                marks=pytest.mark.xfail(raises=ValidationError),
                id="float",
            ),
            pytest.param(
                "str",
                marks=pytest.mark.xfail(raises=ValidationError),
                id="str",
            ),
        ],
    )
    def test_max_runtime(self, simulation: MockSimulation, max_runtime):
        """Tests maximum number of iterations constraint."""
        for _ in range(max_runtime):
            simulation.step(force=True)

        with pytest.raises(
            RuntimeError, match=r"^Maximum[\D]+\(max_iter = 5\)[\D]+exceeded"
        ):
            assert simulation.iterations == 5
            simulation.step(force=True)

    @pytest.mark.parametrize(argnames="name", argvalues=["MockFooSim", None])
    def test_name(self, simulation: MockSimulation, name):
        """Tests if name was correctly set."""
        if name is not None:
            assert simulation.name == name
        else:
            assert simulation.name == str(simulation)

    def test_start(self, simulation: MockSimulation, mocker):
        """Tests if starting a simulation steps through the models."""
        foo_step = mocker.spy(simulation.foo, "step")
        bar_step = mocker.spy(simulation.bar, "step")

        simulation.start()
        assert simulation.is_alive()
        assert simulation.is_running.is_set()

        simulation.has_stepped.wait()
        assert simulation.iterations > 0
        assert simulation.timer.runtime > timedelta(seconds=0)
        assert simulation.timer.mission_runtime > timedelta(seconds=0)

        # Check if all models were called
        assert foo_step.call_count > 0 and bar_step.call_count > 0

        # Ensure a simulation cannot be started twice
        with pytest.raises(RuntimeError, match="can only be started once"):
            simulation.start()

    @staticmethod
    def check_incrementables(simulation: MockSimulation) -> None:
        """Check if``simulation`` is actually paused/stopped.

        Iterations and runtime/mission_time should be constant, these
        are refered to as incrementables.
        """
        n_iter = simulation.iterations
        start_runtime = simulation.timer.runtime
        start_mission_time = simulation.timer.mission_runtime

        # Check if values are constant after some time has passed
        time.sleep(STEP_DURATION)
        assert n_iter == simulation.iterations
        assert simulation.timer.runtime == start_runtime
        assert simulation.timer.mission_runtime == start_mission_time

    def test_pause_play(self, started_simulation: MockSimulation):
        """Tests if the simulation can be paused and resumed."""
        simulation = started_simulation
        simulation.pause()
        simulation.has_stepped.wait()

        assert simulation.is_alive()
        assert not simulation.is_stopped.is_set()
        assert not simulation.is_running.is_set()

        # Ensure simulation is actually paused
        self.check_incrementables(simulation)

        # Ensure simulation can be resumed
        simulation.play()

        assert simulation.is_alive()
        assert not simulation.is_stopped.is_set()
        assert simulation.is_running.is_set()

    def test_stop(self, started_simulation: MockSimulation):
        """Test function of stopping a Simulation."""
        simulation = started_simulation

        simulation.stop()
        simulation.join()
        assert not simulation.is_alive()

        # Ensure timers are stopped along with the Simulation
        self.check_incrementables(simulation)

    def test_stop_raises_runtime_error(self, simulation: MockSimulation):
        """Test if stopping a ``simulation`` before it starts fails."""
        with pytest.raises(RuntimeError, match="cannot be stopped"):
            simulation.stop()

    @pytest.mark.parametrize(
        "n_steps",
        [10, pytest.param(11, marks=pytest.mark.xfail(raises=RuntimeError))],
    )
    def test_step_for(self, simulation: MockSimulation, n_steps):
        """Tests if step_for functions and still raises error."""
        simulation.step_for(n_steps, force=True)
        assert simulation.iterations == n_steps

    @pytest.mark.parametrize("time_step", [1, 0.01])
    def test_time_step(self, simulation: MockSimulation, time_step):
        """Test if ``time_step`` correctl affects timer."""
        simulation.step(force=True)
        assert simulation.timer.mission_runtime == timedelta(seconds=time_step)

    def test_output(self, simulation, mocker):
        """Tests if output is called at the end of a simulation."""
        output = mocker.spy(simulation, "output_collector")
        simulation.start()
        simulation.stop()

        assert output.call_count == 1
