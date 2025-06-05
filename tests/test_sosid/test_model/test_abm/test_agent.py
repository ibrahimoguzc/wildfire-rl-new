# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import copy
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.sosid.model.abm.agent import (
    Agent,
    AgentWithGPSMixin,
    MovingAgent,
    MovingAgentWithGPS,
    StaticAgent,
)
from src.sosid.model.abm.task import TaskStatus
from tests.snippets import ScenarioTestSuite
from tests.test_sosid.test_model.test_transform import (
    BOTTOM,
    BOUNDS,
    LEFT,
    RIGHT,
    TEST_GPS,
    TOP,
)

DISTANCE_SCENARIOS = {
    "argnames": "label, positions",
    "argvalues": [
        ("distance_positive", (3, 4)),
        ("distance_negative", (-3, -4)),
        ("distance_2D", ((3, 4), (1, 0), (10, 0), (-3, -4))),
    ],
}

EXPECTED_DISTANCE = {
    "distance_positive": 5,
    "distance_negative": 5,
    "distance_2D": (5, 1, 10, 5),
}

DISTANCE_GPS_SCENARIOS = {
    "argnames": "label, coordinates, expected_result",
    "argvalues": [
        (
            "northern_hemisphere",
            (
                (53.5361186422262, 9.86936776053907),
                (53.546339534076445, 9.969412735511114),
            ),
            6708,
        ),
        (
            "equator",
            (
                (4.175249514767575, 73.50068138809274),
                (3.533466656769755, 72.9350682106876),
            ),
            95030,
        ),
        (
            "southern_hemisphere",
            (
                (-41.308438535545655, 174.63529446658364),
                (-32.46024387309427, 115.74069828121115),
            ),
            5229000,
        ),
    ],
}

POS_SCENARIOS = {
    "argnames": "label, locations, agent_pos",
    "argvalues": [
        ("pos_outside_interval", ((3, 4), (30, 40)), (0, 0)),
        ("pos_inside_interval", ((3, 4), (30, 40)), (6, 8)),
        ("pos_not_input", ((3, 4), (1, 0), (30, 40)), None),
    ],
}

POS_1 = np.array((3, 4))
POS_2 = np.array((1, 0))
EXPECTED_POSITIONS = {
    "pos_outside_interval": (POS_1, 5.0),
    "pos_inside_interval": (POS_1, 5.0),
    "pos_not_input": (POS_2, 1.0),
}

TIME_STEP = 1

ORIGIN_GPS = BOUNDS[0]
DESTINATION_GPS = BOUNDS[1]


class MockModel:
    def __init__(self, positions: np.ndarray | None):
        class MockTimer:
            def __init__(self):
                self.mission_start = datetime.now()
                self.mission_time = copy.copy(self.mission_start)

        class MockSimulation:
            def __init__(self):
                self.time_step = timedelta(seconds=TIME_STEP)
                self.timer = MockTimer()

            def step_timer(self):
                self.timer.mission_time += self.time_step

        self.simulation = MockSimulation()
        self.positions = positions


class MockAgent:
    """A minimal mock agent"""

    def __init__(self, pos):
        self.pos = pos


class ChildMockAgent(MockAgent):
    """mock agent with testable method and inherits from `MockAgent`."""

    def __init__(self, pos):
        super().__init__(pos)
        self.value = 0

    def add_one(self):
        self.value += 1


@pytest.fixture(scope="function")
def agent_cls():
    """Provides an minimal Agent object as a fixture."""
    return MockAgent


@pytest.fixture(scope="function")
def agent():
    return MockAgent((0, 0))


@pytest.fixture(scope="function")
def child_agent():
    return ChildMockAgent((0, 0))


class AgentTester(ScenarioTestSuite):
    scenarios = DISTANCE_SCENARIOS

    @pytest.mark.parametrize(**DISTANCE_SCENARIOS)
    def test_distance(self, label, positions):
        test_pos = (0, 0)
        expected_result = np.array(EXPECTED_DISTANCE[label])
        assert np.all(
            self.test_obj.distance(np.array(test_pos), np.array(positions))
            == expected_result
        )

    @pytest.mark.parametrize(**DISTANCE_GPS_SCENARIOS)
    def test_distance_gps(self, label, coordinates, expected_result):
        distance = self.test_obj.distance_gps(coordinates[0], coordinates[1])
        assert np.allclose(distance, expected_result, rtol=0.01)

    @pytest.mark.parametrize(**POS_SCENARIOS)
    def test_nearest_position(self, label, locations, agent_pos):
        expected_pos, expected_distance = EXPECTED_POSITIONS[label]
        agent_pos = np.array(agent_pos) if agent_pos else None
        position, _ = self.test_obj.nearest_position(
            np.array(locations), pos=agent_pos
        )
        assert np.all(position == expected_pos)

    @pytest.mark.parametrize(**POS_SCENARIOS)
    def test_nearest_agent(self, agent_cls, label, locations, agent_pos):
        agents = [agent_cls(pos) for pos in locations]
        this_agent = agent_cls(agent_pos) if agent_pos else None
        closest_agent, distance = self.test_obj.nearest_agent(
            agents, this_agent
        )
        expected_pos, _ = EXPECTED_POSITIONS[label]
        assert np.all(closest_agent.pos == expected_pos)

    def test_get_other_agents(self, agent, child_agent):
        self.test_obj.model.agents = [
            agent,
            child_agent,
            self.test_obj,
        ]
        mock_agents = list(self.test_obj.get_other_agents(MockAgent))
        assert agent in mock_agents
        assert child_agent in mock_agents
        assert self.test_obj not in mock_agents
        mock_agents = list(self.test_obj.get_other_agents(Agent))
        assert self.test_obj not in mock_agents
        assert agent not in mock_agents

    def test_run_on_other_agents(self, child_agent):
        test_child_agent = copy.copy(child_agent)
        assert test_child_agent.value == 0
        assert child_agent.value == 0
        self.test_obj.model.agents = [
            test_child_agent,
            child_agent,
            self.test_obj,
        ]
        self.test_obj.run_on_other_agents(
            expression=lambda agent: agent.add_one(), agent_type=ChildMockAgent
        )
        assert test_child_agent.value == 1
        assert child_agent.value == 1


class AgentWithGPSTester(AgentTester):
    def test_gps_coords(self):
        # Test initial GPS Coordinate
        assert np.allclose(self.test_obj.gps_coords, ORIGIN_GPS, rtol=0.0001)
        # Test setting GPS Coordinate
        self.test_obj.gps_coords = TEST_GPS
        assert np.allclose(self.test_obj.gps_coords, TEST_GPS, rtol=0.0001)

    def test_distance(self):
        """Override test for `AgentWithGPS`.

        Tests against `distance_with_gps` result.
        """
        distance = self.test_obj.distance((0, 0), (RIGHT - LEFT, BOTTOM - TOP))
        expected_distance = self.test_obj.distance_gps(
            ORIGIN_GPS, DESTINATION_GPS
        )
        assert np.allclose(distance, expected_distance, atol=0.5)


class TestAgent(AgentTester):
    class MockAgent(Agent):
        def __init__(
            self,
            unique_id,
            model,
            autopopulate,
        ):
            super().__init__(
                unique_id=unique_id,
                model=model,
                autopopulate=autopopulate,
            )

        def step(self):
            pass

    test_pos = np.array((0, 0))
    test_class = MockAgent
    model = MockModel(test_pos)
    test_obj = MockAgent(
        unique_id=0,
        model=model,
        autopopulate=False,
    )


class TestStaticAgent(AgentTester):
    class MockStaticAgent(StaticAgent):
        def __init__(self, unique_id, model):
            super().__init__(unique_id=unique_id, model=model)

        def step(self):
            pass

    test_pos = np.array((0, 0))
    test_class = MockStaticAgent
    model = MockModel(test_pos)
    test_obj = MockStaticAgent(unique_id=0, model=model)


class AirTrafficManagerTester(AgentWithGPSTester):
    def test_is_cleared(self, agent):
        test_agent = copy.copy(agent)
        # Test with agent at front of queue
        self.test_obj.register_agent(test_agent)
        self.test_obj.model.simulation.step_timer()
        assert self.test_obj.is_cleared(test_agent) == False
        self.test_obj.model.simulation.step_timer()
        assert self.test_obj.is_cleared(test_agent) == True
        # Test case with agent not at front of queue
        self.test_obj.register_agent(agent)
        self.test_obj.register_agent(test_agent)
        assert self.test_obj.is_cleared(test_agent) == False

    def test_register_agent(self, agent):
        # Test with input `queue` = None
        self.test_obj.register_agent(agent)
        assert agent in self.test_obj.holding_pattern
        # Test with input `queue` = self.other_queue
        self.test_obj.register_agent(agent, self.test_obj.other_queue)
        assert agent in self.test_obj.other_queue

    def test_deregister_agent(self, agent):
        # Test with input `queue` = None
        self.test_obj.register_agent(agent)
        self.test_obj.deregister_agent(agent)
        assert agent not in self.test_obj.holding_pattern
        # Test with input `queue` = self.other_queue
        self.test_obj.register_agent(agent, self.test_obj.other_queue)
        self.test_obj.deregister_agent(agent, self.test_obj.other_queue)
        assert agent not in self.test_obj.other_queue

    def test_position_in_queue(self, agent):
        # Test with input `queue` = None
        # Clear all agents in queue if any
        self.test_obj.holding_pattern = []
        test_agent = copy.copy(agent)
        self.test_obj.register_agent(agent)
        self.test_obj.register_agent(test_agent)
        self.test_obj.register_agent(agent)
        assert self.test_obj.position_in_queue(test_agent) == 1
        # Test with input `queue` = self.other_queue
        self.test_obj.other_queue = []
        self.test_obj.register_agent(agent, self.test_obj.other_queue)
        self.test_obj.register_agent(test_agent, self.test_obj.other_queue)
        self.test_obj.register_agent(agent, self.test_obj.other_queue)
        assert (
            self.test_obj.position_in_queue(
                test_agent, self.test_obj.other_queue
            )
            == 1
        )


TIME = 5


class MovingAgentTester(AgentTester):
    def test_idle(self):
        """Test `idle` method for proper progression and termination."""
        status = self.test_obj.idle(TIME)
        assert status.value == TaskStatus.IN_PROGRESS.value
        assert self.test_obj.__idle_timer__ is not None
        for _ in range(TIME - 2):
            self.test_obj.idle(timedelta(seconds=TIME))
        status = self.test_obj.idle(timedelta(seconds=TIME))
        assert status.value == TaskStatus.COMPLETE.value
        assert self.test_obj.__idle_timer__ is None

    def test_remaining_idle_time(self):
        """Test incorrect and correct uses of `remaining_idle_time`."""
        # Test incorrect use of method
        with pytest.raises(Exception):
            self.test_obj.remaining_idle_time
        # Test correct use of method
        self.test_obj.idle(TIME)
        assert self.test_obj.remaining_idle_time == 4

    def test_navigate_to(self):
        # Verify starting position
        assert np.all(self.test_obj.pos == (0, 0))
        # Test navigation in progress
        status = self.test_obj.navigate_to((6, 8), 5)
        assert np.all(self.test_obj.pos == (3, 4))
        assert status.value == TaskStatus.IN_PROGRESS.value
        # Test navigation complete
        status = self.test_obj.navigate_to((6, 8), 5)
        assert np.all(self.test_obj.pos == (6, 8))
        assert status.value == TaskStatus.COMPLETE.value

        # Test different direction and speed
        status = self.test_obj.navigate_to((8, 8), 2)
        assert np.all(self.test_obj.pos == (8, 8))
        assert status.value == TaskStatus.COMPLETE.value

    def test_aspect(self):
        self.test_obj.navigate_to((0, 0), 1)
        assert self.test_obj.aspect == -45
        self.test_obj.navigate_to((10, 10), 1)
        assert self.test_obj.aspect == 135


class TestMovingAgent(MovingAgentTester):
    class MockMovingAgent(MovingAgent):
        def __init__(
            self,
            unique_id,
            model,
            pos,
            autopopulate: bool,
        ):
            super().__init__(unique_id, model, pos, autopopulate)

        def icon_size(self):
            pass

    test_pos = np.array((0, 0))
    test_class = MockMovingAgent
    model = MockModel(np.array(test_pos))
    test_obj = MockMovingAgent(
        unique_id=0, model=model, autopopulate=False, pos=test_pos
    )


class TestAgentWithGPS(AgentWithGPSTester):
    class MockAgentWithGPS(AgentWithGPSMixin, Agent):
        def __init__(self, unique_id, model, autopopulate: bool, pos):
            super().__init__(unique_id, model, autopopulate, pos)
            self.top_left_bounds = TOP, LEFT

        def step(self):
            pass

    test_pos = np.array((0, 0))
    test_class = MockAgentWithGPS
    model = MockModel(np.array(test_pos))
    test_obj = MockAgentWithGPS(
        unique_id=0, model=model, autopopulate=False, pos=(0, 0)
    )


class MovingAgentWithGPSTester(AgentWithGPSTester):
    def test_navigate_to_gps(self):
        # Test moving across `gps_to_pos_test_file`
        # Compute velocity based on diagonal distance of map
        self.test_obj.gps_coords = ORIGIN_GPS
        distance = self.test_obj.distance_gps(ORIGIN_GPS, DESTINATION_GPS)
        time_step_factor = 100
        velocity = distance / TIME_STEP / time_step_factor
        for _ in range(time_step_factor + 1):
            self.test_obj.navigate_to_gps(DESTINATION_GPS, velocity)
        assert np.all(self.test_obj.gps_coords == DESTINATION_GPS)


class TestMovingAgentWithGPS(MovingAgentWithGPSTester):
    class MockMovingAgentWithGPS(MovingAgentWithGPS):
        def __init__(
            self,
            unique_id,
            model,
            pos,
            autopopulate: bool,
        ):
            self.top_left_bounds = TOP, LEFT
            super().__init__(
                unique_id,
                model,
                pos=pos,
                autopopulate=autopopulate,
            )

        def step(self):
            pass

        def icon_size(self):
            pass

    test_pos = np.array((0.0, 0.0))
    test_class = MockMovingAgentWithGPS
    model = MockModel(np.array(test_pos))
    test_obj = MockMovingAgentWithGPS(
        unique_id=0, model=model, pos=(0, 0), autopopulate=False
    )
