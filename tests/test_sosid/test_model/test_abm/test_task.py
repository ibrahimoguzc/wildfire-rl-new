# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from collections import deque

import pytest

from sosid.model.abm.task import (
    Task,
    TaskPriority,
    TaskQueue,
    TaskScheduler,
    TaskStatus,
)
from tests.snippets import ScenarioTestSuite


class MockAgent:
    """A mock agent to test task priority."""

    def __init__(self):
        self.tasks = TaskScheduler(self)

    def step(self):
        self.tasks.run_active()

    @Task(priority=TaskPriority.HIGHEST)
    def highest(self):
        return TaskStatus.COMPLETE

    @Task(priority=TaskPriority.HIGH)
    def high(self):
        return TaskStatus.COMPLETE

    @Task  # Testing default value
    def normal(self):
        return TaskStatus.COMPLETE

    @Task(priority=TaskPriority.LOW)
    def low(self):
        return TaskStatus.COMPLETE

    @Task(priority=TaskPriority.LOWEST)
    def lowest(self):
        return TaskStatus.COMPLETE


@pytest.fixture(scope="function")
def agent():
    """Provides an minimal Agent object as a fixture."""
    return MockAgent()


@pytest.fixture(scope="function")
def inheriting_agent_populate():
    """Minimal Agent object which inherits tasks with priorities."""

    class MockInheritingAgent(MockAgent):
        def __init__(self):
            super().__init__()

    return MockInheritingAgent()


@pytest.fixture(scope="function")
def inheriting_agent_set_active():
    """Minimal Agent object which inherits tasks without priorities."""

    class MockAgent:
        def __init__(self):
            self.tasks = TaskScheduler(self, autopopulate=False)
            self.completed = False
            self.failed = False

        def step(self):
            self.tasks.run_active()

        @Task
        def highest(self):
            return TaskStatus.COMPLETE

        @highest.on_complete
        def high(self):
            self.completed = True
            return TaskStatus.COMPLETE

        @Task
        def lowest(self):
            return TaskStatus.FAILED

        @lowest.on_fail
        def low(self):
            self.failed = True
            return TaskStatus.COMPLETE

    class MockInheritingAgent(MockAgent):
        def __init__(self):
            super().__init__()

    return MockInheritingAgent()


class TestTask:
    def test_on_complete(self):
        """Tests `on_complete` callback function."""

        class MockAgentFoo:
            completed = False

            @Task
            def find_bar(self):
                return TaskStatus.COMPLETE

            @find_bar.on_complete
            def update_status(self):
                self.completed = True

        agent = MockAgentFoo()
        agent.find_bar(agent)
        assert agent.completed is True

    def test_on_fail(self):
        """Tests `on_fail` callback function."""

        class MockAgentBar:
            failed = False

            @Task
            def find_foo(self):
                return TaskStatus.FAILED

            @find_foo.on_fail
            def update_status(self):
                self.failed = True

        agent = MockAgentBar()
        agent.find_foo(agent)
        assert agent.failed is True

    def test_run(self):
        """Tests if `run` method functions as a context manager."""

        class MockAgentSpam:
            failed = False

            @Task
            def find_foo(self):
                return TaskStatus.FAILED

            @find_foo.on_fail
            def update_status(self):
                self.failed = True

        agent = MockAgentSpam()
        with agent.find_foo.run(agent) as status:
            # Making sure that the output is stored in status
            assert status is TaskStatus.FAILED

            # Ensuring that the callback has not yet been executed
            assert agent.failed is False

        # Making sure that the callback executes in the outer scope
        assert agent.failed is True

    COMPARISON_TEST_CASES = {
        "argnames": "task_priority, op, other_priority",
        "argvalues": [
            (TaskPriority.LOW, "__le__", TaskPriority.LOW),
            (TaskPriority.NORMAL, "__eq__", TaskPriority.NORMAL),
            (TaskPriority.NORMAL, "__ne__", TaskPriority.HIGH),
            (TaskPriority.HIGH, "__ge__", TaskPriority.HIGH),
            (TaskPriority.HIGHEST, "__gt__", TaskPriority.HIGH),
        ],
    }

    @pytest.mark.parametrize(**COMPARISON_TEST_CASES)
    def test_comparisons(self, task_priority, op, other_priority):
        """Tests if TaskPriority is getting compared correctly."""
        task = TestTaskScheduler.get_mock_task(task_priority)
        other = TestTaskScheduler.get_mock_task(other_priority)
        assert getattr(task, op)(other)


class TestTaskScheduler:
    def test_populate_queue(self, agent):
        """Tests if populated queue is sorted based on priority."""
        result = [task.task_method.__name__ for task in agent.tasks.__queue__]
        assert result == ["lowest", "low", "normal", "high", "highest"]

    def test_inheritance_populate_queue(self, inheriting_agent_populate):
        """Tests if populated queue works with inheritance."""
        result = [
            task.task_method.__name__
            for task in inheriting_agent_populate.tasks.__queue__
        ]
        assert result == ["lowest", "low", "normal", "high", "highest"]

    def test_inheritance_set_active(self, inheriting_agent_set_active):
        """Tests if set active works with inheritance"""
        agent = inheriting_agent_set_active
        agent.tasks.set_active(agent.highest)
        assert agent.tasks.active_task == agent.highest

    def test_inheritance_on_complete(self, inheriting_agent_set_active):
        """Tests if 'on_complete' callback works with inheritance"""
        agent = inheriting_agent_set_active
        agent.highest(agent)
        assert agent.completed is True

    def test_inheritance_on_fail(self, inheriting_agent_set_active):
        """Tests if 'on_fail' callback works with inheritance"""
        agent = inheriting_agent_set_active
        agent.lowest(agent)
        assert agent.failed is True

    def test_task_population_order(self):
        """Tests if same priority tasks are populated sequentially.

        This guarantees that tasks defined first are run first as
        long as they have the same :py:class:`TaskPriority`.
        """

        class MockAgent:
            """A mock agent to test task priority."""

            def __init__(self):
                self.tasks = TaskScheduler(self)

            @Task
            def first(self):
                return TaskStatus.COMPLETE

            @Task
            def second(self):
                return TaskStatus.COMPLETE

            @Task
            def third(self):
                return TaskStatus.COMPLETE

        agent = MockAgent()
        result = [task.task_method.__name__ for task in agent.tasks.__queue__]
        assert result == ["third", "second", "first"]

    def test_idle(self):
        """Tests if ``True`` is returned when there are no tasks."""
        tasks = TaskScheduler(object())
        assert tasks.idle

    def test_active_task(self):
        """Tests if the correct priority task is selected as active."""
        tasks = TaskScheduler(object())
        assert tasks.active_task is None  # Testing return value w/ empty queue

        # Creating mock low and high priority tasks
        low_priority_task = self.get_mock_task(TaskPriority.LOWEST)
        high_priority_task = self.get_mock_task(TaskPriority.HIGHEST)

        # Appending to task queue
        for task in (low_priority_task, high_priority_task):
            tasks.__queue__.append(task)

        # Ensuring that the active task is the one with highest priority
        assert tasks.active_task is high_priority_task

    def test_disable_autopopulate(self):
        """Tests if disabling autopopulate functions correctly."""

        class MockManualAgent:
            """A mock agent to test task priority."""

            def __init__(self):
                self.tasks = TaskScheduler(self, autopopulate=False)

            @Task(priority=TaskPriority.HIGHEST)
            def dont_collect(self):
                return TaskStatus.COMPLETE

        agent = MockManualAgent()
        assert agent.tasks.idle

        # Retrieving tasks manually
        agent.tasks.populate_queue()
        assert not agent.tasks.idle

    ADD_TEST_CASES = {
        "argnames": "priority, expected_index",
        "argvalues": [
            (TaskPriority.LOWEST, 0),
            (TaskPriority.LOW, 1),
            (TaskPriority.NORMAL, 2),
            (TaskPriority.HIGH, 3),
            (TaskPriority.HIGHEST, 4),
        ],
    }

    @pytest.mark.parametrize(**ADD_TEST_CASES)
    def test_add(self, priority, expected_index, agent):
        """Ensuring if task is added to the correct position."""
        mock_task = self.get_mock_task(priority)
        agent.tasks.add(mock_task)
        assert agent.tasks.__queue__[expected_index] is mock_task

    def test_remove(self, agent):
        """Ensuring that tasks are removed correctly."""
        # Creating tasks tuple to allow iteration over mutating deque
        for task in tuple(agent.tasks.__queue__):
            agent.tasks.remove(task)
            assert task not in agent.tasks.__queue__

    SET_ACTIVE_TEST_CASES = {
        "argnames": "priority, expected_index",
        "argvalues": [(p, -1) for p in TaskPriority],
    }

    @pytest.mark.parametrize(**SET_ACTIVE_TEST_CASES)
    def test_set_active(self, priority, expected_index, agent):
        """Testing if task is set active irrespective of priority."""
        mock_task = self.get_mock_task(priority)
        agent.tasks.set_active(mock_task)
        assert agent.tasks.__queue__[expected_index] is mock_task

    def test_set_active_callback(self):
        """Tests whether set_active works when run from a callback."""

        class MockAgentBug:
            def __init__(self):
                self.tasks = TaskScheduler(self, autopopulate=True)

            @Task(priority=TaskPriority.HIGHEST)
            def find_bug(self):
                return TaskStatus.COMPLETE

            @find_bug.on_complete
            def grab_bug(self):
                self.tasks.set_active(self.eat_bug)

            @Task(priority=TaskPriority.LOWEST)
            def eat_bug(self):
                return TaskStatus.COMPLETE

            @Task(priority=TaskPriority.HIGH)
            def digest_bug(self):
                return TaskStatus.FAILED

            @digest_bug.on_fail
            def throw_up(self):
                self.tasks.set_active(self.find_bug)

            def step(self):
                self.tasks.run_active()

        agent = MockAgentBug()
        agent.step()
        agent.step()
        # if set_active ran correctly, the next task is digest_bug
        assert agent.tasks.active_task is agent.digest_bug

        agent.step()
        # If set_active ran from on_fail, next task is find_bug
        assert agent.tasks.active_task is agent.find_bug

    def test_detect_recursion(self):
        """Tests whether infinite recursion is detected."""

        class RecursiveMockAgent:
            def __init__(self):
                self.tasks = TaskScheduler(
                    self, autopopulate=False, recursion_check=True
                )
                self.tasks.set_active(self.no_recursion)

            @Task(priority=TaskPriority.HIGHEST)
            def no_recursion(self):
                return TaskStatus.COMPLETE

            @no_recursion.on_complete
            def set_next_task(self):
                self.tasks.set_active(self.nested_recursive_task)

            @Task(priority=TaskPriority.HIGH)
            def nested_recursive_task(self):
                self.cause_recursion()
                return TaskStatus.COMPLETE

            def cause_recursion(self):
                # Burying set_active call inside a list comprehension
                [self.tasks.set_active(self.no_recursion) for _ in range(1)]

            def step(self):
                self.tasks.run_active()

        agent = RecursiveMockAgent()
        agent.step()  # First task `no_recursion` should run fine

        # Running `nested_recursive_task` should cause a RecursionError
        with pytest.raises(RecursionError):
            agent.step()

    @pytest.mark.parametrize("task_status", [s for s in TaskStatus])
    def test_run_active(self, task_status):
        """Testing running of active tasks with diff. task statuses."""
        tasks = TaskScheduler(object())
        mock_task = self.get_mock_task(
            TaskPriority.NORMAL, task_status=task_status
        )
        tasks.__queue__.append(mock_task)
        assert mock_task in tasks.__queue__
        tasks.run_active()
        if (
            task_status is TaskStatus.COMPLETE
            or task_status is TaskStatus.FAILED
        ):
            assert mock_task not in tasks.__queue__
        else:
            # If a task is in progress it should still be in the queue
            assert mock_task in tasks.__queue__

    def test_task_cycle(self):
        """Tests if tasks can re-populate (cycle) with autopopulate."""

        class MockAutoAgent:
            """A mock agent to test task cycling."""

            def __init__(self):
                self.tasks = TaskScheduler(self, autopopulate=True)

            @Task(priority=TaskPriority.HIGHEST)
            def infinite_task(self):
                return TaskStatus.COMPLETE

        agent = MockAutoAgent()
        agent.tasks.run_active()  # Running task for the first time

        # Testing if task has been completed and removed from queue
        assert agent.infinite_task not in agent.tasks.__queue__

        # Testing if task has been sucessfully re-run
        agent.tasks.run_active()

    def test_ensure_task(self):
        """Testing if expected exception is raised for a non task."""
        valid_task = self.get_mock_task(TaskPriority.NORMAL)
        assert TaskScheduler.ensure_task(valid_task) is valid_task
        with pytest.raises(ValueError):
            TaskScheduler.ensure_task(object())

    @staticmethod
    def get_mock_task(priority, task_status=TaskStatus.IN_PROGRESS):
        """Creates a mock task that can be append/inserted."""

        @Task(priority=priority)
        def mock_task(mock_self):
            return task_status

        return mock_task


class UniqueInt(int):
    """Specializes :py:obj:`int` to return unique hash values."""

    def __hash__(self):  # noqa: D105
        return object.__hash__(self)

    def __repr__(self):  # noqa: D105
        return f"<UniqueInt = {self} object at {hex(id(self))}>"


TASK_QUEUE_SCENARIOS = {
    "argnames": "label, args",
    "argvalues": [
        ("no_argument", ()),
        ("descending_ints", ([UniqueInt(i) for i in reversed(range(4))],)),
    ],
}


class TestTaskQueue(ScenarioTestSuite):
    test_class = TaskQueue
    scenarios = TASK_QUEUE_SCENARIOS

    EXPECTED_DEQUEUE = {
        "no_argument": deque([]),
        # The reverse order list should be sorted on entry
        "descending_ints": deque([0, 1, 2, 3]),
    }

    def test_init(self, scenario):
        """Tests if TaskQueue is initialized sorted."""
        assert scenario.obj == deque(self.EXPECTED_DEQUEUE[scenario.label])

    EXPECTED_INSORT_IDX = {"no_argument": 0, "descending_ints": 1}

    def test_insort(self, scenario):
        """Tests if inserting an object functions as expected."""
        test_int = UniqueInt(1)
        queue = scenario.obj.copy()  # Copying to prevent changing scenario
        queue.insort(test_int)
        assert queue[self.EXPECTED_INSORT_IDX[scenario.label]] is test_int

    EXPECTED_APPEND_IDX = {"no_argument": 0, "descending_ints": 4}

    def test_append(self, scenario):
        """Tests if appending (promoting task to active) functions."""
        test_int = UniqueInt(-1)
        queue = scenario.obj.copy()  # Copying to prevent changing scenario
        queue.append(test_int)
        assert queue[self.EXPECTED_APPEND_IDX[scenario.label]] is test_int
