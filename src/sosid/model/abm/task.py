# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains classes that implement a task system for Agents."""

import bisect
import inspect
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from enum import Enum, unique

# TODO Create a logger detailing task calls per Agent
# TODO Research how to make a PlantUML activity diagram from Agent tasks
# TODO Contemplate possibility of having sub-tasks
# TODO test multiple Tasks sharing the same on_completed callback


@unique
class TaskPriority(Enum):
    """Defines py:class:`Task` priorities."""

    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4


@unique
class TaskStatus(Enum):
    """Defines :py:class:`Task` exit status codes."""

    IN_PROGRESS = -1
    COMPLETE = 0
    FAILED = 1


# TODO consider moving into typedef.py
TaskMethod = Callable[[object], TaskStatus]


# TODO Assign docstring and annotations of task method
class Task:
    """A Decorator for marking a method as :py:class:`Agent` task.

    A task here refers to an prioritized :py:class:`Agent` action that
    mutates the present state of the agent and/or its surroundings
    during a simulation step. A decorated task method and its callbacks
    can be used to chain together complex agent-logic that would be too
    lengthy inside of a single step method.

    The simplest usage of the :py:class:`Task` decorator is to add a
    high-priority task that executes before all other assigned tasks::

        class RescueHelicopter(Agent):
            @Task(priority=TaskPriority.HIGHEST)
            def find_survivors(self):
                self.scan_area()
                if self.survivors_found:
                    return TaskStatus.COMPLETED
                else:
                    return TaskStatus.IN_PROGRESS

            @Task(priority=TaskPriority.LOWEST)
            def return_to_base(self):
                if self.pos != self.model.headquarters:
                    self.fly_to(self.model.headquarters)
                    return TaskStatus.IN_PROGRESS
                else:
                    return TaskStatus.COMPLETED

    One can also add call-back functions to perform an additional
    action within the same simulation step. This can useful for manually
    enforcing what the next task performed by the :py:class:`Agent`
    should be, or for communicating to other agents::

            @find_survivors.on_complete
            def notify_patrol(self):
                for agent in self.model.agents:
                    agent.notify(self.pos)

    In case a rescue_survivors()` task also needs to communicate with
    other agents after completion, same `notify_patrol()` method can be
    decorated multiple times as follows::

            @rescue_survivors.on_complete
            @find_survivors.on_complete
            def notify_patrol(self):
                for agent in self.model.agents:
                    agent.notify(self.pos)

    Args:
        priority: Sets the priority of the task. Defaults to
            :py:attr:`TaskPriority.NORMAL`.
    """

    __slots__ = [
        "complete_method",
        "fail_method",
        "priority",
        "task_method",
    ]

    def __init__(
        self,
        task_method: TaskMethod | None = None,
        *,  # All arguments to the decorator must be positional
        priority: TaskPriority | None = TaskPriority.NORMAL,
    ):
        self.priority = priority
        self.task_method = task_method
        self.complete_method = None
        self.fail_method = None

    def __call__(
        self, method_or_agent: TaskMethod | object
    ) -> object | TaskStatus:
        """Handles the decoration and later the task execution process.

        This method handles two scenarios of the :py:class:`Task`
        decorator. First is when the decorator is called without
        any arguments::

            @Task
            def find_survivors(self):
                self.scan_area()
                if self.survivors_found:
                    return TaskStatus.COMPLETED
                else:
                    return TaskStatus.IN_PROGRESS

        In this case :py:meth`find_survivors` is passed to
        :py:class:`Task` during its initialization and is assigned to
        its :py:attr:`task_method` attribute.

        Second, is if the decorator was instantiated with keyword
        arguments::

            @Task(priority=TaskPriority.HIGHEST)
            def find_survivors(self):
                self.scan_area()
                if self.survivors_found:
                    return TaskStatus.COMPLETED
                else:
                    return TaskStatus.IN_PROGRESS

        In this case, the :py:class:`Task` class is first instatiated
        with the keyword argument ``priority``. Once it becomes an
        object it is used as a decorator where :py:meth:`eat_spam`
        becomes its argument. In this case we simply want to assign the
        decorated :py:meth:`eat_spam` to :py:attr:`test_method` of the
        :py:class:`Task` object and return it. Evaluation of the task
        with a :py:class:`Agent` instance will then happen in the
        subsequent call.

        Args:
            method_or_agent: The decorated :py:attr:`task_method` or an
                :py:class:`Agent` instance depending on the call.

        Returns:
            A :py:class:`Task` with its :py:attr:`test_method` assigned
            if being called as a decorator with keyword arguments.
            Otherwise, a :py:class:`TaskStatus` exit code.

        Note:
            Providing functionality of both handling decoration and
            running of a task breaks the Single Responsibility Principle
            (SRP). However, it allows the user to run tasks manually for
            debugging purposes in a clean way. Therefore, the additional
            overhead required when calling the :py:meth:`run` agent will
            only occur in this debugging purpose. In normal operation
            the :py:meth:`run` context manager should be used instead.
        """
        if self.task_method:  # Post decoration, being called as a task
            agent = method_or_agent
            with self.run(agent) as status:
                return status  # Callbacks run after status is returned
        else:  # Method not yet assigned, most likely called as a decorator
            # update_wrapper(self, method)
            self.task_method = method_or_agent
            return self

    @contextmanager
    def run(self, agent) -> object:
        """Runs :py:class:`Task` for an ``agent`` in a context manager.

        This method allows executing additional code before the
        callbacks are run as follows::
            with Task.run(agent) as status:
                pass  # Any code that needs to run before the callbacks
            # Callbacks run here in the outer scope
        """
        status = self.task_method(agent)
        try:
            yield status
        finally:
            if status is TaskStatus.IN_PROGRESS:
                pass
            elif status is TaskStatus.COMPLETE and self.complete_method:
                self.complete_method(agent)
            elif status is TaskStatus.FAILED and self.fail_method:
                self.fail_method(agent)

    # TODO finish documentation for on_complete and on_fail callbacks
    def on_complete(self, method: TaskMethod) -> TaskMethod:
        """Sets the callback function when a task is complete.

        Returns:
            The decorated on_complete method in order to facitilitate
            multiple decorations.

        """
        self.complete_method = method
        return method

    def on_fail(self, method: TaskMethod) -> TaskMethod:
        """Sets the callback function when a task fails.

        Returns:
            The decorated on_fail method in order to facitilitate
            multiple decorations.

        """
        self.fail_method = method
        return method

    def __repr__(self):
        """Returns a :py:obj:`str` representation of the task."""
        return f"<{self.task_method.__name__} Task object at {hex(id(self))}>"

    # Overriding comparison magic methods to allow sorting of Tasks
    def __lt__(self, other: object):  # noqa: D105
        return self.priority.value < other.priority.value

    def __le__(self, other: object):  # noqa: D105
        return self.priority.value <= other.priority.value

    def __eq__(self, other: object):  # noqa: D105
        return self.priority.value == other.priority.value

    def __ne__(self, other: object):  # noqa: D105
        return self.priority.value != other.priority.value

    def __ge__(self, other: object):  # noqa: D105
        return self.priority.value >= other.priority.value

    def __gt__(self, other: object):  # noqa: D105
        return self.priority.value > other.priority.value


class TaskQueue(deque):
    """High-Performance ordered :py:obj:`deque` with O(1) lookup.

    Tasks are sorted in increasing :py:class:`TaskPriority` during
    initialization and order is maintained during successive calls to
    :py:meth:`insort`. Therefore, tasks with the highest priority are at
    the right of the :py:obj:`deque` (index = -1).

    Note:
        Use :py:meth:`append` to manually disregard
        :py:class:`TaskPriority` and force a :py:class:`Task` to execute
        in the next simulation step.

    """

    def __init__(self, iterable: Iterable | None = None):
        super().__init__(sorted(iterable) if iterable else ())

    def insort(self, task: Task) -> None:
        """Inserts a ``task`` while respecting :py:class:`TaskPriority`.

        If a :py:class:`Task` exists within the :py:obj:`deque` with the
        same :py:class:`TaskPriority` as ``task`` then it will be
        inserted to the left. Therefore, previous tasks in the
        :py:class:`TaskQueue` with the same :py:class:`TaskPriority`
        will be executed first.
        """
        bisect.insort_left(self, task)


class TaskScheduler:
    """Responsible for scheduling and running the tasks of an Agent.

    Args:
        agent: The :py:class:`Agent` instance to schedule tasks for.
        autopopulate: Determines if all :py:class:`Task` defined within
            :py:attr:`agent` should added into the :py:class:`TaskQueue`
            using their :py:class:`TaskPriority`.
        recursion_check: Toggles if a check should be run on the
            call-stack to detect infinite task-recursion caused by
            calling :py:meth`set_active` from within a :py:class:`Task`.
            This option can be disabled to increase performance.
    """

    __slots__ = ["__queue__", "agent", "autopopulate", "recursion_check"]

    def __init__(
        self,
        agent: object,
        autopopulate: bool = True,
        recursion_check: bool = True,
    ):
        self.agent = agent
        self.__queue__ = TaskQueue()
        self.populate_queue() if autopopulate else ()
        self.autopopulate = autopopulate
        self.recursion_check = recursion_check

    def populate_queue(self, task_check: Callable = lambda obj: True):
        """Populates queue with all tasks of :py:attr:`agent`.

        ``task_check``: Takes input lambda function to make additional
            checks on `Task` prior to populating queue.
        """
        for cls in self.agent.__class__.mro():
            for obj in vars(cls).values():
                if isinstance(obj, Task) and task_check(obj):
                    self.__queue__.insort(obj)

    @property
    def idle(self) -> bool:
        """Returns ``True`` if there are no scheduled tasks."""
        return len(self.__queue__) == 0

    @property
    def active_task(self) -> Task:
        """Returns the current active task in the task schedule."""
        return self.__queue__[-1] if not self.idle else None

    def add(self, task: Task) -> None:
        """Adds a :py:class:`Task` with :py:class:`TaskPriority`."""
        self.__queue__.insort(self.ensure_task(task))

    def remove(self, task: Task) -> None:
        """Removes a :py:class:`Task`from the queue."""
        self.__queue__.remove(self.ensure_task(task))

    def set_active(self, task: Task) -> None:
        """Set ``task`` as the current active Task.

        This will overrule the :py:class:`TaskPriority` of ``task``
        and execute it in the next simulation step.

        Warning:
            Calling this method from within a :py:class:`Task` will lead
            to infinite recursion. In this scenario, consider using
            :py:meth:`add` or calling :py:meth:`set_active` from a
            :py:class:`Task` callback method such as
            :py:meth:`Task.on_complete` or :py:meth:`Task.on_fail.`
        """
        self._detect_task_recursion() if self.recursion_check else ()
        self.__queue__.append(self.ensure_task(task))

    def run_active(self):
        """Runs the first :py:class:`Task` within the queue."""
        if self.idle and self.autopopulate:
            self.populate_queue()

        with self.active_task.run(self.agent) as status:
            if status is not TaskStatus.IN_PROGRESS:
                self.__queue__.pop()  # Remove completed task before callbacks

    @staticmethod
    def ensure_task(task: Task) -> Task:
        """Returns a :py:class:`Task` if a :py:class:`Task` is passed.

        Raises:
            ValueError: If a non-task object is passed.

        """
        if isinstance(task, Task):
            return task
        raise ValueError("Cannot set non Task object as current.")

    def _detect_task_recursion(self) -> None:
        """Iterates the call-stack to detect :py:class:`Task` recursion.

        Detection works by assuming that if :py:meth:`set_active` was
        called from within a :py:class:`Task` then a caller in the
        call stack will have an identical :py:class:`types.CodeType`
        object as the current :py:attr:`active_task`.
        """
        if self.active_task is not None:
            # Caller of set-active is 2 frames back in the call stack
            caller = inspect.currentframe().f_back.f_back
            active_task_method = self.active_task.task_method

            # Iteration continues until the end of the call stack or
            # until the agent step method is reached
            while (
                hasattr(caller, "f_back")
                and caller.f_code is not self.agent.step.__code__
            ):
                if caller.f_code is active_task_method.__code__:
                    raise RecursionError(
                        "Infinite task recursion detected: set_active was "
                        "called while running "
                        f"{self.active_task.task_method.__name__} "
                        f"in {self.agent}"
                    )
                caller = caller.f_back
