# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Classes for defining, controlling, recording time of simulations."""

import copy
import random
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from random import Random
from typing import Any, Generic, TypeVar, no_type_check

import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, PositiveInt, model_validator
from pydantic.types import ImportString
from typing_extensions import Self

from sosid.abstract import Controller, Model, Writable
from sosid.environment.environment import BaseEnvironment
from sosid.model.transform import gps_to_pos, pos_to_gps
from sosid.output import Output, OutputFormat, TargetKey
from sosid.typedef import LatLon, Position
from sosid.util.abc import abstractattribute

# TODO create Environment interface so models are not coupled directly


class BaseModel(PydanticBaseModel):
    """Base class for all input parameter classes."""

    model_config = ConfigDict(
        frozen=True,
        ignored_types=(cached_property,),
        validate_default=True,
        extra="ignore",
    )


class Distribution(BaseModel):
    """A probability distribution from the Python Standard Library.

    The expected usage is to access a distribution function from the
    Python Standard Library as follows:

    {"random_fn": random.uniform, "kwargs": {"a": 0, "b": 10}}
    """

    random_fn: ImportString
    kwargs: dict[str, Any]

    def evaluate_distribution(self, rng: random.Random):
        """Evaluate distribution using `random_fn` and kwargs."""
        return self.random_fn(rng, **self.kwargs)


class PositionInput(BaseModel):
    """Position input to be specified as gps coordinate or position."""

    gps_coords: tuple[float, float] | None = None
    pos: tuple[float, float] | None = None

    @model_validator(mode="after")
    def check_position(self) -> Self:
        """Check if either gps_coords or pos is supplied."""
        msg = f"{self.__class__.__name__} requires EITHER gps_coords or pos."
        assert not all(  # noqa: S101
            [self.gps_coords, self.pos]
        ), f"{msg} Both have been supplied."
        assert any(  # noqa: S101
            [self.gps_coords, self.pos]
        ), f"{msg} Neither have been supplied."
        return self

    def get_pos(
        self,
        top_left_bounds: tuple[float, float],
    ) -> Position:
        """Carthesian position based on either gps_coords or pos."""
        if self.gps_coords is not None:
            return gps_to_pos(self.gps_coords, top_left_bounds)
        return np.array(self.pos, dtype=np.float64)

    def get_gps_coords(
        self,
        top_left_bounds: tuple[float, float],
    ) -> LatLon:
        """GPS coordinates based on either gps_coords or pos."""
        if self.gps_coords is not None:
            return np.array(self.gps_coords, dtype=np.float64)
        return pos_to_gps(self.pos, top_left_bounds)

    def check_in_bbox(
        self,
        bbox_pos: tuple[tuple[float, float], tuple[float, float]],
        bbox_gps: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        """Utility to validate input positions."""
        msg = f"{self.__class__.__name__.strip('Input')} location"
        if self.gps_coords:
            msg = f"{msg} {self.gps_coords} exceeds map boundaries"
            assert (  # noqa: S101
                bbox_gps[1][0] <= self.gps_coords[0] <= bbox_gps[0][0]
            ), f"{msg} latitude {bbox_gps}"
            assert (  # noqa: S101
                bbox_gps[0][1] <= self.gps_coords[1] <= bbox_gps[1][1]
            ), f"{msg} longitude {bbox_gps}"
        elif self.pos:
            msg = f"{msg} {self.pos} exceeds map boundaries"
            assert (  # noqa: S101
                bbox_pos[0][0] < self.pos[0] < bbox_pos[1][0]
            ), f"{msg} width: {bbox_pos}"
            assert (  # noqa: S101
                bbox_pos[0][1] < self.pos[1] < bbox_pos[1][1]
            ), f"{msg} height: {bbox_pos}"


class SimulationParameters(BaseModel):
    """A Single source of truth for simulation parameters."""

    # Simulation Parameters
    time_step: float  # SI second
    max_runtime: PositiveInt
    name: str | None = None
    slaving: bool = False
    zoom_scale_factor: float = 1.0
    reset_available: bool = False

    def evaluate_distributions(self, rng: random.Random) -> None:
        """Evaluates probability distributions with ``rng``.

        Any attribute that is callable is treated as a probability
        distribution and must be of the following form::

            random_variable: callable = lambda r: r.uniform(a, b)

        Args:
            rng: A random-number generator that is an instance of
                :py:class:`random`

        """
        for name, value in vars(self).items():
            if isinstance(value, Distribution):
                object.__setattr__(
                    self, name, value.evaluate_distribution(rng)
                )


TIME_NOT_SET = datetime(year=1, month=1, day=1)


class SimulationTimer:
    """Keeps track of runtime for both simulation and real-world time.

    Args:
        is_running: :py:class:`threading.Event` that sets if the
            simulation is currently running.
        is_stopped: :py:class:`threading.Event` that sets if the
            simulation has been stopped (terminated)
        mission_start: Defines the start date and time of the
            simulation. Defaults to the current time.

    Attributes:
        mission_start: Recorded mission start time of the simulation
        mission_time: Current time in the mission
        __start__: Real-world start-time of the simulation
        __recorded_runtime__: Real-world runtime of the simulation

    """

    __slots__ = (
        "is_running",
        "is_stopped",
        "mission_start",
        "mission_time",
        "__start__",
        "__recorded_runtime__",
    )

    def __init__(
        self,
        is_running: threading.Event,
        is_stopped: threading.Event,
        mission_start: datetime = datetime.now(),
    ):
        self.is_running = is_running
        self.is_stopped = is_stopped
        self.mission_start = mission_start
        # Shallow copying mission_start time to be be able to increment
        # it inplace with the :py:attr:`step` method.
        self.mission_time = copy.copy(mission_start)
        self.__start__ = TIME_NOT_SET
        self.__recorded_runtime__ = timedelta()

    def start(self) -> None:
        """Records the time at which simualtion was started/resumed."""
        if not self.is_running.is_set() or self.__start__ == TIME_NOT_SET:
            self.__start__ = datetime.now()

    def pause(self) -> None:
        """Records the time at which simualtion was paused/stopped."""
        # If simulation is running then we can stop
        if self.is_running.is_set() and not self.is_stopped.is_set():
            self.__recorded_runtime__ += datetime.now() - self.__start__

    def step(self, time_step: timedelta) -> None:
        """Advances the mission timer by one ``time_step``."""
        self.mission_time += time_step

    @property
    def runtime(self) -> timedelta:
        """Returns the real-world duration of the running simulation.

        If called while the simulation is currently running, an updated
        value is returned. Otherwise if the simulation is paused,
        subsequent calls will return the :py:attr:`__recorded_runtime__.
        """
        if self.is_running.is_set() and not self.is_stopped.is_set():
            return datetime.now() - self.__start__ + self.__recorded_runtime__
        else:
            return self.__recorded_runtime__

    @property
    def mission_runtime(self) -> timedelta:
        """Returns the mission time duration."""
        return self.mission_time - self.mission_start


SimContext = TypeVar("SimContext", bound=AbstractContextManager)
SimParams = TypeVar("SimParams", bound=SimulationParameters)


class Simulation(
    Generic[SimParams, SimContext], threading.Thread, Controller, Writable
):
    """An instantiation of models with data.

    Once instantiated a :py:class:`Simulation` can be started in
    a new thread by calling :py:meth:`start`. Since the simulation
    runs in a new thread, it can be interrupted and later resumed at
    anytime using the :py:meth:`pause` and :py:meth:`play` methods.

    Args:
        parameters: A single source of truth for data used in a
            a single simulation run.
        seed: The value used to instantiate :py:class:`random.Random`.
            If a simulation is re-run with the same seed value, it will
            produce the exact same result. To add stochasticity, the
            model must be re-run at a design-point with differing
            seed values::

                for i in range(100):
                    Simulation:
        profile: Toggles if pyinstrument should be used to profile the
            current simulation run. If ``True`` profiling statistics
            are displayed in a web-browser.

    Attributes:
        random (random.Random): A Python random-number generator
            instance.
        timer (SimulationTimer): Keeps track of both real-world runtime
            as well as the current time within the simulated mission.
        time_step: The duration of a single step or tick of the
            simulation in SI second.
        iterations (int): Counts the number of iterations performed.
        is_running (treading.Event): An event that controls if the
            simulation is currently running. This allows the current
            simulation run to be paused.

    """

    def __init__(
        self,
        parameters: SimParams,
        seed: int | None = 0,
        context: SimContext | None = None,
    ):
        threading.Thread.__init__(self)
        self.seed = seed
        self.random = Random(seed)
        self.parameters = parameters
        self.parameters.evaluate_distributions(self.random)
        self.context = nullcontext() if context is None else context

        # Settings the simulation time_step
        self.time_step = timedelta(seconds=self.parameters.time_step)

        # Thread lock used to make incrementing operations atomic
        self.lock = threading.Lock()

        # Creating event which enables pausing/resuming the thread
        self.is_running = threading.Event()

        # Creaing event which enables stopping the thread
        self.is_stopped = threading.Event()

        # Creaing event which enables the thread to be slaved by GUI
        self.not_slaved = threading.Event()
        self.not_slaved.set()

        self.timer = SimulationTimer(
            self.is_running,
            self.is_stopped,
            getattr(parameters, "mission_start", datetime.now()),
        )

        # Creating counting variable for storing simulation iterations
        self.iterations = 0

    @cached_property
    def max_iter(self):
        return int(
            np.ceil(self.parameters.max_runtime / self.parameters.time_step)
        )

    @cached_property
    @no_type_check
    def name(self) -> str:
        """Name used to identify the :py:class:`Simulation` instance."""
        return (
            str(self) if self.parameters.name is None else self.parameters.name
        )

    @abstractattribute
    def models(self) -> Sequence[Model]:
        """Sets the execution order of the defined models."""

    @abstractattribute
    def environment(self) -> BaseEnvironment:
        """Sets the environment for simulation."""

    def step(self, force: bool = False) -> None:
        """Advances all models by a single :py:attr:`time_step`.

        This method should be called if the user wants to increment
        the simulation by a single step manually.

        Args:
            force: Determines if the simulation should be advanced
                regardless of iteration and runtime constraints.
        """
        if self.iterations < (max_iter := self.max_iter):
            if (
                self.is_running.is_set()
                and not self.is_stopped.is_set()
                or force
            ):
                self.step_all_models()

                with self.lock:  # Ensure incrementing is atomic
                    self.iterations += 1
                    self.timer.step(self.time_step)
        else:
            if (
                not self.is_running.is_set()
                and self.is_stopped.is_set()
                or not force
            ):
                print(
                    f"Simulation Terminated: Max iterations = {max_iter}"
                    f"reached"
                )
                self.stop()
                self.output_collector()
            else:
                raise RuntimeError(
                    f"Maximum number of iterations (max_iter = {max_iter}) "
                    f"has been exceeded"
                )

    def step_all_models(self) -> None:
        """Steps each model in the simulation."""
        for model in self.models:
            model.step()

    def step_for(self, n_steps: int, force: bool = False) -> None:
        """Steps :py:class:`Simulation` by ``n_steps``.

        Intended for use only with the simulation slaved to the GUI.

        Args:
            n_steps: Determines the number of simulation steps the
                simulation will progress per call.
            force: Determines if the simulation should be advanced
                regardless of iteration and runtime constraints.
        """
        for _ in range(n_steps):
            self.step(force=force)

    def start(self) -> None:
        """Starts the :py:class:`Simulation` thread."""
        self.is_running.set()
        self.timer.start()
        super().start()  # Calls :py:meth:`run`

    def run(self) -> None:
        """Main loop that runs until stopped.

        Note:
            The main loop awaits both the :py:attr:`is_running` and
            :py:attr:`not_slaved` events. Therefore, to savely terminate
            the thread these two events must be set.
        """
        with self.context:
            while not self.is_stopped.is_set():
                self.is_running.wait()
                self.not_slaved.wait()
                self.step()

            # Ensuring :py:attr:`is_running` is false on exit.
            self.is_running.clear()

    def pause(self) -> None:
        """Pauses the :py:class:`Simulation."""
        if not self.is_stopped.is_set():
            with self.lock:
                self.timer.pause()
                self.is_running.clear()  # Makes the event return False

    def play(self) -> None:
        """Plays (resumes) the :py:class:`Simulation`."""
        if not self.is_stopped.is_set():
            with self.lock:
                self.timer.start()
                self.is_running.set()

    def stop(self) -> None:
        """Stops (terminates) the :py:class:`Simulation` thread.

        Note:
            Once the simulation is stopped it cannot be resumed.
        """
        if self.is_alive():
            with self.lock:
                # Main loop (run) is alive and awaiting events to be set
                self.timer.pause()
                self.is_stopped.set()
                self.is_running.set()
                self.not_slaved.set()
            self.output_collector()  # Trigger output to run
        else:
            raise RuntimeError(
                "A simulation that is not started cannot be stopped."
            )

    def reset(self) -> None:
        """Resets the simulation to its initial state."""
        if self.is_alive():
            self.stop()

        with self.lock:
            # Resetting the timer
            self.timer = SimulationTimer(
                self.is_running,
                self.is_stopped,
                getattr(self.parameters, "mission_start", datetime.now()),
            )

            # Resetting iterations
            self.iterations = 0

            # Resetting the random number generator with the same seed
            self.random = Random(self.seed)

            # Re-evaluate parameters if they have stochastic components
            self.parameters.evaluate_distributions(self.random)

            # Reset threading events
            self.is_running.clear()  # Clear running state
            self.is_stopped.clear()  # Clear stopped state, allowing new start
            self.not_slaved.set()  # Ensure simulation is not in slaved state

            for model in reversed(self.models):
                model.reset()

            # Init the new simulation thread
            threading.Thread.__init__(self)

    def flatten_dict(
        self, data: dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> dict[str, Any]:
        """Recursively flattens a nested dictionary, prefixing keys with
        parent keys."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    for i, sub_dict in enumerate(v):
                        items.extend(
                            self.flatten_dict(
                                sub_dict, f"{new_key}{sep}{i}", sep=sep
                            ).items()
                        )
                else:
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    def initialize_outputs(self):
        """Initialize dictionaries based on the TargetKey enum."""
        return {key.value: {} for key in TargetKey}

    def get_output_data(self) -> dict[str, dict[str, Any]]:
        """Collects and flattens simulation and agent output data."""
        output_holder = self.initialize_outputs()
        output_holder[TargetKey.SIMULATION.value] = super().output_collector()
        for model in self.models:
            if o := model.output_collector():
                output_holder[TargetKey.AGENTS.value].update(o)
        return output_holder

    def write_output_data(
        self,
        output_holder: dict,
        output_path: Path,
        output_format: OutputFormat,
    ) -> None:
        """Writes simulation and agent data to disk in the specified format."""
        self._create_output_dir(output_path)
        self._save_simulation_data(output_holder, output_path, output_format)
        self._save_agent_data(output_holder, output_path, output_format)

    def _create_output_dir(self, output_path: Path) -> None:
        """Creates the output directory if it does not exist."""
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)

    def _save_simulation_data(
        self,
        output_holder: dict,
        output_path: Path,
        output_format: OutputFormat,
    ) -> None:
        """Saves simulation data to disk in the specified format."""
        sim_str = f"_{TargetKey.SIMULATION.value}.{output_format.value}"
        sim_output_path = output_path.parent / (output_path.stem + sim_str)
        sim_data = output_holder[TargetKey.SIMULATION.value]
        Output.save_dict(sim_data, sim_output_path, output_format)

    def _save_agent_data(
        self,
        output_holder: dict,
        output_path: Path,
        output_format: OutputFormat,
    ) -> None:
        """Saves agent data to disk in the specified format."""
        for agent_type, agent_data in output_holder[
            TargetKey.AGENTS.value
        ].items():
            flattened_agent_data = self._flatten_agent_data(
                agent_data, agent_type
            )
            agent_str = (
                f"_{TargetKey.AGENTS.value}_{agent_type}.{output_format.value}"
            )
            agent_output_path = output_path.parent / (
                output_path.stem + agent_str
            )
            Output.save_dict(
                flattened_agent_data, agent_output_path, output_format
            )

    def _flatten_agent_data(self, agent_data: dict, agent_type: str) -> dict:
        """Flattens the agent data dictionary."""
        return self.flatten_dict(agent_data, agent_type)

    @Output(target_key=TargetKey.SIMULATION)
    def mission_start(self):
        return self.timer.mission_start

    @Output(target_key=TargetKey.SIMULATION)
    def simulation_runtime(self):  # noqa D102
        return self.timer.runtime.total_seconds()

    @Output(target_key=TargetKey.SIMULATION)
    def total_mission_time(self):
        return self.timer.mission_runtime.total_seconds()

    @Output(target_key=TargetKey.SIMULATION)
    def seed(self):  # noqa D102
        return self.seed
