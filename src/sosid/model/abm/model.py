# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains a modified Model class of the MESA ABM library."""

import random
import time
from itertools import chain
from typing import Any

from mesa import Model as MesaModel
from mesa.space import ContinuousSpace, Grid

# Ensures that the step-method is implemented
from sosid.abstract import Model
from sosid.typedef import Position
from sosid.util.abc import abstractattribute

from .agent import Agent
from .schedule import BaseScheduler

__all__ = ["AgentBasedModel"]


class AgentBasedModel(MesaModel, Model):
    """Modifies :py:class:`mesa.Model` to accept a ``simulation``."""

    def __new__(cls, simulation: object | None = None, *args, **kwargs):
        """Provides :py:class:`mesa.Model` the RNG of ``simulation``."""
        model = object.__new__(cls)  # This only works in Python 3.3 and above
        if simulation:
            model.simulation = simulation
            model.random = simulation.random
        else:
            model._seed = time.time()
            if "seed" in kwargs and kwargs["seed"] is not None:
                model._seed = kwargs["seed"]
            model.random = random.Random(model._seed)
        return model

    def __init__(self, simulation: object | None = None):
        self.running = False
        self.simulation = simulation

    @abstractattribute
    def agents(self) -> list[Agent]:
        """Enforces that at least one :py:class:`Agent` is created."""

    @abstractattribute
    def agents_by_type(self) -> dict[type[Agent], list[Agent]]:
        """Map of :py:class:`Agent` type to its instances."""

    @abstractattribute
    def schedule(self) -> BaseScheduler:
        """Enforces that the user specifies an Activation type."""

    @abstractattribute
    def space(self) -> Grid | ContinuousSpace:
        """Enforces that the user specifies an Space type."""

    def __gui_repr__(self):
        """Optionally one can implement a GUI representation."""
        return [a.__gui_repr__() for a in chain(*self.agents_by_type.values())]

    def add_agent(self, agent: Agent, pos: Position | None = None) -> None:
        """Add :py:class:`Agent` to :py:class:`Simulation`."""
        if self.exists(agent):
            raise RuntimeError(f"{agent} has already been added to the Model")
        if pos is not None:
            self.space.place_agent(agent, pos)
        self.schedule.add(agent)
        self.agents.append(agent)
        self.agents_by_type[type(agent)].append(agent)
        self.simulation.add_to_view(agent)

    def remove_agent(self, agent: Agent) -> None:
        """Remove :py:class:`Agent` from :py:class:`Simulation`."""
        if self.exists(agent):
            self.schedule.remove(agent)
            self.space.remove_agent(agent) if agent.pos is not None else ()
            (a := self.agents).pop(a.index(agent))
            (a := self.agents_by_type[type(agent)]).pop(a.index(agent))
            self.simulation.remove_from_view(agent)
        else:
            raise RuntimeError(f"{agent} does not exist")

    __last_id__ = -1

    def get_unique_id(self) -> int:
        """Generate a unique_id for a new agent."""
        self.__last_id__ += 1
        return self.__last_id__

    def exists(self, agent: Agent) -> bool:
        """Check if ``agent`` already exists in the model."""
        return True if agent in self.agents else False

    def output_collector(self) -> dict[str, Any]:
        agents_by_type = {}
        for agent_type, agents in self.agents_by_type.items():
            agents_output = {}
            n = 0
            for agent in agents:
                if agent_output := agent.output_collector():
                    agent_key = f"{n}"
                    if hasattr(agent, "output_id_name"):
                        agent_key = (
                            f"{n}_{agent.output_id_name.replace('.json', '')}"
                        )
                    agents_output[agent_key] = agent_output
                    n += 1
            if agents_output:
                agents_by_type[agent_type.__name__] = agents_output
        return agents_by_type
