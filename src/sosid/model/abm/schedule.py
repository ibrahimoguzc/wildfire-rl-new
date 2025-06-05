# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains modified scheduler classes of the MESA ABM library."""

from collections import defaultdict

from mesa.model import Model
from mesa.time import BaseScheduler as MesaBaseScheduler


class BaseScheduler(MesaBaseScheduler):
    """Activates agents one at a time, in the order they were added.

    Assumes that each agent added has a *step* method which takes no
    arguments. (This is explicitly meant to replicate the scheduler in
    MESA).
    """

    def __init__(self, model: Model) -> None:
        """Create a new, empty BaseScheduler."""
        self.model = model
        self._agents = {}

    def step(self) -> None:
        """Execute the step of all the agents, one at a time."""
        for agent in self.agent_buffer(shuffled=False):
            agent.step()


class RandomActivation(BaseScheduler):
    """Activates each agent once per step in random order.

    This scheduler reshuffles order of activation in every step.
    This is equivalent to the NetLogo 'ask agents...' and is generally
    the default behavior for an ABM.
    Assumes that all agents have a step(model) method.
    """

    def step(self) -> None:
        """Executes the step of all agents."""
        for agent in self.agent_buffer(shuffled=True):
            agent.step()


class SimultaneousActivation(BaseScheduler):
    """Activates all agents simulatenously.

    This scheduler requires that each agent have two methods: step and
    advance.
    step(): activates the agent and stages any necessary changes, but
        does not apply them yet.
    advance(): applies the changes.
    """

    def step(self) -> None:
        """Step all agents, then advance them."""
        agent_keys = list(self._agents.keys())
        for agent_key in agent_keys:
            self._agents[agent_key].step()
        for agent_key in agent_keys:
            self._agents[agent_key].advance()


# MESA ABM library has StagedActivation method which has not been
# defined here


class RandomActivationByBreed(RandomActivation):
    """Activates each type of agent once per step in random order.

    This scheduler reshuffles order of activation in every step.
    This is equivalent to the NetLogo 'ask breed...' and is generally
    the default behavior for an ABM.
    Assumes that all agents have a step() method.
    """

    def __init__(self, model):  # noqa D102
        super().__init__(model)
        self.agents_by_breed = defaultdict(dict)

    def add(self, agent):
        """Add an Agent object to the schedule.

        Args:
            agent: An Agent to be added to the schedule.
        """
        self._agents[agent.unique_id] = agent
        agent_class = type(agent)
        self.agents_by_breed[agent_class][agent.unique_id] = agent

    def remove(self, agent):
        """Remove all instances of a given agent from the schedule."""
        del self._agents[agent.unique_id]

        agent_class = type(agent)
        del self.agents_by_breed[agent_class][agent.unique_id]

    def step(self, by_breed=True):
        """Executes the step of each agent breed, one at a time.

        Args:
            by_breed: If True, run all agents of a single breed before
                running the next one.
        """
        if by_breed:
            for agent_class in self.agents_by_breed:
                self.step_breed(agent_class)
        else:
            super().step()

    def step_breed(self, breed):
        """Shuffle order and run all agents of a given breed.

        Args:
            breed: Class object of the breed to run.
        """
        agent_keys = list(self.agents_by_breed[breed].keys())
        self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_breed[breed][agent_key].step()

    def get_breed_count(self, breed_class):
        """Returns the number of agents of class breed in the queue."""
        return len(self.agents_by_breed[breed_class].values())
