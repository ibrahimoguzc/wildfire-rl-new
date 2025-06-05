# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains Highest-Level Abstract Class Definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Any,
    ClassVar,
)

from sosid.output import Output
from sosid.util.abc import (
    ABCMeta,
    Component,
    ComponentMeta,
    abstractattribute,
    abstractmethod,
)


class Viewable(metaclass=ABCMeta):
    """Defines an abstract entity that can be viewed in the GUI."""

    @abstractmethod
    def __gui_repr__(self) -> Sequence[dict[str, Any]] | dict[str, Any]:
        """Method that implements the GUI Representation Protocol."""


class WritableMeta(ABCMeta):  # noqa: D101
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]):
        """Gathers :py:class:`Output` decorated methods."""
        if bases:  # Will skip on Writeable base-class
            outputs: set[str] = set()
            for base in reversed(bases):
                outputs.update(base.__dict__.get("__outputs__", set()))
            for attr, value in dct.items():
                if isinstance(value, Output):
                    outputs.add(attr)
                elif attr in outputs:  # Output is overwritten
                    outputs.remove(attr)
            dct["__outputs__"] = outputs
        return super().__new__(cls, name, bases, dct)


class Writable(metaclass=WritableMeta):
    """Base class for all writable simulation outputs."""

    __outputs__: ClassVar[set[str]] = set()

    def output_collector(self) -> dict[str, Any]:
        """Retrieve outputs of current instance."""
        output_dict = {
            output: getattr(self, output) for output in self.__outputs__
        }
        return output_dict


class Model(Viewable, Writable):
    """Abstract Base Class (ABC) for all models."""

    @abstractmethod
    def step(self) -> None:
        """Run the :py:class:`Model` for a single iteration."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the model to its initial state."""


class Viewer(metaclass=ABCMeta):
    """Defines the "view" (GUI) of the MVC architecture."""

    @abstractmethod
    def compileSimulationItems(self) -> None:
        """Compiles :py:class:`Viewable` objects to a render mapping."""

    @abstractmethod
    def addSimulationItem(self, viewable: Viewable) -> None:
        """Adds a ``viewable`` to the :py:class:`Viewer`."""

    @abstractmethod
    def removeSimulationItem(self, viewable: Viewable) -> None:
        """Removes a ``viewable`` from the :py:class:`Viewer`."""


class Controller(metaclass=ABCMeta):
    """Defines the "controller" of the MVC architecture.

    The controller is responsible for interfacing the model(s) and view
    as well as coordinating the execution of the model. The former
    aspect allows the GUI code to be decoupled from that of the model.
    As such it is possible to run the model without a GUI present

    Attributes:
        __view__: An instance of :py:class:`Viewer`
    """

    __view__: Viewer | None = None

    @abstractattribute
    def name(self) -> str:
        """Name used to label the `:py:class:`Viewer` window."""

    @abstractmethod
    def step(self) -> None:
        """Run the program for a single iteration."""

    @abstractmethod
    def step_for(self, n_steps: int) -> None:
        """Run the program for ``n_steps`` iterations."""

    @abstractmethod
    def start(self) -> None:
        """Start the threaded program."""

    @abstractmethod
    def run(self) -> None:
        """Runs the main-loop of the program."""

    @abstractmethod
    def pause(self) -> None:
        """Pauses the execution of the program."""

    @abstractmethod
    def play(self) -> None:
        """Resumes the execution of the program."""

    @abstractmethod
    def stop(self) -> None:
        """Terminates the execution of the program."""

    def refresh_view(self) -> None:
        """Forces the :py:class:`Viewer` to refresh all items."""
        self.__view__.compileSimulationItems() if self.__view__ else ()

    def add_to_view(self, viewable: Viewable) -> None:
        """Adds a ``viewable`` from the :py:class:`Viewer`."""
        self.__view__.addSimulationItem(viewable) if self.__view__ else ()

    def remove_from_view(self, viewable: Viewable) -> None:
        """Removes a ``viewable`` from the :py:class:`Viewer`."""
        self.__view__.removeSimulationItem(viewable) if self.__view__ else ()


class ComponentWritableMeta(ComponentMeta, WritableMeta):
    """Meta class for writable components.

    Combined MetaClass to prevent metaclass conflicts.
    """


class ComponentWritable(Component, Writable, metaclass=ComponentWritableMeta):
    """Base class for all writable simulation output components."""
