# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains abstractions for defining Cellular Automata (CA) models."""

import dataclasses
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sosid.abstract import Model
from sosid.util.abc import abstractattribute, abstractmethod


@dataclass(frozen=True)
class GridData:
    """Contains Numpy.ndarrays of a CellularAutomata model.

    Caution:
        Do not forget to decorate the subclass with the
        :py:function:`dataclasses.dataclass` as follows::

            @dataclass
            class GridDataSubClass(GridData)
                pass

    """

    # TODO emit message to highlight the size of CA arrays
    def preallocate(self, shape, order: str | None = "C") -> None:
        """Overwrites default scalar values with Numpy arrays.

        Args:
            shape: (n_rows, n_cols) of the cellular environment.
                Due to the CA coordiante system, the number of rows,
                `n_rows`, corresponds to the y-axis and the number of
                columns, `n_cols`, corresponds to the x-axis of the
                Global coordinate system.
            order:

        """
        for field in dataclasses.fields(self):
            if issubclass(field.type, np.generic) or field.type is bool:
                value = getattr(self, field.name)
                if not isinstance(value, np.ndarray):
                    array = np.full(
                        shape=shape,
                        fill_value=value,
                        dtype=field.type,
                        order=order,
                    )
                    object.__setattr__(self, field.name, array)

    @cached_property
    def asdict(self):
        """Returns a dictionary version of :py:class:`GridData`.

        This is useful for passing a large-quantity of data arrays to
        a function as follows::

            def too_many_keywords(**GridData.asdict)
                pass

        """
        return dataclasses.asdict(self)


class CellularAutomataModel(Model):
    def __init__(self, simulation: object):
        self.simulation = simulation
        self.__data__.preallocate(self.shape)

    @abstractattribute
    def __data__(self) -> GridData:
        """Contains the data of a CA model as a bunch of arrays."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Defines the number of rows and columns of the CA grid."""
