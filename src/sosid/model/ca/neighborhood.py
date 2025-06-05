# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains Celullar Automata (CA) Neighborhood classes."""

import itertools


class Neighborhood:
    """Defines the base-class for Cellular Automata (CA) neighborhoods.

    Specializations of this base-class can override :py:meth:

    Args:
        radius: Specifies the number of cells away from the center
            that the neighborhood includes

    Keyword Arguments:
        include_center: Sets whether the center cell should be included
            in the neighborhood. Defaults to False.

    Attributes:
        __dim__ (int): Sets the dimension of the Neighborhood.
            Defaults to 2 for 2D.

    """

    __dim__ = 2

    def __init__(self, radius: int, *, include_center: bool | None = False):
        self.radius = radius
        self.include_center = include_center

    @property
    def center(self) -> tuple[int, ...]:
        """Defines the relative location of the centroid.

        This will always return a :py:obj:`tuple` with a length equal to
        the dimension of the neighborhood.

        Returns:
            Location of the centroid in all dimensions

        Examples:
            >>> from sosid.model.ca.neighborhood import Neighborhood
            >>> obj = Neighborhood(radius=1)
            >>> obj.center
            (0, 0)
            >>> obj.__dim__ = 3
            >>> obj.center
            (0, 0, 0)

        """
        return tuple(0 for _ in range(self.__dim__))

    @property
    def limits(self) -> tuple[tuple[int, int]]:
        """Returns the offset limits for each dimension.

        The length of the outer :py:obj:`tuple` is equal to the
        :py:attr:`__dim__` of the Neighborhood, where each inner each
        :py:obj:`tuple` describes the outermost lower and upper offsets.

        Note:
            :ref:`numba.stencil() <numba:stencil-neighborhood>` kernels
            require these limits be specified explicitely for run-time.

        Examples:
            For a standard 2D neighborhood with a radius = 1

            >>> obj = Neighborhood(radius=1)
            >>> obj.limits
            ((-1, 1)(-1, 1))

            For a 3D neighborhood with a radius = 2

            >>> obj = Neighborhood(radius=2)
            >>> obj.__dim__ = 3
            >>> obj.limits
            ((-2, 2)(-2, -2)(-2, 2))

        """
        return tuple((-self.radius, self.radius) for _ in range(self.__dim__))

    @property
    def cell_count(self) -> int:
        """Returns the number of cells within the neighborhood."""
        raise NotImplementedError

    def as_tuple(self) -> tuple[tuple[int, ...]]:
        """Returns a tuple of offsets defining the neighboring cells.

        The tuple of offsets is therefore a relative indexing pattern.
        This is useful for iterating over the neighborhood within a
        :py:func:`numba.jit` decorated function where nested
        :py:obj:`tuple` arguments are allowed.
        """
        offset_range = range(-self.radius, self.radius + 1)
        return tuple(
            filter(
                self.in_neighborhood,
                itertools.product(offset_range, repeat=self.__dim__),
            )
        )

    def in_neighborhood(self, offset: tuple[int, ...]) -> bool:
        """Checks if the current ``offset`` is within the neighborhood.

        Args:
            offset: Offset indices (i_offset, j_offset) relative to the
                center cell such that (i + i_offset, j + j_offset)
                describes the absolute index of the neighboring cell

        Returns:
            True if ``offset`` is within neighborhood, else False

        """
        raise NotImplementedError


class MooreNeighborhood(Neighborhood):
    """Defines a square shaped neighborhood.

    For a neighborhood with :py:attr:`radius` = 1, the square
    neighborhood is comprised of 8 neighbors as shown below by
    :numref:`moore_diagram`

    # TODO replace the figure below

    .. _moore_diagram:
    .. figure:: ../../../docs/source/_static/img/moore.svg
       :width: 85 %

       Moore Neighborhood of Radius = 1 (Wikipedia)


    See Also:
        http://mathworld.wolfram.com/MooreNeighborhood.html

    """

    __dim__ = 2

    @property
    def cell_count(self) -> int:  # noqa D102
        n_cells = (2 * self.radius + 1) ** 2
        return n_cells if self.include_center else n_cells - 1

    def in_neighborhood(self, offset) -> bool:  # noqa D102
        if offset != self.center:
            return all(abs(o) <= self.radius for o in offset)
        return self.include_center


class NeumannNeighborhood(Neighborhood):
    """Defines a cross shaped neighborhood.

    For a neighborhood with :py:attr:`radius` = 1, the square
    neighborhood is comprised of 4 neighbors as shown below by
    :numref:`neumann_diagram`

    # TODO replace the figure below

    .. _neumann_diagram:
    .. figure:: ../../../docs/source/_static/img/neumann.svg
       :width: 85 %

       Neumann Neighborhood of Radius = 1 (Wikipedia)

    See Also:
        http://mathworld.wolfram.com/vonNeumannNeighborhood.html

    """

    __dim__ = 2

    @property
    def cell_count(self) -> int:  # noqa D102
        n_cells = 2 * self.radius * (self.radius + 1) + 1
        return n_cells if self.include_center else n_cells - 1

    def in_neighborhood(self, offset) -> bool:  # noqa D102
        if offset != self.center:
            return sum(abs(o) for o in offset) <= self.radius
        return self.include_center


# Defining default offsets for use in common CA models
MOORE = MooreNeighborhood(radius=1, include_center=False)
MOORE_OFFSETS = MOORE.as_tuple()
NEUMANN = NeumannNeighborhood(radius=1, include_center=False)
NEUMANN_OFFSETS = NEUMANN.as_tuple()
