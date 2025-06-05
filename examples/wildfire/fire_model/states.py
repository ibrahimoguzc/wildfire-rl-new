# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains mappings between readable fire states and their int value.

This module contains the definition of the mapping between the
human-readable wildfire states and their integer value within the
Cellular Automata (CA) Fire-Propagation model of (TODO mention RUI). The
most clean way to implement this mapping is to use the module-level
constants: :py:const:`NONFLAMMABLE`, :py:const:`COMBUSTIBLE`,
:py:const:`EARLY_BURNING`, :py:const:`FULL_BURNING`, and
:py:const:`BURNT`.
# TODO include refernece to the Rui paper above

The reason for selecting to use module level constants is to reduce
verbosity within the Numba functions. For example, if dictionaries
or C-structs defined with Numpy were used instead of accesing a
fire-state by its variable name, at minimum retrieving the fire-states
would require a syntax such as the following::

    if state == states["nonflammable"]:
        pass

This clearly requires more effort than simply calling the module-level
constant::

    if state == NONFLAMMABLE:
        pass

Not to mention that getting Numba to deal with dictionary inputs
requires additional effort and set-up.

Note:
    The :py:const:`NONFLAMMABLE` state has been added to allow quick
    look-up of cells that contain no fuels during the propagation
    calculation. This allows all relevant information to be stored
    inside a single array which reduces memory access time.

"""

from matplotlib import colors

__all__ = [
    "BURNT",
    "COMBUSTIBLE",
    "EARLY_BURNING",
    "EXTINGUISHING",
    "FULL_BURNING",
    "NONFLAMMABLE",
    "SUPPRESSED",
]
# TODO consider using https://docs.python.org/3/library/enum.html

# Docstrings are added after module constants to render values in Sphinx
SUPPRESSED = 0
"""int: A cell that has been suppressed by an agent with suppressant """

NONFLAMMABLE = 1
"""int: A cell with no fuels (i.e. water, roads, rock, etc.)
"""

COMBUSTIBLE = 2
"""int: A cell that has fuel content and can ignite (i.e. farmland,
forest, houses, etc)
"""

EARLY_BURNING = 3
"""int: A cell that has ignited. If a cell transitions to early-burning,
then it will automatically transition to a full burning state at the
next iteration.
"""

FULL_BURNING = 4
"""int: A cell that is fully ignited which has the capability to ignite
neighboring cells. If all neighboring cells are either full burning,
beyond the full burning state, or are non-flammable, then the cell will
start extinguishing in the next iteration.
"""

EXTINGUISHING = 5
"""int: A cell in which the fire is rapidly losing intesity. If a cell
has transitioned to this state, then it will be burnt in the next
iteration.
"""

BURNT = 6
"""int: A cell in which a fire has fully extinguished. It is assumed
that these cells cannot be re-ignited as their fuel content is now zero.
"""

COLOR_TABLE = {
    SUPPRESSED: (78, 82, 79, 255),
    NONFLAMMABLE: (78, 82, 79, 0),
    COMBUSTIBLE: (25, 94, 44, 0),
    EARLY_BURNING: (255, 0, 0, 255),
    FULL_BURNING: (255, 69, 0, 255),
    EXTINGUISHING: (194, 0, 0, 255),
    BURNT: (43, 30, 30, 200),
}
"""Dict[int, Tuple[int, int, int]]: Defines the color mapping between
integer fire states and their respective RGB color.
"""

CMAP = colors.ListedColormap(
    [tuple(c / 255.0 for c in rgb) for rgb in COLOR_TABLE.values()]
)

"""colors.ListedColorMap: Defines the color mapping for use with
matplotlib plots.
"""
