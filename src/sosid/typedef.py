# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains type-hint definitions for use within the entire project."""

from typing import Union
#from collections import namedtuple

import numpy as np
from recordclass import dataobject


LatLon = Union[tuple[float, float], np.ndarray]
""" A 2D point on the Geographic (lat = Northward, lon = Eastward)
Coordinate System

"""

Position = Union[tuple[int, int], np.ndarray]
"""A 2D point on the Global (x = Right, y = Down) coordinate system.

If a Numpy array is used then a position must be a column vector.
Therefore, in a position matrix where multiple positions are specified
accessing a single position can be done with the following indexing::

    position_matrix[:, index]

"""

Index = Union[tuple[int, int], tuple[np.ndarray, np.ndarray]]
"""An index within the CA coordinate system (i = Down, j = Right).

If multiple indices are specified, the format to be adopted should
be a tuple of arrays as used by :py:function:`numpy.ndarray.nonzero`
as this allows direct look-up of multiple values from Numpy arrays.
"""


class GridDescriptor(dataobject):
    """A fundamental description of a Grid.

    Shape is defined in the format (n_rows, n_cols)
    dimensions are defined in the format (width, height)

    This standard description is employed for unit conversions between
    pos and index coordinate systems.
    """

    shape: tuple[int, int]
    dimensions: tuple[float, float]
