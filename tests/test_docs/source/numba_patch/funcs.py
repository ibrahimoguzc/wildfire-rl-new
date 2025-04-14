# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""A sample module with Numba, Cuda, and regular Python Functions."""

import numba
from numba import cuda


@numba.jit(nopython=True)
def jitted_func(a: int) -> int:
    """A Numba JIT compiled Function."""
    return a


@cuda.jit(device=True, inline=True)
def device_func(a: int) -> int:
    """A Cuda Device Function."""
    return a


@cuda.jit
def cuda_func(a: int) -> int:
    """A Cuda Auto-Jit Function."""
    return a


def normal_func(a: int) -> int:
    """A Normal python Function."""
    return a
