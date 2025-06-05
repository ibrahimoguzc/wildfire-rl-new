# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains config options, and custom markers/fixtures for pytest."""

import inspect
from collections.abc import Callable
from typing import Any

import numba
import pytest

from util.numba.annotation import get_signature

JIT_COMPILABLE_TYPES = [numba.core.registry.CPUDispatcher]
"""List[type]: Numba types that can be JIT compiled"""

# -- Custom Options -----------------------------------------------------# noqa


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests. Default: False",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return  # Do not skip if --runslow is given.
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# -- Custom Markers -----------------------------------------------------# noqa

JIT_MARKER = "retrieve_jit_funcs"
"""str: Name of marker that collects :py:func:`numba.jit` decorated
functions
"""


def pytest_configure(config):
    """Registers additional markers for use within pytests."""
    config.addinivalue_line(
        "markers",
        (
            f"{JIT_MARKER}: Marks a test to retrieve all @numba.jit decorated "
            "functions used within the current test module"
        ),
    )


# -- Custom Fixtures ----------------------------------------------------# noqa


# TODO finish docstring
@pytest.fixture(scope="session")
def compile_jit() -> Callable:
    """Returns the function used to JIT compile a ``jit_func``.

    This fixture reduces the boiler-plate required when testing Numba
    JIT-compiled functions. However, it requires that the ``jit_func``
    be correctly annotated. This is because the required signature to
    compile a function decorated with :py:func:`numba.jit`, using JIT
    compilation, is obtained from the type-hints (annotations) of the
    underlying Python function.

    Sample usage is as follows::

        import numba


        @numba.jit(nopython=True)
        def add_two(a: int, b: int) -> int:
            return a + b


        def test_add_two(compile_jit):
            compile_jit(add_two)

    Parametrized testing for all :py:func:`numba.jit` decorated
    functions used within the current test module::

        @pytest.mark.retrieve_jit_funcs
        def test_jit_compile(jit_func, jit_compile):
            jit_compile(jit_func)

    In-cases where a parameterization different than the one provided
    by the ``retrieve_jit_funcs`` marker is required, then a custom
    parameterization can be defined normally as follows::

        JIT_TEST_CASES = {
            "argnames": "jit_func",
            "argvalues": [
                preprocess,
                calc_spread_rate,
                calc_ideal_time_step,
                postprocess,
            ],
        }


        @pytest.mark.parametrize(
            **JIT_TEST_CASES, ids=lambda f: f.py_func.__name__
        )
        def test_JIT_compile(jit_func, compile_jit):
            compile_jit(jit_func)

    Note:
        Currently only functions decorated with the Numba
        :py:func:`numba.jit` decorator can be JIT-compiled with this
        pytest fixture.

    """

    def compiler(jit_func):
        # Checking if provided ``jit_func`` is compilable
        error_msg = f"Provided {jit_func} is not JIT compilable"
        assert type(jit_func) in JIT_COMPILABLE_TYPES, error_msg

        # Enforcing ``jit_func`` uses nopython mode
        error_msg = f"Provided {jit_func} must use `nopython` mode"
        assert jit_func.targetoptions["nopython"]

        # Converting function annotation to Numba signature
        signature = get_signature(jit_func.py_func)
        jit_func.compile(signature)  # Compiling JIT w/ converted signature

    return compiler


@pytest.fixture(scope="session")
def jit_func():
    """Dummy fixture for use with the ``retireve_jit_funcs`` marker.

    Although this fixture does not return anything, it serves the
    purpose of reserving the ``jit_func`` name. This is accomplished by
    raising an error in scenarios where the ``jit_func`` is not passed
    any values from a pytest test parameterization.

    Raises:
        RuntimeError: If fixture is used outside the context of a pytest
            parameterization for which it is intended.

    """
    error_msg = "Test declared without using the ``retrieve_jit_funcs`` marker"
    raise RuntimeError(error_msg)


# -- Overrides to Default pytest Behavior -------------------------------# noqa


def pytest_generate_tests(metafunc) -> None:
    """Overrides generation of tests for ``metafunc`` definitions.

    Currently, the implementation below checks if a test has been
    marked with the "retrieve_jit_funcs" marker as well as if it
    is using the ``jit_func`` fixture. If so, it parametrizes the
    test provided by ``metafunc``.

    Args:
        metafunc: Test inspection helper that is used for generating
            tests based on test configuration or values specified in the
            class or module where a test function is defined

    """
    jit_fixture = "jit_func" in metafunc.fixturenames
    jit_marker = JIT_MARKER in metafunc.definition.keywords
    if jit_fixture and jit_marker:
        jit_funcs = get_jit_funcs(metafunc.module)
        # Adding the collected jit_funcs as (parametrized) test-cases
        metafunc.parametrize(
            "jit_func", jit_funcs, ids=lambda f: f.py_func.__name__
        )


def is_jitfunc(obj: Any) -> bool:
    """Determines if the current `obj` is a valid ``jit_func``."""
    return type(obj) in JIT_COMPILABLE_TYPES


def get_jit_funcs(module: object) -> list[object]:
    """Gets :py:func:`numba.jit` decorated functions in a ``module``.

    Args:
        module: A Python module object (not the module name!)

    """
    jit_members = inspect.getmembers(module, is_jitfunc)
    return [jit_func for name, jit_func in jit_members]
