# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from typing import Optional

import numba
import pytest

from util.numba.annotation import (
    BUILTIN_MAPPING,
    BuiltinConverter,
    GenericAliasConverter,
    get_signature,
)

numba_int = BUILTIN_MAPPING[int]
numba_float = BUILTIN_MAPPING[float]
ANNOTATION_TEST_CASES = {
    "argnames": "annotations, expected_result,",
    "argvalues": [
        (
            {"x": tuple[int, int], "y": float, "return": float},
            numba_float(
                numba.types.Tuple([numba_int, numba_int]), numba_float
            ),
        ),
        (
            {
                "x": numba.types.int64,
                "y": numba.types.int8,
                "return": numba.types.int64,
            },
            numba.types.int64(numba.types.int64, numba.types.int8),
        ),
        (  # Testing None-type and List
            {"x": list[int], "y": int, "return": None},
            numba.types.void(numba.types.List(numba_int), numba_int),
        ),
    ],
}


@pytest.fixture(scope="module")
def test_func():
    """Provides a mock function used in testing annotations."""

    def func(x, y) -> None:
        pass

    return func


@pytest.mark.parametrize(**ANNOTATION_TEST_CASES)
def test_get_signature(test_func, annotations, expected_result):
    """Tests if the annotations can be converted without errors."""
    test_func.__annotations__ = annotations
    signature = get_signature(test_func)
    assert signature, "Signature could not be converted"
    assert str(signature) == str(expected_result)


# All mapings between builtin types and Number types defined by
# BuiltinConverter should not fail
builtin_map_cases = [
    (type_hint, expected_result, False)
    for type_hint, expected_result in BuiltinConverter.mapping.items()
]
BUILTIN_TEST_CASES = {
    "argnames": "type_hint, expected_result, fail_case",
    "argvalues": builtin_map_cases + [(object, KeyError, True)],
}


@pytest.mark.parametrize(**BUILTIN_TEST_CASES)
def test_builtin_converter(type_hint, expected_result, fail_case):
    """Tests if builtin types are correctly converted to Numba types."""
    if fail_case:
        with pytest.raises(expected_result):
            BuiltinConverter(type_hint).get_signature()
    else:
        assert BuiltinConverter(type_hint).get_signature() == expected_result


GENERIC_ALIAS_CASES = {
    "argnames": "type_hint, expected_result, fail_case",
    "argvalues": [
        # Testing tuple w/ integers
        (tuple[int, int], numba.types.Tuple((numba_int, numba_int)), False),
        # Testing nested tuple w/ floats
        (
            tuple[float, tuple[float, float]],
            numba.types.Tuple(
                [numba_float, numba.types.Tuple([numba_float, numba_float])]
            ),
            False,
        ),
        # Testing Optional argument
        (Optional[int], numba.types.optional(numba_int), False),
        # Testing Optional w/ nested arguments
        (
            Optional[tuple[float, float]],
            numba.types.optional(
                numba.types.Tuple([numba_float, numba_float])
            ),
            False,
        ),
        # Testing List w/ Numba types
        (list[numba.int8], numba.types.List(numba.int8), False),
    ],
}


@pytest.mark.parametrize(**GENERIC_ALIAS_CASES)
def test_generic_alias_converter(type_hint, expected_result, fail_case):
    """Tests GenericAliases such as Tuple[int] or Tuple[Tuple[int]]."""
    signature = GenericAliasConverter(type_hint).get_signature()
    assert str(signature) == str(expected_result)
