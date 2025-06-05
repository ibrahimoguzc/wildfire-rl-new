# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import pytest

from sosid.model.routing import evaluate_neighbours, find_best_route

DIJKSTRA_TEST_CASES = {
    "argnames": "origin_destination, expected_result ",
    "argvalues": [([0, 2], [1, 2]), ([0, 3], [1, 2, 3]), ([0, 8], [1, 2, 8])],
}
NEIGHBOURS = {
    0: [1, 7],
    1: [0, 2, 7],
    2: [1, 3, 5, 8],
    3: [2, 4, 5],
    4: [3, 5],
    5: [2, 3, 4, 6],
    6: [5, 7, 8],
    7: [0, 1, 6, 8],
    8: [2, 6, 7],
}
ALL_NODES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
VALUES = {
    "0_1": 4,
    "0_7": 8,
    "1_7": 11,
    "1_2": 8,
    "2_8": 2,
    "2_3": 7,
    "2_5": 4,
    "3_4": 9,
    "3_5": 14,
    "4_5": 10,
    "5_6": 2,
    "6_8": 6,
    "6_7": 1,
    "7_8": 7,
}


def value_func(node_1, node_2):
    if node_2 < node_1:
        return VALUES[f"{node_2}_{node_1}"]
    return VALUES[f"{node_1}_{node_2}"]


@pytest.mark.parametrize(**DIJKSTRA_TEST_CASES)
def test_find_best_route(origin_destination, expected_result):
    origin, destination = origin_destination
    answer = find_best_route(
        origin, destination, value_func, ALL_NODES, NEIGHBOURS
    )
    assert answer == expected_result


NEIGHBOURING_VALUES = {
    #:  0  1  2  3  4  5  6  7  8 node indices
    0: [0, 2, 0, 0, 0, 0, 0, 2, 0],
    1: [2, 0, 2, 0, 0, 0, 0, 2, 0],
    2: [0, 2, 0, 2, 0, 1, 0, 0, 2],
    3: [0, 0, 2, 0, 2, 2, 0, 0, 0],
    4: [0, 0, 0, 2, 0, 2, 0, 0, 0],
    5: [0, 0, 1, 2, 2, 0, 2, 0, 0],
    6: [0, 0, 0, 0, 0, 2, 0, 1, 2],
    7: [2, 2, 0, 0, 0, 0, 1, 0, 1],
    8: [0, 0, 2, 0, 0, 0, 2, 1, 0],
}


def neighbouring_values_func(node_1, node_2, criteria):
    """Checks if nodes are neighbours"""
    value = NEIGHBOURING_VALUES[node_1][node_2]
    if value >= criteria:
        return True
    return False


NEIGHBOURS_2 = {
    0: [1, 7],
    1: [0, 2, 7],
    2: [1, 3, 8],
    3: [2, 4, 5],
    4: [3, 5],
    5: [3, 4, 6],
    6: [5, 8],
    7: [0, 1],
    8: [2, 6],
}
EVALUATE_NEIGHBOURS_TEST_CASES = {
    "argnames": "criteria, expected_neighbours",
    "argvalues": [(1, NEIGHBOURS), (2, NEIGHBOURS_2)],
}


@pytest.mark.parametrize(**EVALUATE_NEIGHBOURS_TEST_CASES)
def test_evaluate_neighbours(criteria, expected_neighbours):
    neighbours = evaluate_neighbours(
        ALL_NODES, neighbouring_values_func, criteria
    )
    assert neighbours == expected_neighbours
