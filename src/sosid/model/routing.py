# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/
import copy
import math
from collections import defaultdict
from collections.abc import Callable


def find_best_route(
    origin_node, destination_node, value_func, all_nodes, neighbouring_nodes
) -> list:
    """Find best route based on ``value_func``.

    Uses Dijkstra algorithm to find best route between ``origin_node``
    and ``destination_node`` based on the value returned from
    ``value_func``.

    Args:
        origin_node: origin or start node
        destination_node: destination or end node
        value_func (_type_): function which returns value from inputs
        ``origin_node`` and ``destination_node``
        all_nodes (list): _description_
        neighbouring_nodes (dict): Dictionary linking each node to its
            neighbouring nodes

    Returns:
        list: Best path to ``destination_node`` from ``origin_node``.
            Excludes ``origin_node``.
    """
    # Initialize
    unvisited_nodes = copy.copy(all_nodes)
    unvisited_nodes.remove(origin_node)
    previous_position = defaultdict(list)
    values = defaultdict(lambda: math.inf)
    node = origin_node
    values[node] = 0
    previous_position[node] = node

    # Terminate when ideal path to target node is found
    while node != destination_node:
        # Compute values for all neighbouring nodes
        for neighbour_node in neighbouring_nodes[node]:
            new_value = value_func(node, neighbour_node)
            previous_node = node
            # Iterate over nodes until origin and sum values
            if previous_node != origin_node:
                new_value += values[previous_node]
                previous_node = previous_position[previous_node]

            if new_value < values[neighbour_node]:
                values[neighbour_node] = new_value
                previous_position[neighbour_node] = node
        # Find ideal next node
        node = min(unvisited_nodes, key=lambda n: values[n])
        unvisited_nodes.remove(node)

    node = destination_node
    node_list = []

    while node != origin_node:
        node_list.append(node)
        node = previous_position[node]
    node_list.reverse()
    return node_list


def evaluate_neighbours(nodes: list, function: Callable, criteria) -> dict:
    """Identifies neighbouring nodes based on ``criteria``.

    Args:
        nodes (list): List of nodes from which to find interconnected
            nodes
        function (function): function which takes in two nodes and a
            criteria value as input, the function computes a value
            between the two nodes, and compares it to criteria. Returns
            True or False, if True, nodes are
            interconnected/neighbouring.
        criteria: Value which governs whether two nodes are neighbours

    Returns:
        dict: Dictionary of nodes as keys and list of neighbours as
            values
    """
    neighbours = defaultdict(list)
    for node in nodes:
        for other_node in nodes:
            if node != other_node and function(node, other_node, criteria):
                neighbours[node].append(other_node)
    return neighbours
