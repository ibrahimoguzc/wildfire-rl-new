# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from sosid.model.abm.agent import Agent
from sosid.model.transform import gps_to_mercator
from sosid.util.validation import check_required_fields

if TYPE_CHECKING:
    from sosid.simulation import PositionInput


def find_k_nearest_airports(
    center_coords: np.ndarray,
    airports_data: list[dict[str, any]],
    k_nearest_airports: int,
) -> list[dict[str, any]]:
    """Find the K-nearest airports to a specified center.

    Returns a list of dictionaries with the airport data nearest to the
    center. If k <= 0, returns all the airport data, sorted according to
    the distance w.r.t. the center.

    Args:
        center_coords: Single GPS Coordinate.
        airports_data: List of dictionaries containing GPS pos of the
                        airports.
        k_nearest_airports: The nr. of the nearest airports to filter.
    """
    # Get airport positions from airports_data and transform to ndarray
    airports_positions = np.array([item["pos"] for item in airports_data])

    # Calculate the distances from the ignition center to all airports
    distances = Agent.distance_gps(center_coords, airports_positions)

    # Merge distances with airports_data
    for airport, distance in zip(airports_data, distances):
        airport["distance"] = distance

    # Sort airport data according to their distances
    airports_data_sorted_by_distance = sorted(
        airports_data, key=lambda x: x["distance"]
    )

    # Return only the first K nearest airports (if k > 0)
    if k_nearest_airports > 0:
        k_nearest_airports_data = airports_data_sorted_by_distance[
            0:k_nearest_airports
        ]
        return k_nearest_airports_data

    return airports_data_sorted_by_distance


def deploy_airport_locations(
    sim_params: object,
    airports: list[dict[str, object], PositionInput],
    airport_input_cls: type[PositionInput],
) -> list[dict[str, object]]:
    """Override simulation parameters with airports' positions.

    If the override_airport_param in terrain_gen_input is set to True,
    it overrides the simulation parameters with the K-nearest airports
    to the (first) ignition center and in a given bounding box
    dimensions.
    """
    check_required_fields(
        ["ignition_centers", "k_nearest_airports", "terrain_inputs"],
        sim_params,
    )
    check_required_fields(
        ["fire_map_coordinates", "airport_locations_file"],
        sim_params,
        "terrain_inputs",
    )
    terrain_inputs = sim_params["terrain_inputs"]
    # Get first ignition center
    ignition_center = sim_params["ignition_centers"][0].get_gps_coords(
        gps_to_mercator(terrain_inputs.fire_map_coordinates[0])[:2][::-1],
    )

    # Get saved airport data
    with open(terrain_inputs.airport_locations_file, "rb") as file:
        airport_data = pickle.load(file, encoding="bytes")

    # Filter by the K-nearest airports and by distance
    airports_data_filtered = find_k_nearest_airports(
        ignition_center,
        airport_data,
        sim_params["k_nearest_airports"],
    )

    # Get GPS positions
    position_array_gps = [
        [float(item["pos"][0]), float(item["pos"][1])]
        for item in airports_data_filtered
    ]

    # Override airport parameters. If nr. new airports < current
    # airports --> it overrides only with the nr. of new airports
    # (agents per base must be == nr. airports)
    for i in range(len(position_array_gps)):
        if i >= len(airports):
            break
        pos = position_array_gps[i]
        pos_x, pos_y = map(float, pos)
        if isinstance((ap := airports[i]), airport_input_cls):
            ap = ap.model_dump()
        ap.pop("gps_coords", None)
        ap.pop("pos", None)
        ap["gps_coords"] = (pos_x, pos_y)
        airports[i] = ap

    return airports
