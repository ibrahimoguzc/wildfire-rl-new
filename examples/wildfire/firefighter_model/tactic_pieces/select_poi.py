"""Tactics for selecting a point of interest (POI) to track."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from examples.wildfire.fire_model.states import BURNT, NONFLAMMABLE, SUPPRESSED
from examples.wildfire.firefighter_model.follower import DestinationType
from sosid.model.abm.task import Task, TaskStatus
from sosid.model.abm.trajectory import generate_straight_trajectory
from sosid.model.transform import gps_to_pos, index_to_pos
from sosid.util.abc import ABC

if TYPE_CHECKING:
    from examples.wildfire.firefighter_model.agents import SuppressionUAV


class SelectPOIType(Enum):
    WATER = "water"
    VIP = "vip"
    VEGETATION = "vegetation"
    TOPOGRAPHY = "topography"
    INDIRECT = "indirect"


class SelectPOITask(ABC, Task):
    def complete_method(self, agent: SuppressionUAV) -> None:
        """Set tracking for selected firefront."""
        fire_pos = gps_to_pos(
            agent.full_trajectory.gps_end, agent.top_left_bounds
        )
        propellant, mass = agent.propulsion.estimate_propellant_for_trajectory(
            agent.full_trajectory
        )
        mass -= agent.payload
        nearest_airport, _ = agent.get_nearest_airport(pos=fire_pos)
        elev_poi = agent.model.simulation.environment.get_elevation(fire_pos)
        elev_base = agent.model.simulation.environment.get_elevation(
            nearest_airport.pos
        )
        to_base_trajectory = generate_straight_trajectory(
            profile=agent.profile_parameters,
            gps_start=agent.full_trajectory.gps_end,
            gps_end=nearest_airport.gps_coords,
            altitude_start=agent.full_trajectory.altitudes[-1],
            altitude_end=elev_base + agent.profile_parameters.landing_altitude,
            elevation_start=elev_poi,
            elevation_end=elev_base,
            include_landing=True,
            include_takeoff=True,
        )
        propellant += agent.propulsion.estimate_propellant_for_trajectory(
            to_base_trajectory, start_mass=mass
        )[0]
        if agent.propulsion.is_propellant_available(propellant):
            agent.tasks.set_active(agent.tactic.track_poi)
        else:
            agent.set_destination(nearest_airport.pos, DestinationType.BASE)
            agent.tasks.set_active(agent.tactic.return_to_base)

    def fail_method(self, agent):
        """Return to base if fire has been extinguished."""
        if isinstance(
            agent.tactic.select_poi, SELECT_POI_TABLE[SelectPOIType.INDIRECT]
        ):
            agent.force_tactic_swap = True
            agent.tasks.set_active(agent.tactic.change)
        else:
            nearest_airport, _ = agent.get_nearest_airport()
            agent.set_destination(nearest_airport.pos, DestinationType.BASE)
            agent.tasks.set_active(agent.tactic.return_to_base)


def _positions_from_agents(agents) -> np.ndarray:
    """Return static agent positions as a two-column array."""
    positions = [
        np.asarray(obj.pos, dtype=float)
        for obj in agents
        if getattr(obj, "pos", None) is not None
    ]
    if not positions:
        return np.empty((0, 2), dtype=float)
    return np.vstack(positions)


def _nearest_position_cost(
    fire_positions: np.ndarray,
    objective_positions: np.ndarray,
    map_diagonal: float,
) -> np.ndarray:
    """Score firefronts by closeness to point objectives."""
    if objective_positions.size == 0:
        return np.zeros(len(fire_positions), dtype=float)

    distances = np.linalg.norm(
        fire_positions[:, None, :] - objective_positions[None, :, :],
        axis=2,
    )
    nearest_distances = np.min(distances, axis=1)
    return np.clip((map_diagonal - nearest_distances) / map_diagonal, 0, 1)


def _select_poi_cache(agent) -> dict:
    """Return a lazy cache scoped to the current fire-state version."""
    model_cache = agent.model.__cache__
    fire_version = getattr(agent.model.wildfire, "fire_state_version", 0)
    cache = model_cache.get("select_poi")
    if cache is None or cache.get("fire_state_version") != fire_version:
        cache = {"fire_state_version": fire_version}
        model_cache["select_poi"] = cache
    return cache


def _cached_fire_positions(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "fire_positions" not in cache:
        cache["fire_positions"] = agent.model.wildfire.fire_positions
    return cache["fire_positions"]


def _cached_burning_indices(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "burning_indices" not in cache:
        cache["burning_indices"] = agent.model.wildfire.burning_indices
    return cache["burning_indices"]


def _cached_map_diagonal(agent) -> float:
    cache = _select_poi_cache(agent)
    if "map_diagonal" not in cache:
        map_shape = np.array(agent.model.simulation.environment.dimensions)
        cache["map_diagonal"] = np.linalg.norm(map_shape)
    return cache["map_diagonal"]


def _cached_objective_positions(agent, cache_key: str, agents) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if cache_key not in cache:
        cache[cache_key] = _positions_from_agents(agents)
    return cache[cache_key]


def _candidate_firefronts(
    agent,
    include_burning_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return currently untracked firefront candidates.

    This mirrors the original selection behavior exactly: destinations already
    assigned to other firefighters are removed one by one, and if that removes
    every candidate the full firefront set is restored.
    """
    full_positions = _cached_fire_positions(agent)
    selected_indices = np.arange(len(full_positions))
    fire_positions = full_positions

    full_burning_indices = None
    burning_indices = None
    if include_burning_indices:
        full_burning_indices = _cached_burning_indices(agent)
        burning_indices = full_burning_indices

    for obj in agent.model.firefighters:
        if obj.destination is not None:
            untracked_pos = (fire_positions != obj.destination).any(axis=1)
            fire_positions = fire_positions[untracked_pos]
            selected_indices = selected_indices[untracked_pos]
            if include_burning_indices:
                burning_indices = burning_indices[untracked_pos]

            # Choose any firefront if all are already taken.
            if not np.size(fire_positions):
                fire_positions = full_positions
                selected_indices = np.arange(len(full_positions))
                if include_burning_indices:
                    burning_indices = full_burning_indices

    return fire_positions, burning_indices, selected_indices


def _cached_water_cost(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "water_cost" not in cache:
        cache["water_cost"] = _nearest_position_cost(
            fire_positions=_cached_fire_positions(agent),
            objective_positions=_cached_objective_positions(
                agent,
                "water_positions",
                agent.model.water_sources,
            ),
            map_diagonal=_cached_map_diagonal(agent),
        )
    return cache["water_cost"]


def _cached_urban_cost(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "urban_cost" not in cache:
        cache["urban_cost"] = _nearest_position_cost(
            fire_positions=_cached_fire_positions(agent),
            objective_positions=_cached_objective_positions(
                agent,
                "urban_positions",
                agent.model.protection_locations,
            ),
            map_diagonal=_cached_map_diagonal(agent),
        )
    return cache["urban_cost"]


def _cached_vip_cone_cost(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "vip_cone_cost" not in cache:
        fire_positions = _cached_fire_positions(agent)
        vip_cost = agent.exponential_cone_func(
            pos=fire_positions,
            vip=(location.pos for location in agent.model.protection_locations),
            map_diagonal=_cached_map_diagonal(agent),
        )
        vip_cost = np.asarray(vip_cost, dtype=float)
        if vip_cost.ndim == 0:
            vip_cost = np.full(len(fire_positions), float(vip_cost), dtype=float)
        cache["vip_cone_cost"] = vip_cost
    return cache["vip_cone_cost"]


def _cached_raw_vegetation_priority(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "raw_vegetation_priority" not in cache:
        cache["raw_vegetation_priority"] = np.array(
            [
                agent.calculate_priority_vegetation(i, j)
                for i, j in _cached_burning_indices(agent)
            ],
            dtype=float,
        )
    return cache["raw_vegetation_priority"]


def _cached_raw_topography_priority(agent) -> np.ndarray:
    cache = _select_poi_cache(agent)
    if "raw_topography_priority" not in cache:
        elevation_data = agent.terrain.elevation.elevation_data
        wind_direction = agent.model.simulation.environment.atmosphere.wind_aspect
        cache["raw_topography_priority"] = np.array(
            [
                agent.calculate_priority_topography(
                    i,
                    j,
                    elevation_data,
                    wind_direction,
                )
                for i, j in _cached_burning_indices(agent)
            ],
            dtype=float,
        )
    return cache["raw_topography_priority"]


def _normalize_priority_slice(agent, raw_priorities: np.ndarray) -> np.ndarray:
    return np.array(agent.normalize_priorities(list(raw_priorities)), dtype=float)


def _distance_cost(agent, fire_positions: np.ndarray, map_diagonal: float) -> np.ndarray:
    fire_distances = agent.distance(agent.pos, fire_positions)
    return (map_diagonal - fire_distances) / map_diagonal


def _destination_from_costs(
    fire_positions: np.ndarray,
    selection_cost: np.ndarray,
) -> np.ndarray:
    min_idx = int(np.argmax(selection_cost))
    return fire_positions[min_idx, :]


def _select_water_destination(agent) -> np.ndarray | None:
    fire_positions, _, selected_indices = _candidate_firefronts(agent)
    if not fire_positions.size:
        return None

    map_diagonal = _cached_map_diagonal(agent)
    distance_cost = _distance_cost(agent, fire_positions, map_diagonal)
    water_cost = _cached_water_cost(agent)[selected_indices]
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * water_cost
    )
    return _destination_from_costs(fire_positions, selection_cost)


def _select_vip_destination(agent) -> np.ndarray | None:
    fire_positions, _, selected_indices = _candidate_firefronts(agent)
    if not fire_positions.size:
        return None

    map_diagonal = _cached_map_diagonal(agent)
    distance_cost = _distance_cost(agent, fire_positions, map_diagonal)
    urban_cost = _cached_urban_cost(agent)[selected_indices]
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * urban_cost
    )
    return _destination_from_costs(fire_positions, selection_cost)


def _select_vegetation_destination(agent) -> np.ndarray | None:
    fire_positions, _burning_indices, selected_indices = _candidate_firefronts(
        agent,
        include_burning_indices=True,
    )
    if not fire_positions.size:
        return None

    map_diagonal = _cached_map_diagonal(agent)
    distance_cost = _distance_cost(agent, fire_positions, map_diagonal)
    vip_cost = _cached_vip_cone_cost(agent)[selected_indices]
    vegetation_cost = _normalize_priority_slice(
        agent,
        _cached_raw_vegetation_priority(agent)[selected_indices],
    )
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * vip_cost
        + agent.parameters.vegetation_cost_weight * vegetation_cost
    )
    return _destination_from_costs(fire_positions, selection_cost)


def _select_topography_destination(agent) -> np.ndarray | None:
    fire_positions, _burning_indices, selected_indices = _candidate_firefronts(
        agent,
        include_burning_indices=True,
    )
    if not fire_positions.size:
        return None

    map_diagonal = _cached_map_diagonal(agent)
    distance_cost = _distance_cost(agent, fire_positions, map_diagonal)
    vip_cost = _cached_vip_cone_cost(agent)[selected_indices]
    topography_cost = _normalize_priority_slice(
        agent,
        _cached_raw_topography_priority(agent)[selected_indices],
    )
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * vip_cost
        + agent.parameters.topography_cost_weight * topography_cost
    )
    return _destination_from_costs(fire_positions, selection_cost)


class WaterSelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "water_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        destination = _select_water_destination(agent)
        if destination is None:
            return TaskStatus.FAILED

        agent.set_destination(destination, DestinationType.FIRE)
        return TaskStatus.COMPLETE


class VIPSelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "vip_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        destination = _select_vip_destination(agent)
        if destination is None:
            return TaskStatus.FAILED

        agent.set_destination(destination, DestinationType.FIRE)
        return TaskStatus.COMPLETE


class VegetationSelectPOI(SelectPOITask):
    """Class that adds a the direct attack for prioritizing the
    vegetation type.
    """

    def __init__(self):
        self.task_method.__func__.__name__ = "vegetation_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and
        suppress.
        """
        destination = _select_vegetation_destination(agent)
        if destination is None:
            return TaskStatus.FAILED

        agent.set_destination(destination, DestinationType.FIRE)
        return TaskStatus.COMPLETE


class TopographySelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "topography_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        destination = _select_topography_destination(agent)
        if destination is None:
            return TaskStatus.FAILED

        agent.set_destination(destination, DestinationType.FIRE)
        return TaskStatus.COMPLETE


class IndirectSelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "indirect_select_poi"
        self.complete_method = super().complete_method
        self.fail_method = super().fail_method

    def task_method(self, agent):
        """Selecting a fire block position (point of interest) to track
        and suppress.
        """
        fire_block_indices = agent.model.fire_block_indices

        if agent.model.fire_encircled:
            return TaskStatus.FAILED

        if len(fire_block_indices) == 0:
            return TaskStatus.FAILED

        # Exclude water / non-combustible areas from the indices
        nonflammable_idx_cnt = 0
        closed = False
        crt_block_index = agent.model.current_block_index
        close_index = agent.model.closing_index
        while agent.model.wildfire.fire_states[
            tuple(fire_block_indices[crt_block_index, :])
        ] in [NONFLAMMABLE, SUPPRESSED, BURNT]:
            nonflammable_idx_cnt += 1
            crt_block_index += 1
            crt_block_index %= len(fire_block_indices)
            if crt_block_index == close_index:
                closed = True
            # Avoid infinite loops. This might happen in exceptional
            # cases where the fire is surrounded all around by
            # nonflammable area.
            if nonflammable_idx_cnt >= len(fire_block_indices):
                agent.model.fire_encircled = True
                return TaskStatus.FAILED

        patch_dimensions = agent.suppression_patch(
            agent.payload, agent.suppressant_flow_rate
        )
        # Try to increment crt_block based on suppression patch size
        # (maximize the non-suppressed area), -1 is there because it
        # maintains 1 index being attached to block index (previous
        # block is suppressed))
        max_patch_offset = crt_block_index + int(min(patch_dimensions) / 2) - 1
        # Ensure that no gaps occur in fire block
        for idx in range(crt_block_index, max_patch_offset):
            idx %= len(fire_block_indices)
            if idx == close_index:
                closed = True
            if agent.model.wildfire.fire_states[
                tuple(fire_block_indices[idx, :])
            ] in [NONFLAMMABLE, SUPPRESSED, BURNT]:
                break
            crt_block_index = idx

        fire_block_position = index_to_pos(
            fire_block_indices[crt_block_index, :],
            grid_description=agent.terrain.grid_description,
            origin=agent.fire_terrain_origin,
        )

        # Increment fire block index for the next agent and contain its
        # value to avoid index out of bounds
        agent.model.current_block_index = crt_block_index + 1
        agent.model.current_block_index %= len(fire_block_indices)

        if closed:
            agent.model.current_block_index = close_index

        agent.set_destination(fire_block_position, DestinationType.FIRE)

        return TaskStatus.COMPLETE


SELECT_POI_TABLE = {
    SelectPOIType.WATER: WaterSelectPOI,
    SelectPOIType.VIP: VIPSelectPOI,
    SelectPOIType.TOPOGRAPHY: TopographySelectPOI,
    SelectPOIType.VEGETATION: VegetationSelectPOI,
    SelectPOIType.INDIRECT: IndirectSelectPOI,
}
