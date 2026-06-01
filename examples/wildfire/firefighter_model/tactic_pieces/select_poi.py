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


class WaterSelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "water_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        fire_positions = agent.model.wildfire.fire_positions
        if not fire_positions.size:
            return TaskStatus.FAILED

        # Selecting only firefronts not tracked by other agents
        for obj in agent.model.firefighters:
            if obj.destination is not None:
                untracked_pos = (fire_positions != obj.destination).any(axis=1)
                fire_positions = fire_positions[untracked_pos]

                # Choose any firefront if all are already taken
                if not np.size(fire_positions):
                    fire_positions = agent.model.wildfire.fire_positions

        # Computing the map diagonal to normalise the cost factors
        map_shape = np.array(agent.model.simulation.environment.dimensions)
        map_diagonal = np.linalg.norm(map_shape)

        # Distance cost factor
        fire_distances = agent.distance(agent.pos, fire_positions)
        distance_cost = (map_diagonal - fire_distances) / map_diagonal

        # Water objective cost factor. This makes the water tactic select
        # burning cells that are threatening water resources.
        water_cost = _nearest_position_cost(
            fire_positions=fire_positions,
            objective_positions=_positions_from_agents(
                agent.model.water_sources
            ),
            map_diagonal=map_diagonal,
        )

        # Fire-front selection based on total cost function
        selection_cost = (
            agent.parameters.distance_cost_weight * distance_cost
            + agent.parameters.vip_cost_weight * water_cost
        )
        min_idx = np.argmax(selection_cost)
        agent.set_destination(fire_positions[min_idx, :], DestinationType.FIRE)
        return TaskStatus.COMPLETE


class VIPSelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "vip_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        fire_positions = agent.model.wildfire.fire_positions
        if not fire_positions.size:
            return TaskStatus.FAILED

        # Selecting only firefronts not tracked by other agents
        for obj in agent.model.firefighters:
            if obj.destination is not None:
                untracked_pos = (fire_positions != obj.destination).any(axis=1)
                fire_positions = fire_positions[untracked_pos]

                # Choose any firefront if all are already taken
                if not np.size(fire_positions):
                    fire_positions = agent.model.wildfire.fire_positions

        # Computing the map diagonal to normalise the cost factors
        map_shape = np.array(agent.model.simulation.environment.dimensions)
        map_diagonal = np.linalg.norm(map_shape)

        # Distance cost factor
        fire_distances = agent.distance(agent.pos, fire_positions)
        distance_cost = (map_diagonal - fire_distances) / map_diagonal

        # Urban objective cost factor. Protection locations are the point
        # objectives used by the VIP tactic.
        urban_cost = _nearest_position_cost(
            fire_positions=fire_positions,
            objective_positions=_positions_from_agents(
                agent.model.protection_locations
            ),
            map_diagonal=map_diagonal,
        )

        # Fire-front selection based on total cost function
        selection_cost = (
            agent.parameters.distance_cost_weight * distance_cost
            + agent.parameters.vip_cost_weight * urban_cost
        )
        min_idx = np.argmax(selection_cost)
        agent.set_destination(fire_positions[min_idx, :], DestinationType.FIRE)
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
        fire_positions = agent.model.wildfire.fire_positions
        burning_indices = agent.model.wildfire.burning_indices
        if not fire_positions.size:
            return TaskStatus.FAILED

        # Selecting only firefronts not tracked by other agents
        for obj in agent.model.firefighters:
            if obj.destination is not None:
                untracked_pos = (fire_positions != obj.destination).any(axis=1)
                fire_positions = fire_positions[untracked_pos]
                burning_indices = burning_indices[untracked_pos]

                # Choose any firefront if all are already taken
                if not np.size(fire_positions):
                    fire_positions = agent.model.wildfire.fire_positions
                    burning_indices = agent.model.wildfire.burning_indices

        # Computing the map diagonal to normalise the cost factors
        map_shape = np.array(agent.model.simulation.environment.dimensions)
        map_diagonal = np.linalg.norm(map_shape)

        # Distance cost factor
        fire_distances = agent.distance(agent.pos, fire_positions)
        distance_cost = (map_diagonal - fire_distances) / map_diagonal

        # Very important points (vip) protection cost factor
        vip_cost = agent.exponential_cone_func(
            pos=fire_positions,
            vip=(
                location.pos for location in agent.model.protection_locations
            ),
            map_diagonal=map_diagonal,
        )

        # Vegetation cost
        vegetation_cost = agent.priority_cost_vegetation(burning_indices)

        # Fire-front selection based on total cost function
        selection_cost = (
            agent.parameters.distance_cost_weight * distance_cost
            + agent.parameters.vip_cost_weight * vip_cost
            + agent.parameters.vegetation_cost_weight * vegetation_cost
        )
        min_idx = np.argmax(selection_cost)
        agent.set_destination(fire_positions[min_idx, :], DestinationType.FIRE)
        return TaskStatus.COMPLETE


class TopographySelectPOI(SelectPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "topography_select_poi"

    def task_method(self, agent):
        """Selecting a firefront (point of interest) to track and suppress."""
        fire_positions = agent.model.wildfire.fire_positions
        burning_indices = agent.model.wildfire.burning_indices

        if not fire_positions.size:
            return TaskStatus.FAILED

        # Selecting only firefronts not tracked by other agents
        for obj in agent.model.firefighters:
            if obj.destination is not None:
                untracked_pos = (fire_positions != obj.destination).any(axis=1)
                fire_positions = fire_positions[untracked_pos]
                burning_indices = burning_indices[untracked_pos]

                # Choose any firefront if all are already taken
                if not np.size(fire_positions):
                    fire_positions = agent.model.wildfire.fire_positions
                    burning_indices = agent.model.wildfire.burning_indices

        # Computing the map diagonal to normalise the cost factors
        map_shape = np.array(agent.model.simulation.environment.dimensions)
        map_diagonal = np.linalg.norm(map_shape)

        # Distance cost factor
        fire_distances = agent.distance(agent.pos, fire_positions)
        distance_cost = (map_diagonal - fire_distances) / map_diagonal

        # Very important points (vip) protection cost factor
        vip_cost = agent.exponential_cone_func(
            pos=fire_positions,
            vip=(
                location.pos for location in agent.model.protection_locations
            ),
            map_diagonal=map_diagonal,
        )

        # Topography cost
        topography_cost = agent.priority_cost_topography(burning_indices)

        # Fire-front selection based on total cost function
        selection_cost = (
            agent.parameters.distance_cost_weight * distance_cost
            + agent.parameters.vip_cost_weight * vip_cost
            + agent.parameters.topography_cost_weight * topography_cost
        )
        min_idx = np.argmax(selection_cost)
        agent.set_destination(fire_positions[min_idx, :], DestinationType.FIRE)
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
