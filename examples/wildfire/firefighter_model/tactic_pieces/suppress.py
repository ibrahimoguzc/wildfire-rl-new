import math
from enum import Enum

import numpy as np

from examples.wildfire.firefighter_model.follower import (
    DestinationType,
    PayloadStatus,
)
from sosid.model.abm.task import Task, TaskStatus
from sosid.model.ca.raster import Ellipse
from sosid.model.transform import pos_to_index
from sosid.util.abc import ABC


class SuppressType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class SuppressTask(ABC, Task):
    def complete_method(self, agent):
        agent.total_suppressions += 1
        agent.payload_status = PayloadStatus.NONE
        if not agent.operational_clearance:
            # Return to base and hold
            airport, _ = agent.nearest_airport
            agent.set_destination(airport.pos, DestinationType.BASE)
            agent.tasks.set_active(agent.tactic.return_to_base)
        else:
            agent.tasks.set_active(agent.tactic.change)


class DirectSuppress(SuppressTask):
    def __init__(self):
        self.task_method.__func__.__name__ = "direct_suppress"

    def task_method(self, agent):
        """Contains the logic for how to suppress a fire-front.

        Once a fire-front is suppressed, other agents are instructed
        to check if their fire-front is still active.
        """
        fire_idx = pos_to_index(
            agent.destination,
            grid_description=agent.terrain.grid_description,
            origin=agent.fire_terrain_origin,
        )
        patch_dimensions = agent.suppression_patch(
            agent.payload, agent.suppressant_flow_rate
        )

        # Runs when both dimensions of the suppression shape are nonzero
        if 0 not in patch_dimensions:
            suppression_area = Ellipse(
                idx=(0, 0),
                major=patch_dimensions[0],
                minor=patch_dimensions[1],
            )
            suppression_area.idx = fire_idx
            aspect = agent.model.wildfire.prop_aspect[fire_idx] + 90

            # Runs when no measurable firefront propagation is observed
            if math.isnan(aspect):
                aspect = agent.max_suppression_aspect(
                    suppression_area, agent.model.wildfire.burning_indices
                )

            suppression_area.aspect = aspect % 360

            agent.model.wildfire.suppress(suppression_area)

        return TaskStatus.COMPLETE


class IndirectSuppress(SuppressTask):
    def __init__(self):
        self.task_method.__func__.__name__ = "indirect_suppress"

    def task_method(self, agent):
        """Contains the logic for how to suppress a fire-front.

        Once a fire-front is suppressed, other agents are instructed
        to check if their fire-front is still active.
        """
        fire_idx = pos_to_index(
            agent.destination,
            grid_description=agent.terrain.grid_description,
            origin=agent.fire_terrain_origin,
        )
        patch_dimensions = agent.suppression_patch(
            agent.payload, agent.suppressant_flow_rate
        )
        # Runs when both dimensions of the suppression shape are nonzero
        dropped = False
        if 0 not in patch_dimensions:
            suppression_area = Ellipse(
                idx=(0, 0),
                major=patch_dimensions[0],
                minor=patch_dimensions[1],
            )
            suppression_area.idx = fire_idx
            aspect = agent.indirect_suppression_aspect(
                suppression_area, agent.model.fire_block_indices
            )
            if not math.isnan(aspect):
                dropped = True
                suppression_area.aspect = aspect % 360

                agent.model.wildfire.suppress(suppression_area)

        if dropped:
            return TaskStatus.COMPLETE

        return TaskStatus.FAILED

    def fail_method(self, agent):
        """Reinstances poi selection based on incomplete fire line.

        When the suppression patch does not connect the fire points
        adequately this is triggered.
        """
        failed_index = pos_to_index(
            agent.destination,
            grid_description=agent.terrain.grid_description,
            origin=agent.fire_terrain_origin,
        )

        # check if failed_position in fire_block
        failed_idx = np.where(
            (agent.model.fire_block_indices == failed_index).all(axis=1)
        )[0]

        # add it to fire_block
        if not failed_idx:
            failed_idx = agent.model.current_block_index
            # add secondary index to buffer failed suppression
            buffer_index = np.array(
                [
                    [failed_index[0] + 1, failed_index[1] + 1],
                    failed_index,
                    [failed_index[0] - 1, failed_index[1] - 1],
                ]
            )
            agent.model.fire_block_indices = np.insert(
                agent.model.fire_block_indices,
                failed_idx,
                buffer_index,
                axis=0,
            )

        agent.tasks.set_active(agent.tactic.select_poi)


SUPPRESS_TABLE = {
    SuppressType.DIRECT: DirectSuppress,
    SuppressType.INDIRECT: IndirectSuppress,
}
