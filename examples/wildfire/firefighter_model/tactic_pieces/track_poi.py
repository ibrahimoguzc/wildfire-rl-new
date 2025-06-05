from enum import Enum

import numpy as np

from examples.wildfire.fire_model.states import BURNT, SUPPRESSED
from examples.wildfire.firefighter_model.follower import DestinationType
from sosid.model.abm.task import Task, TaskStatus
from sosid.model.transform import index_to_pos, pos_to_index
from sosid.util.abc import ABC


class TrackPOIType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    FOLLOW_FIREFRONT = "follow_firefront"


class TrackPOITask(ABC, Task):
    def complete_method(self, agent):
        """Directs agent to select and track a new fire-front."""
        agent.tasks.set_active(agent.tactic.suppress)

    def fail_method(self, agent):
        """Directs agent to select and track a new fire-front."""
        agent.tasks.set_active(agent.tactic.select_poi)


class DirectTrackPOI(TrackPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "direct_track_poi"

    def task_method(self, agent):
        return agent.follower.navigate()


class IndirectTrackPOI(TrackPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "indirect_track_poi"

    def task_method(self, agent):
        """Tracking of the selected fire block position (point of
        interest).
        """
        # Check if fire block index has already been suppressed by other
        # agents
        destination_idx = pos_to_index(
            agent.destination,
            agent.terrain.grid_description,
            agent.fire_terrain_origin,
        )
        destination_state = agent.model.wildfire.retrieve_fire_states(
            destination_idx
        )
        if destination_state in [SUPPRESSED, BURNT]:
            return TaskStatus.FAILED
        return agent.follower.navigate()


class FollowFirefrontTrackPOI(TrackPOITask):
    def __init__(self):
        self.task_method.__func__.__name__ = "follow_firefront_track_poi"

    def task_method(self, agent):
        """Tracking of the selected firefront (point of interest).

        Monitoring the firefront propagation of and navigating towards
        the point with the highest propagation rate, following the point
        alopng the firefront.
        """
        i, j = pos_to_index(
            agent.destination,
            grid_description=agent.terrain.grid_description,
            origin=agent.fire_terrain_origin,
        )
        width = agent.terrain.width_in_cells
        height = agent.terrain.height_in_cells

        # Getting a 5x5 Moore Neighborhood of the spread_rates
        moore_range = np.array((-2, -1, 0, 1, 2))
        i_range, j_range = moore_range + i, moore_range + j
        i_neighborhood, j_neighborhood = np.meshgrid(
            i_range[(i_range >= 0) & (i_range < height)],
            j_range[(j_range >= 0) & (j_range < width)],
            copy=False,
            sparse=True,
            indexing="ij",
        )

        rate_neighborhood = agent.model.wildfire.get_spread_rates()[
            i_neighborhood, j_neighborhood
        ]

        # Checks if there is zero spread rate in Moore neighbourhood
        if np.amax(rate_neighborhood) == 0:
            return TaskStatus.FAILED

        # Tracking fire-front by selecting highest spread-rate index
        max_idx = np.unravel_index(
            np.argmax(rate_neighborhood), rate_neighborhood.shape
        ) + np.array((i - 2, j - 2))

        destination = np.array(
            index_to_pos(
                max_idx,
                grid_description=agent.terrain.grid_description,
                origin=agent.fire_terrain_origin,
            )
        )

        # Converting fire-front index to position and navigating to it
        agent.set_destination(destination, DestinationType.FIRE)

        return agent.follower.navigate()


TRACK_POI_TABLE = {
    TrackPOIType.DIRECT: DirectTrackPOI,
    TrackPOIType.INDIRECT: IndirectTrackPOI,
    TrackPOIType.FOLLOW_FIREFRONT: FollowFirefrontTrackPOI,
}
