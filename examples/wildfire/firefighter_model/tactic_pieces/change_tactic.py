from enum import Enum
from functools import cached_property

from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SELECT_POI_TABLE,
    SelectPOIType,
)
from examples.wildfire.firefighter_model.tactic_pieces.suppress import (
    SUPPRESS_TABLE,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    TRACK_POI_TABLE,
)
from sosid.model.abm.task import Task, TaskStatus
from sosid.util.abc import ABC


class CurrentTactic(Enum):
    MAIN = "main"
    ALTERNATIVE = "alternative"


class ChangeType(Enum):
    NO_CHANGE = "no_change"
    RUNTIME = "runtime"
    DAYTIME = "daytime"
    RESIDENTIAL = "residential"
    BURNT = "burnt_area"
    DISTANCE = "distance"


class ChangeTask(ABC, Task):
    def __init__(self, threshold: int, main, alternative):
        self.task_method.__func__.__name__ = "change_tactic"
        self.threshold = threshold
        self.main = main
        self.alternative = alternative
        self.current_tactic = CurrentTactic.MAIN

    def change_tactic(self, agent, tactic=CurrentTactic.ALTERNATIVE):
        tactic = self.alternative
        if tactic == CurrentTactic.MAIN:
            tactic = self.main

        self.current_tactic = tactic

        agent.tactic.select_poi = SELECT_POI_TABLE[tactic.select_poi]()
        agent.tactic.track_poi = TRACK_POI_TABLE[tactic.track_poi]()
        agent.tactic.suppress = SUPPRESS_TABLE[tactic.suppress]()

    def complete_method(self, agent):
        """Set tracking for selected firefront."""
        agent.tasks.set_active(agent.tactic.select_suppressant_source)


class NoChange(ChangeTask):
    def __init__(self, threshold: int, main, alternative):
        self.task_method.__func__.__name__ = "no_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
        return TaskStatus.COMPLETE


class RuntimeChange(ChangeTask):
    def __init__(
        self, threshold: int, main: SelectPOIType, alternative: SelectPOIType
    ):
        self.task_method.__func__.__name__ = "runtime_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
            return TaskStatus.COMPLETE

        runtime = agent.model.simulation.timer.mission_runtime

        # Convert threshold to seconds as runtime in seconds
        threshold_met = runtime.seconds >= self.threshold * 3600
        match threshold_met, self.current_tactic:
            case True, CurrentTactic.MAIN:
                self.change_tactic(agent)

        return TaskStatus.COMPLETE


class DaytimeChange(ChangeTask):
    def __init__(
        self, threshold: int, main: SelectPOIType, alternative: SelectPOIType
    ):
        self.task_method.__func__.__name__ = "daytime_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
            return TaskStatus.COMPLETE

        hour = agent.current_mission_time.hour

        is_day = self.daytime[0] <= hour < self.daytime[1]
        match is_day, self.current_tactic:
            case True, self.night_tactic:
                self.change_tactic(agent)
            case False, self.day_tactic:
                self.change_tactic(agent)
        return TaskStatus.COMPLETE

    @cached_property
    def day_tactic(self) -> CurrentTactic:
        """Tactic to use during the day."""
        return (
            CurrentTactic.ALTERNATIVE
            if self.threshold[0] < self.threshold[1]
            else CurrentTactic.MAIN
        )

    @cached_property
    def night_tactic(self) -> CurrentTactic:
        """Tactic to use during the night."""
        return (
            CurrentTactic.MAIN
            if self.threshold[0] < self.threshold[1]
            else CurrentTactic.ALTERNATIVE
        )

    @cached_property
    def daytime(self) -> tuple[int, int]:
        """Daytime threshold."""
        return tuple(sorted(self.threshold))


class ResidentialChange(ChangeTask):
    def __init__(self, threshold: int, main, alternative):
        self.task_method.__func__.__name__ = "resident_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
            return TaskStatus.COMPLETE
        res_burnt = agent.model.min_distance_fire_urban
        # convert numpy bool to bool
        threshold_met = bool(res_burnt <= self.threshold)
        match threshold_met, self.current_tactic:
            case True, CurrentTactic.MAIN:
                self.change_tactic(agent)
            case False, CurrentTactic.ALTERNATIVE:
                self.change_tactic(agent)

        return TaskStatus.COMPLETE


class BurntChange(ChangeTask):
    def __init__(self, threshold: int, main, alternative):
        self.task_method.__func__.__name__ = "burnt_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
            return TaskStatus.COMPLETE
        burnt = agent.model.simulation.burnt_area

        threshold_met = burnt >= self.threshold
        match threshold_met, self.current_tactic:
            case True, CurrentTactic.MAIN:
                self.change_tactic(agent)

        return TaskStatus.COMPLETE


class DistanceChange(ChangeTask):
    def __init__(self, threshold: int, main, alternative):
        self.task_method.__func__.__name__ = "distance_change"
        super().__init__(threshold, main, alternative)

    def task_method(self, agent):
        if agent.force_tactic_swap:
            self.change_tactic(agent)
            agent.force_tactic_swap = False
            return TaskStatus.COMPLETE
        firefront_distance_to_wildfire = (
            agent.model.min_distance_fire_fireblock
        )

        # convert numpy bool to bool
        threshold_met = bool(firefront_distance_to_wildfire <= self.threshold)
        match threshold_met, self.current_tactic:
            case True, CurrentTactic.MAIN:
                self.change_tactic(agent)
            case False, CurrentTactic.ALTERNATIVE:
                self.change_tactic(agent)

        return TaskStatus.COMPLETE


CHANGE_TABLE = {
    ChangeType.NO_CHANGE: NoChange,
    ChangeType.RUNTIME: RuntimeChange,
    ChangeType.DAYTIME: DaytimeChange,
    ChangeType.RESIDENTIAL: ResidentialChange,
    ChangeType.BURNT: BurntChange,
    ChangeType.DISTANCE: DistanceChange,
}
