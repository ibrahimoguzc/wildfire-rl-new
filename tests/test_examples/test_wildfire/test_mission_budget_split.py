"""Tests for the --missions training-budget split.

A multi-mission run divides one budget between several maps and trains them as
phases in the order given: the whole fleet finishes the first mission's share
before any worker moves to the next. The parts that decide that are:

* ``_split_budget`` -- apportions the budget so the shares sum back to it,
* ``WildfireHourlyEnv._select_mission_index`` -- picks the first mission whose
  share is not spent yet, counting the simulations the fleet has already started
  plus the ones this worker has begun since that figure was current, and
* ``StopTrainingOnMissionSimulations`` -- ends the run once every mission has
  completed its share, and reports each phase as it finishes.

The env-level tests use a bare object rather than a constructed environment so
they do not have to load terrain.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.wildfire.ppo_runnerv2 import (
    MISSION_BUDGET_SIMULATIONS,
    MISSION_BUDGET_TIMESTEPS,
    StopTrainingOnMissionSimulations,
    WildfireHourlyEnv,
    _episode_counts_by_scenario,
    _split_budget,
)


def _phase_env(
    budgets: dict[str, float],
    *,
    mode: str = MISSION_BUDGET_SIMULATIONS,
    fleet_starts: dict[str, int] | None = None,
    own_starts: dict[str, int] | None = None,
    global_ts: dict[str, int] | None = None,
) -> WildfireHourlyEnv:
    """Minimal stand-in exposing only what _select_mission_index reads."""
    env = object.__new__(WildfireHourlyEnv)
    env._scenario_templates = [(Path(name), None) for name in budgets]
    env._mission_budgets = dict(budgets)
    env._mission_budget_mode = mode
    env._fleet_scenario_ep_starts = dict(fleet_starts or {})
    env._scenario_ep_starts = dict(own_starts or {})
    env._scenario_ep_starts_at_sync = {}
    env._global_scenario_ts_counts = dict(global_ts or {})
    env._pending_mission_start = None
    return env


def _mission_of(env: WildfireHourlyEnv, index: int) -> str:
    return env._scenario_templates[index][0].name


class TestSplitBudget:
    def test_equal_two_way_split(self):
        assert _split_budget(100_000, (1.0, 1.0)) == [50_000, 50_000]

    def test_shares_sum_to_total_when_not_divisible(self):
        shares = _split_budget(100_000, (1.0, 1.0, 1.0))
        assert sum(shares) == 100_000
        assert shares == [33_334, 33_333, 33_333]

    def test_weights_are_relative(self):
        assert _split_budget(300, (2.0, 1.0)) == [200, 100]
        assert _split_budget(300, (4.0, 2.0)) == [200, 100]

    def test_rejects_non_positive_weights(self):
        with pytest.raises(ValueError):
            _split_budget(100, (1.0, 0.0))
        with pytest.raises(ValueError):
            _split_budget(100, ())


class TestEpisodeCountsByScenario:
    def test_counts_only_listed_missions(self):
        rows = [
            {"scenario_name": "Palisades6sp6ev.json"},
            {"scenario_name": "Pyrenees6sp6ev.json"},
            {"scenario_name": "Palisades6sp6ev.json"},
            {"scenario_name": "Salamis6sp6ev.json"},
            {},
        ]
        counts = _episode_counts_by_scenario(
            rows, ("Palisades6sp6ev.json", "Pyrenees6sp6ev.json")
        )
        assert counts == {"Palisades6sp6ev.json": 2, "Pyrenees6sp6ev.json": 1}


class TestSelectMissionIndex:
    def test_first_mission_runs_first(self):
        env = _phase_env({"A.json": 100.0, "B.json": 100.0})
        assert _mission_of(env, env._select_mission_index()) == "A.json"

    def test_stays_on_the_first_mission_until_its_share_is_spent(self):
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            fleet_starts={"A.json": 99},
        )
        assert _mission_of(env, env._select_mission_index()) == "A.json"

    def test_switches_once_the_share_is_spent(self):
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            fleet_starts={"A.json": 100},
        )
        assert _mission_of(env, env._select_mission_index()) == "B.json"

    def test_own_starts_since_the_sync_count_against_the_share(self):
        # The fleet figure is a step stale; without this worker's own starts it
        # would keep the phase going past the share.
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            fleet_starts={"A.json": 90},
            own_starts={"A.json": 10},
        )
        assert _mission_of(env, env._select_mission_index()) == "B.json"

    def test_sync_rebases_own_starts_instead_of_double_counting(self):
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            own_starts={"A.json": 10},
        )
        env.set_fleet_scenario_episode_starts({"A.json": 10})
        assert env.mission_simulations_started("A.json") == 10
        assert _mission_of(env, env._select_mission_index()) == "A.json"

    def test_three_missions_run_in_list_order(self):
        budgets = {"A.json": 10.0, "B.json": 10.0, "C.json": 10.0}
        env = _phase_env(budgets, fleet_starts={"A.json": 10})
        assert _mission_of(env, env._select_mission_index()) == "B.json"
        env = _phase_env(budgets, fleet_starts={"A.json": 10, "B.json": 10})
        assert _mission_of(env, env._select_mission_index()) == "C.json"

    def test_last_mission_keeps_running_once_every_share_is_spent(self):
        # Training that outlives its budget (wall-clock chunk, longer chain)
        # stays in the final phase rather than restarting the sequence.
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            fleet_starts={"A.json": 100, "B.json": 100},
        )
        assert _mission_of(env, env._select_mission_index()) == "B.json"

    def test_timestep_phases_switch_on_the_timestep_counts(self):
        budgets = {"A.json": 1000.0, "B.json": 1000.0}
        env = _phase_env(
            budgets,
            mode=MISSION_BUDGET_TIMESTEPS,
            global_ts={"A.json": 999},
            own_starts={"A.json": 500},  # starts are irrelevant here
        )
        assert _mission_of(env, env._select_mission_index()) == "A.json"
        env = _phase_env(
            budgets,
            mode=MISSION_BUDGET_TIMESTEPS,
            global_ts={"A.json": 1000},
        )
        assert _mission_of(env, env._select_mission_index()) == "B.json"

    def test_resumed_counts_place_a_chained_chunk_in_the_right_phase(self):
        env = _phase_env(
            {"A.json": 100.0, "B.json": 100.0},
            fleet_starts={"A.json": 100, "B.json": 40},
        )
        assert _mission_of(env, env._select_mission_index()) == "B.json"


class TestPendingMissionStart:
    """A reset whose episode never runs must not consume any of a share.

    Two such resets happen per worker before training: the factory seeds the env
    with ``reset(seed=...)`` and SB3 resets the vector env again before the first
    rollout. Counting those (rather than the episodes that actually run) spent
    2x the fleet size out of the first phase's share.
    """

    def test_start_is_counted_at_the_first_step_not_at_reset(self):
        env = _phase_env({"A.json": 5.0, "B.json": 5.0})
        env._pending_mission_start = "A.json"
        assert env.mission_simulations_started("A.json") == 0
        env._count_pending_mission_start()
        assert env.mission_simulations_started("A.json") == 1

    def test_stepping_again_does_not_recount_the_same_episode(self):
        env = _phase_env({"A.json": 5.0, "B.json": 5.0})
        env._pending_mission_start = "A.json"
        env._count_pending_mission_start()
        env._count_pending_mission_start()
        assert env.mission_simulations_started("A.json") == 1

    def test_a_discarded_reset_costs_nothing(self):
        env = _phase_env({"A.json": 5.0, "B.json": 5.0})
        env._pending_mission_start = "A.json"  # factory reset, thrown away
        env._pending_mission_start = "A.json"  # SB3's reset, the one that runs
        env._count_pending_mission_start()
        assert env.mission_simulations_started("A.json") == 1


class TestFleetPhases:
    """The phase boundary has to hold across workers, not just within one.

    Replays the loop the training run performs: workers reset independently,
    and after each step the fleet-wide start counts are collected and handed
    back to every worker. ``resets_per_step`` workers reset in the same step and
    therefore decide from the same figure, which is the one way a phase can
    overrun its share.
    """

    @staticmethod
    def _run(quotas, workers, resets_per_step, seed=0):
        rng = np.random.default_rng(seed)
        envs = [
            _phase_env({name: float(quota) for name, quota in quotas.items()})
            for _ in range(workers)
        ]
        order: list[str] = []
        while len(order) < sum(quotas.values()):
            resetting = rng.choice(workers, size=resets_per_step, replace=False)
            for worker_idx in resetting:
                env = envs[worker_idx]
                picked = _mission_of(env, env._select_mission_index())
                env._pending_mission_start = picked  # what reset() records
                env._count_pending_mission_start()  # what the first step does
                order.append(picked)
            totals: dict[str, int] = {}
            for env in envs:
                for name, value in env.get_scenario_episode_starts().items():
                    totals[name] = totals.get(name, 0) + value
            for env in envs:
                env.set_fleet_scenario_episode_starts(totals)
        return order

    def test_phases_do_not_interleave(self):
        quotas = {"first.json": 200, "second.json": 200}
        order = self._run(quotas, workers=8, resets_per_step=4)
        switch = order.index("second.json")
        # Everything before the switch is the first mission, and nothing after
        # it goes back -- bar the workers that reset in the same step as the
        # switch and so decided from the same figure.
        assert set(order[:switch]) == {"first.json"}
        assert order[switch:].count("first.json") <= 4

    def test_shares_are_hit_within_one_step_of_concurrent_resets(self):
        quotas = {"first.json": 200, "second.json": 200}
        order = self._run(quotas, workers=8, resets_per_step=4)
        assert abs(order.count("first.json") - 200) <= 4
        assert abs(order.count("second.json") - 200) <= 4

    def test_three_phases_run_in_order(self):
        quotas = {"a.json": 100, "b.json": 100, "c.json": 100}
        order = self._run(quotas, workers=16, resets_per_step=8, seed=3)
        first_index = {name: order.index(name) for name in quotas}
        assert first_index["a.json"] < first_index["b.json"] < first_index["c.json"]
        for name, quota in quotas.items():
            assert abs(order.count(name) - quota) <= 8

    def test_uneven_weights_are_respected(self):
        quotas = {"a.json": 320, "b.json": 80}
        order = self._run(quotas, workers=4, resets_per_step=2)
        assert abs(order.count("a.json") - 320) <= 2
        assert abs(order.count("b.json") - 80) <= 2


class TestStopTrainingOnMissionSimulations:
    @staticmethod
    def _callback(quotas, completed=None):
        callback = StopTrainingOnMissionSimulations(quotas, completed=completed)
        callback.locals = {"infos": []}
        # num_timesteps is only read for the phase/stop messages.
        callback.model = SimpleNamespace(num_timesteps=0)
        return callback

    @staticmethod
    def _finish(callback, scenario_name, times=1):
        callback.locals["infos"] = [
            {"episode_summary": {}, "scenario_name": scenario_name}
        ] * times
        return callback._on_step()

    def test_keeps_going_until_every_mission_is_done(self):
        callback = self._callback({"A.json": 2, "B.json": 2})
        assert self._finish(callback, "A.json", times=2) is True
        # A is done, B has not started: the fleet-wide count is 2 of 4.
        assert self._finish(callback, "B.json") is True
        assert self._finish(callback, "B.json") is False

    def test_announces_each_phase_as_it_completes(self, capsys):
        callback = self._callback({"A.json": 2, "B.json": 2})
        self._finish(callback, "A.json", times=2)
        assert "Mission phase complete: A.json" in capsys.readouterr().out

    def test_resumed_counts_carry_the_chain_forward(self):
        callback = self._callback(
            {"A.json": 10, "B.json": 10},
            completed={"A.json": 10, "B.json": 9},
        )
        assert self._finish(callback, "B.json") is False

    def test_ignores_scenarios_outside_the_mission_list(self):
        callback = self._callback({"A.json": 1, "B.json": 1})
        assert self._finish(callback, "C.json") is True
        assert self._finish(callback, "A.json") is True
        assert self._finish(callback, "B.json") is False
