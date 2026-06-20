#!/usr/bin/env python3
"""Profile firefighter ABM work during PPO decision windows.

This script instruments methods at runtime and leaves the training code
unchanged. It is meant for short screening runs that answer where ABM time is
going: scheduler breeds, active aircraft tasks, trajectory/propellant checks,
suppression geometry, and fire-block maintenance.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

from examples.wildfire.fire_model.model import CPUFireModel  # noqa: E402
from examples.wildfire.firefighter_model.agents import (  # noqa: E402
    AirTrafficManager,
    IgnitionCenter,
    ProtectionLocation,
    SuppressionUAV,
    WaterSourceManager,
)
import examples.wildfire.firefighter_model.agents as agents_module  # noqa: E402
from examples.wildfire.firefighter_model.model import FirefighterModel  # noqa: E402
import examples.wildfire.firefighter_model.tactic_pieces.select_poi as select_poi_module  # noqa: E402,E501
from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    AIRCRAFT_GROUP_SIZE,
    CONTROLLED_AGENT_COUNT,
    DEFAULT_DECISION_INTERVAL_MINUTES,
    DEFAULT_STATE_FIRE_FRONTS,
    STATE_SPACE_OLD,
    SUPPORTED_TACTIC_DISTRIBUTIONS,
    TACTIC_DISTRIBUTION_INDIVIDUAL,
    WildfireHourlyEnv,
    _resolve_scenario,
)
from sosid.model.abm.schedule import RandomActivationByBreed  # noqa: E402
from sosid.model.abm.task import Task, TaskStatus  # noqa: E402


STATIC_BREED_NAMES = {
    AirTrafficManager.__name__,
    WaterSourceManager.__name__,
    IgnitionCenter.__name__,
    ProtectionLocation.__name__,
}
SELECT_POI_TASK_NAMES = {
    "water_select_poi",
    "vip_select_poi",
    "vegetation_select_poi",
    "topography_select_poi",
    "indirect_select_poi",
}
SUPPRESS_TASK_NAMES = {
    "direct_suppress",
    "indirect_suppress",
}
HEAVY_KEYS = (
    "generate_trajectory",
    "generate_straight_trajectory",
    "set_destination",
    "estimate_propellant_for_journey",
    "max_suppression_aspect",
    "indirect_suppression_aspect",
    "create_fire_block_ellipse",
    "update_fire_block_ellipse",
    "is_fire_containable",
    "verify_block_indices",
)


class TimingStore:
    """Collect cumulative seconds and call counts by metric key."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def clear(self) -> None:
        self.seconds.clear()
        self.counts.clear()

    def add(self, key: str, seconds: float) -> None:
        self.seconds[key] += seconds
        self.counts[key] += 1

    def merge_from(self, other: "TimingStore") -> None:
        for key, value in other.seconds.items():
            self.seconds[key] += value
        for key, value in other.counts.items():
            self.counts[key] += value


class ABMProfiler:
    """Runtime monkeypatch profiler for ABM hot spots."""

    def __init__(self) -> None:
        self.store = TimingStore()
        self._patches: list[tuple[Any, str, Any]] = []

    def clear(self) -> None:
        self.store.clear()

    def add(self, key: str, seconds: float) -> None:
        self.store.add(key, seconds)

    def patch_attr(self, owner: Any, name: str, value: Any) -> None:
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def _wrap_method(self, cls: type, name: str, key: str) -> None:
        original = getattr(cls, name)
        profiler = self

        def wrapped(instance, *args, **kwargs):
            start = perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                profiler.add(key, perf_counter() - start)

        self.patch_attr(cls, name, wrapped)

    def _wrap_function(self, module: Any, name: str, key: str) -> None:
        original = getattr(module, name)
        profiler = self

        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                profiler.add(key, perf_counter() - start)

        self.patch_attr(module, name, wrapped)

    def install(self) -> None:
        profiler = self

        self._wrap_method(CPUFireModel, "step", "fire_kernel_step")
        self._wrap_method(FirefighterModel, "step", "abm_step_total")
        self._wrap_method(
            FirefighterModel,
            "create_fire_block_ellipse",
            "create_fire_block_ellipse",
        )
        self._wrap_method(
            FirefighterModel,
            "update_fire_block_ellipse",
            "update_fire_block_ellipse",
        )
        self._wrap_method(
            FirefighterModel,
            "is_fire_containable",
            "is_fire_containable",
        )
        self._wrap_method(
            FirefighterModel,
            "verify_block_indices",
            "verify_block_indices",
        )

        original_schedule_step = RandomActivationByBreed.step

        def schedule_step(schedule, *args, **kwargs):
            start = perf_counter()
            try:
                return original_schedule_step(schedule, *args, **kwargs)
            finally:
                profiler.add("abm_schedule_total", perf_counter() - start)

        self.patch_attr(RandomActivationByBreed, "step", schedule_step)

        original_step_breed = RandomActivationByBreed.step_breed

        def step_breed(schedule, breed, *args, **kwargs):
            breed_name = getattr(breed, "__name__", str(breed))
            start = perf_counter()
            try:
                return original_step_breed(schedule, breed, *args, **kwargs)
            finally:
                profiler.add(f"breed:{breed_name}", perf_counter() - start)

        self.patch_attr(RandomActivationByBreed, "step_breed", step_breed)

        original_task_run = Task.run

        @contextmanager
        def task_run(task, agent):
            task_name = getattr(task.task_method, "__name__", "unknown_task")
            total_start = perf_counter()
            method_start = perf_counter()
            status = task.task_method(agent)
            profiler.add(
                f"task_method:{task_name}",
                perf_counter() - method_start,
            )
            try:
                yield status
            finally:
                callback_start = perf_counter()
                callback_ran = False
                if status is TaskStatus.IN_PROGRESS:
                    pass
                elif status is TaskStatus.COMPLETE and task.complete_method:
                    callback_ran = True
                    task.complete_method(agent)
                elif status is TaskStatus.FAILED and task.fail_method:
                    callback_ran = True
                    task.fail_method(agent)
                if callback_ran:
                    profiler.add(
                        f"task_callback:{task_name}",
                        perf_counter() - callback_start,
                    )
                profiler.add(
                    f"task_total:{task_name}",
                    perf_counter() - total_start,
                )

        self.patch_attr(Task, "run", task_run)

        for method_name in (
            "generate_trajectory",
            "set_destination",
            "estimate_propellant_for_journey",
            "max_suppression_aspect",
            "indirect_suppression_aspect",
        ):
            self._wrap_method(SuppressionUAV, method_name, method_name)

        # These modules import the function directly, so patch both aliases.
        self._wrap_function(
            agents_module,
            "generate_straight_trajectory",
            "generate_straight_trajectory",
        )
        self._wrap_function(
            select_poi_module,
            "generate_straight_trajectory",
            "generate_straight_trajectory",
        )

    def uninstall(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def __enter__(self) -> "ABMProfiler":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile firefighter ABM work during PPO decision windows."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["Palisades copy.json", "Pyrenees.json", "Salamis.json"],
        help="Scenario JSON names in inputs/ or absolute/relative paths.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101],
        help="Simulation seeds to profile per scenario.",
    )
    parser.add_argument(
        "--decision-steps",
        type=int,
        default=3,
        help="Maximum PPO decision steps to profile per seed.",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=DEFAULT_DECISION_INTERVAL_MINUTES,
        help="Simulation minutes between PPO decisions.",
    )
    parser.add_argument(
        "--fire-detection-delay-minutes",
        type=float,
        default=None,
        help=(
            "Optional override for initial detection delay. If omitted, "
            "uses the scenario response_time like ppo_runnerv2."
        ),
    )
    parser.add_argument(
        "--state-fire-fronts",
        type=int,
        default=DEFAULT_STATE_FIRE_FRONTS,
        help="State-fire-front count used by the runner state calculation.",
    )
    parser.add_argument(
        "--state-space",
        default=STATE_SPACE_OLD,
        help="Runner state-space option.",
    )
    parser.add_argument(
        "--tactic-distribution",
        choices=SUPPORTED_TACTIC_DISTRIBUTIONS,
        default=TACTIC_DISTRIBUTION_INDIVIDUAL,
        help="How sampled PPO tactic choices are assigned.",
    )
    parser.add_argument(
        "--aircraft-group-size",
        type=int,
        default=AIRCRAFT_GROUP_SIZE,
        help="Aircraft per tactic group when tactic distribution is group.",
    )
    parser.add_argument(
        "--controlled-agent-count",
        type=int,
        default=CONTROLLED_AGENT_COUNT,
        help="Number of aircraft decisions exposed by the runner.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("random", "zeros"),
        default="random",
        help="Use random policy samples or all-zero actions.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of top task/helper rows to print per scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional directory for CSV exports: summary, all metrics, top "
            "breeds, top tasks, and heavy helpers."
        ),
    )
    return parser.parse_args()


def _ms(seconds: float, decisions: int) -> float:
    if decisions <= 0:
        return 0.0
    return 1000.0 * seconds / decisions


def _count(count: int, decisions: int) -> float:
    if decisions <= 0:
        return 0.0
    return count / decisions


def _print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("(no rows)")
        return
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).rjust(widths[idx]) for idx, value in enumerate(row)))


def _sum_prefix(store: TimingStore, prefix: str) -> float:
    return sum(value for key, value in store.seconds.items() if key.startswith(prefix))


def _sum_task_names(store: TimingStore, task_names: set[str]) -> float:
    return sum(store.seconds.get(f"task_total:{name}", 0.0) for name in task_names)


def _top_prefixed(
    store: TimingStore,
    prefix: str,
    decisions: int,
    top_n: int,
) -> list[list[Any]]:
    rows = []
    items = [
        (key.removeprefix(prefix), seconds)
        for key, seconds in store.seconds.items()
        if key.startswith(prefix)
    ]
    for name, seconds in sorted(items, key=lambda item: item[1], reverse=True)[:top_n]:
        key = f"{prefix}{name}"
        rows.append(
            [
                name,
                f"{_ms(seconds, decisions):.2f}",
                f"{_count(store.counts.get(key, 0), decisions):.1f}",
                f"{100.0 * seconds / max(store.seconds.get('rl_step_full', 0.0), 1e-12):.1f}",
            ]
        )
    return rows


def _heavy_rows(store: TimingStore, decisions: int, top_n: int) -> list[list[Any]]:
    rows = []
    for key in HEAVY_KEYS:
        seconds = store.seconds.get(key, 0.0)
        if seconds <= 0.0:
            continue
        rows.append(
            [
                key,
                f"{_ms(seconds, decisions):.2f}",
                f"{_count(store.counts.get(key, 0), decisions):.1f}",
                f"{100.0 * seconds / max(store.seconds.get('rl_step_full', 0.0), 1e-12):.1f}",
            ]
        )
    return sorted(rows, key=lambda row: float(row[1]), reverse=True)[:top_n]


def _metric_dict(
    scenario: str,
    key: str,
    seconds: float,
    calls: int,
    decisions: int,
    full_seconds: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "metric": key,
        "total_seconds": f"{seconds:.9f}",
        "calls": calls,
        "ms_per_decision": f"{_ms(seconds, decisions):.6f}",
        "calls_per_decision": f"{_count(calls, decisions):.6f}",
        "pct_full": f"{100.0 * seconds / max(full_seconds, 1e-12):.6f}",
    }


def _all_metric_dicts(
    scenario: str,
    store: TimingStore,
    decisions: int,
) -> list[dict[str, Any]]:
    full = store.seconds.get("rl_step_full", 0.0)
    rows = [
        _metric_dict(
            scenario,
            key,
            seconds,
            store.counts.get(key, 0),
            decisions,
            full,
        )
        for key, seconds in store.seconds.items()
    ]
    return sorted(rows, key=lambda row: float(row["total_seconds"]), reverse=True)


def _top_prefixed_dicts(
    scenario: str,
    store: TimingStore,
    prefix: str,
    decisions: int,
    top_n: int,
) -> list[dict[str, Any]]:
    full = store.seconds.get("rl_step_full", 0.0)
    items = [
        (key, key.removeprefix(prefix), seconds)
        for key, seconds in store.seconds.items()
        if key.startswith(prefix)
    ]
    rows = []
    for key, name, seconds in sorted(
        items, key=lambda item: item[2], reverse=True
    )[:top_n]:
        row = _metric_dict(
            scenario,
            key,
            seconds,
            store.counts.get(key, 0),
            decisions,
            full,
        )
        row["name"] = name
        rows.append(row)
    return rows


def _heavy_dicts(
    scenario: str,
    store: TimingStore,
    decisions: int,
    top_n: int,
) -> list[dict[str, Any]]:
    full = store.seconds.get("rl_step_full", 0.0)
    rows = []
    for key in HEAVY_KEYS:
        seconds = store.seconds.get(key, 0.0)
        if seconds <= 0.0:
            continue
        row = _metric_dict(
            scenario,
            key,
            seconds,
            store.counts.get(key, 0),
            decisions,
            full,
        )
        row["name"] = key
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["total_seconds"]), reverse=True)[
        :top_n
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_csvs(
    output_dir: Path,
    summary_headers: list[str],
    summary_rows: list[list[str]],
    scenario_results: list[tuple[str, TimingStore, int]],
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "abm_profile_summary.csv",
        [dict(zip(summary_headers, row)) for row in summary_rows],
    )

    all_metric_rows: list[dict[str, Any]] = []
    breed_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    helper_rows: list[dict[str, Any]] = []
    for scenario, store, decisions in scenario_results:
        all_metric_rows.extend(_all_metric_dicts(scenario, store, decisions))
        breed_rows.extend(
            _top_prefixed_dicts(scenario, store, "breed:", decisions, top_n)
        )
        task_rows.extend(
            _top_prefixed_dicts(scenario, store, "task_total:", decisions, top_n)
        )
        helper_rows.extend(_heavy_dicts(scenario, store, decisions, top_n))

    _write_csv(output_dir / "abm_profile_all_metrics.csv", all_metric_rows)
    _write_csv(output_dir / "abm_profile_top_breeds.csv", breed_rows)
    _write_csv(output_dir / "abm_profile_top_tasks.csv", task_rows)
    _write_csv(output_dir / "abm_profile_heavy_helpers.csv", helper_rows)


def _profile_scenario(args: argparse.Namespace, scenario: str) -> tuple[TimingStore, int]:
    aggregate = TimingStore()
    completed_decisions = 0

    with ABMProfiler() as profiler:
        for seed in args.seeds:
            env = WildfireHourlyEnv(
                _resolve_scenario(scenario),
                max_steps=max(args.decision_steps + 1, 1),
                decision_interval_minutes=args.decision_interval_minutes,
                fire_detection_delay_minutes=args.fire_detection_delay_minutes,
                state_fire_fronts=args.state_fire_fronts,
                state_space=args.state_space,
                tactic_distribution=args.tactic_distribution,
                aircraft_group_size=args.aircraft_group_size,
                controlled_agent_count=args.controlled_agent_count,
            )
            try:
                env.action_space.seed(seed)
                env.reset(options={"sim_seed": seed})
                profiler.clear()

                for _ in range(args.decision_steps):
                    if args.action_mode == "zeros":
                        action = np.zeros(env.action_space.shape, dtype=int)
                    else:
                        action = env.action_space.sample()

                    profiler.clear()
                    start = perf_counter()
                    _observation, _reward, terminated, truncated, _info = env.step(
                        action
                    )
                    profiler.add("rl_step_full", perf_counter() - start)
                    aggregate.merge_from(profiler.store)
                    completed_decisions += 1

                    if terminated or truncated:
                        break
            finally:
                env.close()

    return aggregate, completed_decisions


def _scenario_summary_row(
    scenario: str,
    store: TimingStore,
    decisions: int,
) -> list[str]:
    full = store.seconds.get("rl_step_full", 0.0)
    fire = store.seconds.get("fire_kernel_step", 0.0)
    abm = store.seconds.get("abm_step_total", 0.0)
    schedule = store.seconds.get("abm_schedule_total", 0.0)
    block_update = max(abm - schedule, 0.0)
    aircraft = store.seconds.get(f"breed:{SuppressionUAV.__name__}", 0.0)
    static = sum(store.seconds.get(f"breed:{name}", 0.0) for name in STATIC_BREED_NAMES)
    select_poi = _sum_task_names(store, SELECT_POI_TASK_NAMES)
    suppress = _sum_task_names(store, SUPPRESS_TASK_NAMES)
    trajectory = (
        store.seconds.get("generate_trajectory", 0.0)
        + store.seconds.get("generate_straight_trajectory", 0.0)
    )
    return [
        scenario,
        str(decisions),
        f"{_ms(full, decisions):.1f}",
        f"{_ms(fire, decisions):.1f}",
        f"{_ms(abm, decisions):.1f}",
        f"{_ms(schedule, decisions):.1f}",
        f"{_ms(block_update, decisions):.1f}",
        f"{_ms(aircraft, decisions):.1f}",
        f"{_ms(static, decisions):.1f}",
        f"{_ms(select_poi, decisions):.1f}",
        f"{_ms(suppress, decisions):.1f}",
        f"{_ms(trajectory, decisions):.1f}",
        f"{100.0 * abm / max(full, 1e-12):.0f}",
    ]


def main() -> None:
    args = _parse_args()
    if args.decision_steps <= 0:
        raise ValueError("--decision-steps must be > 0")

    scenario_results: list[tuple[str, TimingStore, int]] = []
    for scenario in args.scenarios:
        store, decisions = _profile_scenario(args, scenario)
        scenario_results.append((scenario, store, decisions))

    summary_rows = [
        _scenario_summary_row(scenario, store, decisions)
        for scenario, store, decisions in scenario_results
    ]
    summary_headers = [
        "scenario",
        "n",
        "full",
        "fire",
        "abm",
        "schedule",
        "block",
        "aircraft",
        "static",
        "select",
        "suppress",
        "traj",
        "abm_pct",
    ]
    _print_table(
        "ABM PROFILE SUMMARY (per RL decision)",
        summary_headers,
        summary_rows,
    )

    for scenario, store, decisions in scenario_results:
        _print_table(
            f"BREED TIMINGS: {scenario}",
            ["breed", "ms/decision", "calls/decision", "%full"],
            _top_prefixed(store, "breed:", decisions, args.top_n),
        )
        _print_table(
            f"TASK TIMINGS: {scenario}",
            ["task", "ms/decision", "calls/decision", "%full"],
            _top_prefixed(store, "task_total:", decisions, args.top_n),
        )
        _print_table(
            f"HEAVY HELPERS: {scenario}",
            ["helper", "ms/decision", "calls/decision", "%full"],
            _heavy_rows(store, decisions, args.top_n),
        )

    if args.output_dir:
        _export_csvs(
            args.output_dir,
            summary_headers,
            summary_rows,
            scenario_results,
            args.top_n,
        )
        print(f"\nCSV profile outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
