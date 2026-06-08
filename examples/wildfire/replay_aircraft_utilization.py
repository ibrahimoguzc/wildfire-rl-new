"""Replay PPO decision logs and inspect aircraft utilization.

This script is intentionally diagnostic-only: it does not change training.
It replays the action choices from a decision-step CSV, records aircraft
flight/task state at each decision boundary, and summarizes whether each
aircraft actually left base or sat idle.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.wildfire.firefighter_model.suppression_tactics import (
    SuppresionTactic,
)
from examples.wildfire.ppo_runnerv2 import (
    DEFAULT_STATE_FIRE_FRONTS,
    SCENARIOS_DIR,
    STATE_SPACE_MIXED,
    TACTIC_COMBINATIONS,
    TACTIC_DISTRIBUTION_GROUP,
    WildfireHourlyEnv,
    _resolve_scenario,
)
from sosid.model.abm.trajectory import IN_AIR_FLIGHT_STATES


ACTION_BY_LABELS = {
    (select.value, track.value, suppress.value): idx
    for idx, (select, track, suppress) in enumerate(TACTIC_COMBINATIONS)
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _active_task_name(agent: Any) -> str:
    task = agent.tasks.active_task
    if task is None:
        return "none"
    return getattr(task.task_method, "__name__", str(task.task_method))


def _idle_timer_seconds(agent: Any) -> float:
    timer = getattr(agent, "__idle_timer__", None)
    if timer is None:
        return 0.0
    return float(timer.total_seconds())


def _aircraft_snapshot(agent: Any) -> dict[str, Any]:
    flight_state = agent.flight_state
    return {
        "task": _active_task_name(agent),
        "flight_state": _enum_name(flight_state),
        "in_air": int(flight_state in IN_AIR_FLIGHT_STATES),
        "at_base": int(getattr(agent, "current_base", None) is not None),
        "idle_timer_seconds": round(_idle_timer_seconds(agent), 3),
        "takeoffs": int(getattr(agent, "_utilization_takeoffs", 0)),
        "flight_time_min": round(
            float(getattr(agent, "_cumulative_flight_time", 0.0)) / 60.0,
            3,
        ),
        "distance_flown_km": round(
            float(getattr(agent, "total_distance_covered", 0.0)) / 1000.0,
            3,
        ),
        "suppressions": int(getattr(agent, "total_suppressions", 0)),
        "propellant_refills": int(agent.propulsion.n_propellant_refills),
    }


def _action_from_row(row: dict[str, str], action_decision_count: int) -> np.ndarray:
    action_indices: list[int] = []
    for group_idx in range(action_decision_count):
        labels = (
            row[f"group_{group_idx}_select_poi"],
            row[f"group_{group_idx}_track_poi"],
            row[f"group_{group_idx}_suppress"],
        )
        try:
            action_indices.append(ACTION_BY_LABELS[labels])
        except KeyError as err:
            raise ValueError(
                f"Unsupported tactic combination in row: {labels}"
            ) from err
    return np.asarray(action_indices, dtype=np.int64)


def _read_episode_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    episodes: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sim_idx = _safe_int(row.get("simulation_index"), default=0)
            if sim_idx <= 0:
                continue
            episodes[sim_idx].append(row)

    for rows in episodes.values():
        rows.sort(key=lambda item: _safe_int(item.get("decision_step")))
    return dict(sorted(episodes.items()))


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _install_takeoff_counter() -> None:
    original = SuppresionTactic.initiate_takeoff

    def tracked_initiate_takeoff(agent: Any) -> None:
        agent._utilization_takeoffs = int(  # noqa: SLF001 - diagnostic counter
            getattr(agent, "_utilization_takeoffs", 0)
        ) + 1
        return original(agent)

    SuppresionTactic.initiate_takeoff = staticmethod(tracked_initiate_takeoff)


def _episode_summary(
    simulation_index: int,
    sim_seed: int,
    steps_replayed: int,
    final_info: dict[str, Any],
    agents: list[Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "simulation_index": simulation_index,
        "sim_seed": sim_seed,
        "steps_replayed": steps_replayed,
        "terminated": int("episode_summary" in final_info),
        "episode_minutes": (
            final_info.get("episode_summary", {}).get("total_minutes")
            if "episode_summary" in final_info
            else final_info.get("elapsed_minutes")
        ),
        "total_takeoffs": 0,
        "total_flight_time_min": 0.0,
        "total_distance_flown_km": 0.0,
        "total_suppressions": 0,
        "aircraft_never_left_count": 0,
    }
    for idx, agent in enumerate(agents):
        snapshot = _aircraft_snapshot(agent)
        never_left = (
            snapshot["takeoffs"] == 0
            and snapshot["flight_time_min"] <= 0.0
            and snapshot["distance_flown_km"] <= 0.001
        )
        record[f"agent_{idx}_takeoffs"] = snapshot["takeoffs"]
        record[f"agent_{idx}_never_left"] = int(never_left)
        record[f"agent_{idx}_final_task"] = snapshot["task"]
        record[f"agent_{idx}_final_flight_state"] = snapshot["flight_state"]
        record[f"agent_{idx}_flight_time_min"] = snapshot["flight_time_min"]
        record[f"agent_{idx}_distance_flown_km"] = snapshot["distance_flown_km"]
        record[f"agent_{idx}_suppressions"] = snapshot["suppressions"]
        record[f"agent_{idx}_propellant_refills"] = snapshot["propellant_refills"]
        record["total_takeoffs"] += snapshot["takeoffs"]
        record["total_flight_time_min"] += snapshot["flight_time_min"]
        record["total_distance_flown_km"] += snapshot["distance_flown_km"]
        record["total_suppressions"] += snapshot["suppressions"]
        record["aircraft_never_left_count"] += int(never_left)

    record["total_flight_time_min"] = round(record["total_flight_time_min"], 3)
    record["total_distance_flown_km"] = round(record["total_distance_flown_km"], 3)
    return record


def replay(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _install_takeoff_counter()
    decision_rows_by_episode = _read_episode_rows(args.decision_steps)
    selected_episode_ids = list(decision_rows_by_episode)
    if args.episodes:
        requested = set(args.episodes)
        selected_episode_ids = [
            episode_id for episode_id in selected_episode_ids if episode_id in requested
        ]
    if args.max_episodes is not None:
        selected_episode_ids = selected_episode_ids[: args.max_episodes]

    env = WildfireHourlyEnv(
        _resolve_scenario(args.scenario),
        decision_interval_minutes=args.decision_interval_minutes,
        fire_detection_delay_minutes=args.fire_detection_delay_minutes,
        state_fire_fronts=args.state_fire_fronts,
        state_space=args.state_space,
        tactic_distribution=TACTIC_DISTRIBUTION_GROUP,
        aircraft_group_size=args.aircraft_group_size,
        controlled_agent_count=args.controlled_agent_count,
    )

    decision_trace: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for episode_id in selected_episode_ids:
            rows = decision_rows_by_episode[episode_id]
            if not rows:
                continue
            sim_seed = _safe_int(rows[0].get("sim_seed"))
            env.reset(options={"sim_seed": sim_seed})
            final_info: dict[str, Any] = {}
            steps_replayed = 0
            for row in rows:
                action = _action_from_row(row, env.action_decision_count)
                _, reward, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                steps_replayed += 1
                trace_record: dict[str, Any] = {
                    "simulation_index": episode_id,
                    "sim_seed": sim_seed,
                    "decision_step": info.get("decision_step"),
                    "elapsed_minutes": info.get("elapsed_minutes"),
                    "reward": reward,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                }
                agents = env.sim.firefighters.firefighters[
                    : args.controlled_agent_count
                ]
                for idx, agent in enumerate(agents):
                    for key, value in _aircraft_snapshot(agent).items():
                        trace_record[f"agent_{idx}_{key}"] = value
                decision_trace.append(trace_record)
                if terminated or truncated:
                    break

            agents = env.sim.firefighters.firefighters[: args.controlled_agent_count]
            summaries.append(
                _episode_summary(
                    episode_id,
                    sim_seed,
                    steps_replayed,
                    final_info,
                    agents,
                )
            )
    finally:
        env.close()

    return decision_trace, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay decision-step logs and report aircraft utilization.",
    )
    parser.add_argument(
        "--decision-steps",
        type=Path,
        required=True,
        help="Path to results_*_decision_steps.csv from ppo_runnerv2.",
    )
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON used by the run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCENARIOS_DIR / "outputs" / "aircraft_utilization_replay",
        help="Directory for replay utilization CSV outputs.",
    )
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Specific simulation_index values to replay.",
    )
    parser.add_argument("--decision-interval-minutes", type=int, default=10)
    parser.add_argument("--fire-detection-delay-minutes", type=float, default=30.0)
    parser.add_argument("--state-space", default=STATE_SPACE_MIXED)
    parser.add_argument("--state-fire-fronts", type=int, default=DEFAULT_STATE_FIRE_FRONTS)
    parser.add_argument("--controlled-agent-count", type=int, default=6)
    parser.add_argument("--aircraft-group-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace, summaries = replay(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "aircraft_utilization_decision_trace.csv"
    summary_path = args.output_dir / "aircraft_utilization_episode_summary.csv"
    _write_records(trace_path, trace)
    _write_records(summary_path, summaries)

    total_aircraft = len(summaries) * args.controlled_agent_count
    never_left = sum(_safe_int(row["aircraft_never_left_count"]) for row in summaries)
    total_takeoffs = sum(_safe_int(row["total_takeoffs"]) for row in summaries)
    total_suppressions = sum(_safe_int(row["total_suppressions"]) for row in summaries)

    print(f"Replayed episodes: {len(summaries)}")
    print(f"Aircraft-episodes checked: {total_aircraft}")
    print(f"Aircraft-episodes with no takeoff/flight/distance: {never_left}")
    print(f"Total takeoffs counted: {total_takeoffs}")
    print(f"Total suppressions counted: {total_suppressions}")
    print(f"Decision trace written to: {trace_path}")
    print(f"Episode summary written to: {summary_path}")


if __name__ == "__main__":
    main()
