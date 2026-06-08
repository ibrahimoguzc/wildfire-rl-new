#!/usr/bin/env python3
"""Run wildfire env rollouts and validate observation bounds in [0, 1]."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.ppo_runner import (
    SWITCH_SCENARIO_NAMES,
    WildfireHourlyEnv,
    _resolve_scenario,
)


@dataclass
class FeatureStats:
    min_value: float = float("inf")
    max_value: float = float("-inf")
    below_zero_count: int = 0
    above_one_count: int = 0
    sample_count: int = 0


def _feature_names(env: WildfireHourlyEnv) -> list[str]:
    return list(env.state_feature_names)


def _ensure_len(obs: np.ndarray, names: list[str]) -> None:
    if obs.shape != (len(names),):
        raise ValueError(
            f"Observation shape mismatch. obs={obs.shape}, expected={(len(names),)}."
        )


def _update_stats(
    *,
    obs: np.ndarray,
    names: list[str],
    stats: dict[str, FeatureStats],
    violations: list[dict[str, Any]],
    episode_idx: int,
    step_idx: int,
) -> None:
    for i, name in enumerate(names):
        value = float(obs[i])
        row = stats[name]
        row.min_value = min(row.min_value, value)
        row.max_value = max(row.max_value, value)
        row.sample_count += 1
        if value < 0.0:
            row.below_zero_count += 1
            violations.append(
                {
                    "episode": episode_idx,
                    "step": step_idx,
                    "feature": name,
                    "value": value,
                    "type": "below_zero",
                }
            )
        elif value > 1.0:
            row.above_one_count += 1
            violations.append(
                {
                    "episode": episode_idx,
                    "step": step_idx,
                    "feature": name,
                    "value": value,
                    "type": "above_one",
                }
            )


def _record_state_step(
    *,
    rows: list[dict[str, Any]],
    obs: np.ndarray,
    names: list[str],
    episode_idx: int,
    step_idx: int,
    info: dict[str, Any],
    reward: float | None,
    terminated: bool,
    truncated: bool,
) -> None:
    row: dict[str, Any] = {
        "episode": episode_idx,
        "step": step_idx,
        "decision_step": info.get("decision_step"),
        "elapsed_minutes": info.get("elapsed_minutes"),
        "scenario_name": info.get("scenario_name"),
        "sim_seed": info.get("sim_seed"),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    for i, name in enumerate(names):
        row[name] = float(obs[i])
    rows.append(row)


def _action_for_env(env: WildfireHourlyEnv, action_mode: str) -> np.ndarray:
    if action_mode == "zero":
        return np.zeros(env.action_space.shape, dtype=np.int64)
    return np.asarray(env.action_space.sample(), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run wildfire environment rollouts and validate whether all "
            "observation features stay within [0, 1]."
        )
    )
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON (name in inputs/ or absolute path).",
    )
    parser.add_argument(
        "--switch-scenario",
        action="store_true",
        help="Sample scenarios per reset from Palisades copy, Pyrenees, Salamis.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=200,
        help="Max environment steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed; episode seed = seed + episode index.",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=10,
        help="Decision interval in minutes.",
    )
    parser.add_argument(
        "--fire-detection-delay-minutes",
        type=float,
        default=None,
        help="Optional detection delay override in minutes.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("zero", "random"),
        default="random",
        help="Use deterministic zero actions or random sampled actions.",
    )
    parser.add_argument(
        "--output",
        default="examples/wildfire/data/scenarios/outputs/state_bounds_validation.xlsx",
        help="Output report path (.xlsx recommended).",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.max_steps_per_episode < 1:
        raise ValueError("--max-steps-per-episode must be >= 1")
    if (
        args.fire_detection_delay_minutes is not None
        and args.fire_detection_delay_minutes < 0
    ):
        raise ValueError("--fire-detection-delay-minutes must be >= 0")

    if args.switch_scenario:
        switch_paths = tuple(_resolve_scenario(name) for name in SWITCH_SCENARIO_NAMES)
        scenario_path = switch_paths[0]
    else:
        switch_paths = None
        scenario_path = _resolve_scenario(args.scenario)

    env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        decision_interval_minutes=args.decision_interval_minutes,
        fire_detection_delay_minutes=args.fire_detection_delay_minutes,
        switch_scenario=args.switch_scenario,
        switch_scenario_paths=switch_paths,
    )

    names = _feature_names(env)
    stats = {name: FeatureStats() for name in names}
    violations: list[dict[str, Any]] = []
    state_steps: list[dict[str, Any]] = []

    episodes_run = 0
    total_steps = 0
    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            _ensure_len(obs, names)
            _update_stats(
                obs=obs,
                names=names,
                stats=stats,
                violations=violations,
                episode_idx=episode_idx,
                step_idx=0,
            )
            _record_state_step(
                rows=state_steps,
                obs=obs,
                names=names,
                episode_idx=episode_idx,
                step_idx=0,
                info=info,
                reward=None,
                terminated=False,
                truncated=False,
            )

            for step_idx in range(1, args.max_steps_per_episode + 1):
                action = _action_for_env(env, args.action_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                _ensure_len(obs, names)
                _update_stats(
                    obs=obs,
                    names=names,
                    stats=stats,
                    violations=violations,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                )
                _record_state_step(
                    rows=state_steps,
                    obs=obs,
                    names=names,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    info=info,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                total_steps += 1
                if terminated or truncated:
                    break
            episodes_run += 1
    finally:
        env.close()

    summary_rows: list[dict[str, Any]] = []
    for name in names:
        row = stats[name]
        summary_rows.append(
            {
                "feature": name,
                "min_value": row.min_value,
                "max_value": row.max_value,
                "below_zero_count": row.below_zero_count,
                "above_one_count": row.above_one_count,
                "sample_count": row.sample_count,
                "within_0_1": (row.below_zero_count == 0 and row.above_one_count == 0),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            "pandas is required for Excel output. Install pandas or set a .csv path."
        ) from err

    summary_df = pd.DataFrame(summary_rows)
    violations_df = pd.DataFrame(violations)
    state_steps_df = pd.DataFrame(state_steps)
    run_info_df = pd.DataFrame(
        [
            {
                "episodes_run": episodes_run,
                "steps_run": total_steps,
                "switch_scenario": args.switch_scenario,
                "action_mode": args.action_mode,
                "decision_interval_minutes": args.decision_interval_minutes,
                "fire_detection_delay_minutes_override": args.fire_detection_delay_minutes,
                "output_path": str(output_path),
            }
        ]
    )
    with pd.ExcelWriter(output_path) as writer:
        run_info_df.to_excel(writer, index=False, sheet_name="run_info")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        violations_df.to_excel(writer, index=False, sheet_name="violations")
        state_steps_df.to_excel(writer, index=False, sheet_name="state_steps")

    violated_features = int(
        np.count_nonzero(
            ~(summary_df["within_0_1"].to_numpy(dtype=bool))
        )
    )
    print(f"Validation report written to: {output_path}")
    print(
        f"Episodes={episodes_run}, steps={total_steps}, "
        f"features_out_of_range={violated_features}, "
        f"violation_rows={len(violations)}"
    )


if __name__ == "__main__":
    main()
