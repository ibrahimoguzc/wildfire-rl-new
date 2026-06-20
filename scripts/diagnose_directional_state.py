#!/usr/bin/env python3
"""Summarize how much signal the directional observation block carries.

The diagnostic runs real WildfireHourlyEnv episodes with
state_space="directional" and reports min/max/mean/std/nonzero rates for the
appended per-sector directional features. It is intentionally read-only with
respect to the simulator: no training or model writes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for path in (REPO, REPO / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from examples.wildfire.paths import TERRAIN_DIR
from examples.wildfire.ppo_runnerv2 import (
    DIRECTIONAL_PER_FRONT_FEATURES,
    STATE_SPACE_DIRECTIONAL,
    WildfireHourlyEnv,
    _resolve_scenario,
)


def _resolve_water_set(scenario_path: Path, water_set: str) -> int | None:
    if water_set == "off":
        return None

    selected = int(water_set)
    scenario_data = json.loads(scenario_path.read_text())
    namespace = scenario_data["terrain_inputs"]["file_namespace"]
    water_file = TERRAIN_DIR / f"{namespace}_water_sources_set{selected}.pkl"
    if not water_file.exists():
        raise FileNotFoundError(
            f"--water-set {selected} requires {water_file}, which does not exist."
        )
    return selected


def _directional_feature_indices(names: list[str]) -> dict[str, int]:
    prefixes = tuple(f"{name}_" for name in DIRECTIONAL_PER_FRONT_FEATURES)
    return {
        name: idx
        for idx, name in enumerate(names)
        if name.startswith(prefixes)
    }


def _base_and_slot(feature_name: str) -> tuple[str, int]:
    for base in DIRECTIONAL_PER_FRONT_FEATURES:
        prefix = f"{base}_"
        if feature_name.startswith(prefix):
            return base, int(feature_name.removeprefix(prefix))
    raise ValueError(f"Not a directional feature: {feature_name}")


def _action_for_env(
    env: WildfireHourlyEnv,
    rng: np.random.Generator,
    action_mode: str,
    fixed_action: int,
) -> np.ndarray:
    if action_mode == "zero":
        return np.zeros(env.action_space.shape, dtype=np.int64)
    if action_mode == "fixed":
        return np.full(env.action_space.shape, fixed_action, dtype=np.int64)
    # Gym's sampler is fine, but seeding a local generator keeps this script's
    # action stream independent of any simulator RNG internals.
    return np.asarray(
        [
            rng.integers(0, int(high), dtype=np.int64)
            for high in env.action_space.nvec
        ],
        dtype=np.int64,
    ).reshape(env.action_space.shape)


def _summarize_values(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "sample_count": 0,
            "nonzero_count": 0,
            "nonzero_rate": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.zeros(1, dtype=float)
    nonzero = int(np.count_nonzero(np.abs(finite) > 1e-12))
    return {
        "sample_count": int(finite.size),
        "nonzero_count": nonzero,
        "nonzero_rate": float(nonzero / finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario",
        "state_fire_fronts",
        "feature",
        "slot",
        "sample_count",
        "nonzero_count",
        "nonzero_rate",
        "min",
        "max",
        "mean",
        "std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario_path = _resolve_scenario(args.scenario)
    water_set = _resolve_water_set(scenario_path, args.water_set)

    env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        max_steps=args.steps,
        decision_interval_minutes=args.decision_interval_minutes,
        state_space=STATE_SPACE_DIRECTIONAL,
        state_fire_fronts=args.state_fire_fronts,
        tactic_distribution=args.tactic_distribution,
        aircraft_group_size=args.aircraft_group_size,
        controlled_agent_count=args.controlled_agent_count,
        water_set=water_set,
        cell_size_source=args.cell_size,
    )

    names = list(env.state_feature_names)
    feature_indices = _directional_feature_indices(names)
    values_by_feature: dict[str, list[float]] = {
        name: [] for name in feature_indices
    }
    values_by_base: dict[str, list[float]] = {
        name: [] for name in DIRECTIONAL_PER_FRONT_FEATURES
    }
    episode_summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    env.action_space.seed(args.seed)

    for episode_idx in range(args.episodes):
        sim_seed = (
            args.sim_seed + episode_idx
            if args.sim_seed is not None
            else args.seed + episode_idx
        )
        obs, info = env.reset(seed=args.seed + episode_idx, options={"sim_seed": sim_seed})
        episode_rows = [obs]
        terminated = False
        truncated = False
        last_info = info

        for _step in range(args.steps):
            action = _action_for_env(env, rng, args.action_mode, args.fixed_action)
            obs, _reward, terminated, truncated, last_info = env.step(action)
            episode_rows.append(obs)
            if terminated or truncated:
                break

        episode_array = np.vstack(episode_rows)
        for feature_name, idx in feature_indices.items():
            base, _slot = _base_and_slot(feature_name)
            vals = episode_array[:, idx].astype(float)
            values_by_feature[feature_name].extend(vals.tolist())
            values_by_base[base].extend(vals.tolist())

        episode_summaries.append(
            {
                "episode": episode_idx,
                "sim_seed": sim_seed,
                "observations": int(episode_array.shape[0]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "propagation_factor": float(last_info.get("propagation_factor", 0.0)),
                "scenario": last_info.get("scenario"),
                "scenario_name": last_info.get("scenario_name"),
            }
        )

    scenario_label = scenario_path.stem.split()[0].lower()
    summary_rows: list[dict[str, Any]] = []
    for feature_name in sorted(feature_indices, key=lambda name: feature_indices[name]):
        base, slot = _base_and_slot(feature_name)
        stats = _summarize_values(np.asarray(values_by_feature[feature_name], dtype=float))
        summary_rows.append(
            {
                "scenario": scenario_label,
                "state_fire_fronts": args.state_fire_fronts,
                "feature": base,
                "slot": slot,
                **stats,
            }
        )

    for base in DIRECTIONAL_PER_FRONT_FEATURES:
        stats = _summarize_values(np.asarray(values_by_base[base], dtype=float))
        summary_rows.append(
            {
                "scenario": scenario_label,
                "state_fire_fronts": args.state_fire_fronts,
                "feature": base,
                "slot": "all",
                **stats,
            }
        )

    metadata = {
        "scenario": str(scenario_path),
        "state_feature_count": len(names),
        "directional_feature_count": len(feature_indices),
        "episodes": args.episodes,
        "steps": args.steps,
        "decision_interval_minutes": args.decision_interval_minutes,
        "action_mode": args.action_mode,
        "fixed_action": args.fixed_action,
        "water_set": args.water_set,
        "episode_summaries": episode_summaries,
    }
    return summary_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure sparsity and variance in the directional state block."
    )
    parser.add_argument("--scenario", default="Pyrenees.json")
    parser.add_argument("--state-fire-fronts", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--decision-interval-minutes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sim-seed", type=int, default=None)
    parser.add_argument("--controlled-agent-count", type=int, default=6)
    parser.add_argument(
        "--tactic-distribution",
        default="group",
        choices=("individual", "group"),
    )
    parser.add_argument("--aircraft-group-size", type=int, default=2)
    parser.add_argument("--water-set", default="1", choices=("1", "2", "off"))
    parser.add_argument(
        "--cell-size",
        default="code",
        choices=("json", "code"),
        help="metres/cell for directional geometry: 'code' (default) = actual "
        "projected spacing; 'json' = nominal cell_size from the scenario file",
    )
    parser.add_argument(
        "--action-mode",
        default="random",
        choices=("random", "zero", "fixed"),
    )
    parser.add_argument("--fixed-action", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "examples"
        / "wildfire"
        / "data"
        / "scenarios"
        / "outputs"
        / "directional_state_diagnostics",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.state_fire_fronts <= 0:
        raise ValueError("--state-fire-fronts must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = run_diagnostic(args)
    scenario_stem = Path(metadata["scenario"]).stem.split()[0].lower()
    tag = (
        f"{scenario_stem}_k{args.state_fire_fronts}_"
        f"ep{args.episodes}_steps{args.steps}_{args.action_mode}"
    )
    csv_path = args.output_dir / f"{tag}_summary.csv"
    json_path = args.output_dir / f"{tag}_metadata.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("Aggregate directional feature stats:")
    for row in rows[-len(DIRECTIONAL_PER_FRONT_FEATURES):]:
        print(
            f"  {row['feature']:<18} nonzero={row['nonzero_rate']:.3f} "
            f"mean={row['mean']:.4f} std={row['std']:.4f} "
            f"range=[{row['min']:.4f}, {row['max']:.4f}]"
        )


if __name__ == "__main__":
    main()
