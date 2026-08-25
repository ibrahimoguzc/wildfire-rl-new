#!/usr/bin/env python3
"""Evaluate PPO model(s) and fixed-tactic baselines without training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from examples.wildfire.paths import SCENARIOS_DIR
# ppo_runnerv2, not ppo_runner: every model since mid-2026 was trained with v2,
# and only v2's env accepts state_space / water_set / group_sizes /
# state_fire_fronts / cell_size_source. ppo_runner also still carries the
# ignition cell-size drift (5 m assumed vs the sim's ~6 m grid), which would
# put switch-ignition fires about a kilometre from where training placed them.
from examples.wildfire.ppo_runnerv2 import (
    CONTROLLED_AGENT_COUNT,
    DEFAULT_DECISION_INTERVAL_MINUTES,
    OLD_STATE_FEATURES as STATE_FEATURES,
    SCENARIO_FEATURES,
    SWITCH_SCENARIO_NAMES,
    TACTIC_COMBINATIONS,
    TACTIC_DISTRIBUTION_GROUP,
    WildfireHourlyEnv,
    _resolve_scenario,
    _write_records,
)

# Features added to the state vector after the switch_ignition_1m model
# was trained. When evaluating that legacy model we drop these indices
# from the env's observation so the layout matches what the model expects.
LEGACY_DROPPED_FEATURE_NAMES: tuple[str, ...] = ("ignition_x", "ignition_y")
from examples.wildfire.simulation import WildfireParameters

_WORKER_DECISION_INTERVAL_MINUTES: int = DEFAULT_DECISION_INTERVAL_MINUTES
_WORKER_DETERMINISTIC: bool = True
_WORKER_DEVICE: str = "cpu"
_WORKER_MODEL_CACHE: dict[str, PPO] = {}
_WORKER_FIXED_ACTION_CACHE: dict[str, np.ndarray] = {}
# Env construction options that must match how the model was trained. The
# defaults here reproduce the tester's historical behaviour (state_space="old",
# 3 individually-controlled aircraft, no ignition switching, no water set), so
# older invocations keep working; --state-space and friends override them.
_WORKER_ENV_KWARGS: dict[str, Any] = {}
_WORKER_GROUP_SIZES: tuple[int, ...] | None = None


class ObservationDropPrefixWrapper(gym.ObservationWrapper):
    """Drop a fixed number of leading observation features."""

    def __init__(self, env: gym.Env, drop_count: int):
        super().__init__(env)
        if drop_count <= 0:
            raise ValueError("drop_count must be positive.")
        base_shape = self.env.observation_space.shape
        if base_shape is None or len(base_shape) != 1:
            raise ValueError("Only 1D Box observations are supported.")
        if drop_count >= base_shape[0]:
            raise ValueError("drop_count must be smaller than observation size.")
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation spaces are supported.")

        old_space = self.env.observation_space
        self.drop_count = drop_count
        low = old_space.low[drop_count:]
        high = old_space.high[drop_count:]
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=old_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.drop_count :]


class ObservationDropIndicesWrapper(gym.ObservationWrapper):
    """Drop a specific set of observation indices."""

    def __init__(self, env: gym.Env, drop_indices: tuple[int, ...]):
        super().__init__(env)
        if not drop_indices:
            raise ValueError("drop_indices must be non-empty.")
        base_shape = self.env.observation_space.shape
        if base_shape is None or len(base_shape) != 1:
            raise ValueError("Only 1D Box observations are supported.")
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation spaces are supported.")
        n = base_shape[0]
        if any(i < 0 or i >= n for i in drop_indices):
            raise ValueError(f"drop_indices out of range for obs of size {n}.")

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[list(drop_indices)] = False
        self._keep_mask = keep_mask

        old = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old.low[keep_mask],
            high=old.high[keep_mask],
            dtype=old.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self._keep_mask]


def _terminal_moe(summary: dict[str, Any], reward_total: float) -> float:
    """End-of-simulation MoE, not a sum of per-step rewards.

    The env's ``episode_summary["moe_cumulative_reward"]`` is recomputed at
    termination from the terminal burnt area / cost / emissions / casualties and
    the terminal propagation factor, which is the quantity we want to compare
    policies on. ``reward_total`` (the summed step rewards) is a different
    number, so falling back to it silently would mix two metrics in one column.
    Fail loudly instead - an episode with no summary did not terminate normally.
    """
    if "moe_cumulative_reward" not in summary:
        raise RuntimeError(
            "Episode produced no episode_summary['moe_cumulative_reward']; "
            "cannot report end-of-simulation MoE. The episode likely ended by "
            f"truncation rather than termination (summed step reward was "
            f"{reward_total:.6f})."
        )
    return float(summary["moe_cumulative_reward"])


def _fixed_action_vector_from_scenario(
    scenario_path: Path,
    *,
    controlled_count: int = CONTROLLED_AGENT_COUNT,
    group_sizes: Sequence[int] | None = None,
) -> np.ndarray:
    """Read a scenario's per-aircraft tactics into a fixed action vector.

    Without ``group_sizes`` the action has one entry per controlled aircraft,
    which is what an ungrouped (``tactic_distribution="individual"``) env
    expects. With ``group_sizes`` the env takes one decision per group, so the
    vector has one entry per group and every aircraft in a group must carry the
    same tactic - otherwise the scenario is asking for something the grouped
    action space cannot express, and silently keeping the first aircraft's
    tactic would misreport what was evaluated.
    """
    with scenario_path.open(encoding="utf-8") as handle:
        parameters = WildfireParameters.model_validate_json(handle.read())

    combos_per_aircraft: list[tuple[Any, Any, Any]] = []
    for agent_def in parameters.agents:
        main = agent_def.suppression_tactic.main
        combo = (main.select_poi, main.track_poi, main.suppress)
        count = sum(int(c) for c in agent_def.agents_per_base)
        for _ in range(max(0, count)):
            combos_per_aircraft.append(combo)

    if not combos_per_aircraft:
        raise ValueError(f"No aircraft tactics found in {scenario_path.name}")

    def index_of(combo: tuple[Any, Any, Any]) -> int:
        if combo not in TACTIC_COMBINATIONS:
            raise ValueError(
                "Scenario tactic is not in PPO action space "
                f"for {scenario_path.name}: {combo}"
            )
        return TACTIC_COMBINATIONS.index(combo)

    if group_sizes:
        expected = sum(int(size) for size in group_sizes)
        if expected != len(combos_per_aircraft):
            raise ValueError(
                f"--group-sizes sums to {expected} aircraft but "
                f"{scenario_path.name} defines {len(combos_per_aircraft)}."
            )
        action_indices: list[int] = []
        start = 0
        for group_idx, size in enumerate(group_sizes):
            group = combos_per_aircraft[start:start + int(size)]
            start += int(size)
            if len(set(group)) != 1:
                raise ValueError(
                    f"Group {group_idx} of {scenario_path.name} mixes tactics "
                    f"{sorted({str(c) for c in group})}; a grouped action space "
                    "issues one tactic per group, so each group must be uniform."
                )
            action_indices.append(index_of(group[0]))
        return np.asarray(action_indices, dtype=np.int64)

    return np.asarray(
        [
            index_of(combos_per_aircraft[min(i, len(combos_per_aircraft) - 1)])
            for i in range(controlled_count)
        ],
        dtype=np.int64,
    )


def _run_single_episode_fixed(
    *,
    fixed_action: np.ndarray,
    scenario_path: Path,
    decision_interval_minutes: int,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        decision_interval_minutes=decision_interval_minutes,
        switch_scenario=False,
        **(env_kwargs or {}),
    )
    wall_start = time.perf_counter()
    try:
        _, info = env.reset(seed=seed, options={"sim_seed": seed})
        terminated = False
        truncated = False
        reward_total = 0.0

        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(fixed_action)
            reward_total += float(reward)

        summary = dict(info.get("episode_summary", {}))
        final_metrics = dict(summary.get("final_metrics", info.get("metrics", {})))
        total_minutes = float(summary.get("total_minutes", 0.0))
        return {
            "scenario": info.get("scenario"),
            "scenario_name": info.get("scenario_name"),
            "scenario_path": info.get("scenario_path"),
            "seed": seed,
            # Recorded so a switch-ignition run can be checked for what it
            # claims: all policies in a run share a seed, so they must share an
            # ignition, and the ignitions must vary across runs.
            "ignition_pos": summary.get("ignition_pos"),
            "total_decision_steps": int(summary.get("total_decision_steps", 0)),
            "total_minutes": total_minutes,
            "mission_hours": total_minutes / 60.0,
            "burnt_area_m2": float(final_metrics.get("burnt_area_m2", 0.0)),
            "fire_cost_eur": float(final_metrics.get("fire_cost_eur", 0.0)),
            "casualties": float(final_metrics.get("casualties", 0.0)),
            "emissions_tonnes": float(final_metrics.get("emissions_tonnes", 0.0)),
            "moe_cumulative_reward": float(
                _terminal_moe(summary, reward_total)
            ),
            "propagation_factor": float(summary.get("propagation_factor", 0.0)),
            "episode_reward_sum": reward_total,
            "wall_seconds": time.perf_counter() - wall_start,
        }
    finally:
        env.close()


def _run_single_episode_random(
    *,
    scenario_path: Path,
    decision_interval_minutes: int,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate with random tactic assignment at every decision step."""
    env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        decision_interval_minutes=decision_interval_minutes,
        switch_scenario=False,
        **(env_kwargs or {}),
    )
    wall_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    try:
        _, info = env.reset(seed=seed, options={"sim_seed": seed})
        terminated = False
        truncated = False
        reward_total = 0.0

        # One random tactic tuple per action head at every decision step. The
        # head count comes from the env, not CONTROLLED_AGENT_COUNT: a grouped
        # env (--group-sizes) issues one tactic per group, not per aircraft, and
        # hardcoding the aircraft count silently builds the wrong-length action.
        action_nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        while not (terminated or truncated):
            action = rng.integers(
                low=0,
                high=action_nvec,
                dtype=np.int64,
            )
            _, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)

        summary = dict(info.get("episode_summary", {}))
        final_metrics = dict(summary.get("final_metrics", info.get("metrics", {})))
        total_minutes = float(summary.get("total_minutes", 0.0))
        return {
            "scenario": info.get("scenario"),
            "scenario_name": info.get("scenario_name"),
            "scenario_path": info.get("scenario_path"),
            "seed": seed,
            # Recorded so a switch-ignition run can be checked for what it
            # claims: all policies in a run share a seed, so they must share an
            # ignition, and the ignitions must vary across runs.
            "ignition_pos": summary.get("ignition_pos"),
            "total_decision_steps": int(summary.get("total_decision_steps", 0)),
            "total_minutes": total_minutes,
            "mission_hours": total_minutes / 60.0,
            "burnt_area_m2": float(final_metrics.get("burnt_area_m2", 0.0)),
            "fire_cost_eur": float(final_metrics.get("fire_cost_eur", 0.0)),
            "casualties": float(final_metrics.get("casualties", 0.0)),
            "emissions_tonnes": float(final_metrics.get("emissions_tonnes", 0.0)),
            "moe_cumulative_reward": float(
                _terminal_moe(summary, reward_total)
            ),
            "propagation_factor": float(summary.get("propagation_factor", 0.0)),
            "episode_reward_sum": reward_total,
            "wall_seconds": time.perf_counter() - wall_start,
        }
    finally:
        env.close()


def _print_run_table(rows: list[dict[str, Any]]) -> None:
    print(
        "policy | scenario | run | seed | mission_h | burnt_area_m2 | "
        "cost_eur | casualties | emissions_t | moe"
    )
    for row in rows:
        print(
            f"{row['policy_name']} | {row['scenario_name']} | "
            f"{row['run']} | {row['seed']} | "
            f"{row['mission_hours']:.2f} | {row['burnt_area_m2']:.2f} | "
            f"{row['fire_cost_eur']:.2f} | {row['casualties']:.2f} | "
            f"{row['emissions_tonnes']:.2f} | {row['moe_cumulative_reward']:.4f}"
        )


def _build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("policy_type", "")),
            str(row.get("policy_name", "")),
            str(row.get("scenario_name", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    print("\nPolicy + Scenario summary (mean over runs)")
    print(
        "policy_type | policy_name | scenario_name | runs | mean_burnt_area_m2 | "
        "mean_cost_eur | mean_casualties | mean_emissions_tonnes | "
        "mean_mission_h | mean_moe"
    )
    for (policy_type, policy_name, scenario_name), scenario_rows in grouped.items():
        n = len(scenario_rows)
        mean_burnt = sum(float(r["burnt_area_m2"]) for r in scenario_rows) / n
        mean_cost = sum(float(r["fire_cost_eur"]) for r in scenario_rows) / n
        mean_cas = sum(float(r["casualties"]) for r in scenario_rows) / n
        mean_emissions = (
            sum(float(r["emissions_tonnes"]) for r in scenario_rows) / n
        )
        mean_mission_h = sum(float(r["mission_hours"]) for r in scenario_rows) / n
        mean_moe = (
            sum(float(r["moe_cumulative_reward"]) for r in scenario_rows) / n
        )

        summary_row = {
            "policy_type": policy_type,
            "policy_name": policy_name,
            "scenario_name": scenario_name,
            "runs": n,
            "mean_burnt_area_m2": mean_burnt,
            "mean_cost_eur": mean_cost,
            "mean_casualties": mean_cas,
            "mean_emissions_tonnes": mean_emissions,
            "mean_mission_h": mean_mission_h,
            "mean_moe": mean_moe,
        }
        summary_rows.append(summary_row)
        print(
            f"{policy_type} | {policy_name} | {scenario_name} | {n} | "
            f"{mean_burnt:.2f} | {mean_cost:.2f} | {mean_cas:.2f} | "
            f"{mean_emissions:.2f} | {mean_mission_h:.2f} | {mean_moe:.4f}"
        )

    return summary_rows


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()


def _default_comparison_dir(scenario_paths: Sequence[Path]) -> Path:
    outputs_dir = SCENARIOS_DIR / "outputs"
    if len(scenario_paths) == 1:
        scenario_slug = _slugify(scenario_paths[0].stem)
        return outputs_dir / f"{scenario_slug}_comparison"
    return outputs_dir / "multi_scenario_comparison"


def _scenario_tag(scenario_paths: Sequence[Path]) -> str:
    if len(scenario_paths) == 1:
        return _slugify(scenario_paths[0].stem)
    tags = [_slugify(path.stem) for path in scenario_paths]
    return "_".join(tag for tag in tags if tag) or "multi_scenario"


def _plot_results(
    rows: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_name"]), []).append(row)

    metric_specs = (
        ("moe_cumulative_reward", "MoE", "moe"),
        ("burnt_area_m2", "Burnt Area (m^2)", "burnt_area"),
        ("fire_cost_eur", "Fire Cost (EUR)", "cost"),
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    for scenario_name, scenario_rows in grouped.items():
        per_policy: dict[str, list[dict[str, Any]]] = {}
        for row in scenario_rows:
            per_policy.setdefault(str(row.get("policy_name", "")), []).append(row)
        scenario_slug = _slugify(scenario_name)

        for metric_key, ylabel, metric_slug in metric_specs:
            fig, ax = plt.subplots(figsize=(10, 5))
            for policy_name, policy_rows in sorted(per_policy.items()):
                policy_rows = sorted(policy_rows, key=lambda r: int(r["run"]))
                x = [int(r["run"]) for r in policy_rows]
                y = [float(r[metric_key]) for r in policy_rows]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    label=policy_name,
                )
            ax.set_xlabel("Run")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{scenario_name} - {ylabel} by Run (policy comparison)")
            ax.grid(alpha=0.3)
            if len(per_policy) > 1:
                ax.legend(loc="best")
            fig.tight_layout()
            output_path = plots_dir / f"{scenario_slug}_{metric_slug}_comparison.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot: {output_path}")

        # Additional distribution view for MoE across runs.
        box_labels: list[str] = []
        box_values: list[list[float]] = []
        for policy_name, policy_rows in sorted(per_policy.items()):
            values = [float(r["moe_cumulative_reward"]) for r in policy_rows]
            if not values:
                continue
            box_labels.append(policy_name)
            box_values.append(values)
        if box_values:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(box_values, tick_labels=box_labels, showmeans=True)
            ax.set_xlabel("Policy")
            ax.set_ylabel("MoE")
            ax.set_title(f"{scenario_name} - MoE Distribution by Policy")
            ax.grid(alpha=0.3, axis="y")
            fig.autofmt_xdate(rotation=20)
            fig.tight_layout()
            output_path = plots_dir / f"{scenario_slug}_moe_boxplot.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot: {output_path}")


def _run_single_episode(
    *,
    model: PPO,
    scenario_path: Path,
    decision_interval_minutes: int,
    seed: int,
    deterministic: bool,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_kwargs = env_kwargs or {}
    raw_env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        decision_interval_minutes=decision_interval_minutes,
        switch_scenario=False,
        include_scenario_features=False,
        **env_kwargs,
    )
    model_obs_dim = int(model.observation_space.shape[0])
    env_obs_dim = int(raw_env.observation_space.shape[0])
    env: gym.Env
    if env_obs_dim == model_obs_dim:
        env = raw_env
    elif env_obs_dim + len(SCENARIO_FEATURES) == model_obs_dim:
        raw_env.close()
        env = WildfireHourlyEnv(
            scenario_path=scenario_path,
            decision_interval_minutes=decision_interval_minutes,
            switch_scenario=False,
            include_scenario_features=True,
            **env_kwargs,
        )
    elif env_obs_dim > model_obs_dim:
        drop_count = env_obs_dim - model_obs_dim
        legacy_indices = tuple(
            i for i, name in enumerate(STATE_FEATURES)
            if name in LEGACY_DROPPED_FEATURE_NAMES
        )
        if len(legacy_indices) == drop_count:
            # New features were inserted mid-vector since this legacy model
            # was trained. Drop those exact indices to preserve layout.
            env = ObservationDropIndicesWrapper(raw_env, drop_indices=legacy_indices)
        else:
            # Generic backward compatibility: drop leading features.
            env = ObservationDropPrefixWrapper(raw_env, drop_count=drop_count)
    else:
        raw_env.close()
        raise ValueError(
            "Model expects more observation features than environment provides: "
            f"model={model_obs_dim}, env={env_obs_dim}."
        )
    wall_start = time.perf_counter()
    try:
        obs, info = env.reset(seed=seed, options={"sim_seed": seed})
        terminated = False
        truncated = False
        reward_total = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)

        summary = dict(info.get("episode_summary", {}))
        final_metrics = dict(summary.get("final_metrics", info.get("metrics", {})))
        total_minutes = float(summary.get("total_minutes", 0.0))
        record = {
            "scenario": info.get("scenario"),
            "scenario_name": info.get("scenario_name"),
            "scenario_path": info.get("scenario_path"),
            "seed": seed,
            # Recorded so a switch-ignition run can be checked for what it
            # claims: all policies in a run share a seed, so they must share an
            # ignition, and the ignitions must vary across runs.
            "ignition_pos": summary.get("ignition_pos"),
            "total_decision_steps": int(summary.get("total_decision_steps", 0)),
            "total_minutes": total_minutes,
            "mission_hours": total_minutes / 60.0,
            "burnt_area_m2": float(final_metrics.get("burnt_area_m2", 0.0)),
            "fire_cost_eur": float(final_metrics.get("fire_cost_eur", 0.0)),
            "casualties": float(final_metrics.get("casualties", 0.0)),
            "emissions_tonnes": float(final_metrics.get("emissions_tonnes", 0.0)),
            "moe_cumulative_reward": float(
                _terminal_moe(summary, reward_total)
            ),
            "propagation_factor": float(summary.get("propagation_factor", 0.0)),
            "episode_reward_sum": reward_total,
            "wall_seconds": time.perf_counter() - wall_start,
        }
        return record
    finally:
        env.close()


def _init_eval_worker(
    device: str,
    decision_interval_minutes: int,
    deterministic: bool,
    env_kwargs: dict[str, Any] | None = None,
    group_sizes: tuple[int, ...] | None = None,
) -> None:
    global _WORKER_DEVICE
    global _WORKER_DECISION_INTERVAL_MINUTES
    global _WORKER_DETERMINISTIC
    global _WORKER_MODEL_CACHE
    global _WORKER_FIXED_ACTION_CACHE
    global _WORKER_ENV_KWARGS
    global _WORKER_GROUP_SIZES
    _WORKER_DEVICE = device
    _WORKER_DECISION_INTERVAL_MINUTES = int(decision_interval_minutes)
    _WORKER_DETERMINISTIC = bool(deterministic)
    _WORKER_MODEL_CACHE = {}
    _WORKER_FIXED_ACTION_CACHE = {}
    _WORKER_ENV_KWARGS = dict(env_kwargs or {})
    _WORKER_GROUP_SIZES = group_sizes


def _get_worker_model(model_path: str) -> PPO:
    model = _WORKER_MODEL_CACHE.get(model_path)
    if model is None:
        model = PPO.load(model_path, device=_WORKER_DEVICE, print_system_info=False)
        _WORKER_MODEL_CACHE[model_path] = model
    return model


def _get_worker_fixed_action(tactic_source_path: str) -> np.ndarray:
    action = _WORKER_FIXED_ACTION_CACHE.get(tactic_source_path)
    if action is None:
        action = _fixed_action_vector_from_scenario(
            Path(tactic_source_path),
            controlled_count=int(
                _WORKER_ENV_KWARGS.get(
                    "controlled_agent_count", CONTROLLED_AGENT_COUNT
                )
            ),
            group_sizes=_WORKER_GROUP_SIZES,
        )
        _WORKER_FIXED_ACTION_CACHE[tactic_source_path] = action
    return action


def _eval_task(
    task: tuple[str, str, str, str, int, int],
) -> dict[str, Any]:
    (
        policy_type,
        policy_source,
        policy_name,
        scenario_path_str,
        run_idx,
        seed,
    ) = task
    scenario_path = Path(scenario_path_str)

    if policy_type == "ppo_model":
        model = _get_worker_model(policy_source)
        record = _run_single_episode(
            model=model,
            scenario_path=scenario_path,
            decision_interval_minutes=_WORKER_DECISION_INTERVAL_MINUTES,
            seed=seed,
            deterministic=_WORKER_DETERMINISTIC,
            env_kwargs=_WORKER_ENV_KWARGS,
        )
    elif policy_type == "fixed_tactic":
        fixed_action = _get_worker_fixed_action(policy_source)
        record = _run_single_episode_fixed(
            fixed_action=fixed_action,
            scenario_path=scenario_path,
            decision_interval_minutes=_WORKER_DECISION_INTERVAL_MINUTES,
            seed=seed,
            env_kwargs=_WORKER_ENV_KWARGS,
        )
    elif policy_type == "random_tactic":
        record = _run_single_episode_random(
            scenario_path=scenario_path,
            decision_interval_minutes=_WORKER_DECISION_INTERVAL_MINUTES,
            seed=seed,
            env_kwargs=_WORKER_ENV_KWARGS,
        )
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    record["run"] = run_idx
    record["policy_type"] = policy_type
    record["policy_name"] = policy_name
    record["policy_source"] = policy_source
    return record


def main() -> None:
    default_output_arg = str(SCENARIOS_DIR / "outputs" / "ppo_test_results.csv")
    default_summary_output_arg = str(
        SCENARIOS_DIR / "outputs" / "ppo_test_summary.csv"
    )
    default_plots_dir_arg = str(SCENARIOS_DIR / "outputs" / "ppo_test_plots")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PPO model(s) and/or fixed-tactic baseline(s) on wildfire "
            "scenarios without additional training."
        )
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Single path to a trained PPO .zip model (backward-compatible flag).",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=None,
        help="One or more PPO .zip model paths to evaluate.",
    )
    parser.add_argument(
        "--fixed-tactic-scenarios",
        nargs="*",
        default=[],
        help=(
            "Scenario JSON name/path list used only to derive fixed tactics "
            "(e.g., Palisades_fixed_1.json Palisades_fixed_2.json)."
        ),
    )
    parser.add_argument(
        "--include-random-tactics",
        action="store_true",
        help=(
            "Include a random-tactics baseline: each controlled aircraft "
            "is assigned a random tactic tuple at every decision step."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SWITCH_SCENARIO_NAMES),
        help=(
            "Scenario JSON names/paths to evaluate. "
            "Default: Palisades copy, Pyrenees, Salamis."
        ),
    )
    parser.add_argument(
        "--runs-per-scenario",
        type=int,
        default=50,
        help="Number of evaluation simulations per scenario (default: 50).",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=DEFAULT_DECISION_INTERVAL_MINUTES,
        help="Simulation minutes between agent decisions (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for reproducible evaluation runs.",
    )
    parser.add_argument(
        "--random-seeds",
        action="store_true",
        help=(
            "Sample sim seeds uniformly from [0, --seed-pool) using --seed as the "
            "RNG seed (deterministic given --seed, but non-contiguous). All policies "
            "in the same run still share the same sampled seed."
        ),
    )
    parser.add_argument(
        "--seed-pool",
        type=int,
        default=2**31 - 1,
        help="Upper bound for --random-seeds sampling (default: 2**31 - 1).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default is deterministic policy).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model inference (e.g., cpu, cuda, mps, auto).",
    )
    parser.add_argument(
        "--output",
        default=default_output_arg,
        help="Per-run output path (.csv or .xlsx).",
    )
    parser.add_argument(
        "--results-name",
        default="results",
        help=(
            "Base filename (without extension) for per-run output when --output "
            "is not explicitly set. Scenario tag is appended automatically."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=default_summary_output_arg,
        help="Per-policy/per-scenario summary output path (.csv or .xlsx).",
    )
    parser.add_argument(
        "--summary-name",
        default="summary",
        help=(
            "Base filename (without extension) for summary output when "
            "--summary-output is not explicitly set. Scenario tag is appended "
            "automatically."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for evaluation (default: 1). "
            "Set >1 to parallelize runs."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        default=default_plots_dir_arg,
        help="Directory for plots (3 per scenario).",
    )
    # --- env construction: must match how the model was trained ---------------
    # Defaults reproduce the tester's historical behaviour so older commands
    # keep working. A model trained with anything else will fail on observation
    # or action shape unless these are set to the training values.
    env_group = parser.add_argument_group(
        "environment (match the model's training settings)"
    )
    env_group.add_argument(
        "--state-space",
        default="old",
        help="State-space variant used at training time (e.g. directional-2).",
    )
    env_group.add_argument(
        "--state-fire-fronts",
        type=int,
        default=5,
        help="Number of fire fronts in the state vector (default: 5).",
    )
    env_group.add_argument(
        "--water-set",
        default=None,
        help="Water-source subset: 1, 2 or off. Default: the scenario's own list.",
    )
    env_group.add_argument(
        "--switch-ignition",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4),
        help=(
            "Ignition randomisation mode, matching ppo_runnerv2's "
            "--switch-ignition-N. 0 (default) keeps the scenario's ignition."
        ),
    )
    env_group.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Aircraft group sizes, e.g. '5 5'. One tactic decision is issued "
            "per group. Omit for per-aircraft control."
        ),
    )
    env_group.add_argument(
        "--controlled-agent-count",
        type=int,
        default=None,
        help="Number of controlled aircraft (default: the env's own default).",
    )
    env_group.add_argument(
        "--cell-size",
        default=None,
        choices=("json", "code"),
        help="Cell-size source for the corrected flank geometry.",
    )
    args = parser.parse_args()

    # Only forward what was explicitly asked for, so the env keeps its own
    # defaults for anything untouched.
    env_kwargs: dict[str, Any] = {
        "state_space": args.state_space,
        "state_fire_fronts": args.state_fire_fronts,
        "switch_ignition_mode": args.switch_ignition,
    }
    if args.water_set is not None:
        env_kwargs["water_set"] = args.water_set
    if args.cell_size is not None:
        env_kwargs["cell_size_source"] = args.cell_size
    group_sizes: tuple[int, ...] | None = None
    if args.group_sizes:
        group_sizes = tuple(int(v) for v in args.group_sizes)
        env_kwargs["group_sizes"] = list(group_sizes)
        env_kwargs["tactic_distribution"] = TACTIC_DISTRIBUTION_GROUP
    if args.controlled_agent_count is not None:
        env_kwargs["controlled_agent_count"] = args.controlled_agent_count
    elif group_sizes:
        env_kwargs["controlled_agent_count"] = sum(group_sizes)

    if args.runs_per_scenario < 1:
        raise ValueError("--runs-per-scenario must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    model_inputs: list[str] = []
    if args.model_path:
        model_inputs.append(args.model_path)
    if args.model_paths:
        model_inputs.extend(args.model_paths)

    resolved_model_paths: list[Path] = []
    seen_models: set[str] = set()
    for model_input in model_inputs:
        model_path = Path(model_input)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        resolved = str(model_path.resolve())
        if resolved in seen_models:
            continue
        seen_models.add(resolved)
        resolved_model_paths.append(model_path)

    resolved_fixed_tactic_paths: list[Path] = []
    seen_fixed: set[str] = set()
    for tactic_input in args.fixed_tactic_scenarios:
        tactic_path = _resolve_scenario(tactic_input)
        resolved = str(tactic_path.resolve())
        if resolved in seen_fixed:
            continue
        seen_fixed.add(resolved)
        resolved_fixed_tactic_paths.append(tactic_path)

    policies: list[dict[str, str]] = []
    for model_path in resolved_model_paths:
        policies.append(
            {
                "policy_type": "ppo_model",
                "policy_name": model_path.stem,
                "policy_source": str(model_path),
            }
        )
    for tactic_path in resolved_fixed_tactic_paths:
        policies.append(
            {
                "policy_type": "fixed_tactic",
                "policy_name": f"fixed_{tactic_path.stem}",
                "policy_source": str(tactic_path),
            }
        )
    if args.include_random_tactics:
        policies.append(
            {
                "policy_type": "random_tactic",
                "policy_name": "random_tactics",
                "policy_source": "random",
            }
        )
    if not policies:
        raise ValueError(
            "No evaluation policy specified. Provide --model-path/--model-paths "
            "and/or --fixed-tactic-scenarios."
        )

    scenario_paths = tuple(_resolve_scenario(name) for name in args.scenarios)
    comparison_dir = _default_comparison_dir(scenario_paths)
    scenario_tag = _scenario_tag(scenario_paths)
    results_name = _slugify(args.results_name) or "results"
    summary_name = _slugify(args.summary_name) or "summary"
    if args.output == default_output_arg:
        args.output = str(comparison_dir / f"{results_name}_{scenario_tag}.csv")
    if args.summary_output == default_summary_output_arg:
        args.summary_output = str(comparison_dir / f"{summary_name}_{scenario_tag}.csv")
    if args.plots_dir == default_plots_dir_arg:
        args.plots_dir = str(comparison_dir / "plots")

    rows: list[dict[str, Any]] = []
    deterministic = not args.stochastic
    eval_start = time.perf_counter()

    if args.random_seeds:
        seed_rng = np.random.default_rng(args.seed)
        sampled_seeds = seed_rng.integers(
            0, args.seed_pool, size=(len(scenario_paths), args.runs_per_scenario)
        )

    tasks: list[tuple[str, str, str, str, int, int]] = []
    for scenario_idx, scenario_path in enumerate(scenario_paths):
        for run_idx in range(1, args.runs_per_scenario + 1):
            if args.random_seeds:
                seed = int(sampled_seeds[scenario_idx, run_idx - 1])
            else:
                seed = args.seed + scenario_idx * args.runs_per_scenario + (run_idx - 1)
            for policy in policies:
                tasks.append(
                    (
                        policy["policy_type"],
                        policy["policy_source"],
                        policy["policy_name"],
                        str(scenario_path),
                        run_idx,
                        seed,
                    )
                )

    total_runs = len(tasks)
    print(
        "Policies: "
        + ", ".join(
            f"{p['policy_name']}[{p['policy_type']}]" for p in policies
        )
    )

    if args.num_workers == 1:
        model_cache: dict[str, PPO] = {}
        fixed_action_cache: dict[str, np.ndarray] = {}
        for idx, task in enumerate(tasks, start=1):
            (
                policy_type,
                policy_source,
                policy_name,
                scenario_path_str,
                run_idx,
                seed,
            ) = task
            scenario_path = Path(scenario_path_str)
            print(
                f"[{idx}/{total_runs}] Evaluating {policy_name} on "
                f"{scenario_path.name} run={run_idx} seed={seed}"
            )
            if policy_type == "ppo_model":
                model = model_cache.get(policy_source)
                if model is None:
                    model = PPO.load(
                        policy_source,
                        device=args.device,
                        print_system_info=False,
                    )
                    model_cache[policy_source] = model
                row = _run_single_episode(
                    model=model,
                    scenario_path=scenario_path,
                    decision_interval_minutes=args.decision_interval_minutes,
                    seed=seed,
                    deterministic=deterministic,
                    env_kwargs=env_kwargs,
                )
            elif policy_type == "fixed_tactic":
                fixed_action = fixed_action_cache.get(policy_source)
                if fixed_action is None:
                    fixed_action = _fixed_action_vector_from_scenario(
                        Path(policy_source),
                        controlled_count=int(
                            env_kwargs.get(
                                "controlled_agent_count", CONTROLLED_AGENT_COUNT
                            )
                        ),
                        group_sizes=group_sizes,
                    )
                    fixed_action_cache[policy_source] = fixed_action
                row = _run_single_episode_fixed(
                    fixed_action=fixed_action,
                    scenario_path=scenario_path,
                    decision_interval_minutes=args.decision_interval_minutes,
                    seed=seed,
                    env_kwargs=env_kwargs,
                )
            elif policy_type == "random_tactic":
                row = _run_single_episode_random(
                    scenario_path=scenario_path,
                    decision_interval_minutes=args.decision_interval_minutes,
                    seed=seed,
                    env_kwargs=env_kwargs,
                )
            else:
                raise ValueError(f"Unsupported policy type: {policy_type}")

            row["run"] = run_idx
            row["policy_type"] = policy_type
            row["policy_name"] = policy_name
            row["policy_source"] = policy_source
            rows.append(row)
    else:
        max_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
        print(
            f"Running parallel evaluation with {max_workers} worker process(es)."
        )
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_eval_worker,
            initargs=(
                args.device,
                args.decision_interval_minutes,
                deterministic,
                env_kwargs,
                group_sizes,
            ),
        ) as executor:
            future_map = {executor.submit(_eval_task, task): task for task in tasks}
            completed = 0
            for future in as_completed(future_map):
                completed += 1
                (
                    _policy_type,
                    _policy_source,
                    policy_name,
                    scenario_path_str,
                    run_idx,
                    seed,
                ) = future_map[future]
                print(
                    f"[{completed}/{total_runs}] Evaluated {policy_name} on "
                    f"{Path(scenario_path_str).name} run={run_idx} seed={seed}"
                )
                rows.append(future.result())

    scenario_order = {path.name: idx for idx, path in enumerate(scenario_paths)}
    policy_order = {
        (p["policy_type"], p["policy_name"], p["policy_source"]): idx
        for idx, p in enumerate(policies)
    }
    rows.sort(
        key=lambda r: (
            scenario_order.get(str(r.get("scenario_name", "")), 9999),
            policy_order.get(
                (
                    str(r.get("policy_type", "")),
                    str(r.get("policy_name", "")),
                    str(r.get("policy_source", "")),
                ),
                9999,
            ),
            int(r.get("run", 0)),
        )
    )

    duration = time.perf_counter() - eval_start
    _print_run_table(rows)
    summary_rows = _build_summary(rows)

    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    _write_records(rows, output_path)
    _write_records(summary_rows, summary_output_path)
    _plot_results(rows, Path(args.plots_dir))

    print(f"\nSaved per-run results to {output_path}")
    print(f"Saved summary results to {summary_output_path}")
    print(f"Saved plots to {Path(args.plots_dir)}")
    print(f"Evaluation finished in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
