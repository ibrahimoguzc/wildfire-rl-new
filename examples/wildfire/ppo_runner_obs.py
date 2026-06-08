#!/usr/bin/env python3
"""
Train a PPO agent to select hourly suppression tactics on the Palisades scenario.

The environment exposes the same set of state variables logged by
``hourly_metrics.py`` and expects a discrete tactic selection for the first two
aircraft. Each environment step advances the simulation by one simulated hour;
the reward is the mission-effectiveness score (MoE) accumulated over that hour.

Requirements:
    pip install gymnasium stable-baselines3 torch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence
from datetime import timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from scipy.spatial.distance import cdist
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SELECT_POI_TABLE,
    SelectPOIType,
)
from examples.wildfire.firefighter_model.tactic_pieces.suppress import (
    SUPPRESS_TABLE,
    SuppressType,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    TRACK_POI_TABLE,
    TrackPOIType,
)
from examples.wildfire.firefighter_model.follower import (
    DestinationType,
    PayloadStatus,
)
from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation

# Mission effectiveness scaling constants (per-hour scaling)
experiment_factor = 1
BURN_AREA_NORM = 90_870_000.0/experiment_factor
COST_NORM = 191_106_000_000.0/experiment_factor
EMISSION_NORM = 131_224.0/experiment_factor
MOE_WEIGHT = 0.33
CONTROLLED_AGENT_COUNT = 3
AGENT_FEATURE_COUNT = 6
DEFAULT_DECISION_INTERVAL_MINUTES = 10
#CASUALTY_NORM = 1_655.0


STATE_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "time_to_sunset_min",
    "distance_to_fire_line_m",
    "distance_to_water_m",
    "active_firefront_count",
    "burnt_area_fraction",
    "fire_center_x",
    "fire_center_y",
    "spread_angle_deg",
]

SELECT_OPTIONS: tuple[SelectPOIType, ...] = (
    SelectPOIType.WATER,
    SelectPOIType.VEGETATION,
    SelectPOIType.TOPOGRAPHY,
    SelectPOIType.INDIRECT,
)
TRACK_OPTIONS: tuple[TrackPOIType, ...] = (
    TrackPOIType.DIRECT,
    TrackPOIType.INDIRECT,
    TrackPOIType.FOLLOW_FIREFRONT,
)
SUPPRESS_OPTIONS: tuple[SuppressType, ...] = (
    SuppressType.DIRECT,
    SuppressType.INDIRECT,
)

TACTIC_COMBINATIONS: tuple[
    tuple[SelectPOIType, TrackPOIType, SuppressType], ...
] = tuple(
    (select, track, suppress)
    for select in SELECT_OPTIONS
    for track in TRACK_OPTIONS
    for suppress in SUPPRESS_OPTIONS
)


def _resolve_scenario(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    candidate = SCENARIOS_DIR / "inputs" / path_or_name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to locate scenario file: {path_or_name}")


@dataclass
class Metrics:
    burnt_area: float
    cost: float
    casualties: float
    emissions: float


def metrics_to_dict(metrics: Metrics) -> dict[str, float]:
    return {
        "burnt_area_m2": metrics.burnt_area,
        "fire_cost_eur": metrics.cost,
        "casualties": metrics.casualties,
        "emissions_tonnes": metrics.emissions,
    }


def cumulative_moe_reward(burnt_area: float, cost: float, emissions: float) -> float:
    return 1.0 - (
        MOE_WEIGHT * (burnt_area / BURN_AREA_NORM)
        + MOE_WEIGHT * (cost / COST_NORM)
        + MOE_WEIGHT * (emissions / EMISSION_NORM)
    )


class WildfireHourlyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment bridging the wildfire simulation and PPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: Path,
        max_steps: int | None = None,
        decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
    ): 
        super().__init__()
        self.scenario_path = scenario_path
        with scenario_path.open() as handle:
            self.parameters = WildfireParameters.model_validate_json(handle.read())

        self.decision_interval_minutes = decision_interval_minutes
        self.decision_interval = timedelta(minutes=decision_interval_minutes)
        default_steps = math.ceil(
            self.parameters.max_runtime / self.decision_interval.total_seconds()
        )
        self.max_steps = default_steps if max_steps is None else max_steps

        self.controlled_agent_count = CONTROLLED_AGENT_COUNT
        self.agent_feature_count = AGENT_FEATURE_COUNT
        obs_dim = len(STATE_FEATURES) + self.controlled_agent_count * self.agent_feature_count
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [len(TACTIC_COMBINATIONS)] * self.controlled_agent_count
        )

        self.np_random, _ = gym.utils.seeding.np_random()
        self.sim: WildfireSimulation | None = None
        self.water_positions: np.ndarray = np.empty((0, 2), dtype=float)
        self.fire_grid_area: float = 0.0
        self.current_step: int = 0
        self.prev_metrics: Metrics | None = None
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.last_info: dict[str, Any] = {}
        self.current_sim_seed: int | None = None

    # -- Environment helpers -------------------------------------------------
    def _init_simulation(self, seed: int) -> None:
        self.sim = WildfireSimulation(parameters=self.parameters, seed=seed)
        self.sim.wildfire.ignite(self.sim.ignition_centers)
        water_sources = self.sim.firefighters.water_sources
        self.water_positions = (
            np.array([ws.pos for ws in water_sources], dtype=float)
            if water_sources
            else np.empty((0, 2), dtype=float)
        )
        self.fire_grid_area = (
            self.sim.wildfire.fire_states.size
            * (self.sim.parameters.cell_size**2)
        )
        self.prev_metrics = self._compute_metrics()
        self.current_step = 0

        self.cumulative_reward = 0.0
        self.done = False
        self.current_sim_seed = seed

    def _compute_metrics(self) -> Metrics:
        assert self.sim is not None
        burnt = float(self.sim.wildfire.burnt_area)
        cost = float(self.sim.total_fire_cost)
        casualties = float(self.sim.total_casualties)
        emissions = float(self.sim.total_fire_emissions)
        return Metrics(burnt, cost, casualties, emissions)

    def _mission_time(self) -> float:
        assert self.sim is not None
        return self.sim.timer.mission_runtime.total_seconds()

    def _mission_complete(self) -> bool:
        assert self.sim is not None
        fire_remaining = bool(self.sim.wildfire.fire_positions.size)
        return (not fire_remaining) or self.sim.is_stopped.is_set()

    def _apply_actions(
        self,
        action: Sequence[int],
    ) -> None:
        assert self.sim is not None
        combinations = [TACTIC_COMBINATIONS[idx] for idx in action]
        agents = self.sim.firefighters.firefighters[: self.controlled_agent_count]
        for agent, (select, track, suppress) in zip(
            agents, combinations[: len(agents)], strict=False
        ):
            agent.tactic.select_poi = SELECT_POI_TABLE[select]()
            agent.tactic.track_poi = TRACK_POI_TABLE[track]()
            agent.tactic.suppress = SUPPRESS_TABLE[suppress]()
            agent.force_tactic_swap = True

    def _advance_time_window(self) -> None:
        assert self.sim is not None
        target_time_seconds = (
            ((self.current_step + 1)) * self.decision_interval.total_seconds()
        )
        while (
            self._mission_time() < target_time_seconds
            and self.current_step < self.max_steps
            and not self._mission_complete()
        ):
            try:
                self.sim.step(force=True)
            except RuntimeError as err:
                if "not started cannot be stopped" in str(err):
                    self.done = True
                    break
                raise
        if self._mission_time() >= self.parameters.max_runtime:
            self.done = True

    def _compute_state(self) -> np.ndarray:
        assert self.sim is not None
        atmosphere = self.sim.atmosphere
        mission_time = self.sim.timer.mission_time
        burning_indices = self.sim.wildfire.burning_indices
        burning_count = int(burning_indices.shape[0])

        fire_center_x = math.nan
        fire_center_y = math.nan
        spread_angle = math.nan
        distance_fire_line = math.nan
        distance_water = math.nan

        if burning_count:
            fire_positions = self.sim.wildfire.fire_positions
            centroid = fire_positions.mean(axis=0)
            fire_center_x = float(centroid[0])
            fire_center_y = float(centroid[1])

            centroid_idx = burning_indices.mean(axis=0)
            spread_rates = self.sim.wildfire.get_spread_rates(burning_indices)
            fastest_idx = burning_indices[int(np.argmax(spread_rates))]
            angle = math.degrees(
                math.atan2(
                    fastest_idx[0] - centroid_idx[0],
                    fastest_idx[1] - centroid_idx[1],
                )
            )
            spread_angle = float((angle + 360.0) % 360.0)

            if self.water_positions.size:
                distance_water = float(
                    np.linalg.norm(self.water_positions - centroid, axis=1).min()
                )

            fire_block_indices = self.sim.firefighters.fire_block_indices
            if (
                fire_block_indices is not None
                and self.sim.firefighters.current_block_index > 0
            ):
                segment = fire_block_indices[
                    : self.sim.firefighters.current_block_index
                ]
                if segment.size:
                    distance_fire_line = float(
                        self.sim.parameters.cell_size
                        * cdist(burning_indices, segment).min()
                    )

        burnt_area = float(self.sim.wildfire.burnt_area)
        burnt_fraction = (
            burnt_area / self.fire_grid_area if self.fire_grid_area else 0.0
        )

        time_to_sunset = max(
            (self.sim.atmosphere.next_sunset - mission_time).total_seconds() / 60.0,
            0.0,
        )

        state_features: list[float] = [
            float(atmosphere.temperature),
            float(atmosphere.relative_humidity),
            float(atmosphere.wind_speed),
            float((atmosphere.wind_aspect + 360.0) % 360.0),
            float(time_to_sunset),
            float(distance_fire_line)
            if not math.isnan(distance_fire_line)
            else 0.0,
            float(distance_water) if not math.isnan(distance_water) else 0.0,
            float(burning_count),
            float(burnt_fraction),
            float(fire_center_x) if not math.isnan(fire_center_x) else 0.0,
            float(fire_center_y) if not math.isnan(fire_center_y) else 0.0,
            float(spread_angle) if not math.isnan(spread_angle) else 0.0,
        ]

        env_width, env_height = self.sim.environment.dimensions
        env_width = float(env_width) if env_width else 1.0
        env_height = float(env_height) if env_height else 1.0

        agents = self.sim.firefighters.firefighters
        for idx in range(self.controlled_agent_count):
            if idx < len(agents):
                agent = agents[idx]
                pos_x, pos_y = agent.pos
                norm_x = float(pos_x) / env_width
                norm_y = float(pos_y) / env_height
                norm_x = float(max(0.0, min(norm_x, 1.0)))
                norm_y = float(max(0.0, min(norm_y, 1.0)))
                altitude = float(agent.altitude)
                has_payload = (
                    1.0
                    if agent.payload_status == PayloadStatus.ONBOARD
                    else 0.0
                )

                remaining_propellant_mass = agent.propulsion.propellant_mass
                max_propellant_mass = agent.propulsion.max_propellant_mass
                propellant_fraction = (
                    remaining_propellant_mass / max_propellant_mass
                    if max_propellant_mass > 0
                    else 0.0
                )

                nearest_airport, _ = agent.get_nearest_airport()
                required_propellant = agent.estimate_propellant_for_journey(
                    agent.pos,
                    nearest_airport.pos,
                    DestinationType.BASE,
                )
                required_mass = agent.propulsion.propellant_to_mass(
                    required_propellant
                )
                return_margin = (
                    (remaining_propellant_mass - required_mass)
                    / max_propellant_mass
                    if max_propellant_mass > 0
                    else 0.0
                )
                return_margin = float(max(-1.0, min(return_margin, 1.0)))

                state_features.extend(
                    [
                        norm_x,
                        norm_y,
                        altitude,
                        has_payload,
                        propellant_fraction,
                        return_margin,
                    ]
                )
            else:
                state_features.extend([0.0] * self.agent_feature_count)

        return np.array(state_features, dtype=np.float32)

    def _compute_reward_and_info(
        self,
        new_metrics: Metrics,
    ) -> tuple[float, dict[str, Any]]:
        assert self.prev_metrics is not None
        deltas = Metrics(
            new_metrics.burnt_area - self.prev_metrics.burnt_area,
            new_metrics.cost - self.prev_metrics.cost,
            new_metrics.casualties - self.prev_metrics.casualties,
            new_metrics.emissions - self.prev_metrics.emissions,
        )
        reward = 1.0 - (
            MOE_WEIGHT * (deltas.burnt_area / BURN_AREA_NORM)
            + MOE_WEIGHT * (deltas.cost / COST_NORM)
            + MOE_WEIGHT * (deltas.emissions / EMISSION_NORM)
        )
        decision_step = self.current_step + 1
        elapsed_minutes = decision_step * self.decision_interval.total_seconds() / 60.0
        info = {
            "decision_step": decision_step,
            "elapsed_minutes": elapsed_minutes,
            "decision_interval_minutes": self.decision_interval.total_seconds() / 60.0,
            "metrics": new_metrics,
            "deltas": deltas,
            "reward_moe": reward,
            "sim_seed": self.current_sim_seed,
        }
        self.prev_metrics = new_metrics
        self.cumulative_reward += reward
        return reward, info

    # -- Gym API -------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
        elif self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random()

        sim_seed = options.get("sim_seed") if options else None
        if sim_seed is None:
            sim_seed = int(self.np_random.integers(0, 1_000_000))

        self._init_simulation(int(sim_seed))
        observation = self._compute_state()
        self.last_info = {
            "decision_step": 0,
            "elapsed_minutes": 0.0,
            "decision_interval_minutes": self.decision_interval_minutes,
            "metrics": self.prev_metrics,
            "deltas": Metrics(0.0, 0.0, 0.0, 0.0),
            "reward_moe": 0.0,
            "sim_seed": sim_seed,
        }
        return observation, self.last_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.sim is not None
        if self.done:
            raise RuntimeError("step called on terminated environment.")

        self._apply_actions(action)
        self._advance_time_window()
        metrics = self._compute_metrics()
        reward, info = self._compute_reward_and_info(metrics)
        self.current_step += 1

        terminated = (
            self.done
            or self.current_step >= self.max_steps
            or self._mission_complete()
        )
        truncated = False
        observation = self._compute_state()

        if terminated:
            info["episode_summary"] = {
                "total_decision_steps": self.current_step,
                "total_minutes": self.current_step
                * (self.decision_interval.total_seconds() / 60.0),
                "total_reward": self.cumulative_reward,
                "final_metrics": metrics_to_dict(metrics),
            }

        self.last_info = info
        return observation, reward, terminated, truncated, info



def run_policy_evaluation(
    model: PPO,
    scenario_path: Path,
    seed: int,
    *,
    decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
    collect_steps: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a full simulation with the current policy without learning."""

    env = WildfireHourlyEnv(
        scenario_path,
        decision_interval_minutes=decision_interval_minutes,
    )
    obs, info = env.reset(options={"sim_seed": seed})
    done = False
    step_records: list[dict[str, Any]] = []
    last_info = info
    metrics_dict: dict[str, float] = {}
    deltas_dict: dict[str, float] = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action)
        action_labels = [
            TACTIC_COMBINATIONS[int(idx)] for idx in action
        ]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        metrics = info.get("metrics")
        deltas = info.get("deltas")
        metrics_dict = (
            metrics_to_dict(metrics) if isinstance(metrics, Metrics) else {}
        )
        deltas_dict = (
            metrics_to_dict(deltas) if isinstance(deltas, Metrics) else {}
        )
        if collect_steps:
            record = {
                "phase": "evaluation",
                "env_index": 0,
                "sim_seed": seed,
                "decision_step": info.get("decision_step"),
                "elapsed_minutes": info.get("elapsed_minutes"),
                "decision_interval_minutes": info.get(
                    "decision_interval_minutes", decision_interval_minutes
                ),
                "reward_moe": info.get("reward_moe"),
                "step_reward": reward,
        }
            for idx, combo in enumerate(action_labels[: CONTROLLED_AGENT_COUNT]):
                record[f"agent_{idx}_select_poi"] = combo[0].value
                record[f"agent_{idx}_track_poi"] = combo[1].value
                record[f"agent_{idx}_suppress"] = combo[2].value
            record.update(metrics_dict)
            record.update({f"{key}_delta": value for key, value in deltas_dict.items()})
            step_records.append(record)

        last_info = info

    episode_summary = last_info.get("episode_summary", {})
    final_metrics = episode_summary.get("final_metrics", metrics_dict)
    total_minutes = episode_summary.get("total_minutes", 0.0)
    summary = {
        "sim_seed": seed,
        "total_decision_steps": episode_summary.get("total_decision_steps"),
        "total_minutes": total_minutes,
        "total_hours": total_minutes / 60.0 if total_minutes else 0.0,
        **final_metrics,
    }
    summary["moe_cumulative_reward"] = cumulative_moe_reward(
        summary.get("burnt_area_m2", 0.0),
        summary.get("fire_cost_eur", 0.0),
        summary.get("emissions_tonnes", 0.0),
    )
    summary["decision_interval_minutes"] = decision_interval_minutes
    env.close()
    return summary, step_records


class TrainingLoggerWithEvaluations(BaseCallback):
    """Capture per-step training data and trigger periodic evaluation runs."""

    def __init__(
        self,
        scenario_path: Path,
        period: int = 10,
        decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.period = period
        self.training_step_records: list[dict[str, Any]] = []
        self.evaluation_runs: list[dict[str, Any]] = []
        self.completed_episodes = 0
        self._next_seed = 0
        self.controlled_agent_count = CONTROLLED_AGENT_COUNT
        self.decision_interval_minutes = decision_interval_minutes
        self.decision_interval_minutes = decision_interval_minutes

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        for env_idx, (info, action, reward) in enumerate(
            zip(infos, actions, rewards, strict=False)
        ):
            action = np.asarray(action)
            action_labels = [
                TACTIC_COMBINATIONS[int(idx)] for idx in action
            ]
            metrics = info.get("metrics")
            deltas = info.get("deltas")
            metrics_dict = (
                metrics_to_dict(metrics) if isinstance(metrics, Metrics) else {}
            )
            deltas_dict = (
                metrics_to_dict(deltas) if isinstance(deltas, Metrics) else {}
            )
            record = {
                "phase": "training",
                "env_index": env_idx,
                "sim_seed": info.get("sim_seed"),
                "decision_step": info.get("decision_step"),
                "elapsed_minutes": info.get("elapsed_minutes"),
                "decision_interval_minutes": info.get(
                    "decision_interval_minutes", self.decision_interval_minutes
                ),
                "reward_moe": info.get("reward_moe"),
                "step_reward": reward,
            }
            for idx, combo in enumerate(action_labels[: self.controlled_agent_count]):
                record[f"agent_{idx}_select_poi"] = combo[0].value
                record[f"agent_{idx}_track_poi"] = combo[1].value
                record[f"agent_{idx}_suppress"] = combo[2].value
            record.update(metrics_dict)
            record.update({f"{key}_delta": value for key, value in deltas_dict.items()})
            self.training_step_records.append(record)

            if "episode_summary" in info:
                self.completed_episodes += 1
                if self.completed_episodes % self.period == 0:
                    summary, _ = run_policy_evaluation(
                        self.model,
                        self.scenario_path,
                        seed=self._next_seed,
                        decision_interval_minutes=self.decision_interval_minutes,
                        collect_steps=False,
                    )
                    summary["evaluation_round"] = len(self.evaluation_runs) + 1
                    self.evaluation_runs.append(summary)
                    self._next_seed += 1
        return True

    def _on_training_end(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the Palisades scenario.")
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON (name in inputs/ or absolute path).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Total PPO timesteps to collect during training (default: 1680).",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=DEFAULT_DECISION_INTERVAL_MINUTES,
        help="Simulation minutes between agent decisions (default: 10).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--log-output",
        help="Optional Excel path for action/reward log.",
    )

    parser.add_argument(
        "--eval-summary-output",
        help="Optional Excel path to store periodic evaluation summaries.",
    )
    parser.add_argument(
        "--save-model",
        help="Optional path to save the trained PPO policy.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for PPO (e.g., auto, cpu, cuda, mps).",
    )
    args = parser.parse_args()  

    scenario_path = _resolve_scenario(args.scenario)
    env = WildfireHourlyEnv(
        scenario_path,
        decision_interval_minutes=args.decision_interval_minutes,
    )
    default_timesteps = args.timesteps if args.timesteps is not None else 1680
    total_timesteps = max(default_timesteps, 1)
    rollout_steps = min(total_timesteps, env.max_steps)

    step_logger = TrainingLoggerWithEvaluations(
        scenario_path=scenario_path,
        period=10,
        decision_interval_minutes=args.decision_interval_minutes,
    )
    device = args.device
    if device != "auto":
        try:
            import torch
        except ImportError:
            print("PyTorch not available, falling back to CPU.")
            device = "cpu"
        else:
            if device == "mps" and not torch.backends.mps.is_available():
                print("MPS device requested but not available; falling back to CPU.")
                device = "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA device requested but not available; falling back to CPU.")
                device = "cpu"
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        verbose=1,
        tensorboard_log=None,
        n_steps=rollout_steps,
        device=device,
        policy_kwargs = policy_kwargs,
    )
    start_time = pd.Timestamp.now()
    model.learn(total_timesteps=total_timesteps, callback=step_logger)
    training_duration = (pd.Timestamp.now() - start_time).total_seconds()
    env.close()

    print(f"Training complete in {training_duration:.2f} seconds.")

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Trained model saved to {save_path}")

    if step_logger.evaluation_runs:
        eval_path = (
            Path(args.eval_summary_output)
            if args.eval_summary_output
            else SCENARIOS_DIR / "outputs" / "ppo_periodic_evaluations.xlsx"
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(step_logger.evaluation_runs).to_excel(eval_path, index=False)
        print(f"Periodic evaluation summary written to {eval_path}")

    all_step_records = step_logger.training_step_records
    if all_step_records:
        output_path = (
            Path(args.log_output)
            if args.log_output
            else SCENARIOS_DIR
            / "outputs"
            / "ppo_actions_rewards.xlsx"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_step_records).to_excel(output_path, index=False)
        print(f"\nDetailed action/reward log written to {output_path}")


if __name__ == "__main__":
    main()
