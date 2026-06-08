#!/usr/bin/env python3

"""
bismillahirrahmanirrahim, elhamdulillah, esselatu vesselamu ala rasulillah
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
import csv
import gc
import json
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Sequence
from datetime import timedelta

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _process_rss_bytes() -> int:
    """Return current process RSS in bytes; 0 if unavailable."""
    if _psutil is not None:
        try:
            return int(_psutil.Process().memory_info().rss)
        except Exception:
            return 0
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sosid.environment.terrain import FEATURES_COLOR_TABLE, TerrainTypes
from sosid.model.transform import gps_to_mercator, gps_to_pos, index_to_pos, pos_to_gps


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
from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import (
    IgnitionCenterInput,
    ProtectionLocationInput,
    WildfireParameters,
    WildfireSimulation,
    _TerrainParametersCache,
)

# Scenario-specific normalization values.
# Burnt area is in hectares converted to m^2, cost is in million EUR converted to EUR.
SCENARIO_MOE_NORMS: dict[str, dict[str, float]] = {
    "salamis": {
        "burnt_area_norm": 4146.0 * 10_000.0,
        "cost_norm": 13_993.0 * 1_000_000.0,
        "emission_norm": 714_009.0,
        "casualty_norm": 6_000.0,
    },
    "pyrenees": {
        "burnt_area_norm": 9938.0 * 10_000.0,
        "cost_norm": 17_509.0 * 1_000_000.0,
        "emission_norm": 2_364_064.0,
        "casualty_norm": 10_000.0,
    },
    "palisades": {
        "burnt_area_norm": 9087.0 * 10_000.0,
        "cost_norm": 191_106.0 * 1_000_000.0,
        "emission_norm": 131_224.0,
        "casualty_norm": 300_000.0,
    },
}

MOE_WEIGHT = 0.25
# Upper bound for max-spread-rate normalization. spread_rates are in
# m/min (Rothermel-based; see fire_model/jit_funcs/cpu.py). Typical
# wildfire spread is 5-15 m/min; extreme crown-fire conditions reach
# 30-50 m/min. Values above this cap clip to 1.0 in the state vector.
MAX_SPREAD_RATE_NORM_MPM = 30.0
CONTROLLED_AGENT_COUNT = 5
AGENT_FEATURE_COUNT = 2
DEFAULT_DECISION_INTERVAL_MINUTES = 10
IGNITION_BOUNDARY_MARGIN_RATIO = 0.3
IGNITION_BOX_HALF_SIZE = 50  # half-side of 100×100 candidate box in grid cells
IGNITION_URBAN_BUFFER_M = 500.0  # min distance from any urban cell, meters
# --switch-ignition-2: center map box whose edges sit `IGNITION_V2_MARGIN_RATIO`
# of the map edge length away from each map boundary. Default 0.25 → box edge
# = (1 − 2·0.25) · map_edge = half the map.
IGNITION_V2_MARGIN_RATIO = 0.25

ROLLOUT_STEPS_PER_ENV = 144
NUM_MINIBATCH = 12

if ROLLOUT_STEPS_PER_ENV <= 0:
    raise ValueError("ROLLOUT_STEPS_PER_ENV must be positive.")
if NUM_MINIBATCH <= 0:
    raise ValueError("NUM_MINIBATCH must be positive.")
LOG_INTERVAL_SUMMARY_EPISODES = 100
LOG_INTERVAL_STEPS_EPISODES = 1000
LR_DECAY_EXPONENT = 0.70
POI_CANDIDATE_QUANTILE = 0.95
MAX_POI_CANDIDATES = 4000
SWITCH_SCENARIO_NAMES = (
    "Palisades copy.json",
    "Pyrenees.json",
    "Salamis.json",
)


def _make_lr_schedule(initial_lr: float, decay_exponent: float = LR_DECAY_EXPONENT):
    """Return a callable learning-rate schedule decaying with training progress.

    schedule(p) = initial_lr * p^decay_exponent
        decay_exponent = 0.65 (default): aggressive early decay, slow late
        decay_exponent = 1.0:            linear decay (sb3's default behaviour)
        decay_exponent = 0.0:            constant LR (no decay)
    """
    base_lr = float(initial_lr)
    exponent = float(decay_exponent)

    def schedule(progress_remaining: float) -> float:
        progress = max(progress_remaining, 1e-8)
        return base_lr * (progress**exponent)

    return schedule


def _make_env_factory(
    scenario_path: Path,
    decision_interval_minutes: int,
    fire_detection_delay_minutes: float | None,
    switch_scenario: bool,
    switch_scenario_paths: tuple[Path, ...] | None,
    aircraft_source_scenario_path: Path | None,
    switch_ignition_mode: int,
    seed: int,
    ts_budget_per_scenario: float | None = None,
    profile_timings: bool = False,
    gc_collect_on_reset: bool = False,
    include_scenario_features: bool | None = None,
    controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
):
    def _init() -> WildfireHourlyEnv:
        env = WildfireHourlyEnv(
            scenario_path,
            decision_interval_minutes=decision_interval_minutes,
            fire_detection_delay_minutes=fire_detection_delay_minutes,
            switch_scenario=switch_scenario,
            switch_scenario_paths=switch_scenario_paths,
            aircraft_source_scenario_path=aircraft_source_scenario_path,
            switch_ignition_mode=switch_ignition_mode,
            ts_budget_per_scenario=ts_budget_per_scenario,
            profile_timings=profile_timings,
            gc_collect_on_reset=gc_collect_on_reset,
            include_scenario_features=include_scenario_features,
            controlled_agent_count=controlled_agent_count,
        )
        env.reset(seed=seed)
        return env

    return _init


def _build_vector_env(
    num_envs: int,
    scenario_path: Path,
    decision_interval_minutes: int,
    fire_detection_delay_minutes: float | None,
    switch_scenario: bool,
    switch_scenario_paths: tuple[Path, ...] | None,
    aircraft_source_scenario_path: Path | None,
    switch_ignition_mode: int,
    vec_start_method: str | None = None,
    ts_budget_per_scenario: float | None = None,
    profile_timings: bool = False,
    gc_collect_on_reset: bool = False,
    controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
) -> DummyVecEnv | SubprocVecEnv:
    num_envs = max(1, num_envs)
    base_seed = int(np.random.randint(0, 1_000_000))
    if switch_scenario and switch_scenario_paths:
        # Fix one scenario per worker in round-robin order so workers keep
        # their assigned scenario across episode resets.
        env_fns = [
            _make_env_factory(
                switch_scenario_paths[idx % len(switch_scenario_paths)],
                decision_interval_minutes,
                fire_detection_delay_minutes,
                switch_scenario=False,
                switch_scenario_paths=None,
                aircraft_source_scenario_path=aircraft_source_scenario_path,
                switch_ignition_mode=switch_ignition_mode,
                seed=base_seed + idx,
                ts_budget_per_scenario=None,
                profile_timings=profile_timings,
                gc_collect_on_reset=gc_collect_on_reset,
                include_scenario_features=True,
                controlled_agent_count=controlled_agent_count,
            )
            for idx in range(num_envs)
        ]
    else:
        env_fns = [
            _make_env_factory(
                scenario_path,
                decision_interval_minutes,
                fire_detection_delay_minutes,
                switch_scenario,
                switch_scenario_paths,
                aircraft_source_scenario_path,
                switch_ignition_mode,
                seed=base_seed + idx,
                ts_budget_per_scenario=ts_budget_per_scenario,
                profile_timings=profile_timings,
                gc_collect_on_reset=gc_collect_on_reset,
                include_scenario_features=switch_scenario,
                controlled_agent_count=controlled_agent_count,
            )
            for idx in range(num_envs)
        ]
    if num_envs == 1:
        return DummyVecEnv(env_fns)
    start_method = vec_start_method
    if start_method is None:
        # Prefer fork on Linux to reduce per-worker memory footprint.
        # Fallback to spawn elsewhere.
        start_method = "fork" if sys.platform.startswith("linux") else "spawn"
    return SubprocVecEnv(env_fns, start_method=start_method)


SCENARIO_FEATURES: tuple[str, ...] = (
    "scenario_is_palisades",
    "scenario_is_pyrenees",
    "scenario_is_salamis",
)

STATE_FEATURES = [
    "time_since_detection_min",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "time_to_sunset_min",
    "distance_to_fire_line_m",
    "distance_to_water_m",
    "distance_fire_boundary_to_water",
    "distance_fire_boundary_to_vip",
    "distance_fire_boundary_to_vegetation",
    "distance_fire_boundary_to_topography",
    "distance_fire_boundary_to_indirect",
    "fire_center_x",
    "fire_center_y",
    "leftmost_x",
    "leftmost_y",
    "rightmost_x",
    "rightmost_y",
    "uppermost_x",
    "uppermost_y",
    "lowermost_x",
    "lowermost_y",
    "spread_angle_deg",
    "spread_ray_hit_x",
    "spread_ray_hit_y",
    "max_spread_rate_norm",
    "distance_left_boundary",
    "distance_right_boundary",
    "distance_bottom_boundary",
    "distance_top_boundary",
]


def _agent_feature_names(controlled_agent_count: int) -> tuple[str, ...]:
    names: list[str] = []
    for idx in range(controlled_agent_count):
        names.extend(
            [
                f"agent_{idx}_x",
                f"agent_{idx}_y",
            ]
        )
    return tuple(names)


def _state_feature_names(
    include_scenario_flag: bool,
    controlled_agent_count: int,
) -> tuple[str, ...]:
    feature_names: list[str] = []
    if include_scenario_flag:
        feature_names.extend(SCENARIO_FEATURES)
    feature_names.extend(STATE_FEATURES)
    feature_names.extend(_agent_feature_names(controlled_agent_count))
    return tuple(feature_names)


SELECT_OPTIONS: tuple[SelectPOIType, ...] = (
    SelectPOIType.WATER,
    SelectPOIType.VIP,
    SelectPOIType.VEGETATION,
    SelectPOIType.TOPOGRAPHY,
    SelectPOIType.INDIRECT,
)
TRACK_OPTIONS: tuple[TrackPOIType, ...] = (
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


def _is_allowed_tactic_combination(
    select: SelectPOIType,
    track: TrackPOIType,
    suppress: SuppressType,
) -> bool:
    """Constrain tactic tuples to doctrine-compatible combinations."""
    if select == SelectPOIType.INDIRECT:
        # Force a fully indirect chain for indirect selection.
        return (
            track == TrackPOIType.INDIRECT
            and suppress == SuppressType.INDIRECT
        )
    # For non-indirect selection, allow any available track/suppress method.
    return True


TACTIC_COMBINATIONS = tuple(
    combo
    for combo in TACTIC_COMBINATIONS
    if _is_allowed_tactic_combination(*combo)
)


def _resolve_scenario(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    candidate = SCENARIOS_DIR / "inputs" / path_or_name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to locate scenario file: {path_or_name}")


def _scenario_agent_count(scenario_path: Path) -> int:
    with scenario_path.open() as handle:
        data = json.load(handle)
    return sum(
        sum(int(v) for v in agent.get("agents_per_base", []))
        for agent in data.get("agents", [])
    )


def _validate_switch_scenario_agent_counts(paths: tuple[Path, ...]) -> None:
    counts = {path: _scenario_agent_count(path) for path in paths}
    unique = set(counts.values())
    if len(unique) > 1:
        detail = ", ".join(f"{p.name}: {n}" for p, n in counts.items())
        raise ValueError(
            f"Scenarios have different agent counts ({detail}). "
            "All switch-scenarios must have the same number of agents."
        )


def _resolve_output_path(base_dir: Path, path_or_name: str | None, default_name: str) -> Path:
    if path_or_name:
        candidate = Path(path_or_name)
        if candidate.is_absolute():
            return candidate
        return base_dir / candidate
    return base_dir / default_name


def _numeric_response_time_seconds(parameters: WildfireParameters) -> float:
    response_time = parameters.response_time
    if isinstance(response_time, Real):
        return max(0.0, float(response_time))

    raise TypeError(
        "response_time must be numeric to auto-derive detection delay; "
        "pass --fire-detection-delay-minutes to override."
    )


def _scenario_response_time_seconds(scenario_path: Path) -> float:
    # Terrain metadata cache is global; clear before each parse so each scenario
    # validates against its own map bounds.
    _TerrainParametersCache.metadata = {}
    with scenario_path.open(encoding="utf-8") as handle:
        parameters = WildfireParameters.model_validate_json(handle.read())
    return _numeric_response_time_seconds(parameters)


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


def _ensure_metrics_dict(
    metrics: Metrics | dict[str, float] | None,
) -> dict[str, float]:
    if isinstance(metrics, Metrics):
        return metrics_to_dict(metrics)
    if isinstance(metrics, dict):
        return metrics
    return {}


def _write_records(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write records to CSV by default, or XLSX if explicitly requested."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd  # Lazy import; only needed for Excel output.
        except ImportError as err:
            msg = (
                "Excel output requested but pandas is not installed. "
                "Install pandas or use a .csv output path."
            )
            raise ImportError(msg) from err
        pd.DataFrame(records).to_excel(output_path, index=False)
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _resolve_moe_norms(scenario_path: Path) -> dict[str, float]:
    name = scenario_path.stem.lower()
    for key in SCENARIO_MOE_NORMS:
        if key in name:
            return SCENARIO_MOE_NORMS[key]
    return SCENARIO_MOE_NORMS["palisades"]


def _resolve_scenario_one_hot(scenario_path: Path) -> tuple[float, float, float]:
    name = scenario_path.stem.lower()
    if "palisades" in name:
        return (1.0, 0.0, 0.0)
    if "pyrenees" in name:
        return (0.0, 1.0, 0.0)
    if "salamis" in name:
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0)


def cumulative_moe_reward(
    burnt_area: float,
    cost: float,
    emissions: float,
    casualties: float,
    norms: dict[str, float],
    propagation_factor: float = 0.0,
) -> float:
    base_moe = base_moe_reward(
        burnt_area=burnt_area,
        cost=cost,
        emissions=emissions,
        casualties=casualties,
        norms=norms,
    )
    propagation_penalty = propagation_penalty_value(propagation_flag=propagation_factor)
    penalized_moe = base_moe - propagation_penalty
    return penalized_moe


def base_moe_reward(
    burnt_area: float,
    cost: float,
    emissions: float,
    casualties: float,
    norms: dict[str, float],
) -> float:
    return 1.0 - (
        MOE_WEIGHT * (burnt_area / norms["burnt_area_norm"])
        + MOE_WEIGHT * (cost / norms["cost_norm"])
        + MOE_WEIGHT * (emissions / norms["emission_norm"])
        + MOE_WEIGHT * (casualties / norms["casualty_norm"])
    )


def propagation_penalty_value(*, propagation_flag: float) -> float:
    return float(1.0 if propagation_flag > 0.0 else 0.0)


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _scale_to_unit(value: float, minimum: float, maximum: float) -> float:
    span = float(maximum) - float(minimum)
    if span <= 0.0:
        return 0.0
    scaled = (float(value) - float(minimum)) / span
    return _clip01(scaled)


class WildfireHourlyEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment bridging the wildfire simulation and PPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: Path,
        max_steps: int | None = None,
        decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
        fire_detection_delay_minutes: float | None = None,
        switch_scenario: bool = False,
        switch_scenario_paths: tuple[Path, ...] | None = None,
        aircraft_source_scenario_path: Path | None = None,
        switch_ignition_mode: int = 0,
        ts_budget_per_scenario: float | None = None,
        profile_timings: bool = False,
        gc_collect_on_reset: bool = False,
        include_scenario_features: bool | None = None,
        controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.profile_timings = bool(profile_timings)
        self.gc_collect_on_reset = bool(gc_collect_on_reset)
        self._last_reset_wall_seconds: float | None = None
        self._last_reset_free_burn_wall_seconds: float | None = None
        self._last_rss_before_reset_bytes: int | None = None
        self._last_rss_after_reset_bytes: int | None = None
        self.switch_scenario = switch_scenario
        self.switch_scenario_paths = (
            tuple(switch_scenario_paths) if switch_scenario_paths else tuple()
        )
        self.aircraft_source_scenario_path = aircraft_source_scenario_path
        if switch_ignition_mode not in (0, 1, 2):
            raise ValueError(
                f"switch_ignition_mode must be 0, 1, or 2 (got {switch_ignition_mode})"
            )
        self.switch_ignition_mode = int(switch_ignition_mode)
        self.switch_ignition = self.switch_ignition_mode != 0

        self.decision_interval_minutes = decision_interval_minutes
        self.decision_interval = timedelta(minutes=decision_interval_minutes)
        self.fire_detection_delay_override_minutes = fire_detection_delay_minutes
        if (
            self.fire_detection_delay_override_minutes is not None
            and self.fire_detection_delay_override_minutes < 0.0
        ):
            raise ValueError("fire_detection_delay_minutes must be >= 0")
        self.fire_detection_delay_minutes = 0.0
        self.fire_detection_delay_seconds = 0.0
        self.max_steps_override = max_steps

        if self.switch_scenario:
            if not self.switch_scenario_paths:
                raise ValueError(
                    "switch_scenario=True requires switch_scenario_paths."
                )
            self._scenario_templates: list[tuple[Path, WildfireParameters]] = [
                (path, self._load_parameters(path))
                for path in self.switch_scenario_paths
            ]
            self._scenario_agents: dict[Path, tuple[Any, ...]] = {
                path: self._normalize_agents_for_airports(
                    template.agents,
                    len(template.airports),
                )
                for path, template in self._scenario_templates
            }
            self._scenario_ignition_candidates: dict[Path, np.ndarray] = {}
            if self.switch_ignition:
                for path, template in self._scenario_templates:
                    self._scenario_ignition_candidates[path] = (
                        self._build_ignition_candidate_positions(template)
                    )
            initial_path, initial_template = self._scenario_templates[0]
            self.scenario_path = initial_path
            self.parameters = initial_template.model_copy(
                deep=True,
                update={"agents": self._scenario_agents[initial_path]},
            )
        else:
            self._scenario_templates = []
            self._scenario_agents = {}
            self.parameters = self._load_parameters(scenario_path)
            self._scenario_ignition_candidates = {}
            if self.switch_ignition:
                self._scenario_ignition_candidates[self.scenario_path] = (
                    self._build_ignition_candidate_positions(self.parameters)
                )

        self.fire_detection_delay_seconds = self._resolve_detection_delay_seconds(
            self.parameters
        )
        self.fire_detection_delay_minutes = (
            self.fire_detection_delay_seconds / 60.0
        )
        self.max_steps = self._resolve_max_steps(self.parameters)

        self.controlled_agent_count = int(controlled_agent_count)
        if self.controlled_agent_count <= 0:
            raise ValueError("controlled_agent_count must be > 0")
        self.agent_feature_count = AGENT_FEATURE_COUNT
        self.include_scenario_flag = (
            bool(self.switch_scenario)
            if include_scenario_features is None
            else bool(include_scenario_features)
        )
        self.state_feature_names = _state_feature_names(
            self.include_scenario_flag,
            self.controlled_agent_count,
        )
        obs_dim = len(self.state_feature_names)
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
        self.vip_positions: np.ndarray = np.empty((0, 2), dtype=float)
        self.vegetation_poi_positions: np.ndarray = np.empty((0, 2), dtype=float)
        self.topography_poi_positions: np.ndarray = np.empty((0, 2), dtype=float)
        self.fire_grid_area: float = 0.0
        self.current_step: int = 0
        self.prev_metrics: Metrics | None = None
        self.prev_total_moe: float | None = None
        self.initial_metrics: Metrics | None = None
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.last_info: dict[str, Any] = {}
        self.current_sim_seed: int | None = None
        self.current_scenario_path: Path = self.scenario_path
        self.current_scenario_name: str = self.scenario_path.name
        self.current_scenario_label: str = self.scenario_path.stem.split()[0]
        self.current_ignition_pos: tuple[float, float] | None = None
        self._coord_x_min: float = 0.0
        self._coord_x_max: float = 1.0
        self._coord_y_min: float = 0.0
        self._coord_y_max: float = 1.0
        self._coord_width: float = 1.0
        self._coord_height: float = 1.0
        self._map_diagonal: float = 1.0
        self._fire_line_tree: cKDTree | None = None
        self._fire_line_tree_block_index: int = -1
        self._ts_budget_per_scenario: float | None = ts_budget_per_scenario
        self._global_scenario_ts_counts: dict[str, int] = {}

    def set_global_scenario_counts(self, counts: dict[str, int]) -> None:
        self._global_scenario_ts_counts = dict(counts)

    @staticmethod
    def _protection_location_key(location: ProtectionLocationInput) -> tuple[Any, ...]:
        if location.gps_coords is not None:
            lat, lon = location.gps_coords
            return ("gps", round(float(lat), 8), round(float(lon), 8))
        if location.pos is not None:
            x, y = location.pos
            return ("pos", round(float(x), 3), round(float(y), 3))
        return ("none",)

    def _merge_urban_into_protection(
        self,
        parameters: WildfireParameters,
    ) -> tuple[ProtectionLocationInput, ...]:
        merged: list[ProtectionLocationInput] = []
        seen: set[tuple[Any, ...]] = set()

        for protection in parameters.protection_locations:
            key = self._protection_location_key(protection)
            if key in seen:
                continue
            seen.add(key)
            merged.append(protection)

        for urban in parameters.urban_locations:
            if urban.gps_coords is not None:
                candidate = ProtectionLocationInput(
                    gps_coords=tuple(float(v) for v in urban.gps_coords)
                )
            elif urban.pos is not None:
                candidate = ProtectionLocationInput(
                    pos=tuple(float(v) for v in urban.pos)
                )
            else:
                continue
            key = self._protection_location_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

        return tuple(merged)

    def _load_parameters(self, scenario_path: Path) -> WildfireParameters:
        # Terrain metadata is cached globally in the wildfire simulation module.
        # Clear it before loading each scenario so map bounds are scenario-specific.
        _TerrainParametersCache.metadata = {}
        with scenario_path.open() as handle:
            return WildfireParameters.model_validate_json(handle.read())

    def _resolve_max_steps(self, parameters: WildfireParameters) -> int:
        if self.max_steps_override is not None:
            return self.max_steps_override
        return math.ceil(
            parameters.max_runtime / self.decision_interval.total_seconds()
        )

    @staticmethod
    def _terrain_type_mask(
        feature_data: np.ndarray,
        terrain_type: TerrainTypes,
    ) -> np.ndarray:
        """Return mask for a terrain type from imported feature maps."""
        arr = np.asarray(feature_data)
        if arr.ndim == 2:
            # Indexed terrain map.
            return np.asarray(arr == int(terrain_type), dtype=bool)
        if arr.ndim == 3:
            # RGBA terrain map.
            channels = arr.shape[-1]
            color = np.asarray(
                FEATURES_COLOR_TABLE[terrain_type][:channels], dtype=float
            ).reshape((1, 1, channels))
            return np.all(
                np.isclose(arr.astype(float), color, atol=0.5),
                axis=-1,
            )
        raise ValueError(f"Unsupported terrain feature array shape: {arr.shape}")

    @staticmethod
    def _ignition_center_grid_pos(
        parameters: WildfireParameters,
    ) -> tuple[int, int] | None:
        if not parameters.ignition_centers:
            return None
        center = parameters.ignition_centers[0]
        cell_size = float(parameters.cell_size)
        if cell_size <= 0:
            return None
        try:
            if center.gps_coords is not None:
                tl_merc = gps_to_mercator(
                    parameters.terrain_inputs.fire_map_coordinates[0]
                )
                tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))
                x, y = gps_to_pos(center.gps_coords, tl_bounds)
            elif center.pos is not None:
                x, y = float(center.pos[0]), float(center.pos[1])
            else:
                return None
            return (int(y / cell_size), int(x / cell_size))
        except Exception:
            return None

    def _build_ignition_candidate_positions(
        self,
        parameters: WildfireParameters,
    ) -> np.ndarray:
        """Dispatch to the candidate builder for the active ignition mode."""
        if self.switch_ignition_mode == 2:
            return self._build_ignition_candidate_positions_v2(parameters)
        return self._build_ignition_candidate_positions_v1(parameters)

    @staticmethod
    def _build_ignitable_mask(
        feature_data: np.ndarray,
        cell_size_m: float,
        urban_buffer_m: float = IGNITION_URBAN_BUFFER_M,
    ) -> np.ndarray:
        """Cells permitted as ignition seeds.

        Excludes water, urban, and non-combustible (rock/bare) cells.
        When ``urban_buffer_m`` is positive, also enforces a keep-out from
        urban cells.
        """
        water_mask = WildfireHourlyEnv._terrain_type_mask(
            feature_data, TerrainTypes.WATER
        )
        urban_mask = WildfireHourlyEnv._terrain_type_mask(
            feature_data, TerrainTypes.RESIDENTIAL
        )
        rock_mask = WildfireHourlyEnv._terrain_type_mask(
            feature_data, TerrainTypes.NON_COMBUSTIBLE
        )
        ignitable = ~(water_mask | urban_mask | rock_mask)
        if urban_buffer_m > 0.0 and np.any(urban_mask) and cell_size_m > 0:
            buffer_cells = urban_buffer_m / cell_size_m
            # With urban cells as the zero/background pixels, this gives
            # distance from each non-urban cell to the nearest urban cell.
            distance_cells = ndimage.distance_transform_edt(~urban_mask)
            ignitable &= distance_cells >= buffer_cells
        return ignitable

    def _build_ignition_candidate_positions_v2(
        self,
        parameters: WildfireParameters,
    ) -> np.ndarray:
        """Map-centered box with margin = IGNITION_V2_MARGIN_RATIO * edge length."""
        feature_data = np.asarray(
            np.load(parameters.terrain_inputs.features_file, allow_pickle=False)
        )
        if feature_data.ndim < 2:
            return np.empty((0, 2), dtype=np.int64)
        non_forbidden_mask = self._build_ignitable_mask(
            feature_data, float(parameters.cell_size)
        )
        rows, cols = non_forbidden_mask.shape

        margin = IGNITION_V2_MARGIN_RATIO
        r0 = int(round(rows * margin))
        r1 = int(round(rows * (1.0 - margin)))
        c0 = int(round(cols * margin))
        c1 = int(round(cols * (1.0 - margin)))
        if r0 >= r1 or c0 >= c1:
            return np.empty((0, 2), dtype=np.int64)
        box_mask = np.zeros_like(non_forbidden_mask, dtype=bool)
        box_mask[r0:r1, c0:c1] = True
        valid_mask = non_forbidden_mask & box_mask
        if np.any(valid_mask):
            return np.argwhere(valid_mask).astype(np.int64, copy=False)
        return np.empty((0, 2), dtype=np.int64)

    def _build_ignition_candidate_positions_v1(
        self,
        parameters: WildfireParameters,
    ) -> np.ndarray:
        """100×100 cell box around the scenario's original ignition center."""
        feature_data = np.asarray(
            np.load(parameters.terrain_inputs.features_file, allow_pickle=False)
        )
        if feature_data.ndim < 2:
            return np.empty((0, 2), dtype=np.int64)

        buffered_ignitable_mask = self._build_ignitable_mask(
            feature_data, float(parameters.cell_size)
        )
        hard_ignitable_mask = self._build_ignitable_mask(
            feature_data,
            float(parameters.cell_size),
            urban_buffer_m=0.0,
        )
        rows, cols = hard_ignitable_mask.shape

        # Try a 100×100 cell box around the scenario's original ignition center first.
        center_pos = self._ignition_center_grid_pos(parameters)
        if center_pos is not None:
            cr, cc = center_pos
            r0 = max(0, cr - IGNITION_BOX_HALF_SIZE)
            r1 = min(rows, cr + IGNITION_BOX_HALF_SIZE)
            c0 = max(0, cc - IGNITION_BOX_HALF_SIZE)
            c1 = min(cols, cc + IGNITION_BOX_HALF_SIZE)
            box_mask = np.zeros_like(hard_ignitable_mask, dtype=bool)
            box_mask[r0:r1, c0:c1] = True
            valid_mask = buffered_ignitable_mask & box_mask
            if np.any(valid_mask):
                return np.argwhere(valid_mask).astype(np.int64, copy=False)
            valid_mask = hard_ignitable_mask & box_mask
            if np.any(valid_mask):
                return np.argwhere(valid_mask).astype(np.int64, copy=False)

        # Fallback: margin-based candidates only if the local box has no
        # non-water/non-urban/non-rock cells at all.
        margin_candidates = (
            IGNITION_BOUNDARY_MARGIN_RATIO,
            0.2,
            0.1,
            0.0,
        )
        valid_mask = np.zeros_like(hard_ignitable_mask, dtype=bool)
        for margin_ratio in margin_candidates:
            margin_rows = int(max(0, round(margin_ratio * rows)))
            margin_cols = int(max(0, round(margin_ratio * cols)))
            interior_mask = np.zeros_like(hard_ignitable_mask, dtype=bool)
            row_start, row_end = margin_rows, rows - margin_rows
            col_start, col_end = margin_cols, cols - margin_cols
            if row_start < row_end and col_start < col_end:
                interior_mask[row_start:row_end, col_start:col_end] = True
            valid_mask = buffered_ignitable_mask & interior_mask
            if np.any(valid_mask):
                break
            valid_mask = hard_ignitable_mask & interior_mask
            if np.any(valid_mask):
                break

        if not np.any(valid_mask):
            valid_mask = hard_ignitable_mask
        if not np.any(valid_mask):
            return np.empty((0, 2), dtype=np.int64)

        # (row, col) pairs on fire-map raster.
        return np.argwhere(valid_mask).astype(np.int64, copy=False)

    def _sample_ignition_centers(
        self,
        selected_path: Path,
        selected_parameters: WildfireParameters,
    ) -> tuple[Any, ...]:
        if not self.switch_ignition:
            return selected_parameters.ignition_centers

        candidates = self._scenario_ignition_candidates.get(selected_path)
        if candidates is None:
            candidates = self._build_ignition_candidate_positions(selected_parameters)
            self._scenario_ignition_candidates[selected_path] = candidates

        if candidates.size == 0:
            return selected_parameters.ignition_centers

        idx = int(self.np_random.integers(0, len(candidates)))
        row, col = (int(candidates[idx, 0]), int(candidates[idx, 1]))
        cell_size = float(selected_parameters.cell_size)
        # Convert fire-map index (row, col) -> fire-map position (x, y).
        x = (float(col) + 0.5) * cell_size
        y = (float(row) + 0.5) * cell_size

        fire_top_left_merc = gps_to_mercator(
            selected_parameters.terrain_inputs.fire_map_coordinates[0]
        )
        fire_top_left_bounds = (float(fire_top_left_merc[1]), float(fire_top_left_merc[0]))
        lat, lon = pos_to_gps((x, y), fire_top_left_bounds)
        sampled = IgnitionCenterInput(gps_coords=(float(lat), float(lon)))

        # Safety net: if conversion drifts out of fire-map bounds, fallback.
        try:
            grid_shape = selected_parameters.terrain_inputs.grid_shape
            sampled.check_in_bbox(
                bbox_pos=(
                    (0.0, 0.0),
                    (
                        float(grid_shape[0]) * cell_size,
                        float(grid_shape[1]) * cell_size,
                    ),
                ),
                bbox_gps=selected_parameters.terrain_inputs.fire_map_coordinates,
            )
        except AssertionError:
            return selected_parameters.ignition_centers

        return (sampled,)

    def _normalize_agents_for_airports(
        self,
        agents: Sequence[Any],
        airport_count: int,
    ) -> tuple[Any, ...]:
        normalized_agents: list[Any] = []
        for agent in agents:
            counts = tuple(int(value) for value in agent.agents_per_base)
            if len(counts) < airport_count:
                counts = counts + (0,) * (airport_count - len(counts))
            elif len(counts) > airport_count:
                trimmed = list(counts[:airport_count])
                if airport_count > 0:
                    trimmed[0] += sum(counts[airport_count:])
                counts = tuple(trimmed)
            normalized_agents.append(
                agent.model_copy(
                    deep=True,
                    update={"agents_per_base": counts},
                )
            )
        return tuple(normalized_agents)

    def _select_episode_setup(self) -> tuple[Path, WildfireParameters]:
        if not self.switch_scenario:
            selected_path = self.scenario_path
            selected_parameters = self.parameters.model_copy(deep=True)
        else:
            if not self._scenario_templates:
                raise RuntimeError("No scenario templates available for switching.")

            if self._ts_budget_per_scenario is not None:
                budget = self._ts_budget_per_scenario
                weights = np.array(
                    [
                        max(0.0, budget - self._global_scenario_ts_counts.get(path.name, 0))
                        for path, _ in self._scenario_templates
                    ],
                    dtype=float,
                )
                total = weights.sum()
                if total > 0.0:
                    weights /= total
                else:
                    weights = np.ones(len(self._scenario_templates)) / len(self._scenario_templates)
                choice_idx = int(self.np_random.choice(len(self._scenario_templates), p=weights))
            else:
                choice_idx = int(self.np_random.integers(0, len(self._scenario_templates)))
            selected_path, selected_template = self._scenario_templates[choice_idx]
            selected_parameters = selected_template.model_copy(
                deep=True,
                update={"agents": self._scenario_agents[selected_path]},
            )

        merged_protection = self._merge_urban_into_protection(selected_parameters)
        ignition_centers = self._sample_ignition_centers(
            selected_path,
            selected_parameters,
        )
        selected_parameters = selected_parameters.model_copy(
            deep=True,
            update={
                "protection_locations": merged_protection,
                "ignition_centers": ignition_centers,
            },
        )
        return selected_path, selected_parameters

    # -- Environment helpers -------------------------------------------------
    def _resolve_detection_delay_seconds(
        self,
        parameters: WildfireParameters,
    ) -> float:
        if self.fire_detection_delay_override_minutes is not None:
            return float(self.fire_detection_delay_override_minutes) * 60.0

        return _numeric_response_time_seconds(parameters)

    def _response_time_seconds(self) -> float | None:
        try:
            return _numeric_response_time_seconds(self.parameters)
        except TypeError:
            return None

    def _advance_to_decision_start(self) -> None:
        assert self.sim is not None
        if self.fire_detection_delay_seconds <= 0.0:
            return
        while (
            self._mission_time() < self.fire_detection_delay_seconds
            and self._mission_time() < self.parameters.max_runtime
            and not self._mission_complete()
        ):
            try:
                self.sim.step(force=True)
            except RuntimeError as err:
                if "not started cannot be stopped" in str(err):
                    break
                raise

    def _init_simulation(self, seed: int) -> None:
        # Clear the global terrain metadata cache before each episode setup
        # so that when a new scenario is selected, its correct metadata is loaded.
        _TerrainParametersCache.metadata = {}
        selected_path, selected_parameters = self._select_episode_setup()
        self.current_scenario_path = selected_path
        self.current_scenario_name = selected_path.name
        self.current_scenario_label = (
            selected_parameters.terrain_inputs.file_namespace.split("_", 1)[0]
        )
        self.parameters = selected_parameters
        self.fire_detection_delay_seconds = self._resolve_detection_delay_seconds(
            self.parameters
        )
        self.fire_detection_delay_minutes = (
            self.fire_detection_delay_seconds / 60.0
        )
        self.max_steps = self._resolve_max_steps(self.parameters)
        self.sim = WildfireSimulation(parameters=self.parameters, seed=seed)
        self.sim.wildfire.ignite(self.sim.ignition_centers)
        if self.sim.ignition_centers:
            first_center = self.sim.ignition_centers[0]
            self.current_ignition_pos = (
                float(first_center.pos[0]),
                float(first_center.pos[1]),
            )
        else:
            self.current_ignition_pos = None
        water_sources = self.sim.firefighters.water_sources
        self.water_positions = (
            np.array([ws.pos for ws in water_sources], dtype=float)
            if water_sources
            else np.empty((0, 2), dtype=float)
        )
        self._build_static_poi_positions()
        self._cache_static_state_scalers()
        self._fire_line_tree = None
        self._fire_line_tree_block_index = -1
        self.fire_grid_area = (
            self.sim.wildfire.fire_states.size
            * (self.sim.parameters.cell_size**2)
        )
        self.done = False
        if self.profile_timings:
            free_burn_start = time.perf_counter()
            self._advance_to_decision_start()
            self._last_reset_free_burn_wall_seconds = (
                time.perf_counter() - free_burn_start
            )
        else:
            self._advance_to_decision_start()
        self.prev_metrics = self._compute_metrics()
        self.initial_metrics = self.prev_metrics
        self.prev_total_moe = None
        self.current_step = 0

        self.cumulative_reward = 0.0
        self.current_sim_seed = seed

    def _compute_metrics(self) -> Metrics:
        assert self.sim is not None
        burnt = float(self.sim.wildfire.burnt_area)
        cost = float(self.sim.total_fire_cost)
        casualties = float(self.sim.total_casualties)
        emissions = float(self.sim.total_fire_emissions)
        return Metrics(burnt, cost, casualties, emissions)

    def _select_candidate_indices(
        self, indices: np.ndarray, max_points: int
    ) -> np.ndarray:
        if indices.shape[0] <= max_points:
            return indices
        selected = np.linspace(
            0, indices.shape[0] - 1, num=max_points, dtype=np.int64
        )
        return indices[selected]

    def _coerce_grid_indices(self, indices: np.ndarray) -> np.ndarray:
        """Coerce index arrays to (N, 2) grid pairs [y, x]."""
        assert self.sim is not None
        if indices.size == 0:
            return np.empty((0, 2), dtype=np.int64)

        arr = np.asarray(indices)
        if arr.ndim == 1:
            if arr.shape[0] < 2:
                return np.empty((0, 2), dtype=np.int64)
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.shape[1] < 2:
            return np.empty((0, 2), dtype=np.int64)

        # Some terrain-derived maps can carry an extra channel axis and produce
        # 3D argwhere indices. Resolve them to grid (y, x) before conversion.
        if arr.shape[1] == 2:
            yx = arr[:, :2]
        else:
            rows, cols = self.sim.environment.terrain.grid_description.shape
            first = arr[:, :2]
            last = arr[:, -2:]
            first_valid = np.all(
                (first[:, 0] >= 0)
                & (first[:, 0] < rows)
                & (first[:, 1] >= 0)
                & (first[:, 1] < cols)
            )
            last_valid = np.all(
                (last[:, 0] >= 0)
                & (last[:, 0] < rows)
                & (last[:, 1] >= 0)
                & (last[:, 1] < cols)
            )
            if first_valid:
                yx = first
            elif last_valid:
                yx = last
            else:
                yx = first

        yx = np.asarray(yx, dtype=np.int64)
        if yx.ndim != 2 or yx.shape[1] < 2:
            return np.empty((0, 2), dtype=np.int64)
        return yx[:, :2]

    def _indices_to_positions(self, indices: np.ndarray) -> np.ndarray:
        assert self.sim is not None
        yx = self._coerce_grid_indices(indices)
        if yx.size == 0:
            return np.empty((0, 2), dtype=float)

        y_idx = np.asarray(yx[:, 0], dtype=np.int64)
        x_idx = np.asarray(yx[:, 1], dtype=np.int64)
        positions = index_to_pos(
            (y_idx, x_idx),
            grid_description=self.sim.environment.terrain.grid_description,
            origin=self.sim.environment.terrain.origin,
        )
        return np.asarray(positions, dtype=float)

    def _cache_static_state_scalers(self) -> None:
        """Cache terrain/map constants used for state normalization."""
        assert self.sim is not None

        env_width, env_height = self.sim.environment.dimensions
        x_min, y_min = 0.0, 0.0
        x_max = float(env_width) if env_width else 1.0
        y_max = float(env_height) if env_height else 1.0

        terrain_bounds = getattr(
            self.sim.environment.terrain, "bounding_positions", None
        )
        if terrain_bounds is not None and len(terrain_bounds) == 2:
            lower = np.asarray(terrain_bounds[0], dtype=float).reshape(-1)
            upper = np.asarray(terrain_bounds[1], dtype=float).reshape(-1)
            if lower.size >= 2 and upper.size >= 2:
                x_min = float(min(lower[0], upper[0]))
                x_max = float(max(lower[0], upper[0]))
                y_min = float(min(lower[1], upper[1]))
                y_max = float(max(lower[1], upper[1]))

        self._coord_x_min = x_min
        self._coord_x_max = x_max
        self._coord_y_min = y_min
        self._coord_y_max = y_max
        self._coord_width = max(x_max - x_min, 1.0)
        self._coord_height = max(y_max - y_min, 1.0)
        self._map_diagonal = max(
            float(np.linalg.norm([self._coord_width, self._coord_height])),
            1.0,
        )

    def _distance_to_fire_line_m(self, burning_indices: np.ndarray) -> float:
        """Compute minimum burning-cell distance to active fire line segment in meters."""
        assert self.sim is not None
        fire_block_indices = self.sim.firefighters.fire_block_indices
        if (
            fire_block_indices is None
            or fire_block_indices.size == 0
            or self.sim.firefighters.current_block_index <= 0
        ):
            return math.nan

        current_block_index = int(self.sim.firefighters.current_block_index)
        segment_raw = np.asarray(fire_block_indices[:current_block_index])
        segment = self._coerce_grid_indices(segment_raw)
        burning = self._coerce_grid_indices(np.asarray(burning_indices))
        if segment.size == 0 or burning.size == 0:
            return math.nan

        if (
            self._fire_line_tree is None
            or self._fire_line_tree_block_index != current_block_index
        ):
            self._fire_line_tree = cKDTree(np.asarray(segment, dtype=float))
            self._fire_line_tree_block_index = current_block_index

        distances, _ = self._fire_line_tree.query(
            np.asarray(burning, dtype=float), k=1
        )
        min_grid_dist = float(np.min(np.asarray(distances, dtype=float)))
        return float(self.sim.parameters.cell_size) * min_grid_dist

    def _build_static_poi_positions(self) -> None:
        assert self.sim is not None
        firefighters = self.sim.firefighters

        self.vip_positions = np.array(
            [location.pos for location in firefighters.protection_locations],
            dtype=float,
        )
        if self.vip_positions.size == 0:
            self.vip_positions = np.empty((0, 2), dtype=float)

        priority_map = np.asarray(
            self.sim.environment.terrain.features.priority_map, dtype=float
        )
        valid_priority = np.isfinite(priority_map) & (priority_map > 0.0)
        if np.any(valid_priority):
            veg_threshold = float(
                np.quantile(priority_map[valid_priority], POI_CANDIDATE_QUANTILE)
            )
            vegetation_indices = np.argwhere(priority_map >= veg_threshold)
            vegetation_indices = self._select_candidate_indices(
                vegetation_indices, MAX_POI_CANDIDATES
            )
            self.vegetation_poi_positions = self._indices_to_positions(
                vegetation_indices
            )
        else:
            self.vegetation_poi_positions = np.empty((0, 2), dtype=float)

        elevation = np.asarray(
            self.sim.environment.terrain.elevation.elevation_data, dtype=float
        )
        valid_elevation = np.isfinite(elevation)
        if np.any(valid_elevation):
            topo_threshold = float(
                np.quantile(elevation[valid_elevation], POI_CANDIDATE_QUANTILE)
            )
            topography_indices = np.argwhere(elevation >= topo_threshold)
            topography_indices = self._select_candidate_indices(
                topography_indices, MAX_POI_CANDIDATES
            )
            self.topography_poi_positions = self._indices_to_positions(
                topography_indices
            )
        else:
            self.topography_poi_positions = np.empty((0, 2), dtype=float)

    def _current_indirect_poi_positions(self) -> np.ndarray:
        assert self.sim is not None
        fire_block_indices = self.sim.firefighters.fire_block_indices
        if fire_block_indices is None or fire_block_indices.size == 0:
            return np.empty((0, 2), dtype=float)
        return self._indices_to_positions(np.asarray(fire_block_indices))

    def _distance_boundary_to_poi(
        self,
        boundary_points: np.ndarray,
        poi_points: np.ndarray,
        map_diagonal: float,
    ) -> float:
        if boundary_points.size == 0 or poi_points.size == 0 or map_diagonal <= 0.0:
            return 1.0
        min_dist = float(cdist(boundary_points, poi_points).min())
        return float(min(max(min_dist / map_diagonal, 0.0), 1.0))

    def _mission_time(self) -> float:
        assert self.sim is not None
        return self.sim.timer.mission_runtime.total_seconds()

    def _mission_complete(self) -> bool:
        assert self.sim is not None
        fire_remaining = bool(self.sim.wildfire.fire_positions.size)
        return (not fire_remaining) or self.sim.is_stopped.is_set()

    def _propagation_factor(self) -> float:
        assert self.sim is not None
        return 0.0 if self.sim.wildfire.fire_in_bounds else 1.0

    def _current_moe_norms(self) -> dict[str, float]:
        return _resolve_moe_norms(self.current_scenario_path)

    def _current_scenario_one_hot(self) -> tuple[float, float, float]:
        return _resolve_scenario_one_hot(self.current_scenario_path)

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
            self.fire_detection_delay_seconds
            + ((self.current_step + 1)) * self.decision_interval.total_seconds()
        )
        while (
            self._mission_time() < target_time_seconds
            and self._mission_time() < self.parameters.max_runtime
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
        if (
            self._mission_time() >= self.parameters.max_runtime
            or self._mission_complete()
        ):
            self.done = True

    def _compute_state(self) -> np.ndarray:
        assert self.sim is not None
        atmosphere = self.sim.atmosphere
        mission_time = self.sim.timer.mission_time
        time_since_detection_min = (
            max(
                0.0,
                self.sim.timer.mission_runtime.total_seconds()
                - self.fire_detection_delay_seconds,
            )
            / 60.0
        )
        burning_indices = self.sim.wildfire.burning_indices
        burning_count = int(burning_indices.shape[0])

        fire_center_x = math.nan
        fire_center_y = math.nan
        leftmost_x = math.nan
        leftmost_y = math.nan
        rightmost_x = math.nan
        rightmost_y = math.nan
        uppermost_x = math.nan
        uppermost_y = math.nan
        lowermost_x = math.nan
        lowermost_y = math.nan
        spread_angle = math.nan
        spread_ray_hit_x = math.nan
        spread_ray_hit_y = math.nan
        max_spread_rate = math.nan
        distance_fire_line = math.nan
        distance_water = math.nan
        distance_boundary_to_water = 1.0
        distance_boundary_to_vip = 1.0
        distance_boundary_to_vegetation = 1.0
        distance_boundary_to_topography = 1.0
        distance_boundary_to_indirect = 1.0

        if burning_count:
            fire_positions = self.sim.wildfire.fire_positions
            centroid = fire_positions.mean(axis=0)
            fire_center_x = float(centroid[0])
            fire_center_y = float(centroid[1])

            leftmost_idx = int(np.argmin(fire_positions[:, 0]))
            rightmost_idx = int(np.argmax(fire_positions[:, 0]))
            uppermost_idx = int(np.argmax(fire_positions[:, 1]))
            lowermost_idx = int(np.argmin(fire_positions[:, 1]))
            leftmost_x, leftmost_y = map(float, fire_positions[leftmost_idx])
            rightmost_x, rightmost_y = map(float, fire_positions[rightmost_idx])
            uppermost_x, uppermost_y = map(float, fire_positions[uppermost_idx])
            lowermost_x, lowermost_y = map(float, fire_positions[lowermost_idx])

            centroid_idx = burning_indices.mean(axis=0)
            spread_rates = self.sim.wildfire.get_spread_rates(burning_indices)
            max_spread_rate = float(np.max(spread_rates))
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
                distance_fire_line = self._distance_to_fire_line_m(burning_indices)

        time_to_sunset = max(
            (self.sim.atmosphere.next_sunset - mission_time).total_seconds() / 60.0,
            0.0,
        )

        x_min = self._coord_x_min
        x_max = self._coord_x_max
        y_min = self._coord_y_min
        y_max = self._coord_y_max
        coord_width = self._coord_width
        coord_height = self._coord_height
        map_diagonal = self._map_diagonal

        boundary_left = 0.0
        boundary_right = 0.0
        boundary_bottom = 0.0
        boundary_top = 0.0

        if burning_count:
            fire_positions = self.sim.wildfire.fire_positions
            min_x = float(fire_positions[:, 0].min())
            max_x = float(fire_positions[:, 0].max())
            min_y = float(fire_positions[:, 1].min())
            max_y = float(fire_positions[:, 1].max())
            boundary_left = float(max(min_x - x_min, 0.0))
            boundary_right = float(max(x_max - max_x, 0.0))
            boundary_bottom = float(max(min_y - y_min, 0.0))
            boundary_top = float(max(y_max - max_y, 0.0))

        # Ray cast from fire center in spread direction to first map boundary hit.
        if (
            not math.isnan(fire_center_x)
            and not math.isnan(fire_center_y)
            and not math.isnan(spread_angle)
        ):
            theta = math.radians(spread_angle)
            dir_x = math.cos(theta)
            dir_y = math.sin(theta)
            eps = 1e-9
            candidates: list[float] = []

            if abs(dir_x) > eps:
                t_left = (x_min - fire_center_x) / dir_x
                y_left = fire_center_y + t_left * dir_y
                if t_left >= 0.0 and y_min <= y_left <= y_max:
                    candidates.append(t_left)

                t_right = (x_max - fire_center_x) / dir_x
                y_right = fire_center_y + t_right * dir_y
                if t_right >= 0.0 and y_min <= y_right <= y_max:
                    candidates.append(t_right)

            if abs(dir_y) > eps:
                t_bottom = (y_min - fire_center_y) / dir_y
                x_bottom = fire_center_x + t_bottom * dir_x
                if t_bottom >= 0.0 and x_min <= x_bottom <= x_max:
                    candidates.append(t_bottom)

                t_top = (y_max - fire_center_y) / dir_y
                x_top = fire_center_x + t_top * dir_x
                if t_top >= 0.0 and x_min <= x_top <= x_max:
                    candidates.append(t_top)

            if candidates:
                t_hit = min(candidates)
                spread_ray_hit_x = fire_center_x + t_hit * dir_x
                spread_ray_hit_y = fire_center_y + t_hit * dir_y

        if burning_count:
            boundary_points = np.array(
                [
                    [leftmost_x, leftmost_y],
                    [rightmost_x, rightmost_y],
                    [uppermost_x, uppermost_y],
                    [lowermost_x, lowermost_y],
                ],
                dtype=float,
            )
            distance_boundary_to_water = self._distance_boundary_to_poi(
                boundary_points, self.water_positions, map_diagonal
            )
            distance_boundary_to_vip = self._distance_boundary_to_poi(
                boundary_points, self.vip_positions, map_diagonal
            )
            distance_boundary_to_vegetation = self._distance_boundary_to_poi(
                boundary_points, self.vegetation_poi_positions, map_diagonal
            )
            distance_boundary_to_topography = self._distance_boundary_to_poi(
                boundary_points, self.topography_poi_positions, map_diagonal
            )
            indirect_positions = self._current_indirect_poi_positions()
            distance_boundary_to_indirect = self._distance_boundary_to_poi(
                boundary_points, indirect_positions, map_diagonal
            )

        atmosphere_inputs = self.parameters.atmosphere_inputs
        temp_min, temp_max = -20.0, 50.0
        if hasattr(atmosphere_inputs, "temperature_range"):
            try:
                temperature_values = tuple(
                    float(value) for value in atmosphere_inputs.temperature_range
                )
                if temperature_values:
                    temp_min = min(temperature_values)
                    temp_max = max(temperature_values)
            except Exception:
                pass
        if temp_max <= temp_min:
            temp_max = temp_min + 1.0

        wind_speed_upper = max(20.0, float(atmosphere.wind_speed), 1.0)
        if (
            hasattr(atmosphere_inputs, "wind_run")
            and hasattr(atmosphere_inputs, "sun_times")
        ):
            try:
                wind_run = float(atmosphere_inputs.wind_run)
                sun_rise, sunset = atmosphere_inputs.sun_times
                t1 = float(sun_rise) + 1.0
                t2 = 15.0
                t3 = float(sunset) + 2.0
                sf1 = 4.0 * (t2 - t1)
                sf2 = 4.0 * (t3 - t2)
                wind_min = wind_run * 0.0080
                denom = 3600.0 * (sf1 + sf2)
                wind_amp = 0.0
                if denom > 0.0:
                    wind_amp = (
                        (wind_run - wind_min * 24.0 * 3.6)
                        * 2.0
                        * math.pi
                        * 1000.0
                    ) / denom
                wind_speed_upper = max(1.0, wind_min + abs(wind_amp))
            except Exception:
                wind_speed_upper = max(20.0, float(atmosphere.wind_speed), 1.0)

        day_minutes = 24.0 * 60.0

        time_since_detection_norm = _scale_to_unit(
            float(time_since_detection_min), 0.0, day_minutes
        )
        temperature_norm = _scale_to_unit(
            float(atmosphere.temperature), temp_min, temp_max
        )
        humidity_norm = _scale_to_unit(float(atmosphere.relative_humidity), 0.0, 100.0)
        wind_speed_norm = _scale_to_unit(
            float(atmosphere.wind_speed), 0.0, wind_speed_upper
        )
        wind_direction_norm = _scale_to_unit(
            float((atmosphere.wind_aspect + 360.0) % 360.0), 0.0, 360.0
        )
        time_to_sunset_norm = _scale_to_unit(float(time_to_sunset), 0.0, day_minutes)
        distance_fire_line_norm = (
            _scale_to_unit(float(distance_fire_line), 0.0, map_diagonal)
            if not math.isnan(distance_fire_line)
            else 0.0
        )
        distance_water_norm = (
            _scale_to_unit(float(distance_water), 0.0, map_diagonal)
            if not math.isnan(distance_water)
            else 0.0
        )

        fire_center_x_norm = (
            _scale_to_unit(float(fire_center_x), x_min, x_max)
            if not math.isnan(fire_center_x)
            else 0.0
        )
        fire_center_y_norm = (
            _scale_to_unit(float(fire_center_y), y_min, y_max)
            if not math.isnan(fire_center_y)
            else 0.0
        )
        leftmost_x_norm = (
            _scale_to_unit(float(leftmost_x), x_min, x_max)
            if not math.isnan(leftmost_x)
            else 0.0
        )
        leftmost_y_norm = (
            _scale_to_unit(float(leftmost_y), y_min, y_max)
            if not math.isnan(leftmost_y)
            else 0.0
        )
        rightmost_x_norm = (
            _scale_to_unit(float(rightmost_x), x_min, x_max)
            if not math.isnan(rightmost_x)
            else 0.0
        )
        rightmost_y_norm = (
            _scale_to_unit(float(rightmost_y), y_min, y_max)
            if not math.isnan(rightmost_y)
            else 0.0
        )
        uppermost_x_norm = (
            _scale_to_unit(float(uppermost_x), x_min, x_max)
            if not math.isnan(uppermost_x)
            else 0.0
        )
        uppermost_y_norm = (
            _scale_to_unit(float(uppermost_y), y_min, y_max)
            if not math.isnan(uppermost_y)
            else 0.0
        )
        lowermost_x_norm = (
            _scale_to_unit(float(lowermost_x), x_min, x_max)
            if not math.isnan(lowermost_x)
            else 0.0
        )
        lowermost_y_norm = (
            _scale_to_unit(float(lowermost_y), y_min, y_max)
            if not math.isnan(lowermost_y)
            else 0.0
        )
        spread_angle_norm = (
            _scale_to_unit(float(spread_angle), 0.0, 360.0)
            if not math.isnan(spread_angle)
            else 0.0
        )
        spread_ray_hit_x_norm = (
            _scale_to_unit(float(spread_ray_hit_x), x_min, x_max)
            if not math.isnan(spread_ray_hit_x)
            else 0.0
        )
        spread_ray_hit_y_norm = (
            _scale_to_unit(float(spread_ray_hit_y), y_min, y_max)
            if not math.isnan(spread_ray_hit_y)
            else 0.0
        )
        max_spread_rate_norm = (
            _scale_to_unit(float(max_spread_rate), 0.0, MAX_SPREAD_RATE_NORM_MPM)
            if not math.isnan(max_spread_rate)
            else 0.0
        )
        boundary_left_norm = _scale_to_unit(float(boundary_left), 0.0, coord_width)
        boundary_right_norm = _scale_to_unit(float(boundary_right), 0.0, coord_width)
        boundary_bottom_norm = _scale_to_unit(float(boundary_bottom), 0.0, coord_height)
        boundary_top_norm = _scale_to_unit(float(boundary_top), 0.0, coord_height)

        agents = self.sim.firefighters.firefighters
        state_features: list[float] = []
        if self.include_scenario_flag:
            state_features.extend(
                _clip01(value) for value in self._current_scenario_one_hot()
            )
        state_features.extend(
            [
                time_since_detection_norm,
                temperature_norm,
                humidity_norm,
                wind_speed_norm,
                wind_direction_norm,
                time_to_sunset_norm,
                distance_fire_line_norm,
                distance_water_norm,
                _clip01(distance_boundary_to_water),
                _clip01(distance_boundary_to_vip),
                _clip01(distance_boundary_to_vegetation),
                _clip01(distance_boundary_to_topography),
                _clip01(distance_boundary_to_indirect),
                fire_center_x_norm,
                fire_center_y_norm,
                leftmost_x_norm,
                leftmost_y_norm,
                rightmost_x_norm,
                rightmost_y_norm,
                uppermost_x_norm,
                uppermost_y_norm,
                lowermost_x_norm,
                lowermost_y_norm,
                spread_angle_norm,
                spread_ray_hit_x_norm,
                spread_ray_hit_y_norm,
                max_spread_rate_norm,
                boundary_left_norm,
                boundary_right_norm,
                boundary_bottom_norm,
                boundary_top_norm,
            ]
        )

        for idx in range(self.controlled_agent_count):
            if idx < len(agents):
                agent = agents[idx]
                pos_x, pos_y = agent.pos
                norm_x = _scale_to_unit(float(pos_x), x_min, x_max)
                norm_y = _scale_to_unit(float(pos_y), y_min, y_max)

                state_features.extend(
                    [
                        norm_x,
                        norm_y,
                    ]
                )
            else:
                state_features.extend([0.0] * self.agent_feature_count)

        obs = np.array(state_features, dtype=np.float32)
        # Hard guarantee for normalized observations expected by training:
        # convert NaN/inf to finite values and clamp to [0, 1].
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

    def _compute_reward_and_info(
        self,
        new_metrics: Metrics,
    ) -> tuple[float, dict[str, Any]]:
        assert self.prev_metrics is not None
        assert self.initial_metrics is not None
        deltas = Metrics(
            new_metrics.burnt_area - self.prev_metrics.burnt_area,
            new_metrics.cost - self.prev_metrics.cost,
            new_metrics.casualties - self.prev_metrics.casualties,
            new_metrics.emissions - self.prev_metrics.emissions,
        )
        cumulative_deltas = Metrics(
            new_metrics.burnt_area - self.initial_metrics.burnt_area,
            new_metrics.cost - self.initial_metrics.cost,
            new_metrics.casualties - self.initial_metrics.casualties,
            new_metrics.emissions - self.initial_metrics.emissions,
        )
        propagation_factor = self._propagation_factor()
        moe_norms = self._current_moe_norms()
        current_base_moe = base_moe_reward(
            new_metrics.burnt_area,
            new_metrics.cost,
            new_metrics.emissions,
            new_metrics.casualties,
            norms=moe_norms,
        )
        propagation_penalty = propagation_penalty_value(propagation_flag=propagation_factor)
        current_total_moe = cumulative_moe_reward(
            new_metrics.burnt_area,
            new_metrics.cost,
            new_metrics.emissions,
            new_metrics.casualties,
            norms=moe_norms,
            propagation_factor=propagation_factor,
        )
        previous_total_moe = (
            self.prev_total_moe if self.prev_total_moe is not None else current_total_moe
        )
        delta_moe = current_total_moe - previous_total_moe
        # propagation_penalty is already included in current_total_moe
        reward = delta_moe
        self.prev_total_moe = current_total_moe
        decision_step = self.current_step + 1
        elapsed_minutes = decision_step * self.decision_interval.total_seconds() / 60.0
        info = {
            "decision_step": decision_step,
            "elapsed_minutes": elapsed_minutes,
            "decision_interval_minutes": self.decision_interval.total_seconds() / 60.0,
            "fire_detection_delay_minutes": self.fire_detection_delay_minutes,
            "response_time_seconds": self._response_time_seconds(),
            "metrics": metrics_to_dict(new_metrics),
            "deltas": metrics_to_dict(deltas),
            "cumulative_deltas": metrics_to_dict(cumulative_deltas),
            "reward_moe": reward,
            "base_moe": current_base_moe,
            "delta_moe": delta_moe,
            "sim_seed": self.current_sim_seed,
            "propagation_factor": propagation_factor,
            "propagation_penalty": propagation_penalty,
            "scenario": self.current_scenario_label,
            "scenario_name": self.current_scenario_name,
            "scenario_path": str(self.current_scenario_path),
            "ignition_pos": self.current_ignition_pos,
        }
        self.prev_metrics = new_metrics
        self.cumulative_reward += reward
        info["cumulative_reward"] = self.cumulative_reward
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

        if self.profile_timings:
            self._last_rss_before_reset_bytes = _process_rss_bytes()
        if self.gc_collect_on_reset:
            # Drop the prior simulation reference and force cycle collection
            # before allocating the new one. Tests whether retained sim
            # instances explain per-worker RSS growth across resets.
            self.sim = None
            gc.collect()

        reset_start = time.perf_counter() if self.profile_timings else 0.0
        self._init_simulation(int(sim_seed))
        if self.profile_timings:
            self._last_reset_wall_seconds = time.perf_counter() - reset_start
            self._last_rss_after_reset_bytes = _process_rss_bytes()
        observation = self._compute_state()
        propagation_factor = self._propagation_factor()
        moe_norms = self._current_moe_norms()
        if self.prev_metrics is not None:
            self.prev_total_moe = cumulative_moe_reward(
                self.prev_metrics.burnt_area,
                self.prev_metrics.cost,
                self.prev_metrics.emissions,
                self.prev_metrics.casualties,
                norms=moe_norms,
                propagation_factor=propagation_factor,
            )
        initial_base_moe = (
            base_moe_reward(
                self.prev_metrics.burnt_area,
                self.prev_metrics.cost,
                self.prev_metrics.emissions,
                self.prev_metrics.casualties,
                norms=moe_norms,
            )
            if self.prev_metrics is not None
            else 0.0
        )
        initial_propagation_penalty = propagation_penalty_value(propagation_flag=propagation_factor)
        self.last_info = {
            "decision_step": 0,
            "elapsed_minutes": 0.0,
            "decision_interval_minutes": self.decision_interval_minutes,
            "fire_detection_delay_minutes": self.fire_detection_delay_minutes,
            "response_time_seconds": self._response_time_seconds(),
            "metrics": metrics_to_dict(self.prev_metrics),
            "deltas": metrics_to_dict(Metrics(0.0, 0.0, 0.0, 0.0)),
            "cumulative_deltas": metrics_to_dict(Metrics(0.0, 0.0, 0.0, 0.0)),
            "reward_moe": 0.0,
            "base_moe": initial_base_moe,
            "delta_moe": 0.0,
            "sim_seed": sim_seed,
            "propagation_factor": propagation_factor,
            "propagation_penalty": initial_propagation_penalty,
            "scenario": self.current_scenario_label,
            "scenario_name": self.current_scenario_name,
            "scenario_path": str(self.current_scenario_path),
            "ignition_pos": self.current_ignition_pos,
        }
        if self.profile_timings:
            self.last_info["reset_wall_seconds"] = self._last_reset_wall_seconds
            self.last_info["reset_free_burn_wall_seconds"] = (
                self._last_reset_free_burn_wall_seconds
            )
            self.last_info["step_wall_seconds"] = 0.0
            self.last_info["rss_before_reset_bytes"] = (
                self._last_rss_before_reset_bytes
            )
            self.last_info["rss_after_reset_bytes"] = (
                self._last_rss_after_reset_bytes
            )
        return observation, self.last_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.sim is not None
        if self.done:
            raise RuntimeError("step called on terminated environment.")

        step_start = time.perf_counter() if self.profile_timings else 0.0
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
            propagation_factor = float(info.get("propagation_factor", 0.0))
            moe_norms = self._current_moe_norms()
            final_total_moe = cumulative_moe_reward(
                metrics.burnt_area,
                metrics.cost,
                metrics.emissions,
                metrics.casualties,
                norms=moe_norms,
                propagation_factor=propagation_factor,
            )
            info["reward_moe"] = reward
            info["propagation_factor"] = propagation_factor
            info["episode_summary"] = {
                "total_decision_steps": self.current_step,
                "total_minutes": self.current_step
                * (self.decision_interval.total_seconds() / 60.0),
                "final_metrics": metrics_to_dict(metrics),
                "moe_cumulative_reward": final_total_moe,
                "propagation_factor": propagation_factor,
                "scenario": self.current_scenario_label,
                "scenario_name": self.current_scenario_name,
                "scenario_path": str(self.current_scenario_path),
                "ignition_pos": self.current_ignition_pos,
            }

        if self.profile_timings:
            info["step_wall_seconds"] = time.perf_counter() - step_start
            info["reset_wall_seconds"] = self._last_reset_wall_seconds
            info["reset_free_burn_wall_seconds"] = (
                self._last_reset_free_burn_wall_seconds
            )
            info["rss_before_reset_bytes"] = self._last_rss_before_reset_bytes
            info["rss_after_reset_bytes"] = self._last_rss_after_reset_bytes
        self.last_info = info
        return observation, reward, terminated, truncated, info



class TrainingLogger(BaseCallback):
    """Capture per-step training data and per-episode summaries."""

    def __init__(
        self,
        decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
        progress_file_tag: str = "run",
        output_dir: Path = SCENARIOS_DIR / "outputs",
        profile_timings: bool = False,
        controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
    ):
        super().__init__()
        self.training_step_records: list[dict[str, Any]] = []
        self.episode_summaries: list[dict[str, Any]] = []
        self.completed_episodes = 0
        self.controlled_agent_count = int(controlled_agent_count)
        if self.controlled_agent_count <= 0:
            raise ValueError("controlled_agent_count must be > 0")
        self.decision_interval_minutes = decision_interval_minutes
        self.progress_file_tag = progress_file_tag
        self.output_dir = Path(output_dir)
        self._global_scenario_ts_counts: dict[str, int] = {}
        self.profile_timings = bool(profile_timings)
        # Rolling per-scenario timing accumulators reset each rollout.
        self._timing_step: dict[str, list[float]] = {}
        self._timing_reset_totals: dict[str, list[float]] = {}
        self._timing_reset_free_burn: dict[str, list[float]] = {}
        self._seen_reset_keys: set[tuple[int, str, int | None]] = set()
        # Latest observed RSS per worker (env_idx -> bytes); persists
        # across rollouts so we can detect monotonic growth.
        self._worker_latest_rss: dict[int, int] = {}
        # Per-rollout reset RSS samples per scenario (bytes after reset).
        self._rss_after_resets: dict[str, list[int]] = {}
        self._worker_scenario: dict[int, str] = {}
        self._rollout_index: int = 0

    def _reset_timing_accumulators(self) -> None:
        self._timing_step = {}
        self._timing_reset_totals = {}
        self._timing_reset_free_burn = {}
        self._seen_reset_keys = set()
        self._rss_after_resets = {}

    def _print_timing_summary(self) -> None:
        scenarios = sorted(
            set(self._timing_step)
            | set(self._timing_reset_totals)
            | set(self._timing_reset_free_burn)
            | set(self._rss_after_resets)
        )
        if not scenarios:
            return
        rollout_idx = self._rollout_index
        print(f"[profile-timings] rollout {rollout_idx} summary:")
        header = (
            f"  {'scenario':<24} "
            f"{'step_n':>7} {'step_mean':>10} {'step_p95':>9} "
            f"{'reset_n':>8} {'reset_mean':>11} {'free_burn_mean':>15} "
            f"{'rss_mean_MB':>12} {'rss_max_MB':>11} {'wkr_max_MB':>11}"
        )
        print(header)
        # Latest RSS per worker grouped by scenario.
        worker_rss_by_scenario: dict[str, list[int]] = {}
        for env_idx, rss in self._worker_latest_rss.items():
            scen = self._worker_scenario.get(env_idx, "unknown")
            worker_rss_by_scenario.setdefault(scen, []).append(rss)
        for scenario in scenarios:
            steps = np.asarray(self._timing_step.get(scenario, []), dtype=float)
            resets = np.asarray(self._timing_reset_totals.get(scenario, []), dtype=float)
            frees = np.asarray(
                self._timing_reset_free_burn.get(scenario, []), dtype=float
            )
            rss_samples = np.asarray(
                self._rss_after_resets.get(scenario, []), dtype=float
            )
            worker_rss = np.asarray(
                worker_rss_by_scenario.get(scenario, []), dtype=float
            )
            step_mean = float(steps.mean()) if steps.size else float("nan")
            step_p95 = float(np.quantile(steps, 0.95)) if steps.size else float("nan")
            reset_mean = float(resets.mean()) if resets.size else float("nan")
            free_mean = float(frees.mean()) if frees.size else float("nan")
            rss_mean_mb = (
                float(rss_samples.mean()) / (1024 * 1024)
                if rss_samples.size
                else float("nan")
            )
            rss_max_mb = (
                float(rss_samples.max()) / (1024 * 1024)
                if rss_samples.size
                else float("nan")
            )
            worker_max_mb = (
                float(worker_rss.max()) / (1024 * 1024)
                if worker_rss.size
                else float("nan")
            )
            print(
                f"  {scenario:<24} "
                f"{steps.size:>7d} {step_mean:>10.4f} {step_p95:>9.4f} "
                f"{resets.size:>8d} {reset_mean:>11.4f} {free_mean:>15.4f} "
                f"{rss_mean_mb:>12.1f} {rss_max_mb:>11.1f} {worker_max_mb:>11.1f}"
            )

    def _record_timings(self, env_idx: int, info: dict[str, Any]) -> None:
        scenario = info.get("scenario_name") or info.get("scenario") or "unknown"
        self._worker_scenario[env_idx] = scenario
        step_wall = info.get("step_wall_seconds")
        if step_wall is not None and step_wall > 0.0:
            self._timing_step.setdefault(scenario, []).append(float(step_wall))
        rss_after = info.get("rss_after_reset_bytes")
        if rss_after is not None and rss_after > 0:
            self._worker_latest_rss[env_idx] = int(rss_after)
        reset_total = info.get("reset_wall_seconds")
        if reset_total is not None:
            # Deduplicate: each reset is reported on every step of the episode;
            # key on (env_idx, scenario_name, sim_seed) so we count it once.
            key = (env_idx, scenario, info.get("sim_seed"))
            if key not in self._seen_reset_keys:
                self._seen_reset_keys.add(key)
                self._timing_reset_totals.setdefault(scenario, []).append(
                    float(reset_total)
                )
                free_burn = info.get("reset_free_burn_wall_seconds")
                if free_burn is not None:
                    self._timing_reset_free_burn.setdefault(scenario, []).append(
                        float(free_burn)
                    )
                if rss_after is not None and rss_after > 0:
                    self._rss_after_resets.setdefault(scenario, []).append(
                        int(rss_after)
                    )

    def _on_rollout_end(self) -> None:
        if self._global_scenario_ts_counts:
            self.training_env.env_method(
                "set_global_scenario_counts",
                self._global_scenario_ts_counts,
            )
        if self.profile_timings:
            self._rollout_index += 1
            self._print_timing_summary()
            self._reset_timing_accumulators()

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
            metrics_dict = _ensure_metrics_dict(info.get("metrics"))
            deltas_dict = _ensure_metrics_dict(info.get("deltas"))
            cumulative_dict = _ensure_metrics_dict(info.get("cumulative_deltas"))
            record = {
                "phase": "training",
                "env_index": env_idx,
                "sim_seed": info.get("sim_seed"),
                "scenario": info.get("scenario"),
                "scenario_name": info.get("scenario_name"),
                "scenario_path": info.get("scenario_path"),
                "decision_step": info.get("decision_step"),
                "elapsed_minutes": info.get("elapsed_minutes"),
                "decision_interval_minutes": info.get(
                    "decision_interval_minutes", self.decision_interval_minutes
                ),
                "fire_detection_delay_minutes": info.get(
                    "fire_detection_delay_minutes"
                ),
                "response_time_seconds": info.get("response_time_seconds"),
                "reward_moe": info.get("reward_moe"),
                "base_moe": info.get("base_moe"),
                "step_reward": reward,
                "propagation_factor": info.get("propagation_factor"),
                "propagation_penalty": info.get("propagation_penalty"),
                "last_interval": 1 if "episode_summary" in info else 0,
                "delta_moe": info.get("delta_moe"),
            }
            for idx, combo in enumerate(action_labels[: self.controlled_agent_count]):
                record[f"agent_{idx}_select_poi"] = combo[0].value
                record[f"agent_{idx}_track_poi"] = combo[1].value
                record[f"agent_{idx}_suppress"] = combo[2].value
            record.update(metrics_dict)
            record.update({f"{key}_delta": value for key, value in deltas_dict.items()})
            record.update(
                {f"{key}_cumulative_delta": value for key, value in cumulative_dict.items()}
            )
            if self.profile_timings:
                record["step_wall_seconds"] = info.get("step_wall_seconds")
                record["reset_wall_seconds"] = info.get("reset_wall_seconds")
                record["reset_free_burn_wall_seconds"] = info.get(
                    "reset_free_burn_wall_seconds"
                )
                self._record_timings(env_idx, info)
            self.training_step_records.append(record)

            if "episode_summary" in info:
                self.completed_episodes += 1
                summary_record = dict(info["episode_summary"])
                summary_record["sim_seed"] = info.get("sim_seed")
                summary_record["propagation_factor"] = info.get("propagation_factor")
                summary_record["delta_moe_total"] = info.get("cumulative_reward")
                summary_record["fire_detection_delay_minutes"] = info.get(
                    "fire_detection_delay_minutes"
                )
                summary_record["response_time_seconds"] = info.get(
                    "response_time_seconds"
                )
                summary_record["scenario"] = info.get("scenario")
                summary_record["scenario_name"] = info.get("scenario_name")
                summary_record["scenario_path"] = info.get("scenario_path")
                self.episode_summaries.append(summary_record)
                if (
                    self.completed_episodes % LOG_INTERVAL_SUMMARY_EPISODES
                    == 0
                ):
                    self._export_progress(
                        self.completed_episodes,
                        export_steps=(
                            self.completed_episodes
                            % LOG_INTERVAL_STEPS_EPISODES
                            == 0
                        ),
                        export_summary=True,
                    )
            scenario_name = info.get("scenario_name")
            if scenario_name:
                self._global_scenario_ts_counts[scenario_name] = (
                    self._global_scenario_ts_counts.get(scenario_name, 0) + 1
                )
        return True

    def _on_training_end(self) -> None:
        # Flush any remaining buffered step records to avoid losing the tail chunk.
        if self.training_step_records:
            self._export_progress(
                self.completed_episodes,
                export_steps=True,
                export_summary=False,
            )

    def _export_progress(
        self,
        episode_count: int,
        *,
        export_steps: bool,
        export_summary: bool,
    ) -> None:
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if export_steps and self.training_step_records:
            step_path = output_dir / (
                f"results_{self.progress_file_tag}_{episode_count}_steps.csv"
            )
            batch_index = (
                max(episode_count - 1, 0) // LOG_INTERVAL_STEPS_EPISODES + 1
            )
            for record in self.training_step_records:
                record["batch"] = batch_index
            _write_records(self.training_step_records, step_path)
            print(f"Per-step log written to {step_path}")
            # Prevent unbounded in-memory growth of per-step records.
            self.training_step_records.clear()
        if export_summary and self.episode_summaries:
            summary_path = output_dir / (
                f"results_{self.progress_file_tag}_{episode_count}_summary.csv"
            )
            _write_records(self.episode_summaries, summary_path)
            print(f"Episode summary log written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on one scenario or switch across multiple scenarios."
    )
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
        "--fire-detection-delay-minutes",
        type=float,
        default=None,
        help=(
            "Optional override for delay before first decision step (minutes). "
            "If omitted, uses scenario response_time / 60 for consistency."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--policy-arch",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--switch-scenario",
        action="store_true",
        help=(
            "Use multiple scenarios during training. With num-envs=1, one "
            "scenario is sampled per reset; with num-envs>1, scenarios are "
            "assigned round-robin per worker."
        ),
    )
    parser.add_argument(
        "--switch-scenarios",
        nargs="+",
        metavar="SCENARIO",
        default=None,
        help=(
            "Optional scenario list used for switch-scenario training "
            "(JSON names in inputs/ or absolute paths). "
            "If omitted, uses the default switch set."
        ),
    )
    ignition_group = parser.add_mutually_exclusive_group()
    ignition_group.add_argument(
        "--switch-ignition-1",
        action="store_true",
        help=(
            "Randomize ignition each episode within a 100x100 cell box around "
            "the scenario's original ignition center (excludes water/urban)."
        ),
    )
    ignition_group.add_argument(
        "--switch-ignition-2",
        action="store_true",
        help=(
            "Randomize ignition each episode within a map-centered box whose "
            f"edges sit {IGNITION_V2_MARGIN_RATIO:.0%} of the map edge length "
            "from each map boundary (excludes water/urban)."
        ),
    )
    parser.add_argument(
        "--ignition-inputs",
        nargs="+",
        metavar="IGNITION_SPEC",
        help=(
            "Ignition specifications to be parsed for future "
            "--switch-ignition support."
        ),
    )
    parser.add_argument(
        "--log-output",
        help="Optional output path for action/reward log (.csv recommended).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCENARIOS_DIR / "outputs"),
        help=(
            "Base directory for training outputs (interim progress, final logs, "
            "summary, model save when relative path is provided)."
        ),
    )
    parser.add_argument(
        "--summary-output",
        help=(
            "Optional output path for final per-simulation episode summary "
            "(.csv or .xlsx)."
        ),
    )
    parser.add_argument(
        "--progress-file-tag",
        help=(
            "Optional tag for interim progress files. "
            "Writes results_<tag>_<N>_steps.csv and "
            "results_<tag>_<N>_summary.csv."
        ),
    )
    parser.add_argument(
        "--save-model",
        help="Optional path to save the trained PPO policy.",
    )
    parser.add_argument(
        "--load-model",
        help="Optional path to a saved PPO policy to resume training from.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for PPO (e.g., auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--vec-start-method",
        choices=("auto", "fork", "spawn", "forkserver"),
        default="auto",
        help=(
            "SubprocVecEnv start method. auto uses fork on Linux "
            "(lower memory), spawn otherwise."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel wildfire environments to run (>=1).",
    )
    parser.add_argument(
        "--controlled-agent-count",
        type=int,
        default=CONTROLLED_AGENT_COUNT,
        help=(
            "Number of aircraft the policy controls (firefighters[:N]). "
            f"Default {CONTROLLED_AGENT_COUNT}. The scenario must define at "
            "least this many aircraft."
        ),
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help=(
            "Measure per-step and per-reset wall time and RSS in each env, "
            "log them in the step CSV, and print a per-scenario summary at "
            "the end of every PPO rollout."
        ),
    )
    parser.add_argument(
        "--gc-collect-on-reset",
        action="store_true",
        help=(
            "Drop the prior simulation reference and run gc.collect() at the "
            "start of every reset(). Diagnostic for testing whether reference "
            "cycles between sim/env/atmosphere/etc. cause RSS to grow across "
            "resets."
        ),
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help=(
            "PPO target KL divergence (SB3 short-circuits remaining epochs "
            "if approx_kl exceeds 1.5*target_kl on any minibatch). Healthy "
            "range 0.015-0.02. Default None disables the check."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help=(
            "Number of optimization passes over each rollout's data. "
            "Lower (3-5) reduces policy drift per update; higher (7-10) "
            "extracts more learning per rollout. Default 7."
        ),
    )
    parser.add_argument(
        "--lr-decay-exponent",
        type=float,
        default=LR_DECAY_EXPONENT,
        help=(
            "Exponent of the LR schedule lr(p)=initial*p^exponent. "
            "0.65 (default) decays aggressively early; 1.0 is linear; "
            "0.0 disables decay. Use 1.0 for long runs where you want "
            "the actor to keep moving past iter ~30."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Discount factor for future rewards. 0.99 (default) gives "
            "an effective horizon ~100 steps; 0.95 ~20 steps. Short-"
            "episode regimes (Pyrenees ~6 steps) may benefit from 0.95."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help=(
            "If > 0, save an intermediate model snapshot every N PPO "
            "timesteps to <output-dir>/checkpoints/. SB3 internally divides "
            "by num_envs so the actual cadence is N // num_envs collected "
            "samples per worker. Use to guard against SLURM wall-limit "
            "termination of long runs."
        ),
    )
    args = parser.parse_args()

    if args.policy_arch:
        print(
            "Ignoring legacy --policy-arch="
            f"{args.policy_arch!r}; using fixed pi=[512,256,128] / "
            "vf=[512,256,128]."
        )

    if args.switch_ignition_2:
        switch_ignition_mode = 2
    elif args.switch_ignition_1:
        switch_ignition_mode = 1
    else:
        switch_ignition_mode = 0
    ignition_switch_config = {
        "mode": switch_ignition_mode,
        "inputs": tuple(args.ignition_inputs or ()),
    }
    if switch_ignition_mode == 1:
        print(
            "--switch-ignition-1 enabled: ignition randomized each episode "
            "in a 100x100 cell box around the scenario's original ignition."
        )
    elif switch_ignition_mode == 2:
        print(
            "--switch-ignition-2 enabled: ignition randomized each episode "
            f"in a map-centered box (margin {IGNITION_V2_MARGIN_RATIO:.0%} "
            "of map edge from each boundary)."
        )
    if switch_ignition_mode != 0 and ignition_switch_config["inputs"]:
        print(
            "--ignition-inputs are currently parsed but ignored by "
            "the automatic ignition sampler."
        )

    use_switch_scenario = bool(args.switch_scenario or args.switch_scenarios)
    if use_switch_scenario:
        switch_names = (
            tuple(args.switch_scenarios)
            if args.switch_scenarios
            else SWITCH_SCENARIO_NAMES
        )
        switch_scenario_paths = tuple(_resolve_scenario(name) for name in switch_names)
        if not switch_scenario_paths:
            raise ValueError("--switch-scenarios must contain at least one scenario.")
        aircraft_source_scenario_path = None
        scenario_path = switch_scenario_paths[0]
        switch_names_pretty = ", ".join(path.name for path in switch_scenario_paths)
        if args.num_envs > 1:
            print(
                "--switch-scenario enabled: ignoring --scenario and assigning "
                f"{switch_names_pretty} to workers in round-robin order "
                "using each scenario's fleet definition."
            )
        else:
            print(
                "--switch-scenario enabled: ignoring --scenario and sampling "
                f"from {switch_names_pretty} each reset "
                "using each scenario's fleet definition."
            )
    else:
        scenario_path = _resolve_scenario(args.scenario)
        switch_scenario_paths = None
        aircraft_source_scenario_path = None
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.controlled_agent_count <= 0:
        raise ValueError("--controlled-agent-count must be > 0")
    # The policy commands firefighters[:controlled_agent_count]; validate up
    # front so a mismatched scenario cannot silently shrink the controlled set.
    _fleet_paths = (
        switch_scenario_paths if use_switch_scenario and switch_scenario_paths
        else (scenario_path,)
    )
    for _fleet_path in _fleet_paths:
        _fleet_size = _scenario_agent_count(_fleet_path)
        if _fleet_size < args.controlled_agent_count:
            raise ValueError(
                f"--controlled-agent-count={args.controlled_agent_count} exceeds "
                f"the {_fleet_size} aircraft defined in {Path(_fleet_path).name}."
            )
    if args.decision_interval_minutes <= 0:
        raise ValueError("--decision-interval-minutes must be > 0")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be >= 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in the interval (0, 1]")
    if (
        args.fire_detection_delay_minutes is not None
        and args.fire_detection_delay_minutes < 0
    ):
        raise ValueError("--fire-detection-delay-minutes must be >= 0")
    if args.progress_file_tag:
        progress_file_tag = args.progress_file_tag.strip().replace(" ", "_")
    elif use_switch_scenario:
        progress_file_tag = "switch_scenarios"
    else:
        scenario_tag = scenario_path.stem.lower().replace(" ", "_")
        progress_file_tag = f"single_{scenario_tag}"
    vec_start_method = None if args.vec_start_method == "auto" else args.vec_start_method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_timesteps = args.timesteps if args.timesteps is not None else 1680
    total_timesteps = max(default_timesteps, args.num_envs, 2)

    ts_budget_per_scenario: float | None = None
    if use_switch_scenario and switch_scenario_paths and args.num_envs == 1:
        ts_budget_per_scenario = float(total_timesteps) / len(switch_scenario_paths)
        print(
            f"Scenario balance budget: {total_timesteps} total timesteps / "
            f"{len(switch_scenario_paths)} scenarios = "
            f"{ts_budget_per_scenario:.0f} timesteps per scenario"
        )
    elif use_switch_scenario and switch_scenario_paths and args.num_envs > 1:
        worker_assignments = [
            switch_scenario_paths[idx % len(switch_scenario_paths)].name
            for idx in range(args.num_envs)
        ]
        assignment_counts = {
            path.name: worker_assignments.count(path.name)
            for path in switch_scenario_paths
        }
        print(
            "Worker scenario assignment (round-robin): "
            + ", ".join(
                f"{scenario} -> {count} workers"
                for scenario, count in assignment_counts.items()
            )
        )

    if args.num_envs == 1:
        train_env = WildfireHourlyEnv(
            scenario_path,
            decision_interval_minutes=args.decision_interval_minutes,
            fire_detection_delay_minutes=args.fire_detection_delay_minutes,
            switch_scenario=use_switch_scenario,
            switch_scenario_paths=switch_scenario_paths,
            aircraft_source_scenario_path=aircraft_source_scenario_path,
            switch_ignition_mode=switch_ignition_mode,
            ts_budget_per_scenario=ts_budget_per_scenario,
            profile_timings=args.profile_timings,
            gc_collect_on_reset=args.gc_collect_on_reset,
            include_scenario_features=use_switch_scenario,
            controlled_agent_count=args.controlled_agent_count,
        )
    else:
        train_env = _build_vector_env(
            args.num_envs,
            scenario_path,
            args.decision_interval_minutes,
            args.fire_detection_delay_minutes,
            use_switch_scenario,
            switch_scenario_paths,
            aircraft_source_scenario_path,
            switch_ignition_mode,
            vec_start_method,
            ts_budget_per_scenario=ts_budget_per_scenario,
            profile_timings=args.profile_timings,
            gc_collect_on_reset=args.gc_collect_on_reset,
            controlled_agent_count=args.controlled_agent_count,
        )
    max_steps_allowed = max(1, total_timesteps // args.num_envs)
    rollout_steps = int(min(ROLLOUT_STEPS_PER_ENV, max_steps_allowed))
    effective_batch = rollout_steps * args.num_envs
    dynamic_minibatches = min(NUM_MINIBATCH, max(1, effective_batch))
    batch_size = max(2, effective_batch // dynamic_minibatches)
    print(
        "PPO rollout config: "
        f"num_envs={args.num_envs}, n_steps={rollout_steps}, "
        f"effective_batch={effective_batch}, minibatches={dynamic_minibatches}, "
        f"batch_size={batch_size}"
    )
    if args.fire_detection_delay_minutes is None:
        if use_switch_scenario:
            if args.num_envs > 1:
                print(
                    "Decision start delay: scenario response_time / 60 "
                    "(computed per worker-assigned scenario)."
                )
            else:
                print(
                    "Decision start delay: scenario response_time / 60 "
                    "(computed per selected scenario each episode)."
                )
            for switch_path in switch_scenario_paths or ():
                response_delay_min = _scenario_response_time_seconds(switch_path) / 60.0
                print(f"  {switch_path.name}: {response_delay_min:.1f} minutes")
        else:
            response_delay_min = _scenario_response_time_seconds(scenario_path) / 60.0
            print(
                "Decision start delay: "
                f"{scenario_path.name} response_time / 60 = "
                f"{response_delay_min:.1f} minutes"
            )
    else:
        print(
            f"Decision start delay override: {args.fire_detection_delay_minutes} minutes"
        )
    print(f"Allowed tactic combinations: {len(TACTIC_COMBINATIONS)}")
    print(f"Controlled aircraft: {args.controlled_agent_count}")
    if args.profile_timings:
        print("Profiling: per-step/reset wall time and RSS enabled.")
    if args.gc_collect_on_reset:
        print("Diagnostic: gc.collect() and prior-sim drop at reset start enabled.")
    if args.num_envs > 1:
        effective_vec_method = (
            vec_start_method
            if vec_start_method is not None
            else ("fork" if sys.platform.startswith("linux") else "spawn")
        )
        print(f"Vector env start method: {effective_vec_method}")

    step_logger = TrainingLogger(
        decision_interval_minutes=args.decision_interval_minutes,
        progress_file_tag=progress_file_tag,
        output_dir=output_dir,
        profile_timings=args.profile_timings,
        controlled_agent_count=args.controlled_agent_count,
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
    if args.load_model:
        load_path = Path(args.load_model)
        if not load_path.exists():
            raise FileNotFoundError(f"Unable to load model, path not found: {load_path}")
        model = PPO.load(
            str(load_path),
            env=train_env,
            device=device,
            print_system_info=False,
        )
        model.n_steps = rollout_steps
        model.batch_size = batch_size
        model.learning_rate = _make_lr_schedule(
            args.learning_rate,
            args.lr_decay_exponent,
        )
        model.n_epochs = args.n_epochs
        model.target_kl = args.target_kl
        model.gamma = args.gamma
        model._setup_lr_schedule()
        model._setup_rollout_buffer()
        print(f"Loaded PPO policy from {load_path}")
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        )
        print(
            "Policy arch: independent pi=[512,256,128] / "
            "vf=[512,256,128], Tanh"
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=_make_lr_schedule(args.learning_rate, args.lr_decay_exponent),
            verbose=1,
            tensorboard_log=None,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            target_kl=args.target_kl,
            device=device,
            policy_kwargs=policy_kwargs,
        )
        print(
            f"PPO hyperparams: lr={args.learning_rate} (decay^{args.lr_decay_exponent}), "
            f"n_epochs={args.n_epochs}, gamma={args.gamma}, target_kl={args.target_kl}"
        )
    if args.checkpoint_interval > 0:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # SB3 calls the callback every collected sample, so it divides
        # save_freq by n_envs internally; pass the per-env cadence so the
        # user-supplied number is in total PPO timesteps.
        save_freq_per_env = max(1, args.checkpoint_interval // max(1, args.num_envs))
        ckpt_cb = CheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=str(checkpoint_dir),
            name_prefix=(args.progress_file_tag or "ppo_ckpt"),
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1,
        )
        callbacks = CallbackList([step_logger, ckpt_cb])
        print(
            f"Checkpointing every {args.checkpoint_interval} timesteps "
            f"({save_freq_per_env}/env) to {checkpoint_dir}"
        )
    else:
        callbacks = step_logger

    start_time = time.perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except (KeyboardInterrupt, SystemExit) as err:
        # Best-effort save on signal-driven termination (SLURM time-limit).
        print(f"\nTraining interrupted ({type(err).__name__}); saving rescue model.")
        try:
            rescue_path = output_dir / "rescue_on_interrupt.zip"
            model.save(str(rescue_path))
            print(f"Rescue model saved to {rescue_path}")
        except Exception as save_err:  # noqa: BLE001
            print(f"Rescue save failed: {save_err}")
        raise
    training_duration = time.perf_counter() - start_time
    train_env.close()

    print(f"Training complete in {training_duration:.2f} seconds.")

    if args.save_model:
        save_path = _resolve_output_path(
            output_dir,
            args.save_model,
            "trained_policy.zip",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Trained model saved to {save_path}")

    all_step_records = step_logger.training_step_records
    if all_step_records:
        output_path = _resolve_output_path(
            output_dir,
            args.log_output,
            "sparse_rewards_results.csv",
        )
        _write_records(all_step_records, output_path)
        print(f"\nDetailed action/reward log written to {output_path}")

    if step_logger.episode_summaries:
        summary_path = _resolve_output_path(
            output_dir,
            args.summary_output,
            "training_episode_summaries.csv",
        )
        _write_records(step_logger.episode_summaries, summary_path)
        print(f"Episode training summary written to {summary_path}")

        if args.summary_output is None:
            summary_excel_path = output_dir / "training_episode_summaries.xlsx"
            try:
                _write_records(step_logger.episode_summaries, summary_excel_path)
            except Exception as err:
                print(
                    "Unable to write final episode summary Excel file "
                    f"({summary_excel_path}): {err}"
                )
            else:
                print(f"Episode training summary written to {summary_excel_path}")


if __name__ == "__main__":
    main()
