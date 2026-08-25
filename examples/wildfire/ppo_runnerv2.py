#!/usr/bin/env python3

"""
bismillahirrahmanirrahim, elhamdulillah, esselatu vesselamu ala rasulillah
Train a PPO agent to select hourly suppression tactics on the Palisades scenario.

The environment exposes the same set of state variables logged by
``hourly_metrics.py`` and expects discrete tactic selections for controlled
aircraft. Each environment step advances the simulation by one decision
interval; the reward is the mission-effectiveness score (MoE) accumulated over
that interval.

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
import signal
import time
from typing import Any, Sequence
from datetime import timedelta
import warnings

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
from sosid.environment.terrain import (
    CASUALTIES_TABLE,
    COMBUSTIBILITY_TABLE,
    COSTS_TABLE,
    EMISSIONS_TABLE,
    FEATURES_COLOR_TABLE,
    TerrainTypes,
)
from sosid.model.transform import (
    gps_to_mercator,
    gps_to_pos,
    index_to_pos,
    pos_to_gps,
    pos_to_index,
)
from sosid.typedef import GridDescriptor


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
from examples.wildfire.fire_model.states import (
    BURNT,
    COMBUSTIBLE,
    EARLY_BURNING,
    EXTINGUISHING,
    FULL_BURNING,
    NONFLAMMABLE,
    SUPPRESSED,
)
from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.paths import TERRAIN_DIR
from examples.wildfire.simulation import (
    IgnitionCenterInput,
    M2_TO_HECTARS,
    PEOPLE_PER_HOUSEHOLD,
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
CONTROLLED_AGENT_COUNT = 3
AGENT_FEATURE_COUNT = 3  # per controlled aircraft: x, y, altitude
# Normalization bound for aircraft altitude (m). cruise_altitude in the
# aircraft profile is 3000 m; altitudes range [0, cruise], so 3000 covers the
# full flight envelope.
ALTITUDE_NORM_M = 3000.0
AIRCRAFT_GROUP_SIZE = 3
TACTIC_DISTRIBUTION_INDIVIDUAL = "individual"
TACTIC_DISTRIBUTION_GROUP = "group"
SUPPORTED_TACTIC_DISTRIBUTIONS: tuple[str, ...] = (
    TACTIC_DISTRIBUTION_INDIVIDUAL,
    TACTIC_DISTRIBUTION_GROUP,
)
DEFAULT_DECISION_INTERVAL_MINUTES = 10
# Max number of fastest active fire cells inspected for front-derived state.
# This is an upper bound: if fewer cells are burning, we use what exists.
DEFAULT_STATE_FIRE_FRONTS = 5
# Require a meaningful uphill spread multiplier before a front votes for
# topography_flag. This keeps flat/noisy terrain from looking topographic.
TOPOGRAPHY_SLOPE_FACTOR_THRESHOLD = 1.05
# Vegetation threat rule: inspect a radius-3 Moore neighborhood around each
# selected front and count fuel-bearing cells. The front votes positive when
# the count is strictly greater than this threshold.
VEGETATION_NEIGHBOR_RADIUS = 3
VEGETATION_MIN_COMBUSTIBLE_NEIGHBORS = 5
# A front votes for indirect_flag when it is close enough to the current
# indirect/fire-line plan to make line-following tactics relevant.
INDIRECT_FRONT_DISTANCE_THRESHOLD_M = 250.0
POI_CANDIDATE_QUANTILE = 0.95
MAX_POI_CANDIDATES = 4000
# --- Directional state space (per-flank projected threat) --------------------
# A burning cell is an active-perimeter front candidate when its spread rate is
# above this floor (interior / quenched cells are ~0).
DIRECTIONAL_SPREAD_EPS = 1e-6
# Each flank is projected T = decision_interval_minutes ahead; the threat search
# radius around the landing point scales with the projected reach.
DIRECTIONAL_SEARCH_ALPHA = 0.4
DIRECTIONAL_R_MIN_CELLS = 3
DIRECTIONAL_R_MAX_CELLS = 20
# Hard cap on ray-march length so a fast flank / long interval stays bounded.
DIRECTIONAL_MAX_RAY_CELLS = 400
# Slope multiplier that maps to full topography urgency (must be > 1).
DIRECTIONAL_SLOPE_REF = 1.5
# A flank's representative cell keeps its own propagation bearing only when that
# bearing points outward within this cosine threshold (0.0 == within +/-90 deg of
# the radial outward direction); otherwise the projection uses the outward
# bearing so a landing can never be aimed into the already-burnt interior.
DIRECTIONAL_ALIGN_MIN = 0.0
# VIP urgency is a time-to-impact score: the landing->nearest-VIP distance is
# scaled by how far the flank travels in VIP_HORIZONS decision intervals
# (ROS * VIP_HORIZONS * T metres). The sensing range therefore grows with flank
# speed, and the feature rises while there is still ~1 h of lead time to act.
DIRECTIONAL_VIP_HORIZONS = 6
# Fixed distance scales (metres) for the indirect-line and water channels. The
# earlier code reused the tiny vegetation search radius for the indirect line
# (so it was ~always 0) and the whole map diagonal for water (so it was a near
# constant ~0.97). 1 km is an operationally meaningful approach distance to the
# planned containment line; 5 km is a representative scooper shuttle radius, so
# sectors near water now read distinctly higher than sectors far from it.
DIRECTIONAL_LINE_SCALE_M = 1000.0
DIRECTIONAL_WATER_SCALE_M = 5000.0
# Upper clamp for the slope-factor exponent: math.exp raises OverflowError
# above ~709.8, reached at terrain slopes > ~74.8 deg (cliff cells under
# projected landings). e^50 (~5e21) still dwarfs every threshold the factor
# is compared against, so clamping cannot change any decision.
SLOPE_FACTOR_EXP_MAX = 50.0
# Highest fuel flammability in COMBUSTIBILITY_TABLE (pasture); normalizes the
# vegetation fuel-escalation score to [0, 1].
MAX_COMBUSTIBILITY = 2.0

# --- Cell-size source (--cell-size) ------------------------------------------
# The scenario JSON's nominal cell_size is 5 m, but the sim indexes positions on
# a grid whose spacing is mercator_dimensions/grid_shape (~6-6.8 m away from the
# equator). Geometry that mixes the two (e.g. the directional flank projection)
# is skewed by that factor. --cell-size selects which value the directional
# geometry uses: "json" = the scenario's nominal cell_size; "code" = the actual
# projected metres/cell from ACTUAL_CELL_SIZE_M below (matching the sim's grid).
CELL_SIZE_SOURCE_JSON = "json"
CELL_SIZE_SOURCE_CODE = "code"
SUPPORTED_CELL_SIZE_SOURCES: tuple[str, ...] = (
    CELL_SIZE_SOURCE_JSON,
    CELL_SIZE_SOURCE_CODE,
)
# Actual mean metres/cell (0.5*(dim_x/cols + dim_y/rows)), keyed by terrain
# file_namespace. Computed from mercator_dimensions/grid_shape; cross-checked
# against grid_description at load (see _cache_static_state_scalers).
ACTUAL_CELL_SIZE_M: dict[str, float] = {
    "Pyrenees_2000x1996_5m": 6.8360,
    "Pyrenees_2000x1996_5m_reduced": 6.8360,
    "Palisades_2004x1996_5m": 6.0455,
    "Salamis_2004x1996_5m": 6.3376,
}
# These are the terrain types currently counted by
# WildfireSimulation.area_burnt_by_type(). Keep this tuple aligned with that
# method so the fast PPO reward metrics preserve the existing reward semantics.
DAMAGE_TERRAIN_TYPES: tuple[TerrainTypes, ...] = (
    TerrainTypes.NEEDLE_LITTER,
    TerrainTypes.FALLEN_LEAVES,
    TerrainTypes.GRASSES_WEEDS,
    TerrainTypes.CAREX_FORBS,
    TerrainTypes.PASTURE,
    TerrainTypes.PINUS,
    TerrainTypes.FIELD,
    TerrainTypes.RESIDENTIAL,
)
IGNITION_BOUNDARY_MARGIN_RATIO = 0.3
IGNITION_BOX_HALF_SIZE = 50  # half-side of 100×100 candidate box in grid cells
# --switch-ignition-3: same local box as v1, widened to 500×500 cells. Cells are
# `cell_size` (5 m) of real ground on every map, so this is a 2.5×2.5 km ground
# box regardless of latitude. Sizing in cells (not mercator metres) is what keeps
# it identical across maps: mercator metres inflate by 1/cos(lat), i.e. 21%
# at Palisades vs 36% at Pyrenees.
IGNITION_BOX_HALF_SIZE_V3 = 250
IGNITION_URBAN_BUFFER_M = 350.0  # min distance from any urban cell, meters
# --switch-ignition-4: 400×400 cells = 2.0×2.0 km ground (same cells-are-5m-of-
# ground reasoning as v3), paired with a relaxed 150 m urban keep-out so fires
# may start closer to the wildland-urban interface than modes 1/2/3 allow.
IGNITION_BOX_HALF_SIZE_V4 = 200
IGNITION_URBAN_BUFFER_M_V4 = 150.0
# --switch-ignition-2: center map box whose edges sit `IGNITION_V2_MARGIN_RATIO`
# of the map edge length away from each map boundary. Default 0.25 → box edge
# = (1 − 2·0.25) · map_edge = half the map.
IGNITION_V2_MARGIN_RATIO = 0.30

ROLLOUT_STEPS_PER_ENV = 144
NUM_MINIBATCH = 24

if ROLLOUT_STEPS_PER_ENV <= 0:
    raise ValueError("ROLLOUT_STEPS_PER_ENV must be positive.")
if NUM_MINIBATCH <= 0:
    raise ValueError("NUM_MINIBATCH must be positive.")
LOG_INTERVAL_SUMMARY_EPISODES = 100
LOG_INTERVAL_STEPS_EPISODES = 1000
LR_DECAY_EXPONENT = 0.70
SWITCH_SCENARIO_NAMES = (
    "Palisades copy.json",
    "Pyrenees.json",
    "Salamis.json",
)
# --missions splits one training budget between several maps. Which unit the
# split is expressed in follows the budget flag that sized the run:
# --simulations/--chain-total-simulations give every mission a share of the
# episode count, --timesteps/--chain-total-timesteps a share of the timesteps.
MISSION_BUDGET_SIMULATIONS = "simulations"
MISSION_BUDGET_TIMESTEPS = "timesteps"


def _make_lr_schedule(initial_lr: float, decay_exponent: float = LR_DECAY_EXPONENT):
    """Return a callable learning-rate schedule decaying with training progress.

    schedule(p) = initial_lr * p^decay_exponent, p = progress remaining (1 -> 0)

    Exponents below 1 keep the LR *above* the linear schedule for the whole
    run (p^e > p for 0 < p < 1) and then drop steeply over the last percent:
        decay_exponent = 0.70 (default): gentle decay, late collapse
                                         (0.5x initial at ~63% through)
        decay_exponent = 1.0:            linear decay (sb3's default behaviour)
                                         (0.5x initial at 50% through)
        decay_exponent = 0.0:            constant LR (no decay)
    Lower the exponent to hold the LR up longer; raise it to decay sooner.
    """
    base_lr = float(initial_lr)
    exponent = float(decay_exponent)

    def schedule(progress_remaining: float) -> float:
        progress = max(progress_remaining, 1e-8)
        return base_lr * (progress**exponent)

    return schedule


class EpisodeProgress:
    """Shared episode counter driving a simulation-based LR schedule.

    ``TrainingLogger`` bumps ``completed`` as episodes finish and the schedule
    reads it, so the LR decays over simulations instead of over timesteps.
    Deliberately tiny and free of references to the model, env or callbacks:
    SB3 cloudpickles the schedule (and therefore this object) into the saved
    zip. A resumed chunk seeds ``completed`` from the previous chunk's episode
    count, which is what keeps the decay continuous across SLURM jobs.
    """

    def __init__(self, completed: int = 0, total: int = 1):
        self.completed = int(completed)
        self.total = max(1, int(total))

    def progress_remaining(self) -> float:
        return max(0.0, 1.0 - self.completed / float(self.total))


def _make_episode_lr_schedule(
    initial_lr: float,
    decay_exponent: float,
    progress: EpisodeProgress,
):
    """LR schedule decaying with completed simulations, not timesteps.

    Same curve as ``_make_lr_schedule`` — initial_lr * p^decay_exponent — but
    p comes from ``progress`` rather than from SB3's timestep-derived
    ``progress_remaining``, which is ignored. Used for chains budgeted in
    simulations, where the timestep total is only a loose ceiling (episodes
    end well before ``max_runtime``) and would leave the LR nearly flat.
    """
    base_lr = float(initial_lr)
    exponent = float(decay_exponent)

    def schedule(_progress_remaining: float) -> float:
        progress_value = max(progress.progress_remaining(), 1e-8)
        return base_lr * (progress_value**exponent)

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
    mission_budgets: dict[str, float] | None = None,
    mission_budget_mode: str = MISSION_BUDGET_SIMULATIONS,
    initial_scenario_episode_counts: dict[str, int] | None = None,
    initial_scenario_timestep_counts: dict[str, int] | None = None,
    gc_collect_on_reset: bool = False,
    include_scenario_features: bool | None = None,
    state_fire_fronts: int = DEFAULT_STATE_FIRE_FRONTS,
    state_space: str = "old",
    tactic_distribution: str = TACTIC_DISTRIBUTION_INDIVIDUAL,
    aircraft_group_size: int = AIRCRAFT_GROUP_SIZE,
    group_sizes: Sequence[int] | None = None,
    controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
    water_set: int | None = None,
    cell_size_source: str = "json",
    enable_adaptive_time_step: bool | None = None,
    adaptive_step_size_factor: float | None = None,
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
            mission_budgets=mission_budgets,
            mission_budget_mode=mission_budget_mode,
            initial_scenario_episode_counts=initial_scenario_episode_counts,
            initial_scenario_timestep_counts=initial_scenario_timestep_counts,
            gc_collect_on_reset=gc_collect_on_reset,
            include_scenario_features=include_scenario_features,
            state_fire_fronts=state_fire_fronts,
            state_space=state_space,
            tactic_distribution=tactic_distribution,
            aircraft_group_size=aircraft_group_size,
            group_sizes=group_sizes,
            controlled_agent_count=controlled_agent_count,
            water_set=water_set,
            cell_size_source=cell_size_source,
            enable_adaptive_time_step=enable_adaptive_time_step,
            adaptive_step_size_factor=adaptive_step_size_factor,
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
    mission_budgets: dict[str, float] | None = None,
    mission_budget_mode: str = MISSION_BUDGET_SIMULATIONS,
    initial_scenario_episode_counts: dict[str, int] | None = None,
    initial_scenario_timestep_counts: dict[str, int] | None = None,
    gc_collect_on_reset: bool = False,
    state_fire_fronts: int = DEFAULT_STATE_FIRE_FRONTS,
    state_space: str = "old",
    tactic_distribution: str = TACTIC_DISTRIBUTION_INDIVIDUAL,
    aircraft_group_size: int = AIRCRAFT_GROUP_SIZE,
    group_sizes: Sequence[int] | None = None,
    controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
    water_set: int | None = None,
    cell_size_source: str = "json",
    enable_adaptive_time_step: bool | None = None,
    adaptive_step_size_factor: float | None = None,
) -> DummyVecEnv | SubprocVecEnv:
    num_envs = max(1, num_envs)
    base_seed = int(np.random.randint(0, 1_000_000))
    if switch_scenario and switch_scenario_paths and mission_budgets is None:
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
                gc_collect_on_reset=gc_collect_on_reset,
                include_scenario_features=True,
                state_fire_fronts=state_fire_fronts,
                state_space=state_space,
                tactic_distribution=tactic_distribution,
                aircraft_group_size=aircraft_group_size,
                group_sizes=group_sizes,
                controlled_agent_count=controlled_agent_count,
                water_set=water_set,
                cell_size_source=cell_size_source,
                enable_adaptive_time_step=enable_adaptive_time_step,
                adaptive_step_size_factor=adaptive_step_size_factor,
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
                mission_budgets=mission_budgets,
                mission_budget_mode=mission_budget_mode,
                initial_scenario_episode_counts=initial_scenario_episode_counts,
                initial_scenario_timestep_counts=initial_scenario_timestep_counts,
                gc_collect_on_reset=gc_collect_on_reset,
                include_scenario_features=switch_scenario,
                state_fire_fronts=state_fire_fronts,
                state_space=state_space,
                tactic_distribution=tactic_distribution,
                aircraft_group_size=aircraft_group_size,
                group_sizes=group_sizes,
                controlled_agent_count=controlled_agent_count,
                water_set=water_set,
                cell_size_source=cell_size_source,
                enable_adaptive_time_step=enable_adaptive_time_step,
                adaptive_step_size_factor=adaptive_step_size_factor,
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

STATE_SPACE_OLD = "old"
STATE_SPACE_UPDATED = "updated"
STATE_SPACE_DIRECTIONAL = "directional"
STATE_SPACE_DIRECTIONAL_2 = "directional-2"
SUPPORTED_STATE_SPACES: tuple[str, ...] = (
    STATE_SPACE_OLD,
    STATE_SPACE_UPDATED,
    STATE_SPACE_DIRECTIONAL,
    STATE_SPACE_DIRECTIONAL_2,
)

# Per-flank block appended by the "directional" state space (6 features per
# flank slot, K = state_fire_fronts slots, each slot a compass sector).
DIRECTIONAL_PER_FRONT_FEATURES: tuple[str, ...] = (
    "front_severity",
    "vip_urgency",
    "vegetation_urgency",
    "topography_urgency",
    "indirect_urgency",
    "water_access",
)

AGGREGATE_FRONT_FLAG_FEATURES: tuple[str, ...] = (
    "topography_flag",
    "vegetation_flag",
    "indirect_flag",
    "urban_flag",
    "water_flag",
)

# The "updated" state is the per-aircraft-altitude variant: it excludes the
# fire->water distance, keeps each aircraft's altitude channel, and includes the
# aggregate front flags computed from up to `state_fire_fronts` inspected fronts.
# With 3 aircraft this is 34 + 3*3 = 43 observation features.
UPDATED_STATE_FEATURES = [
    "time_since_detection_min",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "time_to_sunset_min",
    "distance_to_fire_line_m",
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
    *AGGREGATE_FRONT_FLAG_FEATURES,
    "distance_left_boundary",
    "distance_right_boundary",
    "distance_bottom_boundary",
    "distance_top_boundary",
]

# The "old" state mirrors ppo_runner.py (v1)'s original vector: it adds the
# fire-centroid -> nearest-water distance (after distance_to_fire_line_m) and
# drops the per-aircraft altitude channel, so 3 aircraft produce 30 + 2*3 = 36
# features.
OLD_STATE_FEATURES = [
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
    "distance_left_boundary",
    "distance_right_boundary",
    "distance_bottom_boundary",
    "distance_top_boundary",
]

# The "directional-2" core is "old" with two changes. It drops the four
# map-boundary distances, each of which is an exact duplicate of a fire-extreme
# coordinate already in the vector (both are scaled by the same span, so
# distance_left_boundary == leftmost_x, distance_bottom_boundary == lowermost_y,
# distance_right_boundary == 1 - rightmost_x, distance_top_boundary ==
# 1 - uppermost_y). In their place it adds the episode's ignition point, so the
# current fire extent can be read against where the fire started rather than
# against the map edges. 30 - 4 + 2 = 28 core features, then the same per-flank
# block and the altitude-free agent block used by "directional".
DIRECTIONAL2_STATE_FEATURES = [
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
    "ignition_x",
    "ignition_y",
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
]


def _normalize_state_space(state_space: str) -> str:
    normalized = str(state_space).strip().lower()
    if normalized not in SUPPORTED_STATE_SPACES:
        supported = ", ".join(SUPPORTED_STATE_SPACES)
        raise ValueError(
            f"Unsupported state space {state_space!r}. Supported choices: {supported}."
        )
    return normalized


def _normalize_cell_size_source(cell_size_source: str) -> str:
    normalized = str(cell_size_source).strip().lower()
    if normalized not in SUPPORTED_CELL_SIZE_SOURCES:
        supported = ", ".join(SUPPORTED_CELL_SIZE_SOURCES)
        raise ValueError(
            f"Unsupported cell size source {cell_size_source!r}. "
            f"Supported choices: {supported}."
        )
    return normalized


def _directional_per_front_feature_names(state_fire_fronts: int) -> tuple[str, ...]:
    names: list[str] = []
    for front_idx in range(state_fire_fronts):
        for feature_name in DIRECTIONAL_PER_FRONT_FEATURES:
            names.append(f"{feature_name}_{front_idx}")
    return tuple(names)


def _state_core_feature_names(
    state_space: str,
    state_fire_fronts: int,
) -> tuple[str, ...]:
    normalized = _normalize_state_space(state_space)
    if normalized == STATE_SPACE_OLD:
        return tuple(OLD_STATE_FEATURES)
    if normalized == STATE_SPACE_UPDATED:
        return tuple(UPDATED_STATE_FEATURES)
    if normalized == STATE_SPACE_DIRECTIONAL:
        # "directional" = old's 30-feature core, verbatim, + the per-flank block.
        return tuple(OLD_STATE_FEATURES) + _directional_per_front_feature_names(
            state_fire_fronts
        )
    if normalized == STATE_SPACE_DIRECTIONAL_2:
        # "directional-2" = the 28-feature core (no boundary distances, plus the
        # ignition point) + the same per-flank block.
        return tuple(
            DIRECTIONAL2_STATE_FEATURES
        ) + _directional_per_front_feature_names(state_fire_fronts)
    raise RuntimeError(f"Unhandled state space: {state_space!r}")


def _agent_feature_count(state_space: str) -> int:
    # "old", "directional" and "directional-2" drop per-aircraft altitude
    # (x, y only); every other state space keeps the altitude channel.
    if _normalize_state_space(state_space) in (
        STATE_SPACE_OLD,
        STATE_SPACE_DIRECTIONAL,
        STATE_SPACE_DIRECTIONAL_2,
    ):
        return 2
    return AGENT_FEATURE_COUNT


def _agent_feature_names(
    controlled_agent_count: int,
    state_space: str = STATE_SPACE_OLD,
) -> tuple[str, ...]:
    names: list[str] = []
    include_altitude = _normalize_state_space(state_space) not in (
        STATE_SPACE_OLD,
        STATE_SPACE_DIRECTIONAL,
        STATE_SPACE_DIRECTIONAL_2,
    )
    for idx in range(controlled_agent_count):
        names.extend(
            [
                f"agent_{idx}_x",
                f"agent_{idx}_y",
            ]
        )
        if include_altitude:
            names.append(f"agent_{idx}_altitude")
    return tuple(names)


def _state_feature_names(
    include_scenario_flag: bool,
    controlled_agent_count: int,
    state_space: str = STATE_SPACE_OLD,
    state_fire_fronts: int = DEFAULT_STATE_FIRE_FRONTS,
) -> tuple[str, ...]:
    feature_names: list[str] = []
    if include_scenario_flag:
        feature_names.extend(SCENARIO_FEATURES)
    feature_names.extend(_state_core_feature_names(state_space, state_fire_fronts))
    feature_names.extend(_agent_feature_names(controlled_agent_count, state_space))
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


def _normalize_tactic_distribution(tactic_distribution: str) -> str:
    normalized = str(tactic_distribution).strip().lower()
    if normalized not in SUPPORTED_TACTIC_DISTRIBUTIONS:
        supported = ", ".join(SUPPORTED_TACTIC_DISTRIBUTIONS)
        raise ValueError(
            f"Unsupported tactic distribution {tactic_distribution!r}. "
            f"Supported choices: {supported}."
        )
    return normalized


def _validate_group_sizes(
    group_sizes: Sequence[int], controlled_agent_count: int
) -> list[int]:
    """Validate an explicit per-decision group layout.

    ``group_sizes[i]`` is the number of consecutive controlled aircraft that
    share the i-th tactic decision. They are laid over the controlled agents
    in order, so e.g. ``[1, 1, 1, 1, 4]`` on a 4-seaplane + 4-eVTOL fleet
    gives each seaplane its own decision and all four eVTOLs one shared
    decision. Must be positive and sum to ``controlled_agent_count``.
    """
    sizes = [int(s) for s in group_sizes]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(
            f"group_sizes must be a non-empty list of positive ints, "
            f"got {list(group_sizes)!r}"
        )
    if sum(sizes) != controlled_agent_count:
        raise ValueError(
            f"group_sizes {sizes} sum to {sum(sizes)}, but "
            f"controlled_agent_count={controlled_agent_count}"
        )
    return sizes


def _action_decision_count(
    tactic_distribution: str,
    controlled_agent_count: int,
    aircraft_group_size: int,
    group_sizes: Sequence[int] | None = None,
) -> int:
    if controlled_agent_count <= 0:
        raise ValueError("controlled_agent_count must be > 0")
    if group_sizes is not None:
        return len(_validate_group_sizes(group_sizes, controlled_agent_count))
    if aircraft_group_size <= 0:
        raise ValueError("aircraft_group_size must be > 0")
    if _normalize_tactic_distribution(tactic_distribution) == (
        TACTIC_DISTRIBUTION_INDIVIDUAL
    ):
        return controlled_agent_count
    return int(math.ceil(controlled_agent_count / aircraft_group_size))


def _tactic_combinations_from_action(
    action: Sequence[int],
) -> list[tuple[SelectPOIType, TrackPOIType, SuppressType]]:
    action_indices = np.asarray(action, dtype=np.int64).reshape(-1)
    return [TACTIC_COMBINATIONS[int(idx)] for idx in action_indices]


def _expand_tactic_combinations(
    action: Sequence[int],
    tactic_distribution: str,
    controlled_agent_count: int,
    aircraft_group_size: int,
    group_sizes: Sequence[int] | None = None,
) -> list[tuple[SelectPOIType, TrackPOIType, SuppressType]]:
    combinations = _tactic_combinations_from_action(action)
    if group_sizes is not None:
        sizes = _validate_group_sizes(group_sizes, controlled_agent_count)
        expanded = []
        for combination, size in zip(combinations, sizes, strict=False):
            expanded.extend([combination] * size)
        return expanded[:controlled_agent_count]
    if _normalize_tactic_distribution(tactic_distribution) == (
        TACTIC_DISTRIBUTION_INDIVIDUAL
    ):
        return combinations[:controlled_agent_count]

    expanded: list[tuple[SelectPOIType, TrackPOIType, SuppressType]] = []
    for combination in combinations:
        expanded.extend([combination] * aircraft_group_size)
    return expanded[:controlled_agent_count]


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


def _scenario_fleet_composition(scenario_path: Path) -> tuple[tuple[str, int], ...]:
    """Aircraft profiles in scenario order with their totals.

    The policy commands ``firefighters[:controlled_agent_count]`` and
    ``--group-sizes`` slices that list positionally, so one action only means
    the same thing on two missions if both declare the same aircraft in the
    same order.
    """
    with scenario_path.open() as handle:
        data = json.load(handle)
    return tuple(
        (
            str(agent.get("file_name", "")),
            sum(int(count) for count in agent.get("agents_per_base", [])),
        )
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


def _split_budget(total: int, weights: Sequence[float]) -> list[int]:
    """Apportion ``total`` between missions so the shares sum back to ``total``.

    Largest-remainder method: each mission takes floor(total * w_i / sum(w))
    and the leftover units go to the largest fractional parts. A 3-way split of
    100000 simulations therefore comes out 33334/33333/33333 instead of losing
    the remainder to truncation.
    """
    total = int(total)
    weight_values = [float(weight) for weight in weights]
    if not weight_values:
        raise ValueError("mission weights must not be empty")
    if any(weight <= 0.0 for weight in weight_values):
        raise ValueError(f"mission weights must all be > 0 (got {weight_values})")
    weight_sum = sum(weight_values)
    exact = [total * weight / weight_sum for weight in weight_values]
    shares = [int(math.floor(value)) for value in exact]
    remainder = total - sum(shares)
    if remainder > 0:
        order = sorted(
            range(len(shares)),
            key=lambda idx: exact[idx] - shares[idx],
            reverse=True,
        )
        for idx in order[:remainder]:
            shares[idx] += 1
    return shares


def _episode_counts_by_scenario(
    rows: Sequence[dict[str, Any]],
    mission_names: Sequence[str],
) -> dict[str, int]:
    """Count episode-summary rows per scenario file name.

    Used to carry a mission split across a chained run: the resumed summary
    says how much of each mission's share earlier chunks already spent. Rows
    written before the mission split existed carry no ``scenario_name`` (or one
    outside this run's mission list) and are ignored.
    """
    counts = {name: 0 for name in mission_names}
    for row in rows:
        name = row.get("scenario_name")
        if name in counts:
            counts[name] += 1
    return counts


def _timestep_counts_by_scenario(
    rows: Sequence[dict[str, Any]],
    mission_names: Sequence[str],
) -> dict[str, int]:
    """Sum each mission's collected timesteps over episode-summary rows.

    The timestep-budgeted counterpart of ``_episode_counts_by_scenario``: an
    episode contributes its ``total_decision_steps``, which is exactly the PPO
    timesteps it produced. Without this a resumed chunk of a timestep-budgeted
    chain would restart at the first phase.
    """
    counts = {name: 0 for name in mission_names}
    for row in rows:
        name = row.get("scenario_name")
        if name not in counts:
            continue
        try:
            counts[name] += int(float(row.get("total_decision_steps") or 0))
        except (TypeError, ValueError):
            continue
    return counts


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


@dataclass(frozen=True)
class FireFrontDiagnostic:
    """Per-front facts used for topography_flag and future front state."""

    source_i: int
    source_j: int
    projected_i: int | None
    projected_j: int | None
    objective_i: int
    objective_j: int
    spread_rate: float
    topography_priority: float
    slope_factor: float
    forward_combustible_uphill: bool
    has_topography_growth: bool
    vegetation_combustible_neighbor_count: int
    has_vegetation_threat: bool
    distance_to_indirect_line_m: float
    near_indirect_line: bool
    distance_to_urban: float
    distance_to_water: float
    objective_vote: str | None


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


def _read_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    """Read a previously written ``*_summary.csv`` back into records.

    Used by ``--resume-summary`` so a chained chunk continues the episode
    numbering and re-emits the full chain history in its own progress files.
    Values stay strings; nothing downstream of the logger re-reads them as
    numbers, and the CSV round-trips byte-identically for the resumed rows.
    """
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


CHAIN_PROGRESS_FILE = "chain_progress.json"


def _write_chain_progress(output_dir: Path, simulations: int, timesteps: int) -> None:
    """Record how far the chain has come, next to the saved model.

    A simulation-budgeted chain decays its LR over completed episodes, so the
    episode count is training state, not just logging: it has to survive the
    SLURM job boundary alongside the weights. The summary CSV also carries it,
    but only up to the last export, so this sidecar is the authoritative copy.
    """
    (output_dir / CHAIN_PROGRESS_FILE).write_text(
        json.dumps(
            {
                "simulations": int(simulations),
                "timesteps": int(timesteps),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )


def _read_chain_progress(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / CHAIN_PROGRESS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:  # noqa: BLE001 - diagnostic only
        print(f"Ignoring unreadable {path}: {err}")
        return None


def _saved_num_timesteps(model_path: Path) -> int:
    """Read ``num_timesteps`` out of an SB3 zip without loading the policy.

    Lets a chained chunk decide it has nothing left to do before paying for
    environment construction. Returns 0 if the field cannot be read; the
    authoritative value is still whatever ``PPO.load`` restores.
    """
    import zipfile

    try:
        with zipfile.ZipFile(model_path) as archive:
            data = json.loads(archive.read("data"))
    except Exception as err:  # noqa: BLE001 - diagnostic only
        print(f"Could not read num_timesteps from {model_path}: {err}")
        return 0
    return int(data.get("num_timesteps") or 0)


def _latest_summary_path(output_dir: Path, progress_file_tag: str) -> Path | None:
    """Most complete episode summary in ``output_dir``.

    Interim files are written every ``LOG_INTERVAL_SUMMARY_EPISODES`` episodes
    as ``results_<tag>_<N>_summary.csv``, which is also what the plotting
    scripts read, so ``--resume-summary auto`` picks the highest N among them.
    A chunk that ran to completion then writes the full history once more to
    ``training_episode_summaries.csv``, which holds up to 99 episodes the
    interim files missed; prefer it when it is the newer of the two (i.e. the
    previous chunk finished cleanly rather than being interrupted).
    """
    best_path: Path | None = None
    best_count = -1
    for path in output_dir.glob(f"results_{progress_file_tag}_*_summary.csv"):
        tail = path.name[len(f"results_{progress_file_tag}_") : -len("_summary.csv")]
        if not tail.isdigit():
            continue
        count = int(tail)
        if count > best_count:
            best_count = count
            best_path = path
    final_path = output_dir / "training_episode_summaries.csv"
    if final_path.exists() and (
        best_path is None or final_path.stat().st_mtime >= best_path.stat().st_mtime
    ):
        return final_path
    return best_path


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
        mission_budgets: dict[str, float] | None = None,
        mission_budget_mode: str = MISSION_BUDGET_SIMULATIONS,
        initial_scenario_episode_counts: dict[str, int] | None = None,
        initial_scenario_timestep_counts: dict[str, int] | None = None,
        gc_collect_on_reset: bool = False,
        include_scenario_features: bool | None = None,
        state_fire_fronts: int = DEFAULT_STATE_FIRE_FRONTS,
        state_space: str = STATE_SPACE_OLD,
        tactic_distribution: str = TACTIC_DISTRIBUTION_INDIVIDUAL,
        aircraft_group_size: int = AIRCRAFT_GROUP_SIZE,
        group_sizes: Sequence[int] | None = None,
        controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
        water_set: int | None = None,
        cell_size_source: str = CELL_SIZE_SOURCE_JSON,
        enable_adaptive_time_step: bool | None = None,
        adaptive_step_size_factor: float | None = None,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.water_set = water_set
        self.cell_size_source = _normalize_cell_size_source(cell_size_source)
        self.gc_collect_on_reset = bool(gc_collect_on_reset)
        self.state_space = _normalize_state_space(state_space)
        if (
            adaptive_step_size_factor is not None
            and adaptive_step_size_factor <= 0.0
        ):
            raise ValueError("adaptive_step_size_factor must be > 0")
        self.adaptive_step_size_factor = (
            None
            if adaptive_step_size_factor is None
            else float(adaptive_step_size_factor)
        )
        self.enable_adaptive_time_step_override = (
            None
            if enable_adaptive_time_step is None
            else bool(enable_adaptive_time_step)
        )
        self.switch_scenario = switch_scenario
        self.switch_scenario_paths = (
            tuple(switch_scenario_paths) if switch_scenario_paths else tuple()
        )
        self.aircraft_source_scenario_path = aircraft_source_scenario_path
        if switch_ignition_mode not in (0, 1, 2, 3, 4):
            raise ValueError(
                "switch_ignition_mode must be 0, 1, 2, 3, or 4 "
                f"(got {switch_ignition_mode})"
            )
        self.switch_ignition_mode = int(switch_ignition_mode)
        self.switch_ignition = self.switch_ignition_mode != 0
        # Mode 4 relaxes the urban keep-out; every other mode keeps 350 m.
        # Held on the env so the candidate builders AND the post-round-trip
        # re-check in _sample_ignition_centers agree on one value.
        self.ignition_urban_buffer_m = (
            IGNITION_URBAN_BUFFER_M_V4
            if self.switch_ignition_mode == 4
            else IGNITION_URBAN_BUFFER_M
        )

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
        self.state_fire_fronts = int(state_fire_fronts)
        if self.state_fire_fronts <= 0:
            raise ValueError("state_fire_fronts must be > 0")

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
            self._scenario_ignitable_mask: dict[Path, np.ndarray | None] = {}
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
            self._scenario_ignitable_mask = {}
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
        self.agent_feature_count = _agent_feature_count(self.state_space)
        self.tactic_distribution = _normalize_tactic_distribution(
            tactic_distribution
        )
        self.aircraft_group_size = int(aircraft_group_size)
        if self.aircraft_group_size <= 0:
            raise ValueError("aircraft_group_size must be > 0")
        self.group_sizes = (
            _validate_group_sizes(group_sizes, self.controlled_agent_count)
            if group_sizes is not None
            else None
        )
        self.action_decision_count = _action_decision_count(
            self.tactic_distribution,
            self.controlled_agent_count,
            self.aircraft_group_size,
            self.group_sizes,
        )
        self.include_scenario_flag = (
            bool(self.switch_scenario)
            if include_scenario_features is None
            else bool(include_scenario_features)
        )
        self.state_feature_names = _state_feature_names(
            self.include_scenario_flag,
            self.controlled_agent_count,
            self.state_space,
            self.state_fire_fronts,
        )
        obs_dim = len(self.state_feature_names)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [len(TACTIC_COMBINATIONS)] * self.action_decision_count
        )

        self.np_random, _ = gym.utils.seeding.np_random()
        self.sim: WildfireSimulation | None = None
        self.water_positions: np.ndarray = np.empty((0, 2), dtype=float)
        self.urban_positions: np.ndarray = np.empty((0, 2), dtype=float)
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
        self.last_front_diagnostics: list[FireFrontDiagnostic] = []
        self.last_front_summary: dict[str, float | int] = {
            "state_fire_fronts": self.state_fire_fronts,
            "state_fire_front_count": 0,
            "topography_front_positive_count": 0,
            "topography_front_required_count": 0,
            "topography_flag": 0.0,
            "vegetation_front_positive_count": 0,
            "vegetation_front_required_count": 0,
            "vegetation_flag": 0.0,
            "indirect_front_positive_count": 0,
            "indirect_front_required_count": 0,
            "indirect_flag": 0.0,
            "urban_front_count": 0,
            "water_front_count": 0,
            "objective_front_count": 0,
            "urban_flag": 0.0,
            "water_flag": 0.0,
        }
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
        # Metres/cell used by directional geometry; set per scenario in
        # _cache_static_state_scalers based on self.cell_size_source.
        self._cell_size_m: float = 1.0
        self._fire_line_tree: cKDTree | None = None
        self._fire_line_tree_block_index: int = -1
        # Static POI KDTrees for the "directional" state (world-coord frame),
        # rebuilt per scenario in _build_static_poi_positions.
        self._vip_tree: cKDTree | None = None
        self._water_tree: cKDTree | None = None
        self._ts_budget_per_scenario: float | None = ts_budget_per_scenario
        # Timesteps collected per scenario, refreshed from the callback. Seeded
        # for a resumed chunk of a timestep-budgeted mission chain so it picks
        # up the phase it left off in.
        self._global_scenario_ts_counts: dict[str, int] = dict(
            initial_scenario_timestep_counts or {}
        )
        # --missions: each mission's share of the training budget, keyed by
        # scenario file name, in simulations or in timesteps. The missions run
        # as phases in list order -- every worker trains the first mission whose
        # share is not spent yet, so the whole fleet finishes one map before
        # moving to the next (see _select_mission_index).
        self._mission_budgets: dict[str, float] | None = (
            dict(mission_budgets) if mission_budgets else None
        )
        self._mission_budget_mode = mission_budget_mode
        if self._mission_budgets is not None and mission_budget_mode not in (
            MISSION_BUDGET_SIMULATIONS,
            MISSION_BUDGET_TIMESTEPS,
        ):
            raise ValueError(
                "mission_budget_mode must be "
                f"{MISSION_BUDGET_SIMULATIONS!r} or {MISSION_BUDGET_TIMESTEPS!r} "
                f"(got {mission_budget_mode!r})"
            )
        # Simulations started per mission: fleet-wide as of the last sync, this
        # worker's own running total, and the snapshot of that total taken at
        # the sync. The difference of the last two is what this worker has begun
        # since the fleet figure was current, which is what stops a phase from
        # running long between syncs. Seeded with the episodes a chained run's
        # earlier chunks already spent, so a resumed chunk starts in the phase
        # it left off in.
        self._fleet_scenario_ep_starts: dict[str, int] = dict(
            initial_scenario_episode_counts or {}
        )
        self._scenario_ep_starts: dict[str, int] = {}
        self._scenario_ep_starts_at_sync: dict[str, int] = {}
        # Mission chosen by the last reset, counted as started once the episode
        # actually steps. Resets whose episode never runs must not consume any
        # of a phase's share, and there are two of them per worker before
        # training begins: the factory seeds the env with reset(seed=...) and
        # SB3 resets the vector env again before the first rollout.
        self._pending_mission_start: str | None = None

    def set_global_scenario_counts(self, counts: dict[str, int]) -> None:
        self._global_scenario_ts_counts = dict(counts)

    def get_scenario_episode_starts(self) -> dict[str, int]:
        """Simulations this worker has started per mission, since construction."""
        return dict(self._scenario_ep_starts)

    def set_fleet_scenario_episode_starts(self, totals: dict[str, int]) -> None:
        """Adopt the fleet-wide per-mission count of started simulations.

        ``totals`` already includes this worker's starts up to now, so the
        since-sync baseline moves with it instead of being counted twice.
        """
        self._fleet_scenario_ep_starts = dict(totals)
        self._scenario_ep_starts_at_sync = dict(self._scenario_ep_starts)

    def _count_pending_mission_start(self) -> None:
        """Charge the running episode to its mission, once, at its first step.

        Called from ``step``: an episode counts against its phase's share when
        it actually runs, not when it is set up, so the resets whose episode is
        discarded before training (see ``_pending_mission_start``) cost nothing.
        """
        pending = self._pending_mission_start
        if pending is None:
            return
        self._scenario_ep_starts[pending] = (
            self._scenario_ep_starts.get(pending, 0) + 1
        )
        self._pending_mission_start = None

    def mission_simulations_started(self, scenario_name: str) -> int:
        """Best current estimate of fleet-wide starts for one mission.

        The fleet figure plus whatever this worker has begun since it arrived.
        """
        return self._fleet_scenario_ep_starts.get(scenario_name, 0) + (
            self._scenario_ep_starts.get(scenario_name, 0)
            - self._scenario_ep_starts_at_sync.get(scenario_name, 0)
        )

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
            params = WildfireParameters.model_validate_json(handle.read())
        # Apply the CLI water-set selector so water_sources_file resolves to the
        # pre-generated subset ({namespace}_water_sources_set{N}.pkl).
        if self.water_set is not None:
            params = params.model_copy(
                update={
                    "terrain_inputs": params.terrain_inputs.model_copy(
                        update={"water_set": int(self.water_set)}
                    )
                }
            )
        return params

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
    def _grid_description(parameters: WildfireParameters) -> GridDescriptor:
        """Index<->position grid the sim actually uses.

        Built from mercator_dimensions / grid_shape, not cell_size:
        away from the equator the mercator metres per cell differ from
        the nominal cell_size (~6.84 m vs 5 m at Pyrenees, ~1.37x), so
        indexing a mercator position by cell_size mislocates the cell.
        """
        terrain_inputs = parameters.terrain_inputs
        grid_shape = terrain_inputs.grid_shape
        mercator_dimensions = terrain_inputs.meta_data["mercator_dimensions"]
        return GridDescriptor(
            shape=(int(grid_shape[0]), int(grid_shape[1])),
            dimensions=(
                float(mercator_dimensions[0]),
                float(mercator_dimensions[1]),
            ),
        )

    @staticmethod
    def _ignition_center_grid_pos(
        parameters: WildfireParameters,
    ) -> tuple[int, int] | None:
        if not parameters.ignition_centers:
            return None
        center = parameters.ignition_centers[0]
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
            # Index with the sim's real grid (mercator_dimensions /
            # grid_shape), not cell_size: the latter shifts the box
            # center by the mercator factor (~3.7 km at Pyrenees).
            row, col = pos_to_index(
                (float(x), float(y)),
                WildfireHourlyEnv._grid_description(parameters),
            )
            return (int(row), int(col))
        except Exception:
            return None

    def _build_ignition_candidate_positions(
        self,
        parameters: WildfireParameters,
    ) -> np.ndarray:
        """Dispatch to the candidate builder for the active ignition mode."""
        if self.switch_ignition_mode == 2:
            return self._build_ignition_candidate_positions_v2(parameters)
        if self.switch_ignition_mode == 3:
            return self._build_ignition_candidate_positions_v1(
                parameters, half_size=IGNITION_BOX_HALF_SIZE_V3
            )
        if self.switch_ignition_mode == 4:
            return self._build_ignition_candidate_positions_v1(
                parameters, half_size=IGNITION_BOX_HALF_SIZE_V4
            )
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
            feature_data,
            float(parameters.cell_size),
            urban_buffer_m=self.ignition_urban_buffer_m,
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
        half_size: int = IGNITION_BOX_HALF_SIZE,
    ) -> np.ndarray:
        """Square cell box around the scenario's original ignition center.

        ``half_size`` is the box half-side in grid cells: 50 for
        --switch-ignition-1 (100×100 cells, 0.5 km ground) and 250 for
        --switch-ignition-3 (500×500 cells, 2.5 km ground).
        """
        feature_data = np.asarray(
            np.load(parameters.terrain_inputs.features_file, allow_pickle=False)
        )
        if feature_data.ndim < 2:
            return np.empty((0, 2), dtype=np.int64)

        buffered_ignitable_mask = self._build_ignitable_mask(
            feature_data,
            float(parameters.cell_size),
            urban_buffer_m=self.ignition_urban_buffer_m,
        )
        hard_ignitable_mask = self._build_ignitable_mask(
            feature_data,
            float(parameters.cell_size),
            urban_buffer_m=0.0,
        )
        rows, cols = hard_ignitable_mask.shape

        # Try the local box around the scenario's original ignition center first.
        center_pos = self._ignition_center_grid_pos(parameters)
        if center_pos is not None:
            cr, cc = center_pos
            r0 = max(0, cr - half_size)
            r1 = min(rows, cr + half_size)
            c0 = max(0, cc - half_size)
            c1 = min(cols, cc + half_size)
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

        # The fire-map grid cell is NOT exactly ``cell_size`` metres: the sim
        # builds its grid_description from (mercator_dimensions / grid_shape).
        # Encode the candidate cell-centre with that SAME grid so the GPS we
        # emit round-trips back to the intended cell when the sim re-indexes it
        # (gps_to_pos -> pos_to_index). Using cell_size here instead shifts the
        # ignited cell by ~1 km, dropping fires onto unvalidated water/urban.
        terrain_inputs = selected_parameters.terrain_inputs
        cell_size = float(selected_parameters.cell_size)
        grid_shape = terrain_inputs.grid_shape
        # matches the sim's terrain.grid_description so index<->pos is identical
        # on both sides (mercator_dimensions / grid_shape, not cell_size).
        grid_description = self._grid_description(selected_parameters)
        fire_top_left_merc = gps_to_mercator(
            terrain_inputs.fire_map_coordinates[0]
        )
        fire_top_left_bounds = (
            float(fire_top_left_merc[1]),
            float(fire_top_left_merc[0]),
        )
        bbox_pos = (
            (0.0, 0.0),
            (float(grid_shape[0]) * cell_size, float(grid_shape[1]) * cell_size),
        )

        # Reusable ignitable mask for the final logic check (drop water / urban
        # / rock and enforce the IGNITION_URBAN_BUFFER_M urban keep-out). Cached
        # per scenario so we build it at most once.
        ignitable_mask = self._scenario_ignitable_mask.get(selected_path)
        if ignitable_mask is None:
            feature_data = np.asarray(
                np.load(terrain_inputs.features_file, allow_pickle=False)
            )
            ignitable_mask = (
                self._build_ignitable_mask(
                    feature_data,
                    cell_size,
                    urban_buffer_m=self.ignition_urban_buffer_m,
                )
                if feature_data.ndim >= 2
                else None
            )
            self._scenario_ignitable_mask[selected_path] = ignitable_mask
        mask_rows, mask_cols = (
            ignitable_mask.shape if ignitable_mask is not None else (0, 0)
        )

        # Draw candidates without replacement and accept the first whose FINAL
        # ignited cell (after the GPS round-trip the sim performs) still passes
        # the ignitable rule. This guarantees no fire is seeded in water, on an
        # incombustible cell, or within IGNITION_URBAN_BUFFER_M of an urban
        # area; if a draw fails the check, we simply pick another.
        order = self.np_random.permutation(len(candidates))
        for idx in order:
            row, col = int(candidates[idx, 0]), int(candidates[idx, 1])
            x, y = index_to_pos((row, col), grid_description)
            lat, lon = pos_to_gps((float(x), float(y)), fire_top_left_bounds)
            sampled = IgnitionCenterInput(gps_coords=(float(lat), float(lon)))

            # Must stay inside the fire map after the round-trip.
            try:
                sampled.check_in_bbox(
                    bbox_pos=bbox_pos,
                    bbox_gps=terrain_inputs.fire_map_coordinates,
                )
            except AssertionError:
                continue

            # Logic check on the cell the sim will ACTUALLY ignite.
            if ignitable_mask is not None:
                px, py = gps_to_pos((float(lat), float(lon)), fire_top_left_bounds)
                fr, fc = pos_to_index((float(px), float(py)), grid_description)
                if not (0 <= fr < mask_rows and 0 <= fc < mask_cols):
                    continue
                if not bool(ignitable_mask[fr, fc]):
                    continue

            return (sampled,)

        # No candidate survived the check: fall back to the scenario default.
        return selected_parameters.ignition_centers

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

    def _select_mission_index(self) -> int:
        """Index of the mission this episode belongs to: the first unspent one.

        The missions are phases in list order. Every new simulation asks how
        many the fleet has already started (or, for a timestep budget, how many
        timesteps it has already collected) on each mission and takes the first
        one still inside its share, so the whole fleet finishes one map before
        any worker moves to the next.

        The fleet figure is refreshed each step, and this worker's own starts
        since that refresh are added on top (see ``mission_simulations_started``)
        so a phase cannot run long while the figure is in flight. What is left
        is workers whose resets land in the same step: they see the same figure
        and can claim the same last slot, which is why a phase can overrun its
        share by a few simulations. Once every share is spent the last mission
        keeps running, so training that continues past its budget (a wall-clock
        chunk, a longer chain) stays in the final phase instead of restarting
        the sequence.
        """
        budgets = self._mission_budgets or {}
        by_simulations = self._mission_budget_mode == MISSION_BUDGET_SIMULATIONS
        for index, (path, _) in enumerate(self._scenario_templates):
            name = path.name
            spent = (
                self.mission_simulations_started(name)
                if by_simulations
                else self._global_scenario_ts_counts.get(name, 0)
            )
            if float(spent) < float(budgets.get(name, 0.0)):
                return index
        return len(self._scenario_templates) - 1

    def _select_episode_setup(self) -> tuple[Path, WildfireParameters]:
        if not self.switch_scenario:
            selected_path = self.scenario_path
            selected_parameters = self.parameters.model_copy(deep=True)
        else:
            if not self._scenario_templates:
                raise RuntimeError("No scenario templates available for switching.")

            if self._mission_budgets is not None:
                choice_idx = self._select_mission_index()
            elif self._ts_budget_per_scenario is not None:
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
            if self._mission_budgets is not None:
                self._pending_mission_start = selected_path.name
            selected_parameters = selected_template.model_copy(
                deep=True,
                update={"agents": self._scenario_agents[selected_path]},
            )

        merged_protection = self._merge_urban_into_protection(selected_parameters)
        ignition_centers = self._sample_ignition_centers(
            selected_path,
            selected_parameters,
        )
        parameter_updates: dict[str, Any] = {
            "protection_locations": merged_protection,
            "ignition_centers": ignition_centers,
        }
        if self.enable_adaptive_time_step_override is not None:
            parameter_updates["enable_adaptive_time_step"] = (
                self.enable_adaptive_time_step_override
            )
        if self.adaptive_step_size_factor is not None:
            parameter_updates["adaptive_step_size_factor"] = (
                self.adaptive_step_size_factor
            )
        selected_parameters = selected_parameters.model_copy(
            deep=True,
            update=parameter_updates,
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
        self._advance_to_decision_start()
        self.prev_metrics = self._compute_metrics()
        self.initial_metrics = self.prev_metrics
        self.prev_total_moe = None
        self.current_step = 0

        self.cumulative_reward = 0.0
        self.current_sim_seed = seed

    def _compute_metrics(self) -> Metrics:
        """Compute PPO reward metrics without repeated full-grid property scans."""
        assert self.sim is not None
        wildfire = self.sim.wildfire
        fire_states = wildfire.fire_states
        cell_area = float(self.sim.parameters.cell_size**2)

        # Match CPUFireModel.burnt_area exactly: total burnt area counts final
        # burnt states plus cells that burned during suppression.
        state_counts = np.bincount(fire_states.ravel(), minlength=BURNT + 1)
        burnt_cells = (
            int(state_counts[BURNT])
            + int(state_counts[EXTINGUISHING])
            + int(state_counts[FULL_BURNING])
            + int(wildfire.total_suppressed_burn_cells)
        )
        burnt = float(burnt_cells * cell_area)

        # Match burnt_area_for_combustibility: cost/emissions/casualties use a
        # damage mask of full-burning-or-later cells plus currently suppressed
        # cells, then group by initial combustibility.
        damage_mask = (fire_states >= FULL_BURNING) | (fire_states == SUPPRESSED)
        combust = wildfire.initial_combustibilities
        area_by_combustibility: dict[float, float] = {}
        if np.any(damage_mask):
            damaged_combustibilities, counts = np.unique(
                combust[damage_mask],
                return_counts=True,
            )
            area_by_combustibility = {
                np.asarray(value, dtype=combust.dtype).item(): (
                    float(count) * cell_area
                )
                for value, count in zip(damaged_combustibilities, counts)
            }

        total_cost = 0.0
        total_emissions = 0.0
        total_casualties = 0.0
        for terrain_type in DAMAGE_TERRAIN_TYPES:
            # Cast the table value to the combustibility array dtype before
            # lookup. This preserves numpy's current equality behavior for
            # float32 terrain rasters and Python float table values.
            combustibility_key = np.asarray(
                COMBUSTIBILITY_TABLE[terrain_type],
                dtype=combust.dtype,
            ).item()
            terrain_area = area_by_combustibility.get(combustibility_key, 0.0)
            area_ha = terrain_area * M2_TO_HECTARS
            total_cost += COSTS_TABLE[terrain_type] * area_ha
            total_emissions += EMISSIONS_TABLE[terrain_type] * area_ha
            total_casualties += (
                CASUALTIES_TABLE[terrain_type]
                * terrain_area
                * PEOPLE_PER_HOUSEHOLD
                * M2_TO_HECTARS
            )

        return Metrics(
            burnt,
            float(round(total_cost, 2)),
            float(int(total_casualties)),
            float(round(total_emissions, 2)),
        )

    def _compute_metrics_reference(self) -> Metrics:
        """Reference version using the simulation properties used before."""
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

        # Metres/cell for directional geometry. "json" trusts the scenario's
        # nominal cell_size; "code" uses the projected metres/cell baked into
        # ACTUAL_CELL_SIZE_M (matching the sim's index<->position grid).
        nominal_cell_size = float(self.sim.parameters.cell_size)
        if self.cell_size_source == CELL_SIZE_SOURCE_CODE:
            grid_desc = self.sim.environment.terrain.grid_description
            grid_cell_size = 0.5 * (
                grid_desc.dimensions[0] / grid_desc.shape[1]
                + grid_desc.dimensions[1] / grid_desc.shape[0]
            )
            namespace = self.sim.parameters.terrain_inputs.file_namespace
            cell_size = ACTUAL_CELL_SIZE_M.get(namespace)
            if cell_size is None:
                cell_size = grid_cell_size
                warnings.warn(
                    f"--cell-size code: {namespace!r} not in ACTUAL_CELL_SIZE_M; "
                    f"falling back to grid_description value {cell_size:.4f} m.",
                    stacklevel=2,
                )
            elif abs(cell_size - grid_cell_size) > 0.05:
                warnings.warn(
                    f"ACTUAL_CELL_SIZE_M[{namespace!r}]={cell_size:.4f} m disagrees "
                    f"with grid_description {grid_cell_size:.4f} m; the constant may "
                    "be stale.",
                    stacklevel=2,
                )
            self._cell_size_m = float(cell_size)
        else:
            self._cell_size_m = nominal_cell_size

    def _distance_to_fire_line_m(self, burning_indices: np.ndarray) -> float:
        """Compute minimum burning-cell distance to active fire line segment in meters."""
        assert self.sim is not None
        burning = self._coerce_grid_indices(np.asarray(burning_indices))
        if burning.size == 0:
            return math.nan

        distances = self._distances_to_fire_line_m(burning)
        if distances.size == 0:
            return math.nan
        return float(np.min(np.asarray(distances, dtype=float)))

    def _distances_to_fire_line_m(self, indices: np.ndarray) -> np.ndarray:
        """Compute per-cell distances to the active fire line segment in meters."""
        assert self.sim is not None
        fire_block_indices = self.sim.firefighters.fire_block_indices
        if (
            fire_block_indices is None
            or fire_block_indices.size == 0
            or self.sim.firefighters.current_block_index <= 0
        ):
            return np.empty(0, dtype=float)

        current_block_index = int(self.sim.firefighters.current_block_index)
        segment_raw = np.asarray(fire_block_indices[:current_block_index])
        segment = self._coerce_grid_indices(segment_raw)
        points = self._coerce_grid_indices(np.asarray(indices))
        if segment.size == 0 or points.size == 0:
            return np.empty(0, dtype=float)

        if (
            self._fire_line_tree is None
            or self._fire_line_tree_block_index != current_block_index
        ):
            self._fire_line_tree = cKDTree(np.asarray(segment, dtype=float))
            self._fire_line_tree_block_index = current_block_index

        distances, _ = self._fire_line_tree.query(
            np.asarray(points, dtype=float), k=1
        )
        return np.asarray(distances, dtype=float) * float(self.sim.parameters.cell_size)

    def _front_distance_to_fire_line_m(self, source_i: int, source_j: int) -> float:
        distances = self._distances_to_fire_line_m(
            np.array([[source_i, source_j]], dtype=np.int64)
        )
        if distances.size == 0:
            return math.inf
        return float(distances[0])

    def _build_static_poi_positions(self) -> None:
        assert self.sim is not None
        firefighters = self.sim.firefighters

        self.urban_positions = np.array(
            [location.pos for location in firefighters.protection_locations],
            dtype=float,
        )
        if self.urban_positions.size == 0:
            self.urban_positions = np.empty((0, 2), dtype=float)
        # In this runner, protection locations are the urban objectives.
        self.vip_positions = self.urban_positions

        if self.state_space in (STATE_SPACE_DIRECTIONAL, STATE_SPACE_DIRECTIONAL_2):
            # World-coord KDTrees queried at projected landing points.
            self._vip_tree = (
                cKDTree(self.vip_positions) if self.vip_positions.size else None
            )
            self._water_tree = (
                cKDTree(self.water_positions) if self.water_positions.size else None
            )

        self.vegetation_poi_positions = np.empty((0, 2), dtype=float)
        self.topography_poi_positions = np.empty((0, 2), dtype=float)
        if self.state_space not in (
            STATE_SPACE_OLD,
            STATE_SPACE_UPDATED,
            STATE_SPACE_DIRECTIONAL,
            STATE_SPACE_DIRECTIONAL_2,
        ):
            return

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

    @staticmethod
    def _angle_between_degrees(first: float, second: float) -> float:
        if not math.isfinite(first) or not math.isfinite(second):
            return math.nan
        return float(abs((first - second + 180.0) % 360.0 - 180.0))

    @staticmethod
    def _grid_offset_aspect(di: int, dj: int) -> float:
        return float((math.degrees(math.atan2(dj, -di)) + 360.0) % 360.0)

    def _forward_neighbor_indices(
        self,
        source_i: int,
        source_j: int,
        prop_aspect: float,
    ) -> list[tuple[int, int]]:
        """Return one-step forward cone cells for a propagation aspect."""
        assert self.sim is not None
        if not math.isfinite(prop_aspect):
            return []
        height, width = self.sim.wildfire.fire_states.shape
        neighbors: list[tuple[float, int, int]] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                offset_aspect = self._grid_offset_aspect(di, dj)
                angle = self._angle_between_degrees(offset_aspect, prop_aspect)
                if math.isnan(angle) or angle > 45.0:
                    continue
                ni = source_i + di
                nj = source_j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighbors.append((angle, int(ni), int(nj)))
        # The closest-angle cell is our best one-step estimate of where this
        # front is trying to move next; the rest keep future front features open.
        neighbors.sort(key=lambda item: item[0])
        return [(ni, nj) for _, ni, nj in neighbors]

    def _topography_wind_alignment(
        self,
        source_i: int,
        source_j: int,
        neighbor_i: int,
        neighbor_j: int,
        wind_direction: float,
    ) -> float:
        # Mirror the topography SelectPOI heuristic: uphill targets count more
        # when they also sit in the wind-favored direction from the fire cell.
        neighbor_angle = math.degrees(
            math.atan2(source_i - neighbor_i, source_j - neighbor_j)
        )
        angle_difference = abs(180.0 - neighbor_angle - wind_direction)
        return _clip01(1.0 - min(angle_difference, 360.0 - angle_difference) / 180.0)

    def _front_topography_priority_raw(
        self,
        source_i: int,
        source_j: int,
    ) -> float:
        """Match the topography tactic's local uphill-combustible heuristic."""
        assert self.sim is not None
        elevation = np.asarray(
            self.sim.environment.terrain.elevation.elevation_data, dtype=float
        )
        fire_states = self.sim.wildfire.fire_states
        height, width = fire_states.shape
        if not (0 <= source_i < height and 0 <= source_j < width):
            return 0.0

        current_elevation = float(elevation[source_i, source_j])
        if not math.isfinite(current_elevation):
            return 0.0
        wind_direction = float(self.sim.atmosphere.wind_aspect)
        priority = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = source_i + di
                nj = source_j + dj
                if not (0 <= ni < height and 0 <= nj < width):
                    continue
                if fire_states[ni, nj] != COMBUSTIBLE:
                    continue
                neighbor_elevation = float(elevation[ni, nj])
                if not math.isfinite(neighbor_elevation):
                    continue
                elevation_gain = neighbor_elevation - current_elevation
                if elevation_gain <= 0.0:
                    continue
                alignment = self._topography_wind_alignment(
                    source_i,
                    source_j,
                    int(ni),
                    int(nj),
                    wind_direction,
                )
                # Keep this value raw instead of normalized. The flag asks
                # whether a real opportunity exists, not how this cell ranks
                # relative to other fronts in the current step.
                priority = max(priority, alignment * elevation_gain)
        return float(max(priority, 0.0))

    def _front_slope_factor(
        self,
        source_i: int,
        source_j: int,
        prop_aspect: float,
    ) -> float:
        """Return the fire model's slope multiplier at a front source cell."""
        assert self.sim is not None
        if not math.isfinite(prop_aspect):
            return 1.0
        slopes = np.asarray(self.sim.environment.terrain.slopes, dtype=float)
        aspects = np.asarray(self.sim.environment.terrain.aspects, dtype=float)
        height, width = slopes.shape
        if not (0 <= source_i < height and 0 <= source_j < width):
            return 1.0
        terrain_slope = float(slopes[source_i, source_j])
        terrain_aspect = float(aspects[source_i, source_j])
        if not math.isfinite(terrain_slope) or not math.isfinite(terrain_aspect):
            return 1.0
        angle = self._angle_between_degrees(terrain_aspect, prop_aspect)
        if math.isnan(angle):
            return 1.0
        # This is the same multiplier used by the spread model. Values above 1
        # mean slope is accelerating spread; values below 1 mean slope resists it.
        hill_dir = -1.0 if angle < 90.0 else 1.0
        exponent = (
            3.553 * hill_dir * math.tan(1.2 * terrain_slope * math.pi / 180.0)
        )
        return float(math.exp(min(exponent, SLOPE_FACTOR_EXP_MAX)))

    def _has_forward_combustible_uphill(
        self,
        source_i: int,
        source_j: int,
        prop_aspect: float,
    ) -> bool:
        # A front only votes for topography if its likely next cells are both
        # burnable and higher than the current cell.
        assert self.sim is not None
        elevation = np.asarray(
            self.sim.environment.terrain.elevation.elevation_data, dtype=float
        )
        fire_states = self.sim.wildfire.fire_states
        current_elevation = float(elevation[source_i, source_j])
        if not math.isfinite(current_elevation):
            return False
        for ni, nj in self._forward_neighbor_indices(source_i, source_j, prop_aspect):
            if fire_states[ni, nj] != COMBUSTIBLE:
                continue
            neighbor_elevation = float(elevation[ni, nj])
            if math.isfinite(neighbor_elevation) and neighbor_elevation > current_elevation:
                return True
        return False

    def _front_vegetation_neighbor_count(
        self,
        source_i: int,
        source_j: int,
        radius: int = VEGETATION_NEIGHBOR_RADIUS,
    ) -> int:
        # Vegetation flag uses the fuel map directly: count cells with
        # combustibility > 0 in the front's radius-r Moore neighborhood.
        assert self.sim is not None
        combustibilities = np.asarray(
            self.sim.environment.terrain.features.combustibilities,
            dtype=float,
        )
        height, width = combustibilities.shape
        if not (0 <= source_i < height and 0 <= source_j < width):
            return 0

        i_start = max(source_i - radius, 0)
        i_stop = min(source_i + radius + 1, height)
        j_start = max(source_j - radius, 0)
        j_stop = min(source_j + radius + 1, width)

        neighborhood = combustibilities[i_start:i_stop, j_start:j_stop]
        fuel_mask = np.isfinite(neighborhood) & (neighborhood > 0.0)
        center_i = source_i - i_start
        center_j = source_j - j_start
        if 0 <= center_i < fuel_mask.shape[0] and 0 <= center_j < fuel_mask.shape[1]:
            fuel_mask[center_i, center_j] = False
        return int(np.count_nonzero(fuel_mask))

    @staticmethod
    def _nearest_position_distance(
        source_position: np.ndarray,
        target_positions: np.ndarray,
    ) -> float:
        targets = np.asarray(target_positions, dtype=float).reshape(-1, 2)
        source = np.asarray(source_position, dtype=float).reshape(-1)
        if targets.size == 0 or source.size < 2 or not np.all(np.isfinite(source[:2])):
            return math.inf
        distances = np.linalg.norm(targets - source[:2], axis=1)
        if distances.size == 0:
            return math.inf
        return float(np.min(distances))

    def _front_objective_vote(
        self,
        objective_i: int,
        objective_j: int,
    ) -> tuple[float, float, str | None]:
        # The front votes for whichever objective class is closer to where the
        # front is expected to move. Ties intentionally go to urban.
        front_positions = self._indices_to_positions(
            np.array([[objective_i, objective_j]], dtype=np.int64)
        )
        if front_positions.size == 0:
            return math.inf, math.inf, None
        front_position = front_positions[0]
        urban_distance = self._nearest_position_distance(
            front_position,
            self.urban_positions,
        )
        water_distance = self._nearest_position_distance(
            front_position,
            self.water_positions,
        )
        if not math.isfinite(urban_distance) and not math.isfinite(water_distance):
            vote = None
        elif not math.isfinite(urban_distance):
            vote = "water"
        elif not math.isfinite(water_distance):
            vote = "urban"
        elif urban_distance <= water_distance:
            vote = "urban"
        else:
            vote = "water"
        return urban_distance, water_distance, vote

    def _select_fire_front_diagnostics(
        self,
        burning_indices: np.ndarray,
    ) -> list[FireFrontDiagnostic]:
        """Analyze up to `state_fire_fronts` fastest burning cells."""
        assert self.sim is not None
        burning = self._coerce_grid_indices(np.asarray(burning_indices))
        if burning.size == 0:
            return []

        spread_rates = np.asarray(self.sim.wildfire.get_spread_rates(burning), dtype=float)
        finite = np.isfinite(spread_rates)
        if not np.any(finite):
            return []

        valid_indices = burning[finite]
        valid_rates = spread_rates[finite]
        # `state_fire_fronts` is a maximum. Early or small fires may have fewer
        # usable burning cells, so we analyze whatever valid fronts exist.
        front_limit = min(self.state_fire_fronts, valid_rates.size)
        if front_limit < valid_rates.size:
            # Avoid sorting every burning cell when only the top N fronts are
            # needed. Include every cell tied at the threshold so exact-rate
            # ties are deterministic; this only expands to a full sort when
            # many cells truly share the same spread rate.
            threshold = np.partition(valid_rates, -front_limit)[-front_limit]
            candidates = np.flatnonzero(valid_rates >= threshold)
            selected = candidates[np.argsort(valid_rates[candidates])[::-1]][
                :front_limit
            ]
        else:
            selected = np.argsort(valid_rates)[::-1]
        diagnostics: list[FireFrontDiagnostic] = []
        prop_aspects = self.sim.wildfire.prop_aspect
        for order_idx in selected:
            source_i, source_j = map(int, valid_indices[order_idx])
            spread_rate = float(valid_rates[order_idx])
            prop_aspect = float(prop_aspects[source_i, source_j])
            forward_cells = self._forward_neighbor_indices(
                source_i,
                source_j,
                prop_aspect,
            )
            projected_i: int | None = None
            projected_j: int | None = None
            if forward_cells:
                projected_i, projected_j = forward_cells[0]
            objective_i = projected_i if projected_i is not None else source_i
            objective_j = projected_j if projected_j is not None else source_j

            topography_priority = self._front_topography_priority_raw(
                source_i,
                source_j,
            )
            slope_factor = self._front_slope_factor(
                source_i,
                source_j,
                prop_aspect,
            )
            forward_combustible_uphill = self._has_forward_combustible_uphill(
                source_i,
                source_j,
                prop_aspect,
            )
            vegetation_neighbor_count = self._front_vegetation_neighbor_count(
                source_i,
                source_j,
            )
            has_vegetation_threat = (
                vegetation_neighbor_count > VEGETATION_MIN_COMBUSTIBLE_NEIGHBORS
            )
            distance_to_indirect_line = self._front_distance_to_fire_line_m(
                source_i,
                source_j,
            )
            near_indirect_line = (
                math.isfinite(distance_to_indirect_line)
                and distance_to_indirect_line <= INDIRECT_FRONT_DISTANCE_THRESHOLD_M
            )
            urban_distance, water_distance, objective_vote = (
                self._front_objective_vote(objective_i, objective_j)
            )
            # A positive front needs all pieces at once: it must be spreading,
            # have an uphill combustible opportunity, and have slope helping
            # propagation enough to matter.
            has_growth = (
                spread_rate > 0.0
                and topography_priority > 0.0
                and slope_factor >= TOPOGRAPHY_SLOPE_FACTOR_THRESHOLD
                and forward_combustible_uphill
            )
            diagnostics.append(
                FireFrontDiagnostic(
                    source_i=source_i,
                    source_j=source_j,
                    projected_i=projected_i,
                    projected_j=projected_j,
                    objective_i=objective_i,
                    objective_j=objective_j,
                    spread_rate=spread_rate,
                    topography_priority=topography_priority,
                    slope_factor=slope_factor,
                    forward_combustible_uphill=forward_combustible_uphill,
                    has_topography_growth=has_growth,
                    vegetation_combustible_neighbor_count=vegetation_neighbor_count,
                    has_vegetation_threat=has_vegetation_threat,
                    distance_to_indirect_line_m=distance_to_indirect_line,
                    near_indirect_line=near_indirect_line,
                    distance_to_urban=urban_distance,
                    distance_to_water=water_distance,
                    objective_vote=objective_vote,
                )
            )
        return diagnostics

    def _compute_fire_front_summary(
        self,
        burning_indices: np.ndarray,
    ) -> dict[str, float | int]:
        # Convert per-front diagnostics into compact binary state features.
        diagnostics = self._select_fire_front_diagnostics(burning_indices)
        self.last_front_diagnostics = diagnostics
        front_count = len(diagnostics)
        positive_count = sum(
            1 for front in diagnostics if front.has_topography_growth
        )
        required_count = int(math.ceil(front_count / 2.0)) if front_count else 0
        vegetation_positive_count = sum(
            1 for front in diagnostics if front.has_vegetation_threat
        )
        vegetation_required_count = required_count
        indirect_positive_count = sum(
            1 for front in diagnostics if front.near_indirect_line
        )
        indirect_required_count = (front_count // 2) + 1 if front_count else 0
        urban_count = sum(1 for front in diagnostics if front.objective_vote == "urban")
        water_count = sum(1 for front in diagnostics if front.objective_vote == "water")
        objective_count = urban_count + water_count
        topography_flag = (
            1.0
            if front_count > 0 and positive_count >= required_count
            else 0.0
        )
        vegetation_flag = (
            1.0
            if (
                front_count > 0
                and vegetation_positive_count >= vegetation_required_count
            )
            else 0.0
        )
        indirect_flag = (
            1.0
            if front_count > 0 and indirect_positive_count >= indirect_required_count
            else 0.0
        )
        urban_flag = 1.0 if objective_count > 0 and urban_count >= water_count else 0.0
        water_flag = 1.0 if objective_count > 0 and water_count > urban_count else 0.0
        summary: dict[str, float | int] = {
            "state_fire_fronts": self.state_fire_fronts,
            "state_fire_front_count": front_count,
            "topography_front_positive_count": positive_count,
            "topography_front_required_count": required_count,
            "topography_flag": topography_flag,
            "vegetation_front_positive_count": vegetation_positive_count,
            "vegetation_front_required_count": vegetation_required_count,
            "vegetation_flag": vegetation_flag,
            "indirect_front_positive_count": indirect_positive_count,
            "indirect_front_required_count": indirect_required_count,
            "indirect_flag": indirect_flag,
            "urban_front_count": urban_count,
            "water_front_count": water_count,
            "objective_front_count": objective_count,
            "urban_flag": urban_flag,
            "water_flag": water_flag,
        }
        self.last_front_summary = summary
        return summary

    def _mission_time(self) -> float:
        assert self.sim is not None
        return self.sim.timer.mission_runtime.total_seconds()

    def _mission_complete(self) -> bool:
        assert self.sim is not None
        fire_remaining = bool(self.sim.wildfire.burning_indices.shape[0])
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
        combinations = _expand_tactic_combinations(
            action,
            self.tactic_distribution,
            self.controlled_agent_count,
            self.aircraft_group_size,
            self.group_sizes,
        )
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
        if self.state_space in (
            STATE_SPACE_OLD,
            STATE_SPACE_UPDATED,
            STATE_SPACE_DIRECTIONAL,
            STATE_SPACE_DIRECTIONAL_2,
        ):
            return self._compute_old_state()
        raise RuntimeError(f"Unhandled state space: {self.state_space!r}")

    def _clear_fire_front_summary(self) -> None:
        self.last_front_diagnostics = []
        self.last_front_summary = {
            "state_fire_fronts": self.state_fire_fronts,
            "state_fire_front_count": 0,
            "topography_front_positive_count": 0,
            "topography_front_required_count": 0,
            "topography_flag": 0.0,
            "vegetation_front_positive_count": 0,
            "vegetation_front_required_count": 0,
            "vegetation_flag": 0.0,
            "indirect_front_positive_count": 0,
            "indirect_front_required_count": 0,
            "indirect_flag": 0.0,
            "urban_front_count": 0,
            "water_front_count": 0,
            "objective_front_count": 0,
            "urban_flag": 0.0,
            "water_flag": 0.0,
        }

    def _compute_old_state(self) -> np.ndarray:
        assert self.sim is not None
        if self.state_space in (
            STATE_SPACE_OLD,
            STATE_SPACE_DIRECTIONAL,
            STATE_SPACE_DIRECTIONAL_2,
        ):
            self._clear_fire_front_summary()
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
        distance_fire_line = math.nan
        distance_water = math.nan
        distance_boundary_to_water = 1.0
        distance_boundary_to_vip = 1.0
        distance_boundary_to_vegetation = 1.0
        distance_boundary_to_topography = 1.0
        distance_boundary_to_indirect = 1.0
        topography_flag = 0.0
        vegetation_flag = 0.0
        indirect_flag = 0.0
        urban_flag = 0.0
        water_flag = 0.0

        if burning_count:
            if self.state_space == STATE_SPACE_UPDATED:
                front_summary = self._compute_fire_front_summary(burning_indices)
                topography_flag = float(front_summary["topography_flag"])
                vegetation_flag = float(front_summary["vegetation_flag"])
                indirect_flag = float(front_summary["indirect_flag"])
                urban_flag = float(front_summary["urban_flag"])
                water_flag = float(front_summary["water_flag"])
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
        else:
            self._clear_fire_front_summary()

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
        boundary_left_norm = _scale_to_unit(float(boundary_left), 0.0, coord_width)
        boundary_right_norm = _scale_to_unit(float(boundary_right), 0.0, coord_width)
        boundary_bottom_norm = _scale_to_unit(float(boundary_bottom), 0.0, coord_height)
        boundary_top_norm = _scale_to_unit(float(boundary_top), 0.0, coord_height)

        # Ignition point of the current episode, scaled on the same map extent as
        # the fire-geometry coordinates so the two are directly comparable. Zero
        # when the scenario declared no ignition center.
        ignition_x_norm = 0.0
        ignition_y_norm = 0.0
        if self.current_ignition_pos is not None:
            ignition_x_norm = _scale_to_unit(
                float(self.current_ignition_pos[0]), x_min, x_max
            )
            ignition_y_norm = _scale_to_unit(
                float(self.current_ignition_pos[1]), y_min, y_max
            )

        agents = self.sim.firefighters.firefighters
        state_features: list[float] = []
        if self.include_scenario_flag:
            state_features.extend(
                _clip01(value) for value in self._current_scenario_one_hot()
            )
        core_values = [
            time_since_detection_norm,
            temperature_norm,
            humidity_norm,
            wind_speed_norm,
            wind_direction_norm,
            time_to_sunset_norm,
            distance_fire_line_norm,
        ]
        # "old", "directional" and "directional-2" add the fire->water distance
        # right after the fire-line distance; "updated" omits that distance.
        if self.state_space in (
            STATE_SPACE_OLD,
            STATE_SPACE_DIRECTIONAL,
            STATE_SPACE_DIRECTIONAL_2,
        ):
            core_values.append(distance_water_norm)
        core_values.extend(
            [
                _clip01(distance_boundary_to_water),
                _clip01(distance_boundary_to_vip),
                _clip01(distance_boundary_to_vegetation),
                _clip01(distance_boundary_to_topography),
                _clip01(distance_boundary_to_indirect),
            ]
        )
        # "directional-2" carries the ignition point just ahead of the fire
        # geometry it is meant to be read against.
        if self.state_space == STATE_SPACE_DIRECTIONAL_2:
            core_values.extend([ignition_x_norm, ignition_y_norm])
        core_values.extend(
            [
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
            ]
        )
        if self.state_space == STATE_SPACE_UPDATED:
            core_values.extend(
                [
                    _clip01(topography_flag),
                    _clip01(vegetation_flag),
                    _clip01(indirect_flag),
                    _clip01(urban_flag),
                    _clip01(water_flag),
                ]
            )
        # "directional-2" omits the four map-boundary distances: each one is an
        # exact duplicate of a fire-extreme coordinate already emitted above.
        if self.state_space != STATE_SPACE_DIRECTIONAL_2:
            core_values.extend(
                [
                    boundary_left_norm,
                    boundary_right_norm,
                    boundary_bottom_norm,
                    boundary_top_norm,
                ]
            )
        state_features.extend(core_values)

        # Both directional state spaces append the per-flank projected-threat
        # block between the core and the agent block.
        if self.state_space in (STATE_SPACE_DIRECTIONAL, STATE_SPACE_DIRECTIONAL_2):
            state_features.extend(
                self._compute_directional_front_block(burning_indices)
            )

        for idx in range(self.controlled_agent_count):
            if idx < len(agents):
                agent = agents[idx]
                pos_x, pos_y = agent.pos
                agent_values = [
                    _scale_to_unit(float(pos_x), x_min, x_max),
                    _scale_to_unit(float(pos_y), y_min, y_max),
                ]
                # "updated" keeps the per-aircraft altitude channel and adds
                # aggregate front flags; "old" drops both.
                if self.state_space == STATE_SPACE_UPDATED:
                    agent_values.append(
                        _scale_to_unit(float(agent.altitude), 0.0, ALTITUDE_NORM_M)
                    )
                state_features.extend(agent_values)
            else:
                state_features.extend([0.0] * self.agent_feature_count)

        obs = np.array(state_features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

    # -- Directional state (per-flank projected threat) ----------------------
    def _compute_directional_front_block(
        self, burning_indices: np.ndarray
    ) -> list[float]:
        """Return the K*6 per-flank threat block for the directional state.

        Each of ``state_fire_fronts`` slots is one compass sector around the
        fire centroid, zero-padded when empty. For the highest-ROS active cell
        in a sector we project a landing point T = decision_interval_minutes
        ahead and score five threat/resource channels around it.
        """
        assert self.sim is not None
        n_feat = len(DIRECTIONAL_PER_FRONT_FEATURES)
        fronts = int(self.state_fire_fronts)
        block = [0.0] * (fronts * n_feat)

        burning = self._coerce_grid_indices(np.asarray(burning_indices))
        if burning.size == 0:
            return block

        wildfire = self.sim.wildfire
        fire_states = np.asarray(wildfire.fire_states)
        height, width = fire_states.shape
        bi = burning[:, 0].astype(np.int64)
        bj = burning[:, 1].astype(np.int64)
        ros = np.asarray(wildfire.get_spread_rates(burning), dtype=float).reshape(-1)
        aspect = np.asarray(wildfire.prop_aspect)[bi, bj].astype(float)

        active = (
            np.isfinite(ros)
            & (ros > DIRECTIONAL_SPREAD_EPS)
            & np.isfinite(aspect)
            & self._has_combustible_neighbor(bi, bj, fire_states, height, width)
        )
        if not np.any(active):
            return block

        ai = bi[active]
        aj = bj[active]
        a_ros = ros[active]
        a_aspect = aspect[active]

        # Sector = compass bearing from the fire centroid (same convention as
        # _grid_offset_aspect: 0=N, 90=E).
        centroid_i = float(bi.mean())
        centroid_j = float(bj.mean())
        bearing = (
            np.degrees(np.arctan2(aj - centroid_j, -(ai - centroid_i))) + 360.0
        ) % 360.0
        sector_width = 360.0 / fronts
        sectors = np.clip((bearing / sector_width).astype(np.int64), 0, fronts - 1)

        cell_size = float(self._cell_size_m)
        map_diagonal = float(self._map_diagonal)
        horizon_min = float(self.decision_interval_minutes)
        combustibilities = self.sim.environment.terrain.features.combustibilities

        for k in range(fronts):
            in_sector = sectors == k
            if not np.any(in_sector):
                continue
            k_ros = a_ros[in_sector]
            k_i = ai[in_sector]
            k_j = aj[in_sector]
            k_aspect_all = a_aspect[in_sector]
            k_bearing = bearing[in_sector]

            severity = float(
                min(max(float(k_ros.mean()) / MAX_SPREAD_RATE_NORM_MPM, 0.0), 1.0)
            )

            rep = int(np.argmax(k_ros))
            src_i = int(k_i[rep])
            src_j = int(k_j[rep])
            src_ros = float(k_ros[rep])
            # Direction repair: trust the representative's own propagation
            # bearing only when it points outward (cos of the angle to the
            # radial outward bearing exceeds the threshold); otherwise -- when it
            # points inward/lateral, or is NaN (cos NaN -> NaN > x is False) --
            # fall back to the outward bearing so the projected landing can never
            # be aimed into the already-burnt interior.
            src_aspect_raw = float(k_aspect_all[rep])
            src_outward = float(k_bearing[rep])
            align = math.cos(math.radians(src_aspect_raw - src_outward))
            src_dir = (
                src_aspect_raw if align > DIRECTIONAL_ALIGN_MIN else src_outward
            )

            land_i, land_j = self._project_flank_landing(
                src_i, src_j, src_dir, src_ros, horizon_min,
                cell_size, fire_states, height, width,
            )

            reach_cells = (src_ros * horizon_min) / cell_size
            radius_cells = float(
                min(
                    max(DIRECTIONAL_SEARCH_ALPHA * reach_cells, DIRECTIONAL_R_MIN_CELLS),
                    DIRECTIONAL_R_MAX_CELLS,
                )
            )
            radius_m = radius_cells * cell_size

            landing_pos = self._indices_to_positions(
                np.array([[land_i, land_j]], dtype=np.int64)
            )
            # Time-to-impact: scale the landing->VIP distance by the distance the
            # flank covers in VIP_HORIZONS intervals, so urgency rises while there
            # is still lead time (ROS*T and the KD distance are both world metres,
            # so this is independent of the cell-size source).
            vip_scale_m = src_ros * DIRECTIONAL_VIP_HORIZONS * horizon_min
            vip_urgency = self._tree_proximity(
                self._vip_tree, landing_pos, vip_scale_m
            )
            # Water proximity on a fixed shuttle-distance scale (not the whole
            # map diagonal, which pinned every sector near 1).
            water_access = self._tree_proximity(
                self._water_tree, landing_pos, DIRECTIONAL_WATER_SCALE_M
            )

            # Closeness of the projected flank to the planned containment line on
            # a fixed approach-distance scale (not the tiny vegetation radius).
            line_dist = self._front_distance_to_fire_line_m(land_i, land_j)
            indirect_urgency = (
                float(min(max(1.0 - line_dist / DIRECTIONAL_LINE_SCALE_M, 0.0), 1.0))
                if math.isfinite(line_dist)
                else 0.0
            )

            vegetation_urgency = self._vegetation_escalation(
                src_i, src_j, land_i, land_j, radius_cells,
                fire_states, combustibilities, height, width,
            )
            topography_urgency = self._topography_local(land_i, land_j, src_dir)

            base = k * n_feat
            block[base + 0] = severity
            block[base + 1] = vip_urgency
            block[base + 2] = vegetation_urgency
            block[base + 3] = topography_urgency
            block[base + 4] = indirect_urgency
            block[base + 5] = water_access
        return block

    @staticmethod
    def _has_combustible_neighbor(
        bi: np.ndarray,
        bj: np.ndarray,
        fire_states: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Vectorized: True where a burning cell has a combustible 8-neighbor."""
        has = np.zeros(bi.shape[0], dtype=bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = bi + di
                nj = bj + dj
                valid = (ni >= 0) & (ni < height) & (nj >= 0) & (nj < width)
                ni_c = np.clip(ni, 0, height - 1)
                nj_c = np.clip(nj, 0, width - 1)
                has |= valid & (fire_states[ni_c, nj_c] == COMBUSTIBLE)
        return has

    @staticmethod
    def _project_flank_landing(
        i: int,
        j: int,
        aspect: float,
        ros: float,
        horizon_min: float,
        cell_size: float,
        fire_states: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        """Ray-march the flank `horizon_min` ahead, clipped at barriers/edge."""
        if cell_size <= 0.0 or not math.isfinite(aspect):
            return i, j
        reach_cells = (ros * horizon_min) / cell_size
        n_steps = int(min(reach_cells, DIRECTIONAL_MAX_RAY_CELLS))
        if n_steps < 1:
            return i, j
        theta = math.radians(aspect)
        di = -math.cos(theta)
        dj = math.sin(theta)
        steps = np.arange(1, n_steps + 1)
        ii = np.rint(i + steps * di).astype(np.int64)
        jj = np.rint(j + steps * dj).astype(np.int64)
        in_bounds = (ii >= 0) & (ii < height) & (jj >= 0) & (jj < width)
        if not bool(in_bounds[0]):
            return i, j
        if not bool(np.all(in_bounds)):
            limit = int(np.argmin(in_bounds))  # first out-of-bounds step
            ii = ii[:limit]
            jj = jj[:limit]
        if ii.size == 0:
            return i, j
        # Fire only advances into unburned fuel or actively-burning cells; BURNT,
        # EXTINGUISHING, SUPPRESSED and NONFLAMMABLE all stop the projection so a
        # landing is never placed inside the already-burnt interior.
        s = fire_states[ii, jj]
        barrier = ~(
            (s == COMBUSTIBLE) | (s == EARLY_BURNING) | (s == FULL_BURNING)
        )
        if bool(barrier.any()):
            stop = int(np.argmax(barrier))
            if stop == 0:
                return i, j
            return int(ii[stop - 1]), int(jj[stop - 1])
        return int(ii[-1]), int(jj[-1])

    def _tree_proximity(
        self, tree: cKDTree | None, point_pos: np.ndarray, scale: float
    ) -> float:
        """1 - clip(dist(point, nearest tree node)/scale, 0, 1); 0 if no tree."""
        if tree is None or point_pos.size == 0 or scale <= 0.0:
            return 0.0
        dist, _ = tree.query(point_pos[0])
        return float(min(max(1.0 - float(dist) / scale, 0.0), 1.0))

    @staticmethod
    def _vegetation_escalation(
        src_i: int,
        src_j: int,
        land_i: int,
        land_j: int,
        radius_cells: float,
        fire_states: np.ndarray,
        combustibilities: np.ndarray,
        height: int,
        width: int,
    ) -> float:
        """Max flammability of unburned fuel near the landing that is more
        flammable than the flank's current fuel (mirrors the VEGETATION tactic),
        normalized by the most flammable fuel."""
        current = float(combustibilities[src_i, src_j])
        r = int(min(max(int(round(radius_cells)), 1), DIRECTIONAL_R_MAX_CELLS))
        i0 = max(land_i - r, 0)
        i1 = min(land_i + r + 1, height)
        j0 = max(land_j - r, 0)
        j1 = min(land_j + r + 1, width)
        if i0 >= i1 or j0 >= j1:
            return 0.0
        win_states = fire_states[i0:i1, j0:j1]
        win_comb = combustibilities[i0:i1, j0:j1]
        mask = (win_states == COMBUSTIBLE) & (win_comb > current)
        if not np.any(mask):
            return 0.0
        best = float(win_comb[mask].max())
        return float(min(max(best / MAX_COMBUSTIBILITY, 0.0), 1.0))

    def _topography_local(self, land_i: int, land_j: int, aspect: float) -> float:
        """Local uphill drive at the landing, gated on burnable higher ground."""
        if DIRECTIONAL_SLOPE_REF <= 1.0:
            return 0.0
        if not self._has_forward_combustible_uphill(int(land_i), int(land_j), aspect):
            return 0.0
        slope_factor = self._front_slope_factor(int(land_i), int(land_j), aspect)
        if not math.isfinite(slope_factor):
            return 0.0
        value = (slope_factor - 1.0) / (DIRECTIONAL_SLOPE_REF - 1.0)
        return float(min(max(value, 0.0), 1.0))

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
            "tactic_distribution": self.tactic_distribution,
            "aircraft_group_size": self.aircraft_group_size,
            "controlled_agent_count": self.controlled_agent_count,
            "action_decision_count": self.action_decision_count,
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

        if self.gc_collect_on_reset:
            # Drop the prior simulation reference and force cycle collection
            # before allocating the new one. Tests whether retained sim
            # instances explain per-worker RSS growth across resets.
            self.sim = None
            gc.collect()

        self._init_simulation(int(sim_seed))
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
            "tactic_distribution": self.tactic_distribution,
            "aircraft_group_size": self.aircraft_group_size,
            "controlled_agent_count": self.controlled_agent_count,
            "action_decision_count": self.action_decision_count,
        }
        self.last_info.update(self.last_front_summary)
        return observation, self.last_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.sim is not None
        self._count_pending_mission_start()
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
        info.update(self.last_front_summary)

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
                "tactic_distribution": self.tactic_distribution,
                "aircraft_group_size": self.aircraft_group_size,
                "controlled_agent_count": self.controlled_agent_count,
                "action_decision_count": self.action_decision_count,
            }

        self.last_info = info
        return observation, reward, terminated, truncated, info



class StopTrainingOnTotalEpisodes(BaseCallback):
    """Stop once ``max_episodes`` episodes have finished across ALL workers.

    SB3 ships ``StopTrainingOnMaxEpisodes``, but that one multiplies the limit
    by ``num_envs`` (its budget is per-env). ``--simulations`` is a total count,
    so we count ``dones`` across the whole vector env instead.
    """

    def __init__(self, max_episodes: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        if int(max_episodes) <= 0:
            raise ValueError(f"max_episodes must be > 0 (got {max_episodes})")
        self.max_episodes = int(max_episodes)
        self.n_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.n_episodes += int(np.sum(dones))
        if self.n_episodes < self.max_episodes:
            return True
        if self.verbose >= 1:
            print(
                f"Reached --simulations={self.max_episodes} "
                f"({self.n_episodes} episodes finished) after "
                f"{self.num_timesteps} timesteps; stopping training."
            )
        return False


class StopTrainingOnMissionSimulations(BaseCallback):
    """Stop once every mission has run its share of the simulation budget.

    ``StopTrainingOnTotalEpisodes`` stops on the fleet-wide episode count, which
    a multi-mission run can reach with the split still lopsided — the short-
    episode map finishes episodes several times faster than the long one. This
    stops on the per-mission counts instead, so the run ends holding the split
    the CLI asked for. Which mission each simulation runs is the env's decision
    (phases in list order); this only decides when to stop, and reports each
    phase as it completes.

    Counts start at ``completed``, the shares earlier chunks of a chained run
    already spent, so the quotas span the chain rather than each chunk.
    """

    def __init__(
        self,
        quotas: dict[str, int],
        completed: dict[str, int] | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        if not quotas:
            raise ValueError("quotas must not be empty")
        self.quotas = {name: int(quota) for name, quota in quotas.items()}
        self.counts = {
            name: int((completed or {}).get(name, 0)) for name in self.quotas
        }

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or ():
            if "episode_summary" not in info:
                continue
            scenario_name = info.get("scenario_name")
            if scenario_name not in self.counts:
                continue
            self.counts[scenario_name] += 1
            if (
                self.verbose >= 1
                and self.counts[scenario_name] == self.quotas[scenario_name]
            ):
                print(
                    f"Mission phase complete: {scenario_name} finished its "
                    f"{self.quotas[scenario_name]} simulations after "
                    f"{self.num_timesteps} timesteps."
                )
        if any(self.counts[name] < self.quotas[name] for name in self.quotas):
            return True
        if self.verbose >= 1:
            split = ", ".join(
                f"{name}: {self.counts[name]}/{self.quotas[name]}"
                for name in self.quotas
            )
            print(
                f"Mission simulation budget spent ({split}) after "
                f"{self.num_timesteps} timesteps; stopping training."
            )
        return False


class StopTrainingOnChunkTimesteps(BaseCallback):
    """Stop once this chunk has collected ``chunk_timesteps`` new timesteps.

    Chained runs hand ``model.learn()`` the *chain's* remaining budget so SB3's
    ``progress_remaining`` (and therefore the LR schedule) spans the whole
    chain rather than restarting each chunk. The per-chunk budget is enforced
    here instead, counting from the timestep the chunk started at.
    """

    def __init__(self, chunk_timesteps: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        if int(chunk_timesteps) <= 0:
            raise ValueError(f"chunk_timesteps must be > 0 (got {chunk_timesteps})")
        self.chunk_timesteps = int(chunk_timesteps)
        self._start_timesteps: int | None = None

    def _on_training_start(self) -> None:
        self._start_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        start = self._start_timesteps or 0
        collected = int(self.num_timesteps) - start
        if collected < self.chunk_timesteps:
            return True
        if self.verbose >= 1:
            print(
                f"Chunk budget reached: {collected} timesteps collected this "
                f"chunk (>= {self.chunk_timesteps}); total {self.num_timesteps}. "
                "Stopping so the chunk can save and hand off."
            )
        return False


class StopTrainingOnWallClock(BaseCallback):
    """Stop after ``max_hours`` of wall-clock training.

    Safety net for chunked runs: SB3 exits ``learn()`` cleanly, so the normal
    ``--save-model`` and CSV flush still run. Without it a chunk whose budget
    does not fit the queue's wall limit is SIGKILLed and loses everything since
    the last checkpoint.
    """

    def __init__(self, max_hours: float, verbose: int = 1):
        super().__init__(verbose=verbose)
        if float(max_hours) <= 0.0:
            raise ValueError(f"max_hours must be > 0 (got {max_hours})")
        self.max_seconds = float(max_hours) * 3600.0
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()

    def _on_step(self) -> bool:
        if self._start_time is None:
            return True
        elapsed = time.perf_counter() - self._start_time
        if elapsed < self.max_seconds:
            return True
        if self.verbose >= 1:
            print(
                f"Wall-clock budget reached: {elapsed / 3600.0:.2f}h trained "
                f"(limit {self.max_seconds / 3600.0:.2f}h) after "
                f"{self.num_timesteps} timesteps; stopping so the chunk can "
                "save and hand off."
            )
        return False


def _estimate_episode_steps(
    scenario_paths: Sequence[Path],
    decision_interval_minutes: int,
) -> int:
    """Upper bound on decision steps per episode, from the scenario JSONs.

    Mirrors ``WildfireHourlyEnv._resolve_max_steps`` (max_runtime divided by
    the decision interval) but reads ``max_runtime`` straight out of the JSON
    so no terrain has to be loaded. Takes the max across scenarios so the
    derived timestep ceiling covers the longest episode when switching.
    """
    interval_seconds = max(1.0, float(decision_interval_minutes) * 60.0)
    longest = 0
    for path in scenario_paths:
        with open(path, "r", encoding="utf-8") as handle:
            max_runtime = float(json.load(handle)["max_runtime"])
        longest = max(longest, math.ceil(max_runtime / interval_seconds))
    return max(1, longest)


class TrainingLogger(BaseCallback):
    """Capture per-step training data and per-episode summaries."""

    def __init__(
        self,
        decision_interval_minutes: int = DEFAULT_DECISION_INTERVAL_MINUTES,
        tactic_distribution: str = TACTIC_DISTRIBUTION_INDIVIDUAL,
        aircraft_group_size: int = AIRCRAFT_GROUP_SIZE,
        group_sizes: Sequence[int] | None = None,
        controlled_agent_count: int = CONTROLLED_AGENT_COUNT,
        progress_file_tag: str = "run",
        output_dir: Path = SCENARIOS_DIR / "outputs",
        resume_summaries: Sequence[dict[str, Any]] | None = None,
        episode_progress: EpisodeProgress | None = None,
        lr_episode_offset: int | None = None,
        summary_retention: int = 0,
        mission_names: Sequence[str] | None = None,
        mission_budget_mode: str = MISSION_BUDGET_SIMULATIONS,
    ):
        super().__init__()
        self.training_step_records: list[dict[str, Any]] = []
        self.decision_step_records: list[dict[str, Any]] = []
        # Per-env buffers let us export decision rows only after their
        # simulation episode has completed and received a simulation index.
        self._episode_decision_buffers: dict[int, list[dict[str, Any]]] = {}
        # Seeding with a previous chunk's summary rows keeps the episode
        # numbering continuous across a chained run and keeps every progress
        # file a full-history file, which is what the plotting scripts assume
        # when they pick the highest-N summary in a run directory.
        self.episode_summaries: list[dict[str, Any]] = list(resume_summaries or ())
        self.resumed_episodes = len(self.episode_summaries)
        self.completed_episodes = self.resumed_episodes
        # Episode numbering follows the resumed CSV rows so each progress file
        # matches its own contents. The LR offset is tracked separately because
        # it comes from the sidecar written with the weights, which can be up
        # to one export interval ahead of the CSV.
        self.lr_episode_offset = (
            self.resumed_episodes
            if lr_episode_offset is None
            else int(lr_episode_offset)
        )
        self.episode_progress = episode_progress
        if self.episode_progress is not None:
            self.episode_progress.completed = self.lr_episode_offset
        self.controlled_agent_count = int(controlled_agent_count)
        if self.controlled_agent_count <= 0:
            raise ValueError("controlled_agent_count must be > 0")
        self.tactic_distribution = _normalize_tactic_distribution(
            tactic_distribution
        )
        self.aircraft_group_size = int(aircraft_group_size)
        if self.aircraft_group_size <= 0:
            raise ValueError("aircraft_group_size must be > 0")
        self.group_sizes = (
            _validate_group_sizes(group_sizes, self.controlled_agent_count)
            if group_sizes is not None
            else None
        )
        self.action_decision_count = _action_decision_count(
            self.tactic_distribution,
            self.controlled_agent_count,
            self.aircraft_group_size,
            self.group_sizes,
        )
        self.decision_interval_minutes = decision_interval_minutes
        self.summary_retention = max(0, int(summary_retention))
        self.progress_file_tag = progress_file_tag
        self.output_dir = Path(output_dir)
        self._global_scenario_ts_counts: dict[str, int] = {}
        # Per-mission episode tally driving the --missions budget split. Seeded
        # from the resumed summary so a chained run keeps spending the same
        # split rather than restarting it each chunk; counts every mission the
        # run knows about, including those already finished.
        self.mission_names = tuple(mission_names or ())
        self.mission_budget_mode = mission_budget_mode
        self._resumed_scenario_ep_counts: dict[str, int] = (
            _episode_counts_by_scenario(self.episode_summaries, self.mission_names)
            if self.mission_names
            else {}
        )
        self._global_scenario_ep_counts: dict[str, int] = dict(
            self._resumed_scenario_ep_counts
        )

    def scenario_episode_counts(self) -> dict[str, int]:
        """Episodes per mission over the whole chain, this chunk included."""
        return dict(self._global_scenario_ep_counts)

    def episode_count_for_chain(self) -> int:
        """Episodes completed by the whole chain, this chunk included.

        Counts from the LR offset (the previous chunk's authoritative count)
        rather than from the resumed CSV rows, so a gap between the two does
        not accumulate across chunks.
        """
        return self.lr_episode_offset + (
            self.completed_episodes - self.resumed_episodes
        )

    def _on_rollout_end(self) -> None:
        if self._global_scenario_ts_counts:
            self.training_env.env_method(
                "set_global_scenario_counts",
                self._global_scenario_ts_counts,
            )

    def _sync_mission_progress(self) -> None:
        """Give every worker the fleet-wide view its next mission choice needs.

        A worker only knows what it has run itself, so on its own it would keep
        a phase going long past the fleet's share. Collecting the per-worker
        start counts and handing back the totals is what makes the phase
        boundary a fleet-wide decision. Run every step rather than at rollout
        end: a rollout is 144 steps per worker, which is enough episodes for a
        phase to overrun its share by hundreds of simulations. Both calls are
        small dicts over pipes that are idle between steps, next to multi-second
        simulation steps.
        """
        if not self.mission_names:
            return
        if self.mission_budget_mode == MISSION_BUDGET_TIMESTEPS:
            # Timestep-budgeted phases switch on the per-mission timestep
            # counts, which this callback already tallies from the infos.
            self.training_env.env_method(
                "set_global_scenario_counts",
                self._global_scenario_ts_counts,
            )
            return
        totals = dict(self._resumed_scenario_ep_counts)
        for counts in self.training_env.env_method("get_scenario_episode_starts"):
            for name, value in counts.items():
                totals[name] = totals.get(name, 0) + int(value)
        self.training_env.env_method("set_fleet_scenario_episode_starts", totals)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        for env_idx, (info, action, reward) in enumerate(
            zip(infos, actions, rewards, strict=False)
        ):
            action = np.asarray(action)
            action_labels = _tactic_combinations_from_action(action)
            agent_action_labels = _expand_tactic_combinations(
                action,
                self.tactic_distribution,
                self.controlled_agent_count,
                self.aircraft_group_size,
                self.group_sizes,
            )
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
                "tactic_distribution": info.get(
                    "tactic_distribution", self.tactic_distribution
                ),
                "aircraft_group_size": info.get(
                    "aircraft_group_size", self.aircraft_group_size
                ),
                "controlled_agent_count": info.get(
                    "controlled_agent_count", self.controlled_agent_count
                ),
                "action_decision_count": info.get(
                    "action_decision_count", self.action_decision_count
                ),
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
                "cumulative_reward": info.get("cumulative_reward"),
                "propagation_factor": info.get("propagation_factor"),
                "propagation_penalty": info.get("propagation_penalty"),
                "last_interval": 1 if "episode_summary" in info else 0,
                "topography_flag": info.get("topography_flag"),
                "state_fire_fronts": info.get("state_fire_fronts"),
                "state_fire_front_count": info.get("state_fire_front_count"),
                "topography_front_positive_count": info.get(
                    "topography_front_positive_count"
                ),
                "topography_front_required_count": info.get(
                    "topography_front_required_count"
                ),
                "vegetation_flag": info.get("vegetation_flag"),
                "vegetation_front_positive_count": info.get(
                    "vegetation_front_positive_count"
                ),
                "vegetation_front_required_count": info.get(
                    "vegetation_front_required_count"
                ),
                "indirect_flag": info.get("indirect_flag"),
                "indirect_front_positive_count": info.get(
                    "indirect_front_positive_count"
                ),
                "indirect_front_required_count": info.get(
                    "indirect_front_required_count"
                ),
                "urban_flag": info.get("urban_flag"),
                "water_flag": info.get("water_flag"),
                "urban_front_count": info.get("urban_front_count"),
                "water_front_count": info.get("water_front_count"),
                "objective_front_count": info.get("objective_front_count"),
                "delta_moe": info.get("delta_moe"),
            }
            if self.tactic_distribution == TACTIC_DISTRIBUTION_GROUP:
                for idx, combo in enumerate(action_labels):
                    record[f"group_{idx}_select_poi"] = combo[0].value
                    record[f"group_{idx}_track_poi"] = combo[1].value
                    record[f"group_{idx}_suppress"] = combo[2].value
            for idx, combo in enumerate(agent_action_labels):
                record[f"agent_{idx}_select_poi"] = combo[0].value
                record[f"agent_{idx}_track_poi"] = combo[1].value
                record[f"agent_{idx}_suppress"] = combo[2].value
            record.update(metrics_dict)
            record.update({f"{key}_delta": value for key, value in deltas_dict.items()})
            record.update(
                {f"{key}_cumulative_delta": value for key, value in cumulative_dict.items()}
            )
            self.training_step_records.append(record)
            self._episode_decision_buffers.setdefault(env_idx, []).append(
                dict(record)
            )

            if "episode_summary" in info:
                self.completed_episodes += 1
                if self.episode_progress is not None:
                    self.episode_progress.completed = self.lr_episode_offset + (
                        self.completed_episodes - self.resumed_episodes
                    )
                summary_record = dict(info["episode_summary"])
                episode_decisions = self._episode_decision_buffers.pop(env_idx, [])
                for decision_record in episode_decisions:
                    decision_record["simulation_index"] = self.completed_episodes
                    decision_record["episode_total_decision_steps"] = (
                        summary_record.get("total_decision_steps")
                    )
                    decision_record["episode_total_minutes"] = summary_record.get(
                        "total_minutes"
                    )
                self.decision_step_records.extend(episode_decisions)
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
                episode_scenario = info.get("scenario_name")
                if episode_scenario in self._global_scenario_ep_counts:
                    self._global_scenario_ep_counts[episode_scenario] += 1
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
                        export_decision_steps=(
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
        self._sync_mission_progress()
        return True

    def _on_training_end(self) -> None:
        # Flush any remaining buffered step records to avoid losing the tail chunk.
        if self.training_step_records:
            self._export_progress(
                self.completed_episodes,
                export_steps=True,
                export_decision_steps=bool(self.decision_step_records),
                export_summary=False,
            )
        elif self.decision_step_records:
            self._export_progress(
                self.completed_episodes,
                export_steps=False,
                export_decision_steps=True,
                export_summary=False,
            )

    def _export_progress(
        self,
        episode_count: int,
        *,
        export_steps: bool,
        export_decision_steps: bool,
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
        if export_decision_steps and self.decision_step_records:
            decision_step_path = output_dir / (
                f"results_{self.progress_file_tag}_{episode_count}_decision_steps.csv"
            )
            batch_index = (
                max(episode_count - 1, 0) // LOG_INTERVAL_STEPS_EPISODES + 1
            )
            for record in self.decision_step_records:
                record["batch"] = batch_index
            _write_records(self.decision_step_records, decision_step_path)
            print(f"Decision-step log written to {decision_step_path}")
            self.decision_step_records.clear()
        if export_summary and self.episode_summaries:
            summary_path = output_dir / (
                f"results_{self.progress_file_tag}_{episode_count}_summary.csv"
            )
            _write_records(self.episode_summaries, summary_path)
            print(f"Episode summary log written to {summary_path}")
            self._prune_summaries(newest=summary_path)

    def _prune_summaries(self, newest: Path) -> None:
        """Keep only the newest ``summary_retention`` summary snapshots.

        Every snapshot holds the full episode history, so each one is a prefix
        of the next and the older files carry no extra information. Keeping
        them all costs O(N^2) disk: a 150k-episode run writes 1500 snapshots
        averaging ~29 MB, about 43 GB. Pruning runs only after the replacement
        has been written, so nothing is lost if the job dies mid-export.
        """
        if self.summary_retention <= 0:
            return
        prefix = f"results_{self.progress_file_tag}_"
        snapshots: list[tuple[int, Path]] = []
        for path in self.output_dir.glob(f"{prefix}*_summary.csv"):
            tail = path.name[len(prefix) : -len("_summary.csv")]
            if tail.isdigit():
                snapshots.append((int(tail), path))
        snapshots.sort()
        for _, path in snapshots[: -self.summary_retention]:
            if path == newest:
                continue
            try:
                path.unlink()
            except OSError as err:  # noqa: PERF203 - rare, and non-fatal
                print(f"Could not prune old summary {path}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on one scenario or switch across multiple scenarios."
    )
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON (name in inputs/ or absolute path).",
    )
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--timesteps",
        type=int,
        help="Total PPO timesteps to collect during training (default: 1680).",
    )
    budget_group.add_argument(
        "--simulations",
        type=int,
        help=(
            "Train for this many simulations (episodes) in total across all "
            "workers, instead of a fixed timestep budget. The timestep budget "
            "is derived as simulations x the scenario's max episode length "
            "(max_runtime / decision-interval-minutes) and acts as a ceiling; "
            "training stops as soon as the episode count is reached."
        ),
    )
    chain_group = parser.add_mutually_exclusive_group()
    chain_group.add_argument(
        "--chain-total-timesteps",
        type=int,
        help=(
            "Total PPO timesteps for a CHAINED run split across several SLURM "
            "jobs. With this set, --timesteps/--simulations size only THIS "
            "chunk, while the LR schedule, the checkpoint numbering and the "
            "stopping point are driven by the chain total: the chunk resumes "
            "at the loaded model's timestep count and trains until its own "
            "budget is spent. When the chain total is already reached the run "
            "exits immediately printing CHAIN COMPLETE."
        ),
    )
    chain_group.add_argument(
        "--chain-total-simulations",
        type=int,
        help=(
            "Simulation (episode) equivalent of --chain-total-timesteps. "
            "Episodes already completed are counted from the resumed summary "
            "(--resume-summary), so this is a whole-chain episode budget. The "
            "LR schedule uses the derived timestep ceiling (simulations x "
            "longest episode), exactly as a single-job --simulations run does."
        ),
    )
    parser.add_argument(
        "--summary-retention",
        type=int,
        default=0,
        help=(
            "Keep only the newest N results_<tag>_<N>_summary.csv snapshots, "
            "deleting older ones once their replacement is written. Each "
            "snapshot holds the whole episode history, so keeping them all "
            "costs O(episodes^2) disk — about 43 GB over a 150k-episode run. "
            "Default 0 keeps every snapshot (previous behaviour). Use 3 for "
            "long chained runs; the newest file still holds everything, so "
            "the plotting scripts and --resume-summary are unaffected."
        ),
    )
    parser.add_argument(
        "--chain-elapsed-timesteps",
        type=int,
        help=(
            "Declare how far into a --chain-total-timesteps chain the loaded "
            "model already is, overriding the timestep counter stored in its "
            "zip. Needed when adopting a run whose counter does not reflect "
            "true chain progress — e.g. a policy continued with --load-model "
            "before chaining existed, whose counter restarted at 0 each job. "
            "The counter is set to this value, so the LR schedule, the "
            "stopping point and checkpoint numbering all follow it."
        ),
    )
    parser.add_argument(
        "--resume-summary",
        help=(
            "Path to a previous chunk's results_<tag>_<N>_summary.csv, or "
            "'auto' to pick the highest-N summary for --progress-file-tag in "
            "--output-dir. Its rows are prepended to this chunk's episode "
            "summaries so episode numbering and the training curve continue "
            "across the chain instead of restarting at 1."
        ),
    )
    parser.add_argument(
        "--max-train-hours",
        type=float,
        default=None,
        help=(
            "Stop training after this many hours of wall clock and save "
            "normally. Safety net for chunked runs: set it a little under the "
            "queue's wall limit (e.g. 22.5 for a 24h job) so a chunk whose "
            "timestep/simulation budget turns out too large still exits "
            "cleanly instead of being killed by SLURM."
        ),
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=DEFAULT_DECISION_INTERVAL_MINUTES,
        help="Simulation minutes between agent decisions (default: 10).",
    )
    parser.add_argument(
        "--tactic-distribution",
        choices=SUPPORTED_TACTIC_DISTRIBUTIONS,
        default=TACTIC_DISTRIBUTION_INDIVIDUAL,
        help=(
            "How PPO tactic choices are assigned. 'individual' uses one "
            "decision per controlled aircraft; 'group' uses one decision per "
            "--aircraft-group-size sequential aircraft."
        ),
    )
    parser.add_argument(
        "--aircraft-group-size",
        type=int,
        default=AIRCRAFT_GROUP_SIZE,
        help=(
            "Number of sequential aircraft sharing one tactic decision when "
            "--tactic-distribution group is used. Default 2."
        ),
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "Explicit per-decision group layout over the controlled aircraft "
            "in scenario order; overrides --tactic-distribution/"
            "--aircraft-group-size. E.g. '--group-sizes 1 1 1 1 4' on a "
            "4-seaplane + 4-eVTOL fleet gives each seaplane its own tactic "
            "decision and all four eVTOLs one shared decision. Values must "
            "sum to --controlled-agent-count."
        ),
    )
    parser.add_argument(
        "--controlled-agent-count",
        type=int,
        default=CONTROLLED_AGENT_COUNT,
        help=(
            "Number of aircraft the policy controls (firefighters[:N]). "
            f"Default {CONTROLLED_AGENT_COUNT}. The scenario must define at "
            "least this many aircraft. With --tactic-distribution group the "
            "group count is ceil(controlled_agent_count / aircraft_group_size)."
        ),
    )
    parser.add_argument(
        "--water-set",
        choices=["1", "2", "off"],
        default="off",
        help=(
            "Select a pre-generated water-source subset (see "
            "examples/wildfire/generate_water_sets.py). "
            "'1' = bodies the current fleet can scoop (smaller set); "
            "'2' = bodies example_aircraft_1 (6x6) can scoop (larger set); "
            "'off' (default) = the full {namespace}_water_sources.pkl. "
            "Resolves to {namespace}_water_sources_set{N}.pkl, which must exist."
        ),
    )
    parser.add_argument(
        "--cell-size",
        choices=SUPPORTED_CELL_SIZE_SOURCES,
        default=CELL_SIZE_SOURCE_JSON,
        help=(
            "Source of the metres/cell used by directional-state geometry. "
            "'json' (default) = the scenario's nominal cell_size (~5 m), the "
            "legacy behavior; 'code' = the actual projected metres/cell from "
            "ACTUAL_CELL_SIZE_M (~6-6.8 m), matching the sim's index<->position "
            "grid so projected landing distances are not skewed. Only affects "
            "the 'directional' state space."
        ),
    )
    parser.add_argument(
        "--adaptive-step-size-factor",
        type=float,
        default=None,
        help=(
            "Override the adaptive fire-model timestep factor. Larger values "
            "allow bigger internal fire-model steps and can speed simulation, "
            "but may change numerical fire propagation. Only affects scenarios "
            "with enable_adaptive_time_step=true. Default uses the scenario "
            "value, normally 0.125."
        ),
    )
    adaptive_time_step_group = parser.add_mutually_exclusive_group()
    adaptive_time_step_group.add_argument(
        "--enable-adaptive-time-step",
        action="store_true",
        help=(
            "Override the scenario and enable adaptive internal fire-model "
            "timesteps."
        ),
    )
    adaptive_time_step_group.add_argument(
        "--disable-adaptive-time-step",
        action="store_true",
        help=(
            "Override the scenario and disable adaptive internal fire-model "
            "timesteps."
        ),
    )
    parser.add_argument(
        "--state-fire-fronts",
        type=int,
        default=DEFAULT_STATE_FIRE_FRONTS,
        help=(
            "Maximum number of fastest burning cells to inspect for "
            "front-derived state features such as topography_flag, "
            "vegetation_flag, indirect_flag, urban_flag, and water_flag."
        ),
    )
    parser.add_argument(
        "--state-space",
        choices=SUPPORTED_STATE_SPACES,
        default=STATE_SPACE_OLD,
        help=(
            "Observation/state-space definition to use. 'old' matches "
            "ppo_runner.py's original state vector (36 features: includes "
            "fire->water distance, no per-aircraft altitude); 'updated' is the "
            "altitude/front-flag variant (43 features: adds per-aircraft "
            "altitude and aggregate front flags, drops fire->water distance); "
            "'directional' is 'old' plus the per-flank projected-threat block "
            "(K = --state-fire-fronts sectors x 6 channels); 'directional-2' is "
            "'directional' with the four map-boundary distances removed (each "
            "duplicated a fire-extreme coordinate) and the episode's scaled "
            "ignition point added instead."
        ),
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
        "--ent-coef",
        type=float,
        default=0.0,
        help=(
            "PPO entropy coefficient. Use a small positive value such as "
            "0.001 to encourage exploration when the policy collapses early."
        ),
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
    parser.add_argument(
        "--missions",
        nargs="+",
        metavar="SCENARIO",
        default=None,
        help=(
            "Train one policy across several missions in sequence, splitting "
            "the training budget between them (JSON names in inputs/ or "
            "absolute paths, e.g. --missions Palisades6sp6ev.json "
            "Pyrenees6sp6ev.json). The missions run as phases in the order "
            "given: the whole fleet trains the first mission until its share of "
            "the budget is spent, then moves to the next. Every new simulation "
            "checks how many the fleet has already started on each mission, so "
            "a phase ends within a few simulations of its share. The unit of "
            "the split follows the budget flag: --simulations / "
            "--chain-total-simulations divide the episode count, --timesteps / "
            "--chain-total-timesteps the timesteps. Replaces "
            "--switch-scenario/--switch-scenarios, which round-robins whole "
            "workers and mixes the missions throughout."
        ),
    )
    parser.add_argument(
        "--mission-weights",
        nargs="+",
        type=float,
        metavar="W",
        default=None,
        help=(
            "Relative shares of the budget per --missions entry, in the same "
            "order (default: equal shares). Values are normalized, so "
            "'--mission-weights 2 1' gives the first mission's phase two thirds "
            "of the budget."
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
            # argparse runs help strings through %-formatting, so a literal
            # '%' here raises TypeError when --help is printed.
            f"edges sit {IGNITION_V2_MARGIN_RATIO * 100:.0f} percent of the "
            "map edge length "
            "from each map boundary (excludes water/urban)."
        ),
    )
    ignition_group.add_argument(
        "--switch-ignition-3",
        action="store_true",
        help=(
            "Randomize ignition each episode within a "
            f"{2*IGNITION_BOX_HALF_SIZE_V3}x{2*IGNITION_BOX_HALF_SIZE_V3} cell "
            "box (2.5x2.5 km ground) around the scenario's original ignition "
            "center. Same geometry as --switch-ignition-1, widened to sit "
            "between it and --switch-ignition-2 (excludes water/urban)."
        ),
    )
    ignition_group.add_argument(
        "--switch-ignition-4",
        action="store_true",
        help=(
            "Randomize ignition each episode within a "
            f"{2*IGNITION_BOX_HALF_SIZE_V4}x{2*IGNITION_BOX_HALF_SIZE_V4} cell "
            "box (2.0x2.0 km ground) around the scenario's original ignition "
            f"center, with a relaxed {IGNITION_URBAN_BUFFER_M_V4:.0f} m urban "
            f"keep-out instead of the {IGNITION_URBAN_BUFFER_M:.0f} m used by "
            "modes 1/2/3, so fires may start nearer the urban interface."
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
    #thread
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel wildfire environments to run (>=1).",
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
            "Exponent of the LR schedule lr(p)=initial*p^exponent, where p is "
            "the fraction of the run remaining. 0.70 (default) stays above a "
            "linear schedule throughout and collapses only over the last "
            "percent; 1.0 is linear; 0.0 disables decay. Lower the exponent "
            "(e.g. 0.5) for long runs where you want the actor to keep moving "
            "late, raise it towards 1.0 to settle sooner."
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

    if args.switch_ignition_4:
        switch_ignition_mode = 4
    elif args.switch_ignition_3:
        switch_ignition_mode = 3
    elif args.switch_ignition_2:
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
    elif switch_ignition_mode == 3:
        print(
            "--switch-ignition-3 enabled: ignition randomized each episode "
            f"in a {2*IGNITION_BOX_HALF_SIZE_V3}x{2*IGNITION_BOX_HALF_SIZE_V3} "
            "cell box (2.5x2.5 km ground) around the scenario's original "
            "ignition."
        )
    elif switch_ignition_mode == 4:
        print(
            "--switch-ignition-4 enabled: ignition randomized each episode "
            f"in a {2*IGNITION_BOX_HALF_SIZE_V4}x{2*IGNITION_BOX_HALF_SIZE_V4} "
            "cell box (2.0x2.0 km ground) around the scenario's original "
            f"ignition, urban keep-out {IGNITION_URBAN_BUFFER_M_V4:.0f} m."
        )
    if switch_ignition_mode != 0 and ignition_switch_config["inputs"]:
        print(
            "--ignition-inputs are currently parsed but ignored by "
            "the automatic ignition sampler."
        )
    if args.enable_adaptive_time_step:
        adaptive_time_step_override = True
    elif args.disable_adaptive_time_step:
        adaptive_time_step_override = False
    else:
        adaptive_time_step_override = None

    if args.missions and (args.switch_scenario or args.switch_scenarios):
        raise ValueError(
            "--missions and --switch-scenario/--switch-scenarios both select "
            "the scenario set; pass only one. --missions splits the training "
            "budget between the maps, --switch-scenarios round-robins workers."
        )
    if args.mission_weights and not args.missions:
        raise ValueError("--mission-weights requires --missions.")

    mission_paths: tuple[Path, ...] = ()
    mission_weights: tuple[float, ...] = ()
    if args.missions:
        mission_paths = tuple(_resolve_scenario(name) for name in args.missions)
        # Budgets, episode counts and the env-side weighting are all keyed by
        # scenario file name, so a repeated mission would share one share of
        # the budget instead of getting its own.
        duplicate_names = {
            path.name
            for path in mission_paths
            if [p.name for p in mission_paths].count(path.name) > 1
        }
        if duplicate_names:
            raise ValueError(
                "--missions must list distinct scenarios (repeated: "
                + ", ".join(sorted(duplicate_names))
                + ")."
            )
        if args.mission_weights is None:
            mission_weights = tuple(1.0 for _ in mission_paths)
        else:
            if len(args.mission_weights) != len(mission_paths):
                raise ValueError(
                    f"--mission-weights takes one value per mission: got "
                    f"{len(args.mission_weights)} weights for "
                    f"{len(mission_paths)} missions."
                )
            if any(weight <= 0.0 for weight in args.mission_weights):
                raise ValueError("--mission-weights values must all be > 0.")
            mission_weights = tuple(float(w) for w in args.mission_weights)

    use_switch_scenario = bool(
        args.switch_scenario or args.switch_scenarios or mission_paths
    )
    if mission_paths:
        switch_scenario_paths = mission_paths
        aircraft_source_scenario_path = None
        scenario_path = mission_paths[0]
        print(
            "--missions enabled: ignoring --scenario and training in phases: "
            + " -> ".join(path.name for path in mission_paths)
            + ". The fleet spends one mission's budget share before moving to "
            "the next, using each scenario's fleet definition."
        )
    elif use_switch_scenario:
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
    if args.decision_interval_minutes <= 0:
        raise ValueError("--decision-interval-minutes must be > 0")
    if args.aircraft_group_size <= 0:
        raise ValueError("--aircraft-group-size must be > 0")
    if args.controlled_agent_count <= 0:
        raise ValueError("--controlled-agent-count must be > 0")
    # The policy commands firefighters[:controlled_agent_count]; the scenario
    # fleet must be large enough or we would silently control fewer aircraft.
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
    # One policy drives every mission, so a fleet that differs between them
    # silently re-points the same action at different aircraft. Warned about
    # rather than rejected: the fleet sizes are already checked above, and a
    # deliberate mismatch is the user's call.
    if mission_paths:
        _compositions = {
            path: _scenario_fleet_composition(path) for path in mission_paths
        }
        if len(set(_compositions.values())) > 1:
            print(
                "WARNING: --missions fleets differ ("
                + "; ".join(
                    f"{path.name}: "
                    + ", ".join(f"{name}x{count}" for name, count in composition)
                    for path, composition in _compositions.items()
                )
                + "). --controlled-agent-count/--group-sizes slice each "
                "scenario's aircraft list positionally, so the same action "
                "commands different aircraft on different missions."
            )
    # Normalize --water-set ("off" -> None, "1"/"2" -> int) and verify the
    # corresponding pre-generated pkl exists for every scenario in use.
    args.water_set = None if args.water_set == "off" else int(args.water_set)
    if args.water_set is not None:
        for _fleet_path in _fleet_paths:
            _ns = json.loads(Path(_fleet_path).read_text())["terrain_inputs"][
                "file_namespace"
            ]
            _wf = TERRAIN_DIR / f"{_ns}_water_sources_set{args.water_set}.pkl"
            if not _wf.exists():
                raise FileNotFoundError(
                    f"--water-set {args.water_set} requires {_wf}, which is "
                    "missing. Generate it first:\n  python -m "
                    "examples.wildfire.generate_water_sets --scenario "
                    f"\"{Path(_fleet_path).name}\""
                )
    if (
        args.adaptive_step_size_factor is not None
        and args.adaptive_step_size_factor <= 0.0
    ):
        raise ValueError("--adaptive-step-size-factor must be > 0")
    if args.state_fire_fronts <= 0:
        raise ValueError("--state-fire-fronts must be > 0")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be >= 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.ent_coef < 0.0:
        raise ValueError("--ent-coef must be >= 0")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in the interval (0, 1]")
    if (
        args.fire_detection_delay_minutes is not None
        and args.fire_detection_delay_minutes < 0
    ):
        raise ValueError("--fire-detection-delay-minutes must be >= 0")
    if args.progress_file_tag:
        progress_file_tag = args.progress_file_tag.strip().replace(" ", "_")
    elif mission_paths:
        progress_file_tag = "missions_" + "_".join(
            path.stem.lower().replace(" ", "_") for path in mission_paths
        )
    elif use_switch_scenario:
        progress_file_tag = "switch_scenarios"
    else:
        scenario_tag = scenario_path.stem.lower().replace(" ", "_")
        progress_file_tag = f"single_{scenario_tag}"
    vec_start_method = None if args.vec_start_method == "auto" else args.vec_start_method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training length is either a timestep budget (default) or a simulation
    # (episode) count. For --simulations the timestep budget becomes a derived
    # ceiling: simulations x the longest possible episode. Episodes that end
    # early simply consume less of it, and StopTrainingOnTotalEpisodes ends
    # training on the exact episode count before the ceiling is reached.
    episode_steps = _estimate_episode_steps(
        switch_scenario_paths if use_switch_scenario and switch_scenario_paths
        else (scenario_path,),
        args.decision_interval_minutes,
    )
    if args.simulations is not None:
        if args.simulations <= 0:
            raise ValueError(
                f"--simulations must be > 0 (got {args.simulations})"
            )
        default_timesteps = args.simulations * episode_steps
        print(
            f"--simulations {args.simulations}: up to {episode_steps} steps per "
            f"episode -> timestep ceiling {default_timesteps}. Training stops "
            "at the episode count, whichever comes first."
        )
    else:
        default_timesteps = args.timesteps if args.timesteps is not None else 1680
    total_timesteps = max(default_timesteps, args.num_envs, 2)

    # Chained runs: --timesteps/--simulations size this chunk, the --chain-*
    # total sizes the whole multi-job run. The chain total is what gets handed
    # to model.learn(), so SB3's progress_remaining (and the LR schedule that
    # reads it) spans the chain; the chunk budget is enforced by callbacks.
    if args.max_train_hours is not None and args.max_train_hours <= 0.0:
        raise ValueError("--max-train-hours must be > 0")
    chain_total_timesteps: int | None = None
    if args.chain_total_timesteps is not None:
        if args.chain_total_timesteps <= 0:
            raise ValueError(
                f"--chain-total-timesteps must be > 0 "
                f"(got {args.chain_total_timesteps})"
            )
        chain_total_timesteps = args.chain_total_timesteps
        print(f"Chain budget: {chain_total_timesteps} total timesteps.")
    elif args.chain_total_simulations is not None:
        if args.chain_total_simulations <= 0:
            raise ValueError(
                f"--chain-total-simulations must be > 0 "
                f"(got {args.chain_total_simulations})"
            )
        chain_total_timesteps = args.chain_total_simulations * episode_steps
        print(
            f"Chain budget: {args.chain_total_simulations} total simulations "
            f"-> timestep ceiling {chain_total_timesteps} "
            f"({episode_steps} steps per episode)."
        )
    if chain_total_timesteps is not None and not (
        args.timesteps is not None
        or args.simulations is not None
        or args.max_train_hours is not None
    ):
        raise ValueError(
            "Chained runs need a per-chunk bound: pass --timesteps or "
            "--simulations to size the chunk, or --max-train-hours to let it "
            "fill the queue's wall limit. Without one the chunk would train "
            "to the chain total and be killed by SLURM."
        )
    if args.max_train_hours is not None:
        print(f"Wall-clock training cutoff: {args.max_train_hours}h.")

    # Episode history carried over from the previous chunk. Seeds the logger so
    # progress files stay full-history and simulation indices keep counting.
    resume_summaries: list[dict[str, Any]] = []
    if args.resume_summary:
        if args.resume_summary.strip().lower() == "auto":
            resume_summary_path = _latest_summary_path(output_dir, progress_file_tag)
            if resume_summary_path is None:
                print(
                    "--resume-summary auto: no previous "
                    f"results_{progress_file_tag}_<N>_summary.csv under "
                    f"{output_dir}; episode numbering starts at 1."
                )
        else:
            resume_summary_path = Path(args.resume_summary)
            if not resume_summary_path.exists():
                raise FileNotFoundError(
                    f"--resume-summary path not found: {resume_summary_path}"
                )
        if resume_summary_path is not None:
            resume_summaries = _read_summary_rows(resume_summary_path)
            print(
                f"Resuming episode history from {resume_summary_path}: "
                f"{len(resume_summaries)} episodes already completed."
            )

    # Episodes completed by earlier chunks. The sidecar is written with the
    # weights so it cannot lag them; the CSV row count is the fallback for runs
    # that predate it or that were adopted from a non-chained job.
    elapsed_episodes = len(resume_summaries)
    if args.load_model:
        saved_progress = _read_chain_progress(output_dir)
        if saved_progress is not None:
            sidecar_episodes = int(saved_progress.get("simulations") or 0)
            if sidecar_episodes != elapsed_episodes:
                print(
                    f"Episode count: {sidecar_episodes} from "
                    f"{CHAIN_PROGRESS_FILE} (saved with the weights) vs "
                    f"{elapsed_episodes} summary rows; using the sidecar."
                )
            elapsed_episodes = sidecar_episodes

    # A chunk that has nothing left to do exits before building environments so
    # the driver can stop the chain cheaply.
    chain_complete_marker = output_dir / "CHAIN_COMPLETE"
    if chain_total_timesteps is not None:
        elapsed_timesteps = 0
        if args.load_model:
            elapsed_timesteps = _saved_num_timesteps(Path(args.load_model))
            saved_progress = _read_chain_progress(output_dir)
            sidecar_timesteps = int((saved_progress or {}).get("timesteps") or 0)
            if saved_progress is not None and sidecar_timesteps != elapsed_timesteps:
                # The two are written together, so a mismatch means the model
                # was swapped — usually a checkpoint restored by hand. The
                # zip's own counter matches the weights, so it wins.
                print(
                    f"Timestep count: {elapsed_timesteps} in the model zip vs "
                    f"{sidecar_timesteps} in {CHAIN_PROGRESS_FILE}; using the "
                    "model's own counter (it matches the weights)."
                )
        if args.chain_elapsed_timesteps is not None:
            if args.chain_elapsed_timesteps < 0:
                raise ValueError("--chain-elapsed-timesteps must be >= 0")
            print(
                f"Chain position overridden: {elapsed_timesteps} -> "
                f"{args.chain_elapsed_timesteps} timesteps "
                "(--chain-elapsed-timesteps)."
            )
            elapsed_timesteps = args.chain_elapsed_timesteps
        elif (
            args.load_model
            and elapsed_timesteps == 0
            and args.chain_total_simulations is None
        ):
            # Same failure as losing the episode count in a simulation chain:
            # the decay would silently restart at the initial LR.
            raise ValueError(
                f"Resuming a --chain-total-timesteps chain from "
                f"{args.load_model}, but its stored timestep counter is 0, so "
                f"the LR decay would restart at {args.learning_rate}. This "
                "happens with policies continued before chaining existed. "
                "Pass --chain-elapsed-timesteps with the run's true cumulative "
                "timesteps, or delete the saved model to start the chain over."
            )
        if args.chain_total_simulations is not None:
            # Simulation-budgeted chains stop on the episode count; the
            # timestep total is only the ceiling handed to learn().
            chain_done = elapsed_episodes >= args.chain_total_simulations
        else:
            chain_done = elapsed_timesteps >= chain_total_timesteps
        if chain_done:
            print(
                "CHAIN COMPLETE: "
                f"{elapsed_timesteps}/{chain_total_timesteps} timesteps and "
                f"{elapsed_episodes}"
                + (
                    f"/{args.chain_total_simulations}"
                    if args.chain_total_simulations is not None
                    else ""
                )
                + " simulations already done; nothing left to train."
            )
            chain_complete_marker.write_text(
                f"timesteps={elapsed_timesteps}\n"
                f"simulations={elapsed_episodes}\n"
            )
            return

    # In a simulation-budgeted chain the episode count IS the LR state: losing
    # it would silently restart the decay at the initial LR.
    episode_progress: EpisodeProgress | None = None
    if args.chain_total_simulations is not None:
        if args.load_model and elapsed_episodes == 0:
            raise ValueError(
                "Resuming a --chain-total-simulations chain but no completed "
                f"episodes were found: neither {output_dir / CHAIN_PROGRESS_FILE} "
                f"nor a summary CSV for tag '{progress_file_tag}' under "
                f"{output_dir}. Continuing would restart the LR decay at "
                f"{args.learning_rate}. Point --resume-summary at the previous "
                "chunk's summary, or delete the saved model to restart the "
                "chain from scratch."
            )
        episode_progress = EpisodeProgress(
            completed=elapsed_episodes,
            total=args.chain_total_simulations,
        )
        resumed_progress = episode_progress.progress_remaining()
        resumed_lr = args.learning_rate * resumed_progress**args.lr_decay_exponent
        # An exponent of 0 makes p^0 = 1 for every p, so the LR never moves.
        # Saying "-> 0" there would describe a decay the run does not perform.
        if args.lr_decay_exponent == 0.0:
            print(
                "LR is constant over simulations: "
                f"{elapsed_episodes}/{args.chain_total_simulations} done, "
                f"progress_remaining {resumed_progress:.3f}, "
                f"LR held at {resumed_lr:.3e} for all "
                f"{args.chain_total_simulations} simulations "
                "(--lr-decay-exponent 0)."
            )
        else:
            print(
                "LR decays over simulations: "
                f"{elapsed_episodes}/{args.chain_total_simulations} done, "
                f"progress_remaining {resumed_progress:.3f}, "
                f"LR {resumed_lr:.3e} -> 0 at "
                f"{args.chain_total_simulations} simulations "
                f"(exponent {args.lr_decay_exponent})."
            )

    # --missions: turn the run's budget into a per-mission share. Simulation
    # budgets split the episode count, timestep budgets split the timesteps;
    # with --chain-total-* the shares span the whole chain and the episodes
    # earlier chunks already spent are read back from the resumed summary.
    mission_budgets: dict[str, float] | None = None
    mission_budget_mode = MISSION_BUDGET_SIMULATIONS
    mission_quotas: dict[str, int] = {}
    resumed_mission_counts: dict[str, int] = {}
    resumed_mission_timesteps: dict[str, int] = {}
    if mission_paths:
        mission_names = tuple(path.name for path in mission_paths)
        resumed_mission_counts = _episode_counts_by_scenario(
            resume_summaries, mission_names
        )
        resumed_mission_timesteps = _timestep_counts_by_scenario(
            resume_summaries, mission_names
        )
        simulation_budget = (
            args.chain_total_simulations
            if args.chain_total_simulations is not None
            else args.simulations
        )
        if simulation_budget is not None:
            mission_budget_mode = MISSION_BUDGET_SIMULATIONS
            mission_quotas = dict(
                zip(
                    mission_names,
                    _split_budget(simulation_budget, mission_weights),
                    strict=True,
                )
            )
            mission_budgets = {
                name: float(quota) for name, quota in mission_quotas.items()
            }
            budget_label = f"{simulation_budget} simulations"
            unit = "simulations"
        else:
            mission_budget_mode = MISSION_BUDGET_TIMESTEPS
            timestep_budget = (
                chain_total_timesteps
                if chain_total_timesteps is not None
                else total_timesteps
            )
            mission_budgets = {
                name: float(share)
                for name, share in zip(
                    mission_names,
                    _split_budget(timestep_budget, mission_weights),
                    strict=True,
                )
            }
            budget_label = f"{timestep_budget} timesteps"
            unit = "timesteps"
        print(
            f"Mission phase budgets ({budget_label}, weights "
            + "/".join(f"{weight:g}" for weight in mission_weights)
            + "), run in this order: "
            + " then ".join(
                f"{name} -> {mission_budgets[name]:.0f} {unit}"
                + (
                    f" ({resumed_mission_counts[name]} already run)"
                    if resumed_mission_counts.get(name)
                    else ""
                )
                for name in mission_names
            )
        )
        if mission_budget_mode == MISSION_BUDGET_TIMESTEPS:
            print(
                "Note: no --simulations/--chain-total-simulations budget was "
                "given, so the split is by timesteps. Missions with shorter "
                "episodes will run more simulations for the same share."
            )

    # --missions brings its own per-mission budget, set above; what follows is
    # the --switch-scenario path, which balances timesteps only.
    ts_budget_per_scenario: float | None = None
    if not mission_paths and use_switch_scenario and switch_scenario_paths:
        if args.num_envs == 1:
            ts_budget_per_scenario = float(total_timesteps) / len(switch_scenario_paths)
            print(
                f"Scenario balance budget: {total_timesteps} total timesteps / "
                f"{len(switch_scenario_paths)} scenarios = "
                f"{ts_budget_per_scenario:.0f} timesteps per scenario"
            )
        else:
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
            mission_budgets=mission_budgets,
            mission_budget_mode=mission_budget_mode,
            initial_scenario_episode_counts=resumed_mission_counts,
            initial_scenario_timestep_counts=resumed_mission_timesteps,
            gc_collect_on_reset=args.gc_collect_on_reset,
            include_scenario_features=use_switch_scenario,
            state_fire_fronts=args.state_fire_fronts,
            state_space=args.state_space,
            tactic_distribution=args.tactic_distribution,
            aircraft_group_size=args.aircraft_group_size,
            group_sizes=args.group_sizes,
            controlled_agent_count=args.controlled_agent_count,
            water_set=args.water_set,
            cell_size_source=args.cell_size,
            enable_adaptive_time_step=adaptive_time_step_override,
            adaptive_step_size_factor=args.adaptive_step_size_factor,
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
            mission_budgets=mission_budgets,
            mission_budget_mode=mission_budget_mode,
            initial_scenario_episode_counts=resumed_mission_counts,
            initial_scenario_timestep_counts=resumed_mission_timesteps,
            gc_collect_on_reset=args.gc_collect_on_reset,
            state_fire_fronts=args.state_fire_fronts,
            state_space=args.state_space,
            tactic_distribution=args.tactic_distribution,
            aircraft_group_size=args.aircraft_group_size,
            group_sizes=args.group_sizes,
            controlled_agent_count=args.controlled_agent_count,
            water_set=args.water_set,
            cell_size_source=args.cell_size,
            enable_adaptive_time_step=adaptive_time_step_override,
            adaptive_step_size_factor=args.adaptive_step_size_factor,
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
    print(
        "State fire fronts inspected for front-derived flags "
        f"(topography/vegetation/indirect/urban/water): {args.state_fire_fronts}"
    )
    if args.fire_detection_delay_minutes is None:
        if use_switch_scenario:
            if mission_paths or args.num_envs == 1:
                print(
                    "Decision start delay: scenario response_time / 60 "
                    "(computed per selected scenario each episode)."
                )
            else:
                print(
                    "Decision start delay: scenario response_time / 60 "
                    "(computed per worker-assigned scenario)."
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
    if adaptive_time_step_override is None:
        print("Adaptive fire timestep: scenario value.")
    else:
        state = "enabled" if adaptive_time_step_override else "disabled"
        print(f"Adaptive fire timestep override: {state}.")
    if args.adaptive_step_size_factor is None:
        print(
            "Adaptive fire step-size factor: scenario/default value "
            "(active only when enable_adaptive_time_step=true)."
        )
    else:
        print(
            "Adaptive fire step-size factor override: "
            f"{args.adaptive_step_size_factor} "
            "(active only when enable_adaptive_time_step=true)."
        )
    print(f"Allowed tactic combinations: {len(TACTIC_COMBINATIONS)}")
    if args.gc_collect_on_reset:
        print("Diagnostic: gc.collect() and prior-sim drop at reset start enabled.")
    print(
        f"State space: {args.state_space} "
        f"({len(train_env.observation_space.low)} observation features)"
    )
    fleet_desc = "; ".join(
        f"{_fleet_path.name}={_scenario_agent_count(_fleet_path)}"
        for _fleet_path in _fleet_paths
    )
    print(
        f"Fleet size (scenario agents): {fleet_desc}; "
        f"policy-controlled: {args.controlled_agent_count}"
    )
    if args.group_sizes:
        # --group-sizes overrides tactic_distribution/aircraft_group_size, so
        # don't print those: they would misreport how tactics are assigned.
        print(
            f"Tactic distribution: explicit group_sizes={args.group_sizes}; "
            f"controlled_aircraft={args.controlled_agent_count}; "
            f"action_decisions={len(train_env.action_space.nvec)}"
        )
    else:
        print(
            f"Tactic distribution: {args.tactic_distribution}; "
            f"controlled_aircraft={args.controlled_agent_count}; "
            f"aircraft_group_size={args.aircraft_group_size}; "
            f"action_decisions={len(train_env.action_space.nvec)}"
        )
    if args.num_envs > 1:
        effective_vec_method = (
            vec_start_method
            if vec_start_method is not None
            else ("fork" if sys.platform.startswith("linux") else "spawn")
        )
        print(f"Vector env start method: {effective_vec_method}")

    step_logger = TrainingLogger(
        decision_interval_minutes=args.decision_interval_minutes,
        tactic_distribution=args.tactic_distribution,
        aircraft_group_size=args.aircraft_group_size,
        group_sizes=args.group_sizes,
        controlled_agent_count=args.controlled_agent_count,
        progress_file_tag=progress_file_tag,
        output_dir=output_dir,
        resume_summaries=resume_summaries,
        episode_progress=episode_progress,
        lr_episode_offset=elapsed_episodes,
        summary_retention=args.summary_retention,
        mission_names=tuple(path.name for path in mission_paths),
        mission_budget_mode=mission_budget_mode,
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
        model.learning_rate = (
            _make_episode_lr_schedule(
                args.learning_rate,
                args.lr_decay_exponent,
                episode_progress,
            )
            if episode_progress is not None
            else _make_lr_schedule(args.learning_rate, args.lr_decay_exponent)
        )
        model.n_epochs = args.n_epochs
        model.target_kl = args.target_kl
        model.gamma = args.gamma
        model.ent_coef = args.ent_coef
        model._setup_lr_schedule()
        # SB3 2.x exposes no public buffer-rebuild hook (_setup_model would
        # reinitialize the loaded policy weights). Recreate the rollout buffer
        # directly so the updated n_steps/gamma/gae_lambda take effect while the
        # loaded policy is preserved.
        from stable_baselines3.common.buffers import (
            DictRolloutBuffer,
            RolloutBuffer,
        )

        buffer_cls = (
            DictRolloutBuffer
            if isinstance(model.observation_space, spaces.Dict)
            else RolloutBuffer
        )
        model.rollout_buffer = buffer_cls(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
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
            learning_rate=(
                _make_episode_lr_schedule(
                    args.learning_rate,
                    args.lr_decay_exponent,
                    episode_progress,
                )
                if episode_progress is not None
                else _make_lr_schedule(args.learning_rate, args.lr_decay_exponent)
            ),
            verbose=1,
            tensorboard_log=None,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            device=device,
            policy_kwargs=policy_kwargs,
        )
        print(
            f"PPO hyperparams: lr={args.learning_rate} (decay^{args.lr_decay_exponent}), "
            f"n_epochs={args.n_epochs}, gamma={args.gamma}, "
            f"ent_coef={args.ent_coef}, target_kl={args.target_kl}"
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
        active_callbacks = [step_logger, ckpt_cb]
        print(
            f"Checkpointing every {args.checkpoint_interval} timesteps "
            f"({save_freq_per_env}/env) to {checkpoint_dir}"
        )
    else:
        active_callbacks = [step_logger]

    # Episode budget for this chunk: the smaller of the per-chunk --simulations
    # and whatever the chain has left after the resumed episodes.
    chunk_episode_budget: int | None = args.simulations
    if args.chain_total_simulations is not None:
        chain_episodes_left = args.chain_total_simulations - elapsed_episodes
        chunk_episode_budget = (
            chain_episodes_left
            if chunk_episode_budget is None
            else min(chunk_episode_budget, chain_episodes_left)
        )
    if chunk_episode_budget is not None:
        active_callbacks.append(StopTrainingOnTotalEpisodes(chunk_episode_budget))
    # With simulation-budgeted phases the run ends when every mission has
    # completed its share, which is a step or two after the last phase's
    # simulations were started. The total-episode stop above still caps the
    # chunk, so a chunked run stops at whichever comes first.
    if mission_quotas:
        active_callbacks.append(
            StopTrainingOnMissionSimulations(
                mission_quotas,
                completed=resumed_mission_counts,
            )
        )

    # In chain mode learn() gets the chain's remaining budget so the LR
    # schedule decays across the whole chain; the chunk is bounded here.
    learn_timesteps = total_timesteps
    reset_num_timesteps = True
    if chain_total_timesteps is not None:
        if args.load_model and args.chain_elapsed_timesteps is not None:
            # SB3 derives progress_remaining from num_timesteps, so the
            # override has to move the counter itself, not just our arithmetic.
            model.num_timesteps = int(args.chain_elapsed_timesteps)
        elapsed_timesteps = int(model.num_timesteps) if args.load_model else 0
        learn_timesteps = max(1, chain_total_timesteps - elapsed_timesteps)
        reset_num_timesteps = not args.load_model
        if args.timesteps is not None:
            active_callbacks.append(
                StopTrainingOnChunkTimesteps(min(total_timesteps, learn_timesteps))
            )
        if args.timesteps is not None:
            chunk_cap = f"{min(total_timesteps, learn_timesteps)} timesteps"
        elif chunk_episode_budget is not None:
            chunk_cap = f"{chunk_episode_budget} simulations"
        else:
            chunk_cap = f"{args.max_train_hours}h wall clock"
        if episode_progress is not None:
            # LR is driven by simulations here; it was reported above.
            lr_note = (
                f"LR follows simulations ({elapsed_episodes}/"
                f"{args.chain_total_simulations})."
            )
        else:
            chunk_progress = 1.0 - elapsed_timesteps / float(chain_total_timesteps)
            lr_note = (
                "LR resumes at "
                f"{args.learning_rate * chunk_progress ** args.lr_decay_exponent:.3g} "
                f"(progress_remaining {chunk_progress:.3f})."
            )
        print(
            f"Chain position: {elapsed_timesteps}/{chain_total_timesteps} "
            f"timesteps done ({elapsed_episodes} simulations); "
            f"{learn_timesteps} left in the chain, this chunk capped at "
            f"{chunk_cap}. {lr_note}"
        )

    if args.max_train_hours is not None:
        active_callbacks.append(StopTrainingOnWallClock(args.max_train_hours))

    callbacks = (
        active_callbacks[0]
        if len(active_callbacks) == 1
        else CallbackList(active_callbacks)
    )

    # SLURM sends SIGTERM before SIGKILL at the wall limit. Python's default
    # SIGTERM disposition kills the process outright, which would skip the
    # rescue save below; turning it into SystemExit routes it there instead.
    def _terminate(signum, _frame):  # noqa: ANN001 - signal handler signature
        raise SystemExit(f"received signal {signum}")

    for _sig in (signal.SIGTERM, signal.SIGUSR1):
        try:
            signal.signal(_sig, _terminate)
        except (ValueError, OSError) as err:  # noqa: PERF203 - startup only
            print(f"Could not install handler for {_sig}: {err}")

    start_time = time.perf_counter()
    try:
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
    except (KeyboardInterrupt, SystemExit) as err:
        # Best-effort save on signal-driven termination (SLURM time-limit).
        print(f"\nTraining interrupted ({type(err).__name__}); saving rescue model.")
        try:
            rescue_path = output_dir / "rescue_on_interrupt.zip"
            model.save(str(rescue_path))
            print(f"Rescue model saved to {rescue_path}")
            if args.save_model:
                # Advance the chunk's own output too, so a chained run can
                # resume from the interrupted chunk instead of replaying it.
                rescue_save_path = _resolve_output_path(
                    output_dir,
                    args.save_model,
                    "trained_policy.zip",
                )
                rescue_save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(rescue_save_path))
                print(f"Rescue model also saved to {rescue_save_path}")
                if chain_total_timesteps is not None:
                    _write_chain_progress(
                        output_dir,
                        simulations=step_logger.episode_count_for_chain(),
                        timesteps=int(model.num_timesteps),
                    )
        except Exception as save_err:  # noqa: BLE001
            print(f"Rescue save failed: {save_err}")
        try:
            # Summary only: the per-step CSVs can be tens of MB and SLURM
            # follows SIGTERM with SIGKILL after ~30s.
            step_logger._export_progress(
                step_logger.completed_episodes,
                export_steps=False,
                export_decision_steps=False,
                export_summary=True,
            )
        except Exception as flush_err:  # noqa: BLE001
            print(f"Rescue summary flush failed: {flush_err}")
        raise
    training_duration = time.perf_counter() - start_time
    train_env.close()

    print(f"Training complete in {training_duration:.2f} seconds.")

    if mission_paths:
        run_mission_counts = step_logger.scenario_episode_counts()
        total_mission_episodes = sum(run_mission_counts.values()) or 1
        print(
            "Mission phases actually run"
            + (
                " (whole chain, resumed episodes included)"
                if any(resumed_mission_counts.values())
                else ""
            )
            + ": "
            + ", ".join(
                f"{path.name}: {run_mission_counts.get(path.name, 0)} simulations"
                + (
                    f"/{mission_quotas[path.name]}"
                    if path.name in mission_quotas
                    else ""
                )
                + f" ({100.0 * run_mission_counts.get(path.name, 0) / total_mission_episodes:.1f}%)"
                for path in mission_paths
            )
        )

    if args.save_model:
        save_path = _resolve_output_path(
            output_dir,
            args.save_model,
            "trained_policy.zip",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Trained model saved to {save_path}")

    if chain_total_timesteps is not None:
        chain_episodes = step_logger.episode_count_for_chain()
        _write_chain_progress(
            output_dir,
            simulations=chain_episodes,
            timesteps=int(model.num_timesteps),
        )
        if args.chain_total_simulations is not None:
            chain_done = chain_episodes >= args.chain_total_simulations
        else:
            chain_done = int(model.num_timesteps) >= chain_total_timesteps
        if chain_done:
            # Lets the driver stop without submitting a job just to find out.
            chain_complete_marker.write_text(
                f"timesteps={model.num_timesteps}\n"
                f"simulations={chain_episodes}\n"
            )
            print(
                "CHAIN COMPLETE: "
                f"{model.num_timesteps}/{chain_total_timesteps} timesteps, "
                f"{chain_episodes} simulations. "
                f"Marker written to {chain_complete_marker}"
            )
        else:
            print(
                "Chunk finished; chain continues at "
                f"{model.num_timesteps}/{chain_total_timesteps} timesteps "
                f"({chain_episodes} simulations)."
            )

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
