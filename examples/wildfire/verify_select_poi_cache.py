#!/usr/bin/env python3
"""Verify lazy select_poi caching preserves tactic destinations.

The cached selectors compute shared firefront costs once per fire-state version
and only for the tactic that asks for them. This script compares each cached
selector against the old inline formula and checks that unrelated cache keys are
not populated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (  # noqa: E402
    _nearest_position_cost,
    _positions_from_agents,
    _select_topography_destination,
    _select_vegetation_destination,
    _select_vip_destination,
    _select_water_destination,
)
from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    DEFAULT_DECISION_INTERVAL_MINUTES,
    DEFAULT_STATE_FIRE_FRONTS,
    STATE_SPACE_SMALL,
    WildfireHourlyEnv,
    _resolve_scenario,
)


TACTICS = {
    "water": _select_water_destination,
    "vip": _select_vip_destination,
    "vegetation": _select_vegetation_destination,
    "topography": _select_topography_destination,
}

UNRELATED_CACHE_KEYS = {
    "water": {
        "urban_positions",
        "urban_cost",
        "vip_cone_cost",
        "raw_vegetation_priority",
        "raw_topography_priority",
    },
    "vip": {
        "water_positions",
        "water_cost",
        "vip_cone_cost",
        "raw_vegetation_priority",
        "raw_topography_priority",
    },
    "vegetation": {
        "water_positions",
        "water_cost",
        "urban_positions",
        "urban_cost",
        "raw_topography_priority",
    },
    "topography": {
        "water_positions",
        "water_cost",
        "urban_positions",
        "urban_cost",
        "raw_vegetation_priority",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cached select_poi destinations with old formulas."
    )
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON name in inputs/ or an absolute/relative path.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303],
        help="Simulation seeds to test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Decision steps to advance after the reset checks.",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=DEFAULT_DECISION_INTERVAL_MINUTES,
        help="Decision interval used by the test environment.",
    )
    parser.add_argument(
        "--fire-detection-delay-minutes",
        type=float,
        default=0.0,
        help="Use 0.0 to inspect the state immediately after ignition.",
    )
    return parser.parse_args()


def _clear_destinations(agent) -> None:
    for firefighter in agent.model.firefighters:
        firefighter.full_trajectory = None


def _set_tracking_case(agent, case: str) -> None:
    _clear_destinations(agent)
    fire_positions = agent.model.wildfire.fire_positions
    if case == "none" or not fire_positions.size:
        return

    destination_owner = (
        agent.model.firefighters[1]
        if len(agent.model.firefighters) > 1
        else agent
    )
    destination_owner._destination_pos = np.array(fire_positions[0], dtype=float)


def _reference_candidates(agent, include_burning_indices: bool = False):
    fire_positions = agent.model.wildfire.fire_positions
    burning_indices = (
        agent.model.wildfire.burning_indices
        if include_burning_indices
        else None
    )
    if not fire_positions.size:
        return fire_positions, burning_indices

    for obj in agent.model.firefighters:
        if obj.destination is not None:
            untracked_pos = (fire_positions != obj.destination).any(axis=1)
            fire_positions = fire_positions[untracked_pos]
            if include_burning_indices:
                burning_indices = burning_indices[untracked_pos]

            if not np.size(fire_positions):
                fire_positions = agent.model.wildfire.fire_positions
                if include_burning_indices:
                    burning_indices = agent.model.wildfire.burning_indices

    return fire_positions, burning_indices


def _reference_water(agent):
    fire_positions, _ = _reference_candidates(agent)
    if not fire_positions.size:
        return None

    map_diagonal = np.linalg.norm(
        np.array(agent.model.simulation.environment.dimensions)
    )
    fire_distances = agent.distance(agent.pos, fire_positions)
    distance_cost = (map_diagonal - fire_distances) / map_diagonal
    water_cost = _nearest_position_cost(
        fire_positions=fire_positions,
        objective_positions=_positions_from_agents(agent.model.water_sources),
        map_diagonal=map_diagonal,
    )
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * water_cost
    )
    return fire_positions[int(np.argmax(selection_cost)), :]


def _reference_vip(agent):
    fire_positions, _ = _reference_candidates(agent)
    if not fire_positions.size:
        return None

    map_diagonal = np.linalg.norm(
        np.array(agent.model.simulation.environment.dimensions)
    )
    fire_distances = agent.distance(agent.pos, fire_positions)
    distance_cost = (map_diagonal - fire_distances) / map_diagonal
    urban_cost = _nearest_position_cost(
        fire_positions=fire_positions,
        objective_positions=_positions_from_agents(
            agent.model.protection_locations
        ),
        map_diagonal=map_diagonal,
    )
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * urban_cost
    )
    return fire_positions[int(np.argmax(selection_cost)), :]


def _reference_vegetation(agent):
    fire_positions, burning_indices = _reference_candidates(
        agent,
        include_burning_indices=True,
    )
    if not fire_positions.size:
        return None

    map_diagonal = np.linalg.norm(
        np.array(agent.model.simulation.environment.dimensions)
    )
    fire_distances = agent.distance(agent.pos, fire_positions)
    distance_cost = (map_diagonal - fire_distances) / map_diagonal
    vip_cost = agent.exponential_cone_func(
        pos=fire_positions,
        vip=(location.pos for location in agent.model.protection_locations),
        map_diagonal=map_diagonal,
    )
    vegetation_cost = agent.priority_cost_vegetation(burning_indices)
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * vip_cost
        + agent.parameters.vegetation_cost_weight * vegetation_cost
    )
    return fire_positions[int(np.argmax(selection_cost)), :]


def _reference_topography(agent):
    fire_positions, burning_indices = _reference_candidates(
        agent,
        include_burning_indices=True,
    )
    if not fire_positions.size:
        return None

    map_diagonal = np.linalg.norm(
        np.array(agent.model.simulation.environment.dimensions)
    )
    fire_distances = agent.distance(agent.pos, fire_positions)
    distance_cost = (map_diagonal - fire_distances) / map_diagonal
    vip_cost = agent.exponential_cone_func(
        pos=fire_positions,
        vip=(location.pos for location in agent.model.protection_locations),
        map_diagonal=map_diagonal,
    )
    topography_cost = agent.priority_cost_topography(burning_indices)
    selection_cost = (
        agent.parameters.distance_cost_weight * distance_cost
        + agent.parameters.vip_cost_weight * vip_cost
        + agent.parameters.topography_cost_weight * topography_cost
    )
    return fire_positions[int(np.argmax(selection_cost)), :]


REFERENCES = {
    "water": _reference_water,
    "vip": _reference_vip,
    "vegetation": _reference_vegetation,
    "topography": _reference_topography,
}


def _assert_same_destination(label: str, cached, reference) -> None:
    if cached is None or reference is None:
        if cached is not reference:
            raise AssertionError(
                f"{label}: one selector failed and the other did not: "
                f"cached={cached}, reference={reference}"
            )
        return
    if not np.allclose(np.asarray(cached), np.asarray(reference)):
        raise AssertionError(
            f"{label}: destination mismatch: "
            f"cached={cached}, reference={reference}"
        )


def _assert_lazy_cache(label: str, agent, tactic: str) -> None:
    cache = agent.model.__cache__.get("select_poi", {})
    unexpected = sorted(UNRELATED_CACHE_KEYS[tactic] & set(cache))
    if unexpected:
        raise AssertionError(
            f"{label}: {tactic} populated unrelated cache keys {unexpected}"
        )


def _verify_tactics(env: WildfireHourlyEnv, label: str) -> None:
    assert env.sim is not None
    firefighters = env.sim.firefighters.firefighters
    if not firefighters:
        raise AssertionError(f"{label}: scenario has no firefighters")
    agent = firefighters[0]

    for tracking_case in ("none", "one_tracked"):
        for tactic, selector in TACTICS.items():
            _set_tracking_case(agent, tracking_case)
            reference = REFERENCES[tactic](agent)

            # The selector cache is intentionally per fire-state snapshot, but
            # each tactic check starts clean so lazy-key assertions are precise.
            agent.model.__cache__.clear()
            _set_tracking_case(agent, tracking_case)
            cached = selector(agent)

            check_label = f"{label} {tracking_case} {tactic}"
            _assert_same_destination(check_label, cached, reference)
            _assert_lazy_cache(check_label, agent, tactic)

    _clear_destinations(agent)
    print(f"{label}: PASS")


def main() -> None:
    args = _parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    env = WildfireHourlyEnv(
        _resolve_scenario(args.scenario),
        max_steps=max(args.steps + 1, 1),
        decision_interval_minutes=args.decision_interval_minutes,
        fire_detection_delay_minutes=args.fire_detection_delay_minutes,
        state_fire_fronts=DEFAULT_STATE_FIRE_FRONTS,
        state_space=STATE_SPACE_SMALL,
    )

    try:
        for seed in args.seeds:
            env.reset(options={"sim_seed": seed})
            _verify_tactics(env, f"seed {seed} reset")

            for step in range(args.steps):
                action = np.zeros(env.action_space.shape, dtype=int)
                _observation, _reward, terminated, truncated, _info = env.step(action)
                _verify_tactics(env, f"seed {seed} step {step + 1}")
                if terminated or truncated:
                    break
    finally:
        env.close()

    print("PASS: cached select_poi destinations match old formulas lazily.")


if __name__ == "__main__":
    main()
