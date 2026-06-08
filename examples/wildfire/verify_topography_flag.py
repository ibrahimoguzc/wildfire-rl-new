#!/usr/bin/env python3
"""Verify front-derived state features in ppo_runnerv2.

The script runs a few short simulations, prints the inspected fire fronts, and
checks that the observation, info dict, and per-front diagnostics agree.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    DEFAULT_STATE_FIRE_FRONTS,
    INDIRECT_FRONT_DISTANCE_THRESHOLD_M,
    TOPOGRAPHY_SLOPE_FACTOR_THRESHOLD,
    VEGETATION_MIN_COMBUSTIBLE_NEIGHBORS,
    WildfireHourlyEnv,
    _resolve_scenario,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check front-derived flags and diagnostics in ppo_runnerv2."
    )
    parser.add_argument(
        "--scenario",
        default="Palisades copy.json",
        help="Scenario JSON name in inputs/ or an absolute/relative path.",
    )
    parser.add_argument(
        "--state-fire-fronts",
        type=int,
        default=DEFAULT_STATE_FIRE_FRONTS,
        help="Maximum number of fastest burning cells to inspect.",
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
        default=3,
        help="Decision steps to run after each reset.",
    )
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=10,
        help="Decision interval used by the test environment.",
    )
    parser.add_argument(
        "--fire-detection-delay-minutes",
        type=float,
        default=0.0,
        help="Use 0.0 to inspect the state immediately after ignition.",
    )
    return parser.parse_args()


def _check_state(
    env: WildfireHourlyEnv,
    feature_indices: dict[str, int],
    label: str,
    observation: np.ndarray,
    info: dict,
) -> None:
    summary = env.last_front_summary
    fronts = env.last_front_diagnostics

    obs_topography_flag = float(observation[feature_indices["topography_flag"]])
    obs_vegetation_flag = float(observation[feature_indices["vegetation_flag"]])
    obs_indirect_flag = float(observation[feature_indices["indirect_flag"]])
    obs_urban_flag = float(observation[feature_indices["urban_flag"]])
    obs_water_flag = float(observation[feature_indices["water_flag"]])
    info_topography_flag = float(info["topography_flag"])
    info_vegetation_flag = float(info["vegetation_flag"])
    info_indirect_flag = float(info["indirect_flag"])
    info_urban_flag = float(info["urban_flag"])
    info_water_flag = float(info["water_flag"])
    count = int(summary["state_fire_front_count"])
    max_count = int(summary["state_fire_fronts"])
    positive = int(summary["topography_front_positive_count"])
    required = int(summary["topography_front_required_count"])
    vegetation_positive = int(summary["vegetation_front_positive_count"])
    vegetation_required = int(summary["vegetation_front_required_count"])
    indirect_positive = int(summary["indirect_front_positive_count"])
    indirect_required = int(summary["indirect_front_required_count"])
    urban_count = int(summary["urban_front_count"])
    water_count = int(summary["water_front_count"])
    objective_count = int(summary["objective_front_count"])

    # Majority rule: ties on even front counts pass because required is ceil(n/2).
    expected_topography_flag = 1.0 if count > 0 and positive >= required else 0.0
    expected_vegetation_flag = (
        1.0 if count > 0 and vegetation_positive >= vegetation_required else 0.0
    )
    expected_indirect_flag = (
        1.0 if count > 0 and indirect_positive >= indirect_required else 0.0
    )
    assert indirect_required == ((count // 2) + 1 if count else 0), (
        label,
        "indirect required count should be strict majority",
        indirect_required,
        count,
    )
    expected_urban_flag = (
        1.0 if objective_count > 0 and urban_count >= water_count else 0.0
    )
    expected_water_flag = (
        1.0 if objective_count > 0 and water_count > urban_count else 0.0
    )

    for name, value in (
        ("obs_topography_flag", obs_topography_flag),
        ("obs_vegetation_flag", obs_vegetation_flag),
        ("obs_indirect_flag", obs_indirect_flag),
        ("obs_urban_flag", obs_urban_flag),
        ("obs_water_flag", obs_water_flag),
        ("info_topography_flag", info_topography_flag),
        ("info_vegetation_flag", info_vegetation_flag),
        ("info_indirect_flag", info_indirect_flag),
        ("info_urban_flag", info_urban_flag),
        ("info_water_flag", info_water_flag),
    ):
        assert value in (0.0, 1.0), (label, name, "is not binary", value)

    assert obs_topography_flag == info_topography_flag == expected_topography_flag, (
        label,
        "topography_flag mismatch",
        obs_topography_flag,
        info_topography_flag,
        expected_topography_flag,
    )
    assert obs_vegetation_flag == info_vegetation_flag == expected_vegetation_flag, (
        label,
        "vegetation_flag mismatch",
        obs_vegetation_flag,
        info_vegetation_flag,
        expected_vegetation_flag,
    )
    assert obs_indirect_flag == info_indirect_flag == expected_indirect_flag, (
        label,
        "indirect_flag mismatch",
        obs_indirect_flag,
        info_indirect_flag,
        expected_indirect_flag,
    )
    assert obs_urban_flag == info_urban_flag == expected_urban_flag, (
        label,
        "urban_flag mismatch",
        obs_urban_flag,
        info_urban_flag,
        expected_urban_flag,
    )
    assert obs_water_flag == info_water_flag == expected_water_flag, (
        label,
        "water_flag mismatch",
        obs_water_flag,
        info_water_flag,
        expected_water_flag,
    )
    assert count <= max_count, (label, "front count exceeds configured max", count)
    assert positive == sum(front.has_topography_growth for front in fronts), (
        label,
        "positive count disagrees with diagnostics",
    )
    assert vegetation_positive == sum(front.has_vegetation_threat for front in fronts), (
        label,
        "vegetation count disagrees with diagnostics",
    )
    assert indirect_positive == sum(front.near_indirect_line for front in fronts), (
        label,
        "indirect count disagrees with diagnostics",
    )
    assert urban_count == sum(front.objective_vote == "urban" for front in fronts), (
        label,
        "urban count disagrees with diagnostics",
    )
    assert water_count == sum(front.objective_vote == "water" for front in fronts), (
        label,
        "water count disagrees with diagnostics",
    )
    assert objective_count == urban_count + water_count, (
        label,
        "objective count disagrees with vote counts",
    )
    if objective_count > 0:
        assert obs_urban_flag + obs_water_flag == 1.0, (
            label,
            "urban/water flags should be mutually exclusive",
        )
    else:
        assert obs_urban_flag == 0.0 and obs_water_flag == 0.0, (
            label,
            "urban/water flags should both be zero without objective votes",
        )

    print(
        f"{label}: topo={info_topography_flag:.0f}, "
        f"veg={info_vegetation_flag:.0f}, "
        f"indirect={info_indirect_flag:.0f}, "
        f"urban={info_urban_flag:.0f}, water={info_water_flag:.0f}, "
        f"fronts={count}/{max_count}, topo_positive={positive}/{required}, "
        f"veg_positive={vegetation_positive}/{vegetation_required}, "
        f"indirect_positive={indirect_positive}/{indirect_required}, "
        f"objective_votes urban={urban_count} water={water_count}"
    )

    for rank, front in enumerate(fronts, start=1):
        expected_topography_vote = (
            front.spread_rate > 0.0
            and front.topography_priority > 0.0
            and front.slope_factor >= TOPOGRAPHY_SLOPE_FACTOR_THRESHOLD
            and front.forward_combustible_uphill
        )
        assert front.has_topography_growth == expected_topography_vote, (
            label,
            "front topography vote mismatch",
            rank,
        )
        expected_vegetation_count = env._front_vegetation_neighbor_count(
            front.source_i,
            front.source_j,
        )
        assert (
            front.vegetation_combustible_neighbor_count
            == expected_vegetation_count
        ), (
            label,
            "front vegetation count mismatch",
            rank,
            front.vegetation_combustible_neighbor_count,
            expected_vegetation_count,
        )
        expected_vegetation_vote = (
            front.vegetation_combustible_neighbor_count
            > VEGETATION_MIN_COMBUSTIBLE_NEIGHBORS
        )
        assert front.has_vegetation_threat == expected_vegetation_vote, (
            label,
            "front vegetation vote mismatch",
            rank,
        )
        expected_indirect_distance = env._front_distance_to_fire_line_m(
            front.source_i,
            front.source_j,
        )
        if math.isfinite(expected_indirect_distance):
            assert math.isclose(
                front.distance_to_indirect_line_m,
                expected_indirect_distance,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ), (
                label,
                "front indirect distance mismatch",
                rank,
                front.distance_to_indirect_line_m,
                expected_indirect_distance,
            )
        else:
            assert not math.isfinite(front.distance_to_indirect_line_m), (
                label,
                "front indirect distance mismatch",
                rank,
                front.distance_to_indirect_line_m,
                expected_indirect_distance,
            )
        expected_indirect_vote = (
            math.isfinite(front.distance_to_indirect_line_m)
            and front.distance_to_indirect_line_m
            <= INDIRECT_FRONT_DISTANCE_THRESHOLD_M
        )
        assert front.near_indirect_line == expected_indirect_vote, (
            label,
            "front indirect vote mismatch",
            rank,
        )
        if not math.isfinite(front.distance_to_urban) and not math.isfinite(
            front.distance_to_water
        ):
            expected_objective_vote = None
        elif not math.isfinite(front.distance_to_urban):
            expected_objective_vote = "water"
        elif not math.isfinite(front.distance_to_water):
            expected_objective_vote = "urban"
        elif front.distance_to_urban <= front.distance_to_water:
            expected_objective_vote = "urban"
        else:
            expected_objective_vote = "water"
        assert front.objective_vote == expected_objective_vote, (
            label,
            "front objective vote mismatch",
            rank,
        )

        print(
            f"  #{rank}: source=({front.source_i},{front.source_j}) "
            f"projected=({front.projected_i},{front.projected_j}) "
            f"objective_cell=({front.objective_i},{front.objective_j}) "
            f"rate={front.spread_rate:.4f} "
            f"topo_priority={front.topography_priority:.4f} "
            f"slope_factor={front.slope_factor:.3f} "
            f"forward_uphill={front.forward_combustible_uphill} "
            f"topo_vote={front.has_topography_growth} "
            f"veg_count={front.vegetation_combustible_neighbor_count} "
            f"veg_vote={front.has_vegetation_threat} "
            f"indirect_dist={front.distance_to_indirect_line_m:.1f} "
            f"indirect_vote={front.near_indirect_line} "
            f"urban_dist={front.distance_to_urban:.1f} "
            f"water_dist={front.distance_to_water:.1f} "
            f"objective_vote={front.objective_vote}"
        )


def main() -> None:
    args = _parse_args()
    if args.state_fire_fronts <= 0:
        raise ValueError("--state-fire-fronts must be > 0")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    env = WildfireHourlyEnv(
        _resolve_scenario(args.scenario),
        max_steps=max(args.steps + 1, 1),
        decision_interval_minutes=args.decision_interval_minutes,
        fire_detection_delay_minutes=args.fire_detection_delay_minutes,
        state_fire_fronts=args.state_fire_fronts,
    )

    try:
        feature_indices = {
            name: env.state_feature_names.index(name)
            for name in (
                "topography_flag",
                "vegetation_flag",
                "indirect_flag",
                "urban_flag",
                "water_flag",
            )
        }
        for seed in args.seeds:
            observation, info = env.reset(options={"sim_seed": seed})
            _check_state(env, feature_indices, f"seed {seed} reset", observation, info)

            for step in range(args.steps):
                action = np.zeros(env.action_space.shape, dtype=int)
                observation, _reward, terminated, truncated, info = env.step(action)
                _check_state(
                    env,
                    feature_indices,
                    f"seed {seed} step {step + 1}",
                    observation,
                    info,
                )
                if terminated or truncated:
                    break
    finally:
        env.close()

    print("PASS: front-derived flags and diagnostics are internally consistent.")


if __name__ == "__main__":
    main()
