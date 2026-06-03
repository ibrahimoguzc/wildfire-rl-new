#!/usr/bin/env python3
"""Verify the fast PPO metric calculation against the original properties.

The training runner now computes burnt area, fire cost, casualties, and
emissions directly in ppo_runnerv2 to avoid repeated full-grid scans. This
script runs a few short seeded simulations and checks that the fast metrics
match the original simulation-property metrics at reset and after each decision
step.
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
    DEFAULT_DECISION_INTERVAL_MINUTES,
    DEFAULT_STATE_FIRE_FRONTS,
    STATE_SPACE_SMALL,
    SUPPORTED_STATE_SPACES,
    SUPPORTED_TACTIC_DISTRIBUTIONS,
    TACTIC_DISTRIBUTION_INDIVIDUAL,
    Metrics,
    WildfireHourlyEnv,
    _resolve_scenario,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ppo_runnerv2 fast reward metrics against the original "
            "simulation-property metrics."
        )
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
        default=2,
        help="Decision steps to run after each reset.",
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
        help="Use 0.0 to compare metrics immediately after ignition.",
    )
    parser.add_argument(
        "--state-fire-fronts",
        type=int,
        default=DEFAULT_STATE_FIRE_FRONTS,
        help="Maximum number of fire fronts used by the environment state.",
    )
    parser.add_argument(
        "--state-space",
        choices=SUPPORTED_STATE_SPACES,
        default=STATE_SPACE_SMALL,
        help="State-space option for the test environment.",
    )
    parser.add_argument(
        "--tactic-distribution",
        choices=SUPPORTED_TACTIC_DISTRIBUTIONS,
        default=TACTIC_DISTRIBUTION_INDIVIDUAL,
        help="Action-distribution option for the test environment.",
    )
    parser.add_argument(
        "--aircraft-group-size",
        type=int,
        default=2,
        help="Group size when --tactic-distribution group is used.",
    )
    parser.add_argument(
        "--controlled-agent-count",
        type=int,
        default=6,
        help="Number of controlled aircraft decisions exposed by the runner.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute/relative tolerance for metric comparison.",
    )
    return parser.parse_args()


def _metric_items(metrics: Metrics) -> tuple[tuple[str, float], ...]:
    return (
        ("burnt_area", metrics.burnt_area),
        ("cost", metrics.cost),
        ("casualties", metrics.casualties),
        ("emissions", metrics.emissions),
    )


def _assert_metrics_match(
    label: str,
    fast: Metrics,
    reference: Metrics,
    tolerance: float,
) -> None:
    for (name, fast_value), (_ref_name, reference_value) in zip(
        _metric_items(fast),
        _metric_items(reference),
    ):
        if not math.isclose(
            fast_value,
            reference_value,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise AssertionError(
                f"{label}: {name} mismatch: "
                f"fast={fast_value}, reference={reference_value}"
            )

    print(
        f"{label}: PASS "
        f"burnt={fast.burnt_area:.2f} m2, "
        f"cost={fast.cost:.2f}, "
        f"casualties={fast.casualties:.0f}, "
        f"emissions={fast.emissions:.2f}"
    )


def main() -> None:
    args = _parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.state_fire_fronts <= 0:
        raise ValueError("--state-fire-fronts must be > 0")

    env = WildfireHourlyEnv(
        _resolve_scenario(args.scenario),
        max_steps=max(args.steps + 1, 1),
        decision_interval_minutes=args.decision_interval_minutes,
        fire_detection_delay_minutes=args.fire_detection_delay_minutes,
        state_fire_fronts=args.state_fire_fronts,
        state_space=args.state_space,
        tactic_distribution=args.tactic_distribution,
        aircraft_group_size=args.aircraft_group_size,
        controlled_agent_count=args.controlled_agent_count,
    )

    try:
        for seed in args.seeds:
            env.reset(options={"sim_seed": seed})
            _assert_metrics_match(
                f"seed {seed} reset",
                env._compute_metrics(),
                env._compute_metrics_reference(),
                args.tolerance,
            )

            for step in range(args.steps):
                # Zero actions are deterministic and sufficient here because
                # the test checks metric equivalence, not tactic quality.
                action = np.zeros(env.action_space.shape, dtype=int)
                (
                    _observation,
                    _reward,
                    terminated,
                    truncated,
                    _info,
                ) = env.step(action)
                _assert_metrics_match(
                    f"seed {seed} step {step + 1}",
                    env._compute_metrics(),
                    env._compute_metrics_reference(),
                    args.tolerance,
                )
                if terminated or truncated:
                    break
    finally:
        env.close()

    print("PASS: fast PPO metrics match the original simulation-property metrics.")


if __name__ == "__main__":
    main()
