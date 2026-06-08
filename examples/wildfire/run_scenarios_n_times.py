#!/usr/bin/env python3
"""Run selected wildfire scenarios N times each and display final results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.firefighter_model.follower import PayloadStatus
from examples.wildfire.simulation import (
    WildfireParameters,
    WildfireSimulation,
    _TerrainParametersCache,
)

DEFAULT_SCENARIOS = (
    "Palisades copy.json",
    "Salamis.json",
    "Pyrenees.json",
)

MOE_WEIGHT = 0.25

# Scenario-specific normalization values provided by user.
# Burnt area is supplied in hectares and converted to m^2.
# Cost is supplied in million euros and converted to euros.
SCENARIO_MOE_NORMS = {
    "salamis": {
        "burnt_area_norm": 4146.0 * 10_000.0,
        "cost_norm": 13_993.0 * 1_000_000.0,
        "emission_norm": 714_009.0,
        "casualty_norm": 10_000.0,
    },
    "pyrenees": {
        "burnt_area_norm": 9938.0 * 10_000.0,
        "cost_norm": 17_509.0 * 1_000_000.0,
        "emission_norm": 2_364_064.0,
        "casualty_norm": 15_000.0,
    },
    "palisades": {
        "burnt_area_norm": 9087.0 * 10_000.0,
        "cost_norm": 191_106.0 * 1_000_000.0,
        "emission_norm": 131_224.0,
        "casualty_norm": 300_000.0,
    },
}


def _resolve_scenario(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    candidate = SCENARIOS_DIR / "inputs" / path_or_name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to locate scenario file: {path_or_name}")


def _write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd  # Lazy import for optional Excel output.
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


def _compute_moe(
    *,
    burnt_area_m2: float,
    cost_eur: float,
    emissions_tonnes: float,
    casualties: float,
    fire_in_bounds: bool,
    norms: dict[str, float],
) -> tuple[float, float, float]:
    base_moe = 1.0 - (
        MOE_WEIGHT * (burnt_area_m2 / norms["burnt_area_norm"])
        + MOE_WEIGHT * (cost_eur / norms["cost_norm"])
        + MOE_WEIGHT * (emissions_tonnes / norms["emission_norm"])
        + MOE_WEIGHT * (casualties / norms["casualty_norm"])
    )
    propagation_factor = 0.0 if fire_in_bounds else 1.0
    penalized_moe = base_moe - propagation_factor
    return base_moe, propagation_factor, penalized_moe


def _finalize_sortie_stats(
    scenario_name: str,
    run_idx: int,
    sortie_records: list[dict[str, Any]],
) -> dict[str, float]:
    if not sortie_records:
        return {
            "sortie_count": 0.0,
            "sortie_mean_minutes": 0.0,
            "sortie_min_minutes": 0.0,
            "sortie_max_minutes": 0.0,
        }
    durations = [float(row["sortie_duration_min"]) for row in sortie_records]
    return {
        "sortie_count": float(len(sortie_records)),
        "sortie_mean_minutes": sum(durations) / len(durations),
        "sortie_min_minutes": min(durations),
        "sortie_max_minutes": max(durations),
    }


def _run_single_simulation(
    scenario_path: Path,
    seed: int,
    run_idx: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Terrain metadata is cached globally; clear between scenarios so each
    # file validates against its own map bounds.
    _TerrainParametersCache.metadata = {}
    with scenario_path.open() as handle:
        parameters = WildfireParameters.model_validate_json(handle.read())
    parameters = parameters.model_copy(update={"run_headless": True})

    simulation = WildfireSimulation(parameters=parameters, seed=seed)
    # Match PPO env behavior: explicitly ignite before stepping.
    simulation.wildfire.ignite(simulation.ignition_centers)
    wall_start = time.perf_counter()
    firefighters = tuple(simulation.firefighters.firefighters)
    sortie_records: list[dict[str, Any]] = []
    tracker: dict[int, dict[str, Any]] = {}
    for agent_idx, agent in enumerate(firefighters):
        is_onboard = agent.payload_status == PayloadStatus.ONBOARD
        tracker[agent.unique_id] = {
            "agent_idx": agent_idx,
            "last_status_onboard": is_onboard,
            "last_resupply_runtime_s": 0.0 if is_onboard else None,
            "drop_seen_since_resupply": False,
            "sortie_counter": 0,
        }

    while simulation.iterations < simulation.max_iter:
        if simulation.is_stopped.is_set():
            break
        try:
            simulation.step(force=True)
        except RuntimeError as err:
            if "not started cannot be stopped" in str(err):
                break
            raise

        runtime_s = float(simulation.timer.mission_runtime.total_seconds())
        for agent in firefighters:
            state = tracker[agent.unique_id]
            is_onboard = agent.payload_status == PayloadStatus.ONBOARD
            was_onboard = bool(state["last_status_onboard"])

            if was_onboard and not is_onboard:
                state["drop_seen_since_resupply"] = True

            if (not was_onboard) and is_onboard:
                if (
                    state["drop_seen_since_resupply"]
                    and state["last_resupply_runtime_s"] is not None
                ):
                    state["sortie_counter"] += 1
                    sortie_duration_min = (
                        runtime_s - float(state["last_resupply_runtime_s"])
                    ) / 60.0
                    # Refill source inference:
                    # - Airport/base refills set current_base and enter ENERGIZE.
                    # - Water scooping refills occur around LOITER without base assignment.
                    flight_state_name = getattr(
                        getattr(agent, "flight_state", None), "name", ""
                    )
                    if (
                        getattr(agent, "current_base", None) is not None
                        or flight_state_name == "ENERGIZE"
                    ):
                        refill_location = "airport"
                    elif flight_state_name == "LOITER":
                        refill_location = "water"
                    else:
                        refill_location = "water"
                    sortie_records.append(
                        {
                            "scenario_name": scenario_path.name,
                            "seed": seed,
                            "run": run_idx,
                            "agent_idx": state["agent_idx"],
                            "agent_unique_id": int(agent.unique_id),
                            "sortie_index": int(state["sortie_counter"]),
                            "sortie_duration_min": sortie_duration_min,
                            "mission_runtime_min_at_refill": runtime_s / 60.0,
                            "refill_location": refill_location,
                            "flight_state_at_refill": flight_state_name,
                        }
                    )
                state["last_resupply_runtime_s"] = runtime_s
                state["drop_seen_since_resupply"] = False

            state["last_status_onboard"] = is_onboard

    wall_duration_seconds = time.perf_counter() - wall_start

    mission_runtime_seconds = simulation.timer.mission_runtime.total_seconds()
    scenario_label = parameters.terrain_inputs.file_namespace.split("_", 1)[0]
    burnt_area_m2 = float(simulation.wildfire.burnt_area)
    fire_cost_eur = float(simulation.total_fire_cost)
    casualties = float(simulation.total_casualties)
    emissions_tonnes = float(simulation.total_fire_emissions)
    fire_in_bounds = bool(simulation.wildfire.fire_in_bounds)
    moe_norms = _resolve_moe_norms(scenario_path)
    moe_base, propagation_factor, moe_penalized = _compute_moe(
        burnt_area_m2=burnt_area_m2,
        cost_eur=fire_cost_eur,
        emissions_tonnes=emissions_tonnes,
        casualties=casualties,
        fire_in_bounds=fire_in_bounds,
        norms=moe_norms,
    )
    sortie_stats = _finalize_sortie_stats(
        scenario_name=scenario_path.name,
        run_idx=run_idx,
        sortie_records=sortie_records,
    )

    return {
        "scenario": scenario_label,
        "scenario_name": scenario_path.name,
        "scenario_path": str(scenario_path),
        "seed": seed,
        "mission_runtime_seconds": mission_runtime_seconds,
        "wall_duration_seconds": wall_duration_seconds,
        "burnt_area_m2": burnt_area_m2,
        "fire_cost_eur": fire_cost_eur,
        "casualties": casualties,
        "emissions_tonnes": emissions_tonnes,
        "fire_in_bounds": fire_in_bounds,
        "moe_base": moe_base,
        "propagation_factor": propagation_factor,
        "moe_cumulative_reward": moe_penalized,
        "moe_burnt_area_norm_m2": moe_norms["burnt_area_norm"],
        "moe_cost_norm_eur": moe_norms["cost_norm"],
        "moe_emission_norm_tonnes": moe_norms["emission_norm"],
        "moe_casualty_norm": moe_norms["casualty_norm"],
        **sortie_stats,
    }, sortie_records


def _print_results(
    records: list[dict[str, Any]],
    sortie_records: list[dict[str, Any]],
) -> None:
    print("\nPer-simulation results")
    print(
        "scenario | run | seed | mission_h | burnt_area_m2 | "
        "cost_eur | casualties | emissions_t | moe | sorties | avg_sortie_min | fire_in_bounds"
    )
    for row in records:
        print(
            f"{row['scenario']} | {row['run']} | {row['seed']} | "
            f"{row['mission_runtime_seconds'] / 3600:.2f} | "
            f"{row['burnt_area_m2']:.2f} | {row['fire_cost_eur']:.2f} | "
            f"{row['casualties']:.2f} | {row['emissions_tonnes']:.2f} | "
            f"{row['moe_cumulative_reward']:.4f} | "
            f"{int(row['sortie_count'])} | {row['sortie_mean_minutes']:.2f} | "
            f"{row['fire_in_bounds']}"
        )

    print("\nScenario summary (mean over runs)")
    scenarios = sorted({row["scenario_name"] for row in records})
    print(
        "scenario_name | runs | mean_burnt_area_m2 | mean_cost_eur | "
        "mean_casualties | mean_emissions_tonnes | mean_mission_h | "
        "mean_moe | mean_sorties | mean_sortie_min"
    )
    for scenario_name in scenarios:
        rows = [row for row in records if row["scenario_name"] == scenario_name]
        n = len(rows)
        mean_burnt = sum(float(row["burnt_area_m2"]) for row in rows) / n
        mean_cost = sum(float(row["fire_cost_eur"]) for row in rows) / n
        mean_cas = sum(float(row["casualties"]) for row in rows) / n
        mean_emissions = sum(float(row["emissions_tonnes"]) for row in rows) / n
        mean_mission_h = (
            sum(float(row["mission_runtime_seconds"]) for row in rows) / n / 3600.0
        )
        mean_moe = sum(float(row["moe_cumulative_reward"]) for row in rows) / n
        mean_sorties = sum(float(row["sortie_count"]) for row in rows) / n
        mean_sortie_min = sum(float(row["sortie_mean_minutes"]) for row in rows) / n
        print(
            f"{scenario_name} | {n} | {mean_burnt:.2f} | {mean_cost:.2f} | "
            f"{mean_cas:.2f} | {mean_emissions:.2f} | {mean_mission_h:.2f} | "
            f"{mean_moe:.4f} | "
            f"{mean_sorties:.2f} | {mean_sortie_min:.2f}"
        )

    if sortie_records:
        print("\nSortie durations")
        print(
            "scenario_name | run | seed | agent_idx | sortie_index | "
            "sortie_duration_min | refill_location"
        )
        for sortie in sortie_records:
            print(
                f"{sortie['scenario_name']} | {sortie['run']} | {sortie['seed']} | "
                f"{sortie['agent_idx']} | {sortie['sortie_index']} | "
                f"{sortie['sortie_duration_min']:.2f} | {sortie['refill_location']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Palisades/Salamis/Pyrenees N times each and display results."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; each simulation uses an incremented seed.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario files (name in inputs/ or absolute path).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for per-simulation results (.csv or .xlsx).",
    )
    parser.add_argument(
        "--sortie-output",
        help=(
            "Optional output path for per-sortie durations "
            "(.csv or .xlsx)."
        ),
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    scenario_paths = [_resolve_scenario(name) for name in args.scenarios]
    records: list[dict[str, Any]] = []
    sortie_records: list[dict[str, Any]] = []

    sim_counter = 0
    total_runs = len(scenario_paths) * args.runs
    for scenario_path in scenario_paths:
        for run_idx in range(args.runs):
            seed = args.seed + sim_counter
            sim_counter += 1
            print(
                f"Running {scenario_path.name} "
                f"({run_idx + 1}/{args.runs}, global {sim_counter}/{total_runs}) "
                f"with seed={seed}"
            )
            result, run_sorties = _run_single_simulation(
                scenario_path, seed, run_idx + 1
            )
            result["run"] = run_idx + 1
            records.append(result)
            sortie_records.extend(run_sorties)

    _print_results(records, sortie_records)

    if args.output:
        output_path = Path(args.output)
        _write_records(records, output_path)
        print(f"\nSaved per-simulation results to {output_path}")

    if args.sortie_output:
        sortie_output_path = Path(args.sortie_output)
        _write_records(sortie_records, sortie_output_path)
        print(f"Saved per-sortie results to {sortie_output_path}")


if __name__ == "__main__":
    main()
