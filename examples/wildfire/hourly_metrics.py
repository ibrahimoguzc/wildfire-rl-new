#!/usr/bin/env python3
"""
Run the wildfire simulation and log hourly state variables plus KPIs to Excel.

The script reproduces the simulation setup used in examples/wildfire/main.py,
but steps the simulation manually so that we can sample the requested signals
at the start of every simulated hour. Output is written to an Excel workbook
containing one row per sampled hour.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SELECT_POI_TABLE,
)
from examples.wildfire.firefighter_model.tactic_pieces.suppress import (
    SUPPRESS_TABLE,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    TRACK_POI_TABLE,
)
from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation

BURN_AREA_NORM = 220000.0
COST_NORM = 1005000.0
EMISSION_NORM = 130.0
CASUALTY_NORM = 240.0


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    candidate = SCENARIOS_DIR / "inputs" / raw_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not locate scenario file: {raw_path}")


def _resolve_output_path(raw_path: str | None, input_path: Path) -> Path:
    if raw_path:
        path = Path(raw_path)
    else:
        default_name = input_path.stem + "_hourly_metrics.xlsx"
        path = SCENARIOS_DIR / "outputs" / default_name
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_water_positions(sim: WildfireSimulation) -> np.ndarray:
    sources = sim.firefighters.water_sources
    if not sources:
        return np.empty((0, 2), dtype=float)
    return np.array([source.pos for source in sources], dtype=float)


def _collect_snapshot(
    sim: WildfireSimulation,
    hour_index: int,
    fire_grid_area: float,
    water_positions: np.ndarray,
    prev_totals: dict[str, float],
    tactic_maps: dict[str, dict[type, str]],
) -> dict[str, Any]:
    """Gather hourly state variables and KPI deltas."""
    mission_time = sim.timer.mission_time
    runtime_seconds = sim.timer.mission_runtime.total_seconds()
    atmosphere = sim.atmosphere

    burning_indices = sim.wildfire.burning_indices
    burning_count = int(burning_indices.shape[0])

    fire_center_x = math.nan
    fire_center_y = math.nan
    spread_angle = math.nan
    distance_fire_line = math.nan
    distance_water = math.nan

    if burning_count:
        fire_positions = sim.wildfire.fire_positions
        centroid = fire_positions.mean(axis=0)
        fire_center_x = float(centroid[0])
        fire_center_y = float(centroid[1])

        centroid_idx = burning_indices.mean(axis=0)
        spread_rates = sim.wildfire.get_spread_rates(burning_indices)
        fastest_idx = burning_indices[int(np.argmax(spread_rates))]
        angle = math.degrees(
            math.atan2(
                fastest_idx[0] - centroid_idx[0],
                fastest_idx[1] - centroid_idx[1],
            )
        )
        spread_angle = float((angle + 360.0) % 360.0)

        if water_positions.size:
            distance_water = float(
                np.linalg.norm(water_positions - centroid, axis=1).min()
            )

        fire_block_indices = sim.firefighters.fire_block_indices
        if (
            fire_block_indices is not None
            and sim.firefighters.current_block_index > 0
        ):
            segment = fire_block_indices[: sim.firefighters.current_block_index]
            if segment.size:
                distance_fire_line = float(
                    sim.parameters.cell_size
                    * cdist(burning_indices, segment).min()
                )

    burnt_area = float(sim.wildfire.burnt_area)
    burnt_fraction = (
        burnt_area / fire_grid_area if fire_grid_area else math.nan
    )

    total_cost = float(sim.total_fire_cost)
    total_casualties = float(sim.total_casualties)
    total_emissions = float(sim.total_fire_emissions)

    snapshot = {
        "hour": hour_index,
        "mission_time": mission_time.isoformat(),
        "runtime_seconds": runtime_seconds,
        "temperature_c": float(atmosphere.temperature),
        "humidity_pct": float(atmosphere.relative_humidity),
        "wind_speed_ms": float(atmosphere.wind_speed),
        "wind_direction_deg": float((atmosphere.wind_aspect + 360.0) % 360.0),
        "time_to_sunset_min": max(
            (atmosphere.next_sunset - mission_time).total_seconds() / 60.0,
            0.0,
        ),
        "distance_to_fire_line_m": distance_fire_line,
        "distance_to_water_m": distance_water,
        "active_firefront_count": burning_count,
        "burnt_area_fraction": burnt_fraction,
        "burnt_area_m2": burnt_area,
        "burnt_area_delta_m2": burnt_area - prev_totals["burnt_area"],
        "fire_cost_eur": total_cost,
        "fire_cost_delta_eur": total_cost - prev_totals["cost"],
        "casualties": total_casualties,
        "casualties_delta": total_casualties - prev_totals["casualties"],
        "emissions_tonnes": total_emissions,
        "emissions_delta_tonnes": (
            total_emissions - prev_totals["emissions"]
        ),
        "fire_center_x": fire_center_x,
        "fire_center_y": fire_center_y,
        "spread_angle_deg": spread_angle,
    }

    burnt_delta = snapshot["burnt_area_delta_m2"]
    cost_delta = snapshot["fire_cost_delta_eur"]
    emission_delta = snapshot["emissions_delta_tonnes"]
    casualty_delta = snapshot["casualties_delta"]
    reward = 1.0 - (
        0.25 * (burnt_delta / BURN_AREA_NORM)
        + 0.25 * (cost_delta / COST_NORM)
        + 0.25 * (emission_delta / EMISSION_NORM)
        + 0.25 * (casualty_delta / CASUALTY_NORM)
    )
    snapshot["reward_moe"] = reward

    for agent_idx, agent in enumerate(sim.firefighters.firefighters):
        select_label = tactic_maps["select"].get(
            agent.tactic.select_poi.__class__, "unknown"
        )
        track_label = tactic_maps["track"].get(
            agent.tactic.track_poi.__class__, "unknown"
        )
        suppress_label = tactic_maps["suppress"].get(
            agent.tactic.suppress.__class__, "unknown"
        )
        snapshot[f"agent_{agent_idx}_select_poi"] = select_label
        snapshot[f"agent_{agent_idx}_track_poi"] = track_label
        snapshot[f"agent_{agent_idx}_suppress"] = suppress_label

    prev_totals.update(
        burnt_area=burnt_area,
        cost=total_cost,
        casualties=total_casualties,
        emissions=total_emissions,
    )

    return snapshot


def run_hourly_sampling(
    sim: WildfireSimulation,
    max_hours: int,
) -> list[dict[str, Any]]:
    """Advance the simulation and collect hourly metrics."""
    water_positions = _prepare_water_positions(sim)
    fire_grid_area = sim.wildfire.fire_states.size * (
        sim.parameters.cell_size**2
    )

    records: list[dict[str, Any]] = []
    prev_totals = {
        "burnt_area": 0.0,
        "cost": 0.0,
        "casualties": 0.0,
        "emissions": 0.0,
    }

    tactic_maps = {
        "select": {cls: enum.value for enum, cls in SELECT_POI_TABLE.items()},
        "track": {cls: enum.value for enum, cls in TRACK_POI_TABLE.items()},
        "suppress": {
            cls: enum.value for enum, cls in SUPPRESS_TABLE.items()
        },
    }

    records.append(
        _collect_snapshot(
            sim=sim,
            hour_index=0,
            fire_grid_area=fire_grid_area,
            water_positions=water_positions,
            prev_totals=prev_totals,
            tactic_maps=tactic_maps,
        )
    )

    next_hour = 1
    next_sample_time = next_hour * 3600.0
    mission_completed = False

    while next_hour <= max_hours and sim.iterations < sim.max_iter:
        try:
            sim.step(force=True)
        except RuntimeError as err:
            if "not started cannot be stopped" in str(err):
                mission_completed = True
                break
            raise

        runtime_seconds = sim.timer.mission_runtime.total_seconds()
        while runtime_seconds >= next_sample_time and next_hour <= max_hours:
            records.append(
                _collect_snapshot(
                    sim=sim,
                    hour_index=next_hour,
                    fire_grid_area=fire_grid_area,
                    water_positions=water_positions,
                    prev_totals=prev_totals,
                    tactic_maps=tactic_maps,
                )
            )
            next_hour += 1
            next_sample_time = next_hour * 3600.0

    if mission_completed:
        runtime_seconds = sim.timer.mission_runtime.total_seconds()
        while runtime_seconds >= next_sample_time and next_hour <= max_hours:
            records.append(
                _collect_snapshot(
                    sim=sim,
                    hour_index=next_hour,
                    fire_grid_area=fire_grid_area,
                    water_positions=water_positions,
                    prev_totals=prev_totals,
                    tactic_maps=tactic_maps,
                )
            )
            next_hour += 1
            next_sample_time = next_hour * 3600.0

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wildfire simulation and export hourly metrics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Scenario JSON file name or path (e.g. 'Palisades copy.json').",
    )
    parser.add_argument(
        "--output",
        help="Optional Excel file path for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed to replicate the run (default: 1).",
    )

    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    output_path = _resolve_output_path(args.output, input_path)

    with input_path.open() as handle:
        parameters = WildfireParameters.model_validate_json(handle.read())

    simulation = WildfireSimulation(parameters=parameters, seed=args.seed)
    simulation.wildfire.ignite(simulation.ignition_centers)

    max_hours = int(parameters.max_runtime // 3600)
    hourly_records = run_hourly_sampling(simulation, max_hours=max_hours)

    df = pd.DataFrame(hourly_records)
    state_columns = [
        "hour",
        "mission_time",
        "runtime_seconds",
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
    state_columns = [col for col in state_columns if col in df.columns]

    action_columns = sorted(
        col for col in df.columns if col.startswith("agent_")
    )

    kpi_columns = [
        "burnt_area_m2",
        "burnt_area_delta_m2",
        "fire_cost_eur",
        "fire_cost_delta_eur",
        "casualties",
        "casualties_delta",
        "emissions_tonnes",
        "emissions_delta_tonnes",
    ]
    kpi_columns = [col for col in kpi_columns if col in df.columns]

    reward_columns = [col for col in ["reward_moe"] if col in df.columns]

    ordered_columns = (
        state_columns + action_columns + kpi_columns + reward_columns
    )
    remaining_columns = [
        col for col in df.columns if col not in ordered_columns
    ]
    df = df[ordered_columns + remaining_columns]

    df.to_excel(output_path, index=False)
    print(f"Hourly metrics written to {output_path}")


if __name__ == "__main__":
    main()
