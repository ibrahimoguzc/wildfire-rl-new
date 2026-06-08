#!/usr/bin/env python3
"""Detect candidate urban locations from terrain residential regions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import (
    ExtendedTerrainParameters,
    TerrainParameters,
    WildfireParameters,
    WildfireSimulation,
    _TerrainParametersCache,
)

DEFAULT_SCENARIOS = (
    "Palisades copy.json",
    "Salamis.json",
    "Pyrenees.json",
)
OUTPUT_ROUND_DECIMALS = 8
OUTPUT_BOUND_EPS = 1e-7
SCENARIO_LOCATION_KEYS = (
    "ignition_centers",
    "airports",
    "water_sources",
    "protection_locations",
    "urban_locations",
)


def _resolve_scenario(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    candidate = SCENARIOS_DIR / "inputs" / path_or_name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to locate scenario file: {path_or_name}")


def _idx_to_gps(
    row: float,
    col: float,
    rows: int,
    cols: int,
    coordinates: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    # coordinates are expected as [[top_lat, left_lon], [bottom_lat, right_lon]]
    top_lat, left_lon = float(coordinates[0][0]), float(coordinates[0][1])
    bottom_lat, right_lon = float(coordinates[1][0]), float(coordinates[1][1])
    row_frac = 0.0 if rows <= 1 else float(row) / float(rows - 1)
    col_frac = 0.0 if cols <= 1 else float(col) / float(cols - 1)
    lat = top_lat + row_frac * (bottom_lat - top_lat)
    lon = left_lon + col_frac * (right_lon - left_lon)
    return lat, lon


def _clamp_gps_to_bbox(
    lat: float,
    lon: float,
    bbox: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    lat_max, lon_min = float(bbox[0][0]), float(bbox[0][1])
    lat_min, lon_max = float(bbox[1][0]), float(bbox[1][1])
    eps = OUTPUT_BOUND_EPS
    clamped_lat = min(max(float(lat), lat_min + eps), lat_max - eps)
    clamped_lon = min(max(float(lon), lon_min + eps), lon_max - eps)
    return clamped_lat, clamped_lon


def _load_parameters_relaxed(scenario_path: Path) -> WildfireParameters:
    data = json.loads(scenario_path.read_text())

    map_in_map = bool(data.get("map_in_map", False))
    terrain_inputs = data.get("terrain_inputs", {})
    _TerrainParametersCache.metadata = {}
    if map_in_map:
        terrain_parameters = ExtendedTerrainParameters.model_validate(terrain_inputs)
    else:
        terrain_parameters = TerrainParameters.model_validate(terrain_inputs)
    bbox = terrain_parameters.coordinates

    # Keep validation robust even when existing scenario location fields contain
    # tiny map-edge rounding overshoots from prior exports.
    for key in SCENARIO_LOCATION_KEYS:
        items = data.get(key)
        if items is None:
            if key in {"protection_locations", "urban_locations"}:
                data[key] = []
            continue
        normalized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            gps = candidate.get("gps_coords")
            if gps is not None and len(gps) == 2:
                lat, lon = _clamp_gps_to_bbox(float(gps[0]), float(gps[1]), bbox)
                candidate["gps_coords"] = [lat, lon]
            normalized.append(candidate)
        data[key] = normalized

    _TerrainParametersCache.metadata = {}
    return WildfireParameters.model_validate(data)


def _build_location(
    *,
    row: float,
    col: float,
    rows: int,
    cols: int,
    coordinates: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
    radius_y_m: float,
    radius_x_m: float,
    angle_deg: float = 0.0,
) -> dict[str, Any]:
    lat, lon = _idx_to_gps(row, col, rows, cols, coordinates)
    lat, lon = _clamp_gps_to_bbox(lat, lon, coordinates)
    return {
        "gps_coords": [
            round(lat, OUTPUT_ROUND_DECIMALS),
            round(lon, OUTPUT_ROUND_DECIMALS),
        ],
        "radius": [round(max(1.0, radius_y_m), 1), round(max(1.0, radius_x_m), 1)],
        "angle": int(round(float(angle_deg))) % 180,
    }


def _component_ellipse_params(
    *,
    full_rows: np.ndarray,
    full_cols: np.ndarray,
    row_center: float,
    col_center: float,
    cell_size_m: float,
) -> tuple[float, float, float]:
    """Fit an oriented ellipse that covers a connected urban component."""
    if full_rows.size == 0 or full_cols.size == 0:
        return (1.0, 1.0, 0.0)

    # Work in metric coordinates around component center.
    x_m = (full_cols.astype(np.float64) - float(col_center)) * float(cell_size_m)
    y_m = (full_rows.astype(np.float64) - float(row_center)) * float(cell_size_m)
    points = np.column_stack((x_m, y_m))

    if points.shape[0] < 2:
        return (1.0, 1.0, 0.0)

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return (1.0, 1.0, 0.0)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    major_vec = evecs[:, order[0]]
    minor_vec = evecs[:, order[1]]

    proj_major = points @ major_vec
    proj_minor = points @ minor_vec

    radius_major = max(
        1.0,
        0.5 * (float(np.max(proj_major)) - float(np.min(proj_major))),
    )
    radius_minor = max(
        1.0,
        0.5 * (float(np.max(proj_minor)) - float(np.min(proj_minor))),
    )

    # Angle in degrees of major axis in map x/y frame.
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    angle_deg = angle_deg % 180.0
    return (radius_major, radius_minor, angle_deg)


def _estimate_cover_points(
    *,
    area_cells: int,
    cell_size_m: float,
    coverage_radius_m: float,
    max_points_per_component: int,
) -> int:
    if max_points_per_component <= 1:
        return 1
    if coverage_radius_m <= 0.0:
        return 1
    area_m2 = float(area_cells) * (cell_size_m**2)
    # Hex-like packing efficiency for overlapping circular influence regions.
    effective_cover_area = max(1.0, math.pi * (coverage_radius_m**2) * 0.72)
    estimated = int(math.ceil(area_m2 / effective_cover_area))
    return max(1, min(estimated, max_points_per_component))


def _pick_cover_points(
    *,
    full_rows: np.ndarray,
    full_cols: np.ndarray,
    core_row: float,
    core_col: float,
    target_count: int,
    max_candidate_cells: int,
) -> np.ndarray:
    if target_count <= 0 or full_rows.size == 0 or full_cols.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    coords = np.column_stack((full_rows, full_cols)).astype(np.float64)
    if coords.shape[0] > max_candidate_cells:
        step = int(math.ceil(coords.shape[0] / max_candidate_cells))
        coords = coords[::step]

    selected: list[np.ndarray] = [np.array([core_row, core_col], dtype=np.float64)]
    min_dist2 = np.sum((coords - selected[0]) ** 2, axis=1)

    for _ in range(1, target_count):
        next_idx = int(np.argmax(min_dist2))
        next_point = coords[next_idx]
        selected.append(next_point)
        dist2 = np.sum((coords - next_point) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        if float(np.max(min_dist2)) < 1.0:
            break

    return np.asarray(selected, dtype=np.float64)


def _select_spread_cores(
    cores: list[dict[str, Any]], max_count: int
) -> list[dict[str, Any]]:
    """Pick `max_count` cores spread across the map via farthest-point sampling.

    Seeds with the largest-area core, then greedily adds the core most distant
    from any already-selected one. Operates in GPS space — adequate when the
    map bbox is small enough that lat/lon distortion is negligible.
    """
    if max_count <= 0 or len(cores) <= max_count:
        return cores
    points = np.array(
        [[float(c["gps_coords"][0]), float(c["gps_coords"][1])] for c in cores]
    )
    selected_idx = [0]
    min_dist2 = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(1, max_count):
        next_idx = int(np.argmax(min_dist2))
        if min_dist2[next_idx] <= 0.0:
            break
        selected_idx.append(next_idx)
        dist2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return [cores[i] for i in selected_idx]


def _extract_candidates(
    scenario_path: Path,
    *,
    min_cells: int,
    core_min_cells: int,
    max_locations: int,
    max_cores: int,
    strategy: str,
    coverage_radius_m: float,
    max_points_per_component: int,
    max_candidate_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = _load_parameters_relaxed(scenario_path)
    simulation = WildfireSimulation(parameters=parameters, seed=0)
    terrain = simulation.environment.terrain
    urban_mask = terrain.urban_areas
    if urban_mask is None or not np.any(urban_mask):
        return [], []

    labels, num_labels = ndimage.label(urban_mask.astype(np.uint8))
    if num_labels == 0:
        return [], []

    rows, cols = urban_mask.shape
    cell_size = float(parameters.cell_size)
    coordinates = parameters.terrain_inputs.coordinates

    components: list[dict[str, Any]] = []
    cores_by_area: list[tuple[int, dict[str, Any]]] = []
    slices = ndimage.find_objects(labels)
    for label_idx, obj_slice in enumerate(slices, start=1):
        if obj_slice is None:
            continue
        component = labels[obj_slice] == label_idx
        area_cells = int(np.count_nonzero(component))

        comp_rows, comp_cols = np.where(component)
        row_offset, col_offset = obj_slice[0].start, obj_slice[1].start
        full_rows = comp_rows + row_offset
        full_cols = comp_cols + col_offset

        row_center = float(np.mean(full_rows))
        col_center = float(np.mean(full_cols))

        radius_x_m, radius_y_m, angle_deg = _component_ellipse_params(
            full_rows=full_rows,
            full_cols=full_cols,
            row_center=row_center,
            col_center=col_center,
            cell_size_m=cell_size,
        )

        # Urban core: the point with largest distance to non-urban cells.
        dt = ndimage.distance_transform_edt(component)
        core_local_idx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        core_row = float(core_local_idx[0] + row_offset)
        core_col = float(core_local_idx[1] + col_offset)
        core_radius_m = float(max(1.0, dt[core_local_idx] * cell_size))

        center_location = _build_location(
            row=row_center,
            col=col_center,
            rows=rows,
            cols=cols,
            coordinates=coordinates,
            radius_y_m=radius_y_m,
            radius_x_m=radius_x_m,
            angle_deg=angle_deg,
        )
        core_location = _build_location(
            row=core_row,
            col=core_col,
            rows=rows,
            cols=cols,
            coordinates=coordinates,
            radius_y_m=core_radius_m,
            radius_x_m=core_radius_m,
        )
        if area_cells >= core_min_cells:
            cores_by_area.append((area_cells, core_location))

        if area_cells < min_cells:
            continue

        if strategy == "component-centers":
            locations = [center_location]
        elif strategy == "cover":
            target_count = _estimate_cover_points(
                area_cells=area_cells,
                cell_size_m=cell_size,
                coverage_radius_m=coverage_radius_m,
                max_points_per_component=max_points_per_component,
            )
            selected_points = _pick_cover_points(
                full_rows=full_rows.astype(np.float64),
                full_cols=full_cols.astype(np.float64),
                core_row=core_row,
                core_col=core_col,
                target_count=target_count,
                max_candidate_cells=max_candidate_cells,
            )
            locations = [
                _build_location(
                    row=float(pt[0]),
                    col=float(pt[1]),
                    rows=rows,
                    cols=cols,
                    coordinates=coordinates,
                    radius_y_m=coverage_radius_m,
                    radius_x_m=coverage_radius_m,
                    angle_deg=angle_deg,
                )
                for pt in selected_points
            ]
        else:
            raise ValueError(
                "strategy must be one of: component-centers, cover"
            )

        components.append(
            {
                "locations": locations,
                "core": core_location,
                "_area_cells": area_cells,
            }
        )

    components.sort(key=lambda x: int(x["_area_cells"]), reverse=True)
    cores_by_area.sort(key=lambda x: x[0], reverse=True)

    flattened_locations: list[dict[str, Any]] = []
    cores: list[dict[str, Any]] = [core for _, core in cores_by_area]
    if max_cores > 0:
        cores = _select_spread_cores(cores, max_cores)
    for comp in components:
        for location in comp["locations"]:
            if len(flattened_locations) >= max_locations:
                break
            flattened_locations.append(location)
        if len(flattened_locations) >= max_locations:
            break

    return flattened_locations, cores


def _location_in_bbox(
    location: dict[str, Any],
    bbox: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    gps = location.get("gps_coords")
    if gps is None or len(gps) != 2:
        return False
    lat = float(gps[0])
    lon = float(gps[1])
    lat_max, lon_min = float(bbox[0][0]), float(bbox[0][1])
    lat_min, lon_max = float(bbox[1][0]), float(bbox[1][1])
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def _assert_within_bounds(
    *,
    locations: list[dict[str, Any]],
    bbox: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
    label: str,
) -> None:
    out_of_bounds = [
        (idx, location.get("gps_coords"))
        for idx, location in enumerate(locations)
        if not _location_in_bbox(location, bbox)
    ]
    if out_of_bounds:
        sample = ", ".join(
            f"{idx}:{coords}" for idx, coords in out_of_bounds[:5]
        )
        raise ValueError(
            f"{label} contains {len(out_of_bounds)} out-of-bounds points "
            f"(sample {sample})."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Determine candidate urban_locations from residential terrain "
            "regions for each scenario."
        )
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario files (name in inputs/ or absolute path).",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=250,
        help="Minimum connected residential cells to keep as a candidate.",
    )
    parser.add_argument(
        "--core-min-cells",
        type=int,
        default=1,
        help=(
            "Minimum connected residential cells to keep as an urban core "
            "(defaults low to include many cores)."
        ),
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=20,
        help="Maximum candidate urban locations to output per scenario.",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=0,
        help=(
            "If > 0, keep only this many urban cores per scenario, selected "
            "via farthest-point sampling so they span the map."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("component-centers", "cover"),
        default="component-centers",
        help=(
            "component-centers: one center per connected urban component; "
            "cover: urban core + spread points to roughly cover each component."
        ),
    )
    parser.add_argument(
        "--coverage-radius-m",
        type=float,
        default=900.0,
        help="Influence radius used per output point in --strategy cover.",
    )
    parser.add_argument(
        "--max-points-per-component",
        type=int,
        default=8,
        help="Maximum number of coverage points sampled per urban component.",
    )
    parser.add_argument(
        "--max-candidate-cells",
        type=int,
        default=12_000,
        help="Upper bound on sampled cells per component when selecting cover points.",
    )
    parser.add_argument(
        "--cores-output",
        help=(
            "Optional JSON output file for per-scenario urban core points "
            "(one core per connected component)."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file for all scenarios.",
    )
    args = parser.parse_args()

    if args.min_cells < 1:
        raise ValueError("--min-cells must be >= 1")
    if args.core_min_cells < 1:
        raise ValueError("--core-min-cells must be >= 1")
    if args.max_locations < 1:
        raise ValueError("--max-locations must be >= 1")
    if args.coverage_radius_m <= 0.0:
        raise ValueError("--coverage-radius-m must be > 0")
    if args.max_points_per_component < 1:
        raise ValueError("--max-points-per-component must be >= 1")
    if args.max_candidate_cells < 1:
        raise ValueError("--max-candidate-cells must be >= 1")

    scenario_paths = [_resolve_scenario(item) for item in args.scenarios]
    result: dict[str, Any] = {}
    cores_result: dict[str, Any] = {}
    for path in scenario_paths:
        candidates, cores = _extract_candidates(
            path,
            min_cells=args.min_cells,
            core_min_cells=args.core_min_cells,
            max_locations=args.max_locations,
            max_cores=args.max_cores,
            strategy=args.strategy,
            coverage_radius_m=args.coverage_radius_m,
            max_points_per_component=args.max_points_per_component,
            max_candidate_cells=args.max_candidate_cells,
        )

        _TerrainParametersCache.metadata = {}
        terrain_data = json.loads(path.read_text()).get("terrain_inputs", {})
        if bool(json.loads(path.read_text()).get("map_in_map", False)):
            bbox = ExtendedTerrainParameters.model_validate(terrain_data).coordinates
        else:
            bbox = TerrainParameters.model_validate(terrain_data).coordinates
        _assert_within_bounds(
            locations=candidates,
            bbox=bbox,
            label=f"{path.name} candidate urban_locations",
        )
        _assert_within_bounds(
            locations=cores,
            bbox=bbox,
            label=f"{path.name} urban cores",
        )

        result[path.name] = candidates
        cores_result[path.name] = cores
        print(
            f"\n{path.name}: {len(candidates)} candidate urban_locations "
            f"(cores found: {len(cores)})"
        )
        print(json.dumps(candidates, indent=2))
        print("Urban cores:")
        print(json.dumps(cores, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {output_path}")

    if args.cores_output:
        cores_path = Path(args.cores_output)
        cores_path.parent.mkdir(parents=True, exist_ok=True)
        cores_path.write_text(json.dumps(cores_result, indent=2))
        print(f"Saved cores to {cores_path}")


if __name__ == "__main__":
    main()
