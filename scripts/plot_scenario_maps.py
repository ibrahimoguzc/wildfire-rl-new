"""Plot terrain colormaps for Palisades / Pyrenees / Salamis side-by-side
with their ignition centers and the random-ignition sampling box.

Usage:
    python3 scripts/plot_scenario_maps.py
    python3 scripts/plot_scenario_maps.py --out /tmp/scenarios.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "examples" / "wildfire" / "data" / "scenarios" / "inputs"
TERRAIN_DIR = REPO_ROOT / "examples" / "wildfire" / "data" / "terrain"

# Half-side of the ignition sampling box in fire-map cells (matches
# IGNITION_BOX_HALF_SIZE in ppo_runner.py).
IGNITION_BOX_HALF_SIZE = 50

SCENARIOS = [
    ("Palisades", "Palisades.json"),
    ("Pyrenees",  "Pyrenees.json"),
    ("Salamis",   "Salamis.json"),
]


def load_meta(label: str) -> dict:
    return json.loads((TERRAIN_DIR / f"{label}.meta").read_text())


def load_scenario(scenario_file: str) -> dict:
    return json.loads((SCENARIOS_DIR / scenario_file).read_text())


def fire_map_bounds(meta: dict) -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) for the fire map.

    Matches simulation.py:410-413 which sorts the meta `coordinates` to
    produce [[max_lat, min_lon], [min_lat, max_lon]].
    """
    coords = np.array(meta["coordinates"])
    lats = sorted(coords[:, 0])
    lons = sorted(coords[:, 1])
    return lats[0], lats[1], lons[0], lons[1]


def gps_to_grid(
    lat: float,
    lon: float,
    bounds: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> tuple[float, float]:
    """Linear lat/lon -> (row, col). Accurate to a few cells at this scale."""
    min_lat, max_lat, min_lon, max_lon = bounds
    rows, cols = grid_shape
    col = (lon - min_lon) / (max_lon - min_lon) * cols
    row = (max_lat - lat) / (max_lat - min_lat) * rows
    return row, col


def plot_one(ax, label: str, scenario_file: str) -> None:
    scenario = load_scenario(scenario_file)
    meta = load_meta(label)

    namespace = scenario["terrain_inputs"]["file_namespace"]
    colormap = np.load(TERRAIN_DIR / f"{namespace}_terrain_colormap.npy")
    rgb = colormap[..., :3].astype(np.uint8)

    bounds = fire_map_bounds(meta)
    rows, cols = rgb.shape[:2]

    ax.imshow(rgb, origin="upper")
    ax.set_title(
        f"{label}\n{namespace}  ({rows}×{cols} cells, {scenario['terrain_inputs']['cell_size']}m)"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    centers = scenario.get("ignition_centers", [])
    for center in centers:
        gps = center.get("gps_coords")
        if gps is None:
            continue
        lat, lon = gps
        row, col = gps_to_grid(lat, lon, bounds, (rows, cols))
        if not (0 <= row < rows and 0 <= col < cols):
            ax.set_xlabel(f"ignition off-grid: ({lat:.4f}, {lon:.4f})")
            continue
        ax.plot(col, row, marker="x", color="red", markersize=14, mew=3, zorder=3)
        rect = plt.Rectangle(
            (col - IGNITION_BOX_HALF_SIZE, row - IGNITION_BOX_HALF_SIZE),
            2 * IGNITION_BOX_HALF_SIZE,
            2 * IGNITION_BOX_HALF_SIZE,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)

    comb = np.load(TERRAIN_DIR / f"{namespace}_terrain_combustibilities.npy")
    flammable_pct = (comb > 0).mean() * 100
    ax.set_xlabel(f"flammable cells: {flammable_pct:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "scenario_maps.png"),
        help="Output PNG path (default: scenario_maps.png in repo root).",
    )
    args = p.parse_args()

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 7))
    for ax, (label, scenario_file) in zip(axes, SCENARIOS):
        plot_one(ax, label, scenario_file)

    fig.suptitle(
        "Terrain colormap with ignition center (×) and 100×100-cell sampling box (dashed)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
