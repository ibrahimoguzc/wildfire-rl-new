"""Plot terrain colormaps for the --switch-scenario set with the
--switch-ignition-2 (map-centered) sampling box overlaid.

Mirror of `plot_scenario_maps.py`, but for the SWITCH_SCENARIO_NAMES set
("Palisades copy.json", "Pyrenees.json", "Salamis.json") and the ignition
range used by `--switch-ignition-2`: a map-centered box whose edges sit
IGNITION_V2_MARGIN_RATIO of the map edge length away from each boundary
(default 0.25 -> central 50%x50% of the map).

Usage:
    python3 scripts/plot_switch_scenario_maps.py
    python3 scripts/plot_switch_scenario_maps.py --out /tmp/switch_scenarios.png
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

# Matches IGNITION_V2_MARGIN_RATIO in ppo_runner.py.
IGNITION_V2_MARGIN_RATIO = 0.25

# Matches SWITCH_SCENARIO_NAMES in ppo_runner.py.
SWITCH_SCENARIOS = [
    ("Palisades copy", "Palisades copy.json"),
    ("Pyrenees",       "Pyrenees.json"),
    ("Salamis",        "Salamis.json"),
]


def load_scenario(scenario_file: str) -> dict:
    return json.loads((SCENARIOS_DIR / scenario_file).read_text())


def load_meta_for_namespace(namespace: str) -> dict | None:
    """Locate the .meta file for a terrain namespace (e.g. 'Palisades_2004x1996_5m' -> 'Palisades.meta')."""
    base = namespace.split("_", 1)[0]
    meta_path = TERRAIN_DIR / f"{base}.meta"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def fire_map_bounds(meta: dict) -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) for the fire map."""
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
    min_lat, max_lat, min_lon, max_lon = bounds
    rows, cols = grid_shape
    col = (lon - min_lon) / (max_lon - min_lon) * cols
    row = (max_lat - lat) / (max_lat - min_lat) * rows
    return row, col


def plot_one(ax, label: str, scenario_file: str) -> None:
    scenario = load_scenario(scenario_file)
    namespace = scenario["terrain_inputs"]["file_namespace"]
    colormap = np.load(TERRAIN_DIR / f"{namespace}_terrain_colormap.npy")
    rgb = colormap[..., :3].astype(np.uint8)
    rows, cols = rgb.shape[:2]

    ax.imshow(rgb, origin="upper")
    ax.set_title(
        f"{label}\n{namespace}  ({rows}x{cols} cells, "
        f"{scenario['terrain_inputs']['cell_size']}m)"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Map-centered ignition box (switch-ignition-2): edges at margin and
    # (1 - margin) of each map dimension.
    margin = IGNITION_V2_MARGIN_RATIO
    r0 = int(round(rows * margin))
    r1 = int(round(rows * (1.0 - margin)))
    c0 = int(round(cols * margin))
    c1 = int(round(cols * (1.0 - margin)))
    rect = plt.Rectangle(
        (c0, r0),
        c1 - c0,
        r1 - r0,
        fill=False,
        edgecolor="red",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.plot(
        (c0 + c1) / 2,
        (r0 + r1) / 2,
        marker="+",
        color="red",
        markersize=14,
        mew=2,
        zorder=3,
    )

    # Original scenario ignition center, for reference (small ring).
    meta = load_meta_for_namespace(namespace)
    if meta is not None:
        bounds = fire_map_bounds(meta)
        for center in scenario.get("ignition_centers", []):
            gps = center.get("gps_coords")
            if gps is None:
                continue
            lat, lon = gps
            row, col = gps_to_grid(lat, lon, bounds, (rows, cols))
            if 0 <= row < rows and 0 <= col < cols:
                ax.plot(
                    col,
                    row,
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="white",
                    markeredgewidth=2,
                    markersize=10,
                    zorder=3,
                )

    comb = np.load(TERRAIN_DIR / f"{namespace}_terrain_combustibilities.npy")
    flammable_pct = (comb > 0).mean() * 100
    box_w_cells = c1 - c0
    box_h_cells = r1 - r0
    ax.set_xlabel(
        f"flammable cells: {flammable_pct:.1f}%  |  "
        f"box: {box_h_cells}x{box_w_cells} cells "
        f"({(1 - 2 * margin) * 100:.0f}% x {(1 - 2 * margin) * 100:.0f}% of map)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "switch_scenario_maps.png"),
        help="Output PNG path (default: switch_scenario_maps.png in repo root).",
    )
    args = p.parse_args()

    fig, axes = plt.subplots(1, len(SWITCH_SCENARIOS), figsize=(18, 7))
    for ax, (label, scenario_file) in zip(axes, SWITCH_SCENARIOS):
        plot_one(ax, label, scenario_file)

    fig.suptitle(
        "Switch-scenario terrain colormaps with --switch-ignition-2 sampling box "
        f"(margin {IGNITION_V2_MARGIN_RATIO:.0%}, dashed)  "
        "and original ignition center (white ring, for reference)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
