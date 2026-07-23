#!/usr/bin/env python
"""Side-by-side switch-ignition candidate areas: Palisades (left) vs Pyrenees
(right), one figure per mode.

Reuses the real WildfireHourlyEnv sampling path (ppo_runnerv2) so the plotted
candidate cells and sampled ignitions match exactly what training would seed.

Usage:
  ~/.conda/envs/rl-env/bin/python scripts/plot_switch_ignition_pairs.py \
      --samples 3000
Produces:
  examples/wildfire/snapshots/switch_ignition_1_palisades_pyrenees.png
  examples/wildfire/snapshots/switch_ignition_2_palisades_pyrenees.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import gymnasium  # noqa: E402

from sosid.environment.terrain import (  # noqa: E402
    FEATURES_COLOR_TABLE,
    TerrainTypes,
)
from sosid.model.transform import (  # noqa: E402
    gps_to_mercator,
    gps_to_pos,
    pos_to_index,
)
from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    WildfireHourlyEnv,
    _resolve_scenario,
)

DEFAULT_OUTPUT_DIR = REPO / "examples" / "wildfire" / "snapshots"
SCENARIOS = ("Palisades.json", "Pyrenees.json")  # left, right

_MAX_TYPE = max(int(t) for t in TerrainTypes)
_COLORS = [(0.5, 0.5, 0.5, 1.0)] * (_MAX_TYPE + 1)
for _t, _rgba in FEATURES_COLOR_TABLE.items():
    _COLORS[int(_t)] = tuple(c / 255.0 for c in _rgba)
TERRAIN_CMAP = ListedColormap(_COLORS)


def _grid_index(gps, tl_bounds, grid_desc):
    px, py = gps_to_pos((float(gps[0]), float(gps[1])), tl_bounds)
    return pos_to_index((float(px), float(py)), grid_desc)


def collect(scenario: str, mode: int, samples: int, seed: int):
    """Return everything needed to draw one scenario panel for a mode."""
    path = _resolve_scenario(scenario)
    env = WildfireHourlyEnv(
        scenario_path=path,
        switch_ignition_mode=mode,
        controlled_agent_count=1,
        state_space="old",
    )
    env.np_random, _ = gymnasium.utils.seeding.np_random(seed)

    params = env.parameters
    ti = params.terrain_inputs
    feature_data = np.asarray(np.load(ti.features_file, allow_pickle=False))
    if feature_data.ndim == 3:
        feature_data = np.argmax(feature_data, axis=-1)
    rows, cols = feature_data.shape[:2]
    grid_desc = env._grid_description(params)
    tl_merc = gps_to_mercator(ti.fire_map_coordinates[0])
    tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))

    candidates = env._scenario_ignition_candidates.get(path)
    if candidates is None or len(candidates) == 0:
        raise RuntimeError(f"{scenario} mode {mode}: no candidates (would fall back)")

    # Real ground cell size (mercator) -> km^2 per cell.
    gs = ti.grid_shape
    md = ti.meta_data["mercator_dimensions"]
    cell_y, cell_x = md[1] / gs[0], md[0] / gs[1]
    area_km2 = len(candidates) * cell_y * cell_x / 1e6

    center_rc = None
    if params.ignition_centers and params.ignition_centers[0].gps_coords:
        center_rc = _grid_index(
            params.ignition_centers[0].gps_coords, tl_bounds, grid_desc
        )

    default_gps = (
        tuple(params.ignition_centers[0].gps_coords)
        if params.ignition_centers and params.ignition_centers[0].gps_coords
        else None
    )
    sr, sc = [], []
    for _ in range(samples):
        sampled = env._sample_ignition_centers(path, params)
        gps = tuple(sampled[0].gps_coords)
        if default_gps is not None and gps == default_gps:
            continue
        r, c = _grid_index(gps, tl_bounds, grid_desc)
        if 0 <= r < rows and 0 <= c < cols:
            sr.append(r)
            sc.append(c)

    box = (
        int(candidates[:, 0].min()), int(candidates[:, 0].max()),
        int(candidates[:, 1].min()), int(candidates[:, 1].max()),
    )
    return {
        "name": path.stem.split()[0],
        "feature_data": feature_data,
        "rows": rows, "cols": cols,
        "box": box,
        "center_rc": center_rc,
        "sr": np.asarray(sr), "sc": np.asarray(sc),
        "n_cand": len(candidates),
        "area_km2": area_km2,
    }


def draw(ax, d, mode: int, full_map: bool = False):
    fd = d["feature_data"]
    box = d["box"]
    rows, cols = d["rows"], d["cols"]
    if full_map:
        i0, i1, j0, j1 = 0, rows, 0, cols
        box_lw = 2.5
    else:
        extent = max(box[1] - box[0], box[3] - box[2])
        margin = max(40, int(0.25 * extent))
        i0, i1 = max(0, box[0] - margin), min(rows, box[1] + margin + 1)
        j0, j1 = max(0, box[2] - margin), min(cols, box[3] + margin + 1)
        box_lw = 1.8

    ax.imshow(fd, cmap=TERRAIN_CMAP, vmin=-0.5, vmax=_MAX_TYPE + 0.5,
              origin="upper", interpolation="nearest")
    ax.add_patch(Rectangle(
        (box[2], box[0]), box[3] - box[2], box[1] - box[0],
        fill=False, ec="red", lw=box_lw, label="candidate box"))
    if d["sr"].size:
        ax.scatter(d["sc"], d["sr"], s=6, c="black", alpha=0.30,
                   linewidths=0, label=f"sampled ignitions (n={d['sr'].size})")
    if d["center_rc"] is not None:
        ax.plot(d["center_rc"][1], d["center_rc"][0], marker="*", ms=16,
                mfc="yellow", mec="black", mew=1.0, label="true ignition")
    ax.set_xlim(j0, j1)
    ax.set_ylim(i1, i0)  # origin upper
    ax.set_title(
        f"{d['name']} — switch-ignition-{mode}\n"
        f"{d['n_cand']:,} ignitable cells ≈ {d['area_km2']:.2f} km²")
    ax.set_xlabel("grid col (j)")
    ax.set_ylabel("grid row (i)")

    present = np.unique(fd[i0:i1, j0:j1])
    handles = ax.get_legend_handles_labels()[0]
    handles += [
        Patch(fc=TERRAIN_CMAP(int(t)), label=TerrainTypes(int(t)).name)
        for t in present if 0 <= int(t) <= _MAX_TYPE
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2],
                    choices=(1, 2))
    ap.add_argument("--full-map", action="store_true",
                    help="show the whole map (box positioned in context) "
                         "instead of zooming on the box")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_fullmap" if args.full_map else ""
    for mode in args.modes:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
        for ax, scen in zip(axes, SCENARIOS):
            d = collect(scen, mode, args.samples, args.seed)
            print(f"mode {mode} {d['name']}: {d['n_cand']:,} cells "
                  f"≈ {d['area_km2']:.2f} km², plotted {d['sr'].size}")
            draw(ax, d, mode, full_map=args.full_map)
        view = "whole map" if args.full_map else "zoom on box"
        fig.suptitle(
            f"switch-ignition-{mode} candidate area ({view}) — "
            f"Palisades (left) vs Pyrenees (right)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = (args.output_dir
               / f"switch_ignition_{mode}_palisades_pyrenees{suffix}.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
