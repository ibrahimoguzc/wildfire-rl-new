#!/usr/bin/env python
"""Sample switch-ignition points and scatter them on the terrain map.

Builds a real WildfireHourlyEnv with the requested switch-ignition mode, draws
many ignition centres through the actual ``_sample_ignition_centers`` path
(the same code training uses), converts each to a grid cell, and plots them
over the scenario's terrain-type raster. Shows the candidate (ignition) box,
the scenario's true ignition centre, and the sampled cloud -- a full-map
context panel plus a zoom on the box.

Usage:
  ~/.conda/envs/rl-env/bin/python scripts/visualize_switch_ignition.py \
      --scenario Pyrenees6p.json --mode 1 --samples 3000
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

# Terrain-type colormap (index == TerrainTypes value).
_MAX_TYPE = max(int(t) for t in TerrainTypes)
_COLORS = [(0.5, 0.5, 0.5, 1.0)] * (_MAX_TYPE + 1)
for _t, _rgba in FEATURES_COLOR_TABLE.items():
    _COLORS[int(_t)] = tuple(c / 255.0 for c in _rgba)
TERRAIN_CMAP = ListedColormap(_COLORS)


def _grid_index(gps, tl_bounds, grid_desc):
    px, py = gps_to_pos((float(gps[0]), float(gps[1])), tl_bounds)
    return pos_to_index((float(px), float(py)), grid_desc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter sampled switch-ignition points on the terrain map."
    )
    parser.add_argument("--scenario", default="Pyrenees6p.json")
    parser.add_argument("--mode", type=int, default=1, choices=(1, 2),
                        help="switch-ignition mode (1 or 2)")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    path = _resolve_scenario(args.scenario)
    env = WildfireHourlyEnv(
        scenario_path=path,
        switch_ignition_mode=args.mode,
        controlled_agent_count=1,
        state_space="old",
    )
    env.np_random, _ = gymnasium.utils.seeding.np_random(args.seed)

    params = env.parameters
    ti = params.terrain_inputs
    feature_data = np.asarray(np.load(ti.features_file, allow_pickle=False))
    if feature_data.ndim == 3:  # collapse a one-hot/type axis to an index map
        feature_data = np.argmax(feature_data, axis=-1)
    rows, cols = feature_data.shape[:2]
    grid_desc = env._grid_description(params)
    tl_merc = gps_to_mercator(ti.fire_map_coordinates[0])
    tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))

    candidates = env._scenario_ignition_candidates.get(path)
    if candidates is None or len(candidates) == 0:
        print("No ignition candidates; switch-ignition would fall back to "
              "the scenario default. Nothing to scatter.")
        return

    # True scenario ignition centre.
    center_rc = None
    if params.ignition_centers and params.ignition_centers[0].gps_coords:
        center_rc = _grid_index(
            params.ignition_centers[0].gps_coords, tl_bounds, grid_desc
        )

    # Sample ignition centres through the real path.
    sr, sc, fallback, oob = [], [], 0, 0
    default_gps = (
        tuple(params.ignition_centers[0].gps_coords)
        if params.ignition_centers and params.ignition_centers[0].gps_coords
        else None
    )
    for _ in range(args.samples):
        sampled = env._sample_ignition_centers(path, params)
        gps = tuple(sampled[0].gps_coords)
        if default_gps is not None and gps == default_gps:
            fallback += 1
            continue
        r, c = _grid_index(gps, tl_bounds, grid_desc)
        if not (0 <= r < rows and 0 <= c < cols):
            oob += 1
            continue
        sr.append(r)
        sc.append(c)
    sr = np.asarray(sr)
    sc = np.asarray(sc)

    cand_r = candidates[:, 0]
    cand_c = candidates[:, 1]
    box = (int(cand_r.min()), int(cand_r.max()),
           int(cand_c.min()), int(cand_c.max()))
    print(f"scenario={path.name} mode={args.mode} samples={args.samples}")
    print(f"  candidate box rows[{box[0]},{box[1]}] cols[{box[2]},{box[3]}] "
          f"({len(candidates)} cells)")
    print(f"  plotted {sr.size}  fallback {fallback}  out-of-grid {oob}")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 7.5))

    # --- full-map context ---
    ax_full.imshow(feature_data, cmap=TERRAIN_CMAP, vmin=-0.5,
                   vmax=_MAX_TYPE + 0.5, origin="upper", interpolation="nearest")
    margin = max(40, int(0.25 * max(box[1] - box[0], box[3] - box[2])))
    ax_full.add_patch(Rectangle(
        (box[2] - margin, box[0] - margin),
        (box[3] - box[2]) + 2 * margin, (box[1] - box[0]) + 2 * margin,
        fill=False, ec="red", lw=1.5))
    ax_full.set_title(f"{path.name}  full terrain (red = ignition box)")
    ax_full.set_xlabel("grid col (j)")
    ax_full.set_ylabel("grid row (i)")

    # --- zoom on the ignition box ---
    i0, i1 = max(0, box[0] - margin), min(rows, box[1] + margin + 1)
    j0, j1 = max(0, box[2] - margin), min(cols, box[3] + margin + 1)
    ax_zoom.imshow(feature_data, cmap=TERRAIN_CMAP, vmin=-0.5,
                   vmax=_MAX_TYPE + 0.5, origin="upper", interpolation="nearest")
    ax_zoom.add_patch(Rectangle(
        (box[2], box[0]), box[3] - box[2], box[1] - box[0],
        fill=False, ec="red", lw=1.5, label="ignition box (candidates)"))
    if sr.size:
        ax_zoom.scatter(sc, sr, s=8, c="black", alpha=0.35, linewidths=0,
                        label=f"sampled ignitions (n={sr.size})")
    if center_rc is not None:
        ax_zoom.plot(center_rc[1], center_rc[0], marker="*", ms=18,
                     mfc="yellow", mec="black", mew=1.0,
                     label="scenario ignition centre")
    ax_zoom.set_xlim(j0, j1)
    ax_zoom.set_ylim(i1, i0)  # origin upper
    ax_zoom.set_title(
        f"switch-ignition-{args.mode}: {sr.size} sampled ignition cells")
    ax_zoom.set_xlabel("grid col (j)")
    ax_zoom.set_ylabel("grid row (i)")

    # Legend: markers + the terrain types actually visible in the zoom.
    present = np.unique(feature_data[i0:i1, j0:j1])
    handles = ax_zoom.get_legend_handles_labels()[0]
    handles += [
        Patch(fc=TERRAIN_CMAP(int(t)), label=TerrainTypes(int(t)).name)
        for t in present if 0 <= int(t) <= _MAX_TYPE
    ]
    ax_zoom.legend(handles=handles, loc="upper right", fontsize=7,
                   framealpha=0.9)

    fig.tight_layout()
    stem = path.stem.split()[0].lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{stem}_switch_ignition_{args.mode}_n{sr.size}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
