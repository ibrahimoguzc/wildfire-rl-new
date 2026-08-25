#!/usr/bin/env python3
"""Render ignition-sampling diagnostic maps into ``snapshots/``.

Reconstructs the ``*_switch_ignition_<mode>_n<N>.png`` snapshots: for a
scenario + switch-ignition mode it draws the terrain-typed fire raster,
overlays the ignition candidate box (red), scatters N ignition cells
sampled through the *real* ``_sample_ignition_centers`` path, and marks
the scenario's true ignition centre (yellow star).

Two panels per figure:
  * left  — the whole fire map,
  * right — a zoom onto the candidate box.

Usage:
  python -m examples.wildfire.make_ignition_snapshots \
      --scenario Pyrenees.json Palisades.json --mode 1 --samples 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sosid.environment.terrain import FEATURES_COLOR_TABLE, TerrainTypes
from sosid.model.transform import gps_to_mercator, gps_to_pos, pos_to_index

from examples.wildfire.ppo_runnerv2 import WildfireHourlyEnv, _resolve_scenario

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"

# Legend order (only types actually present are shown).
LEGEND_ORDER = (
    TerrainTypes.WATER,
    TerrainTypes.FIELD,
    TerrainTypes.NEEDLE_LITTER,
    TerrainTypes.PINUS,
    TerrainTypes.GRASSES_WEEDS,
    TerrainTypes.CAREX_FORBS,
    TerrainTypes.PASTURE,
    TerrainTypes.FALLEN_LEAVES,
    TerrainTypes.NON_COMBUSTIBLE,
    TerrainTypes.RESIDENTIAL,
)


def _type_index_raster(feature_data: np.ndarray) -> np.ndarray:
    """Collapse the feature raster to a (rows, cols) terrain-type index map."""
    arr = np.asarray(feature_data)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    # RGBA raster: match each pixel to the nearest entry in the colour table.
    channels = arr.shape[-1]
    flat = arr.reshape(-1, channels).astype(float)
    idx = np.zeros(flat.shape[0], dtype=np.int64)
    best = np.full(flat.shape[0], np.inf)
    for ttype, colour in FEATURES_COLOR_TABLE.items():
        c = np.asarray(colour[:channels], dtype=float)
        d = np.sum((flat - c) ** 2, axis=1)
        hit = d < best
        idx[hit] = int(ttype)
        best[hit] = d[hit]
    return idx.reshape(arr.shape[:2])


def _rgb_image(type_idx: np.ndarray) -> np.ndarray:
    """Build an (rows, cols, 3) uint8 image from the terrain-type index map."""
    rows, cols = type_idx.shape
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    for ttype, colour in FEATURES_COLOR_TABLE.items():
        img[type_idx == int(ttype)] = colour[:3]
    return img


def _sample_cells(env, path, params, n_samples, tl_bounds, grid_desc, shape):
    """Sample N ignitions and round-trip them to fire-raster cells."""
    rows, cols = shape
    default_gps = None
    if params.ignition_centers and params.ignition_centers[0].gps_coords:
        default_gps = tuple(params.ignition_centers[0].gps_coords)
    cells = []
    fallback = 0
    for _ in range(n_samples):
        sampled = env._sample_ignition_centers(path, params)
        gps = tuple(sampled[0].gps_coords)
        if default_gps is not None and gps == default_gps:
            fallback += 1
            continue
        px, py = gps_to_pos((float(gps[0]), float(gps[1])), tl_bounds)
        fr, fc = pos_to_index((float(px), float(py)), grid_desc)
        if 0 <= fr < rows and 0 <= fc < cols:
            cells.append((fr, fc))
    return np.asarray(cells, dtype=np.int64), fallback


def render_scenario(scenario_name: str, mode: int, n_samples: int) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    path = _resolve_scenario(scenario_name)
    env = WildfireHourlyEnv(
        scenario_path=path,
        switch_ignition_mode=mode,
        controlled_agent_count=6,
        state_space="old",
    )
    env.np_random, _ = __import__("gymnasium").utils.seeding.np_random(12345)

    params = env.parameters
    ti = params.terrain_inputs
    feature_data = np.asarray(np.load(ti.features_file, allow_pickle=False))
    type_idx = _type_index_raster(feature_data)
    rows, cols = type_idx.shape
    rgb = _rgb_image(type_idx)

    grid_desc = env._grid_description(params)
    tl_merc = gps_to_mercator(ti.fire_map_coordinates[0])
    tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))

    candidates = env._scenario_ignition_candidates.get(path)
    if candidates is None or len(candidates) == 0:
        raise SystemExit(f"{scenario_name}: no ignition candidates for mode {mode}")
    cr, cc = candidates[:, 0], candidates[:, 1]
    r0, r1, c0, c1 = int(cr.min()), int(cr.max()), int(cc.min()), int(cc.max())

    cells, fallback = _sample_cells(
        env, path, params, n_samples, tl_bounds, grid_desc, (rows, cols)
    )

    true_cell = env._ignition_center_grid_pos(params)

    present = [t for t in LEGEND_ORDER if np.any(type_idx == int(t))]
    box_kw = dict(edgecolor="red", facecolor="none", linewidth=2.0)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 9))

    # --- left: whole map ---
    ax_full.imshow(rgb, origin="upper", interpolation="nearest")
    ax_full.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, **box_kw))
    if true_cell is not None:
        ax_full.plot(true_cell[1], true_cell[0], marker="*", markersize=18,
                     color="gold", markeredgecolor="black", linestyle="none")
    ax_full.set_title(f"{path.stem}  full terrain (red = ignition box)")
    ax_full.set_xlabel("grid col (j)")
    ax_full.set_ylabel("grid row (i)")

    # --- right: zoom on the candidate box ---
    pad_r = max(20, int(0.4 * (r1 - r0)))
    pad_c = max(20, int(0.4 * (c1 - c0)))
    ax_zoom.imshow(rgb, origin="upper", interpolation="nearest")
    ax_zoom.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, **box_kw))
    if len(cells):
        ax_zoom.scatter(cells[:, 1], cells[:, 0], s=4, c="black", alpha=0.35,
                        linewidths=0)
    if true_cell is not None:
        ax_zoom.plot(true_cell[1], true_cell[0], marker="*", markersize=20,
                     color="gold", markeredgecolor="black", linestyle="none")
    ax_zoom.set_xlim(c0 - pad_c, c1 + pad_c)
    ax_zoom.set_ylim(r1 + pad_r, r0 - pad_r)  # inverted for image origin
    ax_zoom.set_title(
        f"switch-ignition-{mode}: {len(cells)} sampled ignition cells")
    ax_zoom.set_xlabel("grid col (j)")
    ax_zoom.set_ylabel("grid row (i)")

    handles = [
        Patch(facecolor="none", edgecolor="red", linewidth=2.0,
              label="ignition box (candidates)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=6, label=f"sampled ignitions (n={len(cells)})"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14,
               label="scenario ignition centre"),
    ]
    handles += [Patch(facecolor=np.array(FEATURES_COLOR_TABLE[t][:3]) / 255,
                      edgecolor="none", label=t.name) for t in present]
    ax_zoom.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOTS_DIR / (
        f"{path.stem.lower()}_switch_ignition_{mode}_n{n_samples}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  fallback to default: {fallback}/{n_samples}   "
          f"candidate cells: {len(candidates):,}")
    print(f"Saved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", nargs="+",
                    default=["Pyrenees.json", "Palisades.json"])
    ap.add_argument("--mode", type=int, default=1, choices=(1, 2, 3, 4))
    ap.add_argument("--samples", type=int, default=3000)
    args = ap.parse_args()
    for scen in args.scenario:
        print(f"=== {scen}  mode {args.mode} ===")
        render_scenario(scen, args.mode, args.samples)


if __name__ == "__main__":
    main()
