#!/usr/bin/env python
"""Three scenario terrain maps side by side with the --switch-ignition-4
sampling area marked.

Two things are drawn per map, deliberately distinct:
  * the dashed rectangle: the nominal 400x400-cell candidate box centred on the
    scenario's true ignition (IGNITION_BOX_HALF_SIZE_V4 = 200);
  * the shaded cells: the ACTUAL sampling area - the env's precomputed
    ignition candidates, i.e. the box minus water/non-ignitable terrain and
    minus the 150 m urban keep-out (IGNITION_URBAN_BUFFER_M_V4). This is what
    training really samples from, taken from WildfireHourlyEnv itself rather
    than re-derived here.

Usage:
    python scripts/plot_scenario_maps_sw4.py --out final_plots/scenario_maps_sw4.png
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import gymnasium  # noqa: E402
from sosid.environment.terrain import FEATURES_COLOR_TABLE, TerrainTypes  # noqa: E402
from sosid.model.transform import gps_to_mercator, gps_to_pos, pos_to_index  # noqa: E402
from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    WildfireHourlyEnv, _resolve_scenario,
    IGNITION_BOX_HALF_SIZE_V4, IGNITION_URBAN_BUFFER_M_V4,
)

_MAX = max(int(t) for t in TerrainTypes)
_COLORS = [(0.5, 0.5, 0.5, 1.0)] * (_MAX + 1)
for t, rgba in FEATURES_COLOR_TABLE.items():
    _COLORS[int(t)] = tuple(c / 255.0 for c in rgba)
CMAP = ListedColormap(_COLORS)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BOX_HUE, CAND_HUE = "#d03b3b", "#d03b3b"


def panel(ax, scenario_name: str, prefix: str = "") -> None:
    path = _resolve_scenario(scenario_name)
    env = WildfireHourlyEnv(scenario_path=path, switch_ignition_mode=4,
                            controlled_agent_count=1, state_space="old")
    env.np_random, _ = gymnasium.utils.seeding.np_random(0)
    params = env.parameters
    ti = params.terrain_inputs
    feat = np.asarray(np.load(ti.features_file, allow_pickle=False))
    if feat.ndim == 3:
        feat = np.argmax(feat, axis=-1)
    rows, cols = feat.shape[:2]
    grid = env._grid_description(params)
    tl = gps_to_mercator(ti.fire_map_coordinates[0])
    tlb = (float(tl[1]), float(tl[0]))

    ax.imshow(feat, cmap=CMAP, origin="upper", interpolation="nearest",
              vmin=0, vmax=_MAX)

    # true ignition
    gps = params.ignition_centers[0].gps_coords
    r0, c0 = pos_to_index(tuple(map(float, gps_to_pos(tuple(map(float, gps)), tlb))), grid)
    h = IGNITION_BOX_HALF_SIZE_V4
    ax.add_patch(Rectangle((c0 - h, r0 - h), 2 * h, 2 * h, fill=False,
                           edgecolor=BOX_HUE, linestyle=(0, (5, 3)),
                           linewidth=1.6, zorder=4))
    # actual candidate cells (box minus water + 150 m urban keep-out)
    cand = env._scenario_ignition_candidates.get(path)
    mask = np.zeros((rows, cols), dtype=bool)
    n_cand = 0
    if cand is not None and len(cand):
        cr = np.clip(cand[:, 0], 0, rows - 1)
        cc = np.clip(cand[:, 1], 0, cols - 1)
        mask[cr, cc] = True
        n_cand = len(cand)
    overlay = np.zeros((rows, cols, 4))
    overlay[mask] = matplotlib.colors.to_rgba(CAND_HUE, alpha=0.38)
    ax.imshow(overlay, origin="upper", interpolation="nearest", zorder=3)


    label = scenario_name.replace("5sp5ev.json", "")
    ax.set_title(f"{prefix}{label}", fontsize=17, color=INK, loc="center", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+",
                    default=["Palisades5sp5ev.json", "Pyrenees5sp5ev.json",
                             "Salamis5sp5ev.json"])
    ap.add_argument("--out", default="final_plots/scenario_maps_sw4.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.scenarios), figsize=(6.2 * len(args.scenarios), 6.6))
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, sc) in enumerate(zip(np.atleast_1d(axes), args.scenarios)):
        panel(ax, sc, prefix=f"{chr(ord('a') + i)}) ")
    handles = [
        Patch(facecolor="none", edgecolor=BOX_HUE, linestyle="--",
              label=f"switch-ignition-4 box ({2*IGNITION_BOX_HALF_SIZE_V4}×{2*IGNITION_BOX_HALF_SIZE_V4} cells)"),
        Patch(facecolor=CAND_HUE, alpha=0.38,
              label=f"ignitable area (box − water − {IGNITION_URBAN_BUFFER_M_V4:.0f} m urban keep-out)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=14, bbox_to_anchor=(0.5, -0.005))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
