#!/usr/bin/env python3
"""Evaluation boxplots for the theater-level study.

Three panels (Palisades, Pyrenees, Salamis) of MoE distributions from the
7-way evaluation (500 simulations per arm per map), restricted to the three
training instances shown in training_curves_grid -- same colors -- plus the
Random Tactics and Fixed Tactics baselines in neutral gray.

Writes final_plots/eval_boxplots_theater.{pdf,png}.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "examples/wildfire/data/scenarios/outputs/eval_7way_3maps_754407/results.csv"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"
# Identity colors carried over from training_curves_grid (validated slots 1-3).
C_PAL = "#2a78d6"
C_PYR = "#eb6834"
C_TH = "#1baf7a"
# Baselines are benchmarks, not training instances: neutral gray, identified by
# their axis position/label (boxplot identity is positional, color is a cue).
C_BASE_FILL = "#d8d6ce"
C_BASE_EDGE = "#75736c"

ARMS = [
    ("arm1_palisades_only_93k8", "Palisades\nonly", C_PAL),
    ("arm2_pyrenees_only_100k", "Pyrenees\nonly", C_PYR),
    ("arm5_concurrent_100k", "Theater\nlevel", C_TH),
    ("random_tactics", "Random\nTactics", None),
    ("fixed_Pyrenees5sp5ev_heuristic", "Fixed\nTactics", None),
]
PANELS = [("Palisades", "Palisades"), ("Pyrenees", "Pyrenees"),
          ("Salamis", "Salamis (unseen)")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=REPO / "final_plots")
    ap.add_argument("--png-dpi", type=int, default=130)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with EVAL.open() as handle:
        rows = list(csv.DictReader(handle))
    data: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        data.setdefault((r["scenario"], r["policy_name"]), []).append(
            float(r["moe_cumulative_reward"])
        )

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.4), facecolor=SURFACE, sharey=True,
    )
    for ax, (scen, title) in zip(axes, PANELS):
        series = [np.asarray(data[(scen, arm)]) for arm, _l, _c in ARMS]
        bp = ax.boxplot(
            series,
            positions=np.arange(len(ARMS)),
            widths=0.58,
            patch_artist=True,
            whis=1.5,
            showmeans=True,
            meanprops=dict(marker="D", markersize=4.5, markerfacecolor=SURFACE,
                           markeredgewidth=1.1),
            medianprops=dict(color=INK, lw=1.4),
            flierprops=dict(marker="o", markersize=2.2, alpha=0.28,
                            markeredgecolor="none"),
            boxprops=dict(lw=1.3),
            whiskerprops=dict(lw=1.1),
            capprops=dict(lw=1.1),
        )
        for i, (_arm, _label, color) in enumerate(ARMS):
            edge = color if color else C_BASE_EDGE
            fill = (
                matplotlib.colors.to_rgba(color, 0.22) if color else C_BASE_FILL
            )
            bp["boxes"][i].set(facecolor=fill, edgecolor=edge)
            for part in ("whiskers", "caps"):
                for artist in bp[part][2 * i:2 * i + 2]:
                    artist.set(color=edge)
            bp["means"][i].set(markeredgecolor=edge)
            bp["fliers"][i].set(markerfacecolor=edge)

        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color=GRID, lw=0.9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels([label for _a, label, _c in ARMS], fontsize=9,
                           color=INK_MUTED)
        ax.set_title(title, fontsize=11, color=INK, weight="bold",
                     loc="left", pad=8)

    axes[0].set_ylabel("MoE", fontsize=10, color=INK_MUTED)
    axes[0].set_ylim(-0.45, 1.08)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.13,
                        wspace=0.10)
    for ext in ("pdf", "png"):
        path = args.outdir / f"eval_boxplots_theater.{ext}"
        fig.savefig(path, format=ext, facecolor=SURFACE,
                    **({"dpi": args.png_dpi} if ext == "png" else {}))
        print(f"wrote {path}")

    # stats for the accompanying text
    print(f"\n{'arm':<34}" + "".join(f"{s:>12}" for s, _ in PANELS))
    for arm, _label, _c in ARMS:
        means = [np.mean(data[(s, arm)]) for s, _ in PANELS]
        print(f"{arm:<34}" + "".join(f"{m:>12.3f}" for m in means))


if __name__ == "__main__":
    main()
