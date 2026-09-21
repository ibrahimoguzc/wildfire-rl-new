#!/usr/bin/env python3
"""Regret figure: how far each policy falls short of the best policy in each
context (Palisades, Pyrenees, and the two pooled), in containment points.

Data: the 250-seed final evaluation (`final_evaluation/results.csv`), the
same five policies and entity colors as the rest of final_plots/eval_2.
Regret = best containment in that context minus the policy's containment;
the best policy scores 0 by definition. The theater-level agent's three
bars are near-zero - it is within half a point of the local best in every
context - while every alternative concedes real performance somewhere.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "final_evaluation/results.csv"
OUTDIR = REPO / "final_plots/eval_2"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

POLICIES = [  # top-to-bottom, same order and hues as the boxplots
    ("random_tactics", "Random tactics", "#8a8985"),
    ("fixed_Pyrenees5sp5ev_heuristic", "Fixed tactics", "#8a8985"),
    ("arm1_palisades_only", "Palisades only", "#2a78d6"),
    ("arm2_pyrenees_only", "Pyrenees only", "#eb6834"),
    ("arm5_concurrent", "Theater level (concurrent)", "#1baf7a"),
]

CONTEXTS = [
    ("a) Palisades", ["Palisades5sp5ev.json"]),
    ("b) Pyrenees", ["Pyrenees5sp5ev.json"]),
    ("c) Pooled", ["Palisades5sp5ev.json", "Pyrenees5sp5ev.json"]),
]


def main() -> None:
    with RESULTS.open() as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), facecolor=SURFACE,
                             sharey=True)
    n_rows = len(POLICIES)
    for ax, (title, scenarios) in zip(axes, CONTEXTS):
        keep = [r for r in rows if r["scenario_name"] in scenarios]
        contain = {
            p: np.mean([float(r["propagation_factor"]) != 1.0 for r in keep
                        if r["policy_name"] == p]) * 100.0
            for p, _, _ in POLICIES
        }
        best = max(contain.values())
        ax.set_facecolor(SURFACE)
        for i, (key, label, hue) in enumerate(POLICIES):
            y = n_rows - i
            regret = best - contain[key]
            ax.barh(y, regret, height=0.62, color=hue, zorder=3)
            ax.annotate(f"{regret:.1f}", xy=(regret, y), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=9, weight="bold", color=hue if regret > 0
                        else INK_MUTED)
        ax.set_yticks([n_rows - i for i in range(n_rows)])
        ax.set_yticklabels([lab for _, lab, _ in POLICIES], fontsize=10,
                           color=INK)
        ax.set_xlim(0, 14.5)
        ax.set_ylim(0.4, n_rows + 0.6)
        ax.grid(True, axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
        ax.tick_params(axis="y", colors=INK)
        ax.set_title(title, fontsize=11.5, color=INK, weight="bold",
                     loc="left", pad=8)
        ax.set_xlabel("regret (containment points)", fontsize=9.5,
                      color=INK_MUTED)

    fig.text(0.005, 0.985,
             "Shortfall from the best policy in each context",
             fontsize=12.5, color=INK, weight="bold", ha="left", va="top")
    fig.text(0.005, 0.005,
             "Regret = best containment rate in the context minus the policy's own "
             "(0 = best there). 250 paired ignitions per map; pooled = both maps, "
             "500 pairs. Salamis excluded (saturated).",
             fontsize=8.5, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    for ext in ("png", "pdf"):
        path = OUTDIR / f"regret.{ext}"
        fig.savefig(path, format=ext, facecolor=SURFACE, bbox_inches="tight",
                    **({"dpi": 130} if ext == "png" else {}))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
