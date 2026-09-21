#!/usr/bin/env python3
"""Horizontal MoE boxplots from the FINAL EVALUATION (job 756122, 250 paired
seeds per map), restricted to the five headline policies.

Three figures into final_plots/eval_2: Palisades, Pyrenees, and the two maps
pooled (Salamis excluded - it is saturated and dilutes every contrast).

Color follows the ENTITY, matching the training figures exactly: Palisades-only
blue, Pyrenees-only orange, theater-level (concurrent) aqua - the validated
categorical trio - with both baselines in the neutral gray. Identity is never
color-alone: every row is named on the y axis. Red line = mean (the MoE
distribution is bimodal, so the mean, not the median, separates policies).
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

import sys

REPO = Path(__file__).resolve().parent.parent
# argv[1]: results CSV (default: the 250-seed final evaluation)
# argv[2]: filename suffix, e.g. "_100sims" (default: none)
RESULTS = (Path(sys.argv[1]) if len(sys.argv) > 1
           else REPO / "final_evaluation/results.csv")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUTDIR = REPO / "final_plots/eval_2"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"
NEUTRAL = "#8a8985"
MEAN_RED = "#d03b3b"

# Fixed entity -> hue map (identical to the training-curve figures).
POLICIES = [  # top-to-bottom display order
    ("random_tactics", "Random tactics", NEUTRAL),
    ("fixed_Pyrenees5sp5ev_heuristic", "Fixed tactics", NEUTRAL),
    ("arm1_palisades_only", "Palisades only", "#2a78d6"),
    ("arm2_pyrenees_only", "Pyrenees only", "#eb6834"),
    ("arm5_concurrent", "Theater level (concurrent)", "#1baf7a"),
]


def load() -> list[dict[str, str]]:
    with RESULTS.open() as handle:
        return list(csv.DictReader(handle))


def panel(rows: list[dict[str, str]], scenarios: list[str], title: str,
          subtitle: str, out_stem: str) -> None:
    keep = [r for r in rows if r["scenario_name"] in scenarios]
    if not keep:
        print(f"skip {out_stem}: no rows for {scenarios}")
        return
    # Tolerate CSVs that carry only a subset of the five policies (e.g. the
    # 3-policy confirmatory runs); absent policies simply have no row.
    policies = [(p, lab, hue) for p, lab, hue in POLICIES
                if any(r["policy_name"] == p for r in keep)]
    data = {p: np.array([float(r["moe_cumulative_reward"]) for r in keep
                         if r["policy_name"] == p])
            for p, _, _ in policies}
    n_per = {v.size for v in data.values()}
    assert len(n_per) == 1, f"unequal policy row counts: {n_per}"
    subtitle = subtitle.format(n=n_per.pop())

    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(886047)

    n_rows = len(policies)
    for i, (key, label, hue) in enumerate(policies):
        y = n_rows - i  # first policy at the top
        vals = data[key]
        bp = ax.boxplot(
            vals, positions=[y], vert=False, widths=0.58, patch_artist=True,
            showfliers=False, zorder=3,
            boxprops=dict(facecolor=SURFACE, edgecolor=hue, linewidth=1.6),
            whiskerprops=dict(color=hue, linewidth=1.4),
            capprops=dict(color=hue, linewidth=1.4),
            medianprops=dict(color=hue, linewidth=1.4),
        )
        bp["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(hue, 0.12))
        ax.scatter(vals, y + rng.uniform(-0.16, 0.16, vals.size),
                   s=7, color=hue, alpha=0.35, linewidths=0, zorder=2)
        mean = float(vals.mean())
        ax.plot([mean, mean], [y - 0.29, y + 0.29], color=MEAN_RED,
                linewidth=2.2, zorder=4)
        ax.annotate(f"{mean:+.3f}", xy=(mean, y + 0.33), ha="center",
                    va="bottom", fontsize=8.5, color=MEAN_RED,
                    annotation_clip=False)

    ax.set_yticks([n_rows - i for i in range(n_rows)])
    ax.set_yticklabels([label for _, label, _ in policies], fontsize=10,
                       color=INK)
    ax.set_ylim(0.4, n_rows + 0.75)
    ax.set_xlabel("MoE", fontsize=10, color=INK_MUTED)
    lo = min(v.min() for v in data.values())
    hi = max(v.max() for v in data.values())
    if lo < 0.0 < hi:
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=1)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.tick_params(axis="y", colors=INK)

    fig.text(0.005, 0.975, title, fontsize=12.5, color=INK, weight="bold",
             ha="left", va="top")
    fig.text(0.005, 0.925, subtitle, fontsize=9, color=INK_MUTED,
             ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    for ext in ("png", "pdf"):
        path = OUTDIR / f"{out_stem}{SUFFIX}.{ext}"
        fig.savefig(path, format=ext, facecolor=SURFACE,
                    bbox_inches="tight", **({"dpi": 130} if ext == "png" else {}))
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load()
    note = "red line = mean · box = IQR with 1.5 IQR whiskers · dots = individual simulations"
    panel(rows, ["Palisades5sp5ev.json"], "Palisades",
          "{n} paired ignitions · switch-ignition-4, water set 2 · " + note,
          "eval_palisades")
    panel(rows, ["Pyrenees5sp5ev.json"], "Pyrenees",
          "{n} paired ignitions · switch-ignition-4, water set 2 · " + note,
          "eval_pyrenees")
    present = {r["scenario_name"] for r in rows}
    if {"Palisades5sp5ev.json", "Pyrenees5sp5ev.json"} <= present:
        panel(rows, ["Palisades5sp5ev.json", "Pyrenees5sp5ev.json"],
              "Pooled: Palisades + Pyrenees",
              "{n} paired ignitions pooled (seeds paired within each map) · " + note,
              "eval_pooled")
    else:
        print("skip eval_pooled: needs both maps in the CSV")


if __name__ == "__main__":
    main()
