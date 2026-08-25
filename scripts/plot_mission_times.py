#!/usr/bin/env python
"""Plot mission duration over training for one chained run.

    python scripts/plot_mission_times.py <chain-output-dir> [--window 2000]
                                         [--out ...] [--title ...] [--xlabel ...]

Splits by outcome, because the aggregate hides the story: as the policy
improves, both contained and escaped missions get longer while the mix shifts
towards the (shorter) contained ones, so the overall mean barely moves.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MINUTES_COL = "total_minutes"
ESCAPE_COL = "propagation_factor"  # 1.0 = fire left the map

# dataviz reference palette, categorical slots 1-3 (light), fixed order.
ALL_HUE, ESCAPED_HUE, CONTAINED_HUE = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def latest_summary(folder: Path) -> Path:
    files = glob.glob(os.path.join(folder, "results_*_summary.csv"))
    if not files:
        fallback = folder / "training_episode_summaries.csv"
        if fallback.exists():
            return fallback
        raise SystemExit(f"no summary CSV in {folder}")
    return Path(max(files, key=lambda p: int(re.search(r"_(\d+)_summary\.csv$", p).group(1))))


def masked_moving_average(values: np.ndarray, mask: np.ndarray, window: int):
    """Mean of `values` over a sliding window, counting only masked entries.

    Returns (x, y) with windows that contain no qualifying episode dropped, so
    a subgroup that vanishes for a stretch leaves a gap rather than a zero.
    """
    kernel = np.ones(window)
    total = np.convolve(np.where(mask, values, 0.0), kernel, mode="valid")
    count = np.convolve(mask.astype(float), kernel, mode="valid")
    x = np.arange(window - 1, window - 1 + total.size)
    ok = count > 0
    return x[ok], total[ok] / count[ok]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain_dir")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="simulation")
    args = parser.parse_args()

    folder = Path(args.chain_dir)
    csv_path = latest_summary(folder)
    frame = pd.read_csv(csv_path)
    minutes = pd.to_numeric(frame[MINUTES_COL], errors="coerce").to_numpy()
    escaped = pd.to_numeric(frame[ESCAPE_COL], errors="coerce").to_numpy() == 1.0
    keep = ~np.isnan(minutes)
    minutes, escaped = minutes[keep], escaped[keep]
    n = minutes.size
    w = args.window

    groups = [
        ("All missions", np.ones(n, dtype=bool), ALL_HUE),
        ("Fire escaped", escaped, ESCAPED_HUE),
        ("Fire contained", ~escaped, CONTAINED_HUE),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    ends = []
    for label, mask, hue in groups:
        x, y = masked_moving_average(minutes, mask, w)
        ax.plot(x, y, color=hue, linewidth=2.0, label=label, zorder=3)
        ends.append((y[-1], f"{label}: {y[0]:.0f} -> {y[-1]:.0f} min"))

    ax.set_ylabel(f"mission duration (minutes)\n({w}-sim moving average)",
                  color=INK, fontsize=10)
    ax.set_xlabel(args.xlabel, color=INK, fontsize=10)

    # Direct labels at the right edge, spaced so they cannot collide.
    lo, hi = ax.get_ylim()
    items = sorted(ends)
    frac = [(y - lo) / (hi - lo) for y, _ in items]
    for i in range(1, len(frac)):
        frac[i] = max(frac[i], frac[i - 1] + 0.07)
    for (_, text), f in zip(items, frac):
        ax.annotate(text, xy=(1.0, f), xycoords="axes fraction",
                    xytext=(8, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=9, color=INK, annotation_clip=False)

    # Centre-left: escaped sits high, contained low, so the middle band is free.
    legend = ax.legend(loc="center left", frameon=False, fontsize=9, ncol=1,
                       handlelength=1.6, borderaxespad=0.8)
    for text in legend.get_texts():
        text.set_color(INK)

    title = args.title or folder.name.replace("_", " ")
    height = fig.get_figheight()
    fig.suptitle(title, x=0.006, y=1 - 0.28 / height, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.text(0.006, 1 - 0.56 / height,
             f"{n:,} simulations  ·  escaped {escaped.mean():.1%}  ·  "
             f"overall mean {np.nanmean(minutes):.0f} min  ·  source {csv_path.name}",
             ha="left", fontsize=9, color=INK_MUTED)

    out = Path(args.out) if args.out else Path("verification_plots") / f"{folder.name}_mission_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.80, 1 - 0.85 / height))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    for label, mask, _ in groups:
        x, y = masked_moving_average(minutes, mask, w)
        print(f"{label:16} n={mask.sum():>7,}  {y[0]:6.1f} -> {y[-1]:6.1f} min "
              f"({y[-1] - y[0]:+.1f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
