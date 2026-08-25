#!/usr/bin/env python
"""Plot the training curve of a chained run as one continuous series.

    python scripts/plot_chain_progress.py <chain-output-dir> [--window 2000]
                                          [--out graphs/<name>.png]

Chained runs write every chunk into one folder, and each interim summary holds
the full history, so the highest-N ``results_*_summary.csv`` is the whole run.
Reward and fire-escape rate are different units, so they get one panel each
rather than a second y-axis.

Chunk seams are read from the job logs when they are available (the runner
prints "LR decays over simulations: <n>/<total> done" at each resume) so the
plot can show that the curve passes through them without a step.
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

REWARD_COL = "moe_cumulative_reward"
ESCAPE_COL = "propagation_factor"  # 1.0 = fire left the map, 0.0 = stayed in bounds

# dataviz reference palette, categorical slots 1 and 2 (light mode).
# Validated: worst adjacent CVD dE 24.7, normal-vision dE 33.6, all checks pass.
REWARD_HUE = "#2a78d6"
ESCAPE_HUE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"


def latest_summary(folder: Path) -> Path:
    """Highest-N summary in the folder — the full history for a chained run."""
    files = glob.glob(os.path.join(folder, "results_*_summary.csv"))
    if not files:
        fallback = folder / "training_episode_summaries.csv"
        if fallback.exists():
            return fallback
        raise SystemExit(f"no summary CSV in {folder}")

    def count(path: str) -> int:
        match = re.search(r"_(\d+)_summary\.csv$", path)
        return int(match.group(1)) if match else -1

    return Path(max(files, key=count))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, max(1, values.size))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def chunk_seams(run_name: str, limit: int) -> list[int]:
    """Simulation counts at which a chunk resumed, from the SLURM logs."""
    seams: set[int] = set()
    pattern = re.compile(r"LR decays over simulations: (\d+)/(\d+) done")
    for log in glob.glob("logs/slurm-*.out"):
        try:
            with open(log, "r", encoding="utf-8", errors="ignore") as handle:
                head = [next(handle, "") for _ in range(400)]
        except OSError:
            continue
        text = "".join(head)
        if run_name not in text:
            continue
        match = pattern.search(text)
        if match and 0 < int(match.group(1)) < limit:
            seams.add(int(match.group(1)))
    return sorted(seams)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain_dir", help="outputs/<run>_chain directory")
    parser.add_argument("--window", type=int, default=2000, help="moving-average window")
    parser.add_argument("--out", help="output PNG (default graphs/<run>_progress.png)")
    parser.add_argument("--title", help="override the figure title")
    parser.add_argument("--xlabel", default="simulation")
    parser.add_argument(
        "--ylabel-moe", dest="ylabel_moe", default=None,
        help="override the MoE axis label (default also names the MA window)",
    )
    parser.add_argument(
        "--metric",
        choices=("both", "moe", "escape"),
        default="both",
        help="which panel(s) to draw: MoE reward, fire-escape rate, or both",
    )
    args = parser.parse_args()

    folder = Path(args.chain_dir)
    csv_path = latest_summary(folder)
    frame = pd.read_csv(csv_path)
    reward = pd.to_numeric(frame[REWARD_COL], errors="coerce").to_numpy()
    escaped = pd.to_numeric(frame[ESCAPE_COL], errors="coerce").to_numpy() == 1.0
    keep = ~np.isnan(reward)
    reward, escaped = reward[keep], escaped[keep]
    n = reward.size

    window = args.window
    reward_ma = moving_average(reward, window)
    escape_ma = moving_average(escaped.astype(float), window)
    x = np.arange(window - 1, window - 1 + reward_ma.size)

    run_name = folder.name
    seams = chunk_seams(run_name.replace("_chain", ""), n)

    want_moe = args.metric in ("both", "moe")
    want_escape = args.metric in ("both", "escape")
    if args.metric == "both":
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 height_ratios=[1.15, 1])
        ax_r, ax_e = axes
    else:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        axes = (ax,)
        ax_r = ax if want_moe else None
        ax_e = ax if want_escape else None
    fig.patch.set_facecolor(SURFACE)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        for seam in seams:
            ax.axvline(seam, color=INK_MUTED, linewidth=0.8, alpha=0.35,
                       linestyle=(0, (4, 3)), zorder=1)

    # No per-episode scatter: the reward is effectively binary, so 100k+ raw
    # points fill the panel as a solid block and bury the trend.
    if want_moe:
        ax_r.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=2)
        ax_r.plot(x, reward_ma, color=REWARD_HUE, linewidth=2.0, zorder=3)
        ax_r.set_ylabel(
            args.ylabel_moe or f"MoE reward\n({window}-sim moving average)",
            color=INK, fontsize=10,
        )
        pad_r = max(0.02, 0.12 * (reward_ma.max() - reward_ma.min()))
        ax_r.set_ylim(reward_ma.min() - pad_r, reward_ma.max() + pad_r * 2.2)

    escape_pct = escape_ma * 100.0
    if want_escape:
        ax_e.plot(x, escape_pct, color=ESCAPE_HUE, linewidth=2.0, zorder=3)
        ax_e.set_ylabel(f"fire escaped the map\n(%, {window}-sim moving average)",
                        color=INK, fontsize=10)
        pad_e = max(0.5, 0.12 * (escape_pct.max() - escape_pct.min()))
        ax_e.set_ylim(escape_pct.min() - pad_e * 2.0, escape_pct.max() + pad_e)
    axes[-1].set_xlabel(args.xlabel, color=INK, fontsize=10)

    # Direct labels instead of a legend: one series per panel. Label the two
    # ends; only call out the peak when it is not simply the current value.
    def label(ax, xi, yi, text, dx, dy, ha):
        ax.annotate(text, xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=9, color=INK)

    if want_moe:
        label(ax_r, x[0], reward_ma[0], f"start {reward_ma[0]:+.3f}", 10, -14, "left")
        label(ax_r, x[-1], reward_ma[-1], f"now {reward_ma[-1]:+.3f}", -6, 8, "right")
        peak = int(reward_ma.argmax())
        if x[-1] - x[peak] > 0.05 * n:
            label(ax_r, x[peak], reward_ma[peak],
                  f"peak {reward_ma[peak]:+.3f} @ {x[peak]:,}", 0, 10, "center")

    if want_escape:
        label(ax_e, x[0], escape_pct[0], f"start {escape_pct[0]:.1f}%", 10, 6, "left")
        label(ax_e, x[-1], escape_pct[-1], f"now {escape_pct[-1]:.1f}%", -6, -14, "right")
        best = int(escape_ma.argmin())
        if x[-1] - x[best] > 0.05 * n:
            label(ax_e, x[best], escape_pct[best],
                  f"best {escape_pct[best]:.1f}% @ {x[best]:,}", 0, -14, "center")

    if seams:
        axes[0].annotate(
            "chunk resume", xy=(seams[0], axes[0].get_ylim()[1]),
            xytext=(4, -8), textcoords="offset points",
            ha="left", va="top", fontsize=8, color=INK_MUTED,
        )

    title = args.title or run_name.replace("_", " ")
    # Header spacing in inches, not figure fractions: the single-panel figure is
    # shorter, and fixed fractions put the subtitle on top of the title.
    height = fig.get_figheight()
    fig.suptitle(title, x=0.008, y=1 - 0.28 / height, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.text(0.008, 1 - 0.56 / height,
             f"{n:,} simulations across {len(seams) + 1} SLURM jobs  ·  "
             f"source {csv_path.name}",
             ha="left", fontsize=9, color=INK_MUTED)

    suffix = "" if args.metric == "both" else f"_{args.metric}"
    out = Path(args.out) if args.out else Path("graphs") / f"{run_name}{suffix}_progress.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"{n:,} simulations, {len(seams)} chunk seams at {seams}")
    print(f"reward  MA{window}: {reward_ma[0]:+.4f} -> {reward_ma[-1]:+.4f} "
          f"(peak {reward_ma.max():+.4f})")
    print(f"escaped MA{window}: {escape_ma[0]:.1%} -> {escape_ma[-1]:.1%} "
          f"(best {escape_ma.min():.1%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
