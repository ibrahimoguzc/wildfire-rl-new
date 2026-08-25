#!/usr/bin/env python
"""Overlay the training curves of several chained runs.

    python scripts/plot_chain_compare.py <dir>=<label> [<dir>=<label> ...] \
        [--window 2000] [--out verification_plots/compare.png] [--title ...]

Reward and fire-escape rate are different units, so they get one panel each
rather than a second y-axis. Every series carries a direct label as well as a
legend entry, so identity never depends on colour alone.
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
ESCAPE_COL = "propagation_factor"  # 1.0 = fire left the map

# dataviz reference palette, categorical slots 1-4 (light), used in fixed order.
# Validated: worst adjacent CVD dE 9.1, normal-vision dE 22.9. Contrast WARN on
# slots 3-4 is relieved by the direct labels every series carries.
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"


def latest_summary(folder: Path) -> Path:
    files = glob.glob(os.path.join(folder, "results_*_summary.csv"))
    if not files:
        fallback = folder / "training_episode_summaries.csv"
        if fallback.exists():
            return fallback
        raise SystemExit(f"no summary CSV in {folder}")

    def count(path: str) -> int:
        m = re.search(r"_(\d+)_summary\.csv$", path)
        return int(m.group(1)) if m else -1

    return Path(max(files, key=count))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, max(1, values.size))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def place_labels(ax, entries, min_gap_frac=0.055):
    """Draw end-of-line labels, pushed apart so they never overlap.

    entries: list of (y_value, text, colour). Positions are nudged in axis
    fraction space, then drawn in ink - the colour rides on the line itself.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo or 1.0
    items = sorted(entries, key=lambda e: e[0])
    frac = [(y - lo) / span for y, _, _ in items]
    for i in range(1, len(frac)):  # push up
        frac[i] = max(frac[i], frac[i - 1] + min_gap_frac)
    overflow = frac[-1] - 1.0
    if overflow > 0:  # then slide the whole stack back down
        frac = [f - overflow for f in frac]
    for (y, text, _), f in zip(items, frac):
        ax.annotate(text, xy=(1.0, f), xycoords="axes fraction",
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=INK,
                    annotation_clip=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", metavar="DIR=LABEL")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--out", default="verification_plots/chain_compare.png")
    parser.add_argument("--title", default="Chained training runs")
    parser.add_argument(
        "--metric",
        choices=("both", "moe", "escape"),
        default="both",
        help="which panel(s) to draw: MoE reward, fire-escape rate, or both",
    )
    parser.add_argument("--xlabel", default="simulation")
    parser.add_argument("--ylabel-moe", dest="ylabel_moe", default=None,
                        help="override the MoE axis label (default names the MA window)")
    parser.add_argument(
        "--slots",
        help=(
            "comma-separated 1-based categorical slots, one per run, e.g. '3,2,4'. "
            "Use it to keep a run's colour identical to an earlier figure that "
            "included more runs - colour should follow the run, not its position."
        ),
    )
    parser.add_argument(
        "--linestyles",
        nargs="+",
        help=(
            "one of solid/dashed/dotted/dashdot per run. Lets two runs that are "
            "the same experiment share a categorical hue and differ by dash - a "
            "composite encoding, which is how to show more series than the "
            "validated palette has slots without inventing a hue."
        ),
    )
    parser.add_argument(
        "--phase",
        help=(
            "keep only rows whose scenario_name matches this (e.g. "
            "Pyrenees5sp5ev.json) and re-index from 0, so a multi-mission run's "
            "phase can be laid over single-map runs on a common x axis"
        ),
    )
    parser.add_argument(
        "--max-sims", dest="max_sims", type=int,
        help=(
            "truncate every run to this many simulations before averaging, so "
            "runs of different lengths are compared over the same budget. Runs "
            "shorter than it are left as they are."
        ),
    )
    args = parser.parse_args()

    # More runs than validated slots is allowed only with an explicit --slots
    # map, i.e. when the author has deliberately assigned hues (normally pairing
    # runs that are one experiment and separating them by --linestyles).
    if len(args.runs) > len(HUES) and not args.slots:
        raise SystemExit(
            f"{len(args.runs)} runs but only {len(HUES)} validated categorical "
            "slots. Pass --slots to assign hues explicitly (repeat a slot for "
            "runs that are the same experiment) and --linestyles to separate "
            "them, or split the figure."
        )
    if args.slots:
        idx = [int(v) - 1 for v in args.slots.split(",")]
        if len(idx) != len(args.runs) or not all(0 <= i < len(HUES) for i in idx):
            raise SystemExit(f"--slots needs {len(args.runs)} values in 1..{len(HUES)}")
        hues = [HUES[i] for i in idx]
    else:
        hues = HUES[: len(args.runs)]

    styles = args.linestyles or ["solid"] * len(args.runs)
    if len(styles) != len(args.runs):
        raise SystemExit(f"--linestyles needs {len(args.runs)} values")

    series = []
    for spec in args.runs:
        path, _, label = spec.partition("=")
        folder = Path(path)
        csv_path = latest_summary(folder)
        frame = pd.read_csv(csv_path)
        reward = pd.to_numeric(frame[REWARD_COL], errors="coerce").to_numpy()
        escaped = pd.to_numeric(frame[ESCAPE_COL], errors="coerce").to_numpy() == 1.0
        keep = ~np.isnan(reward)
        reward, escaped = reward[keep], escaped[keep].astype(float)
        if args.phase and "scenario_name" in frame:
            names = frame["scenario_name"].to_numpy()[keep]
            sel = names == args.phase
            if not sel.any():
                raise SystemExit(
                    f"--phase {args.phase!r} matched nothing in {folder.name}; "
                    f"present: {sorted(set(names))}"
                )
            reward, escaped = reward[sel], escaped[sel]
        if args.max_sims:
            reward, escaped = reward[: args.max_sims], escaped[: args.max_sims]
        series.append({
            "label": label or folder.name,
            "n": reward.size,
            "reward": moving_average(reward, args.window),
            "escape": moving_average(escaped, args.window) * 100.0,
        })

    want_moe = args.metric in ("both", "moe")
    want_escape = args.metric in ("both", "escape")
    if args.metric == "both":
        fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
        ax_r, ax_e = axes
    else:
        fig, only = plt.subplots(figsize=(14, 6))
        axes = (only,)
        ax_r = only if want_moe else None
        ax_e = only if want_escape else None
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

    window = args.window
    for hue, style, s in zip(hues, styles, series):
        x = np.arange(window - 1, window - 1 + s["reward"].size)
        if want_moe:
            ax_r.plot(x, s["reward"], color=hue, linewidth=2.0, linestyle=style,
                      label=s["label"], zorder=3)
        if want_escape:
            ax_e.plot(x, s["escape"], color=hue, linewidth=2.0, linestyle=style,
                      label=s["label"], zorder=3)

    if want_moe:
        ax_r.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=2)
        ax_r.set_ylabel(args.ylabel_moe or f"MoE reward\n({window}-sim moving average)",
                        color=INK, fontsize=10)
        place_labels(ax_r, [(s["reward"][-1], f"{s['reward'][-1]:+.3f}", h)
                            for h, s in zip(hues, series)])
    if want_escape:
        ax_e.set_ylabel(f"fire escaped the map\n(%, {window}-sim moving average)",
                        color=INK, fontsize=10)
        place_labels(ax_e, [(s["escape"][-1], f"{s['escape'][-1]:.1f}%", h)
                            for h, s in zip(hues, series)])
    axes[-1].set_xlabel(args.xlabel, color=INK, fontsize=10)

    # "best" rather than a fixed corner: centre-left is the empty band when a
    # saturated run is plotted against a learning one, but two learning runs of
    # different lengths both climb through it and the legend lands on the lines.
    legend = axes[0].legend(loc="best", frameon=False, fontsize=9, ncol=2,
                            handlelength=1.6, borderaxespad=0.8)
    for text in legend.get_texts():
        text.set_color(INK)

    height = fig.get_figheight()
    fig.suptitle(args.title, x=0.006, y=1 - 0.28 / height, ha="left",
                 fontsize=13, color=INK, weight="bold")
    fig.text(0.006, 1 - 0.56 / height,
             "  ·  ".join(f"{s['label']}: {s['n']:,} sims" for s in series),
             ha="left", fontsize=9, color=INK_MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.93, 1 - 0.85 / height))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    for s in series:
        print(f"{s['label']:26} {s['n']:>7,} sims  reward {s['reward'][0]:+.4f} -> "
              f"{s['reward'][-1]:+.4f}   escape {s['escape'][0]:.1f}% -> {s['escape'][-1]:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
