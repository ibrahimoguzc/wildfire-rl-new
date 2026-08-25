#!/usr/bin/env python
"""Boxplot the end-of-simulation MoE of several policies from a ppo_tester CSV.

    python scripts/plot_eval_boxplots.py <results.csv> [--out <png>]
                                         [--metric moe_cumulative_reward]

One box per policy. ``moe_cumulative_reward`` in a ppo_tester results CSV is
the MoE recomputed at episode termination from the terminal metrics, not a sum
of per-step rewards, so each point is one finished simulation.

Per-run points are drawn over the boxes on purpose: on Pyrenees the MoE is
close to binary (the fire-escaped indicator dominates it), so a box alone
reports a median sitting in a gap where almost no simulation actually lands.
The strip shows the two clusters the box hides.

Runs are paired across policies by seed - ppo_tester gives every policy the
same seed for a given run index - so the printed comparison is a paired one,
which is far tighter than comparing the two marginal distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# dataviz reference palette, categorical slots 1-3 (light mode).
# Validated: worst adjacent CVD dE 9.2 (deutan), normal-vision dE 27.6, all
# checks pass. The contrast WARN on the green is relieved by the direct axis
# labels and the printed values under every box.
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Status-critical red, used here as an annotation mark (the mean), not as a
# series identity. 4.68:1 on the light surface. It is the only red in the
# figure and it is named in the subtitle and printed under every box, so it
# never carries meaning by colour alone.
MEAN_HUE = "#d03b3b"
# Neutral for non-learned baselines under --color-by-type: colour then encodes
# "is this a trained policy?", and the axis labels carry individual identity,
# so nothing depends on telling two greys apart.
BASELINE_HUE = "#8a8985"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

ESCAPE_COL = "propagation_factor"  # 1.0 = fire left the map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="per-run CSV from ppo_tester")
    parser.add_argument("--metric", default="moe_cumulative_reward")
    parser.add_argument("--out", default="verification_plots/eval_boxplots.png")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--note",
        help=(
            "extra clause for the subtitle, e.g. what a short axis label like "
            "'Fixed tactics' actually stands for"
        ),
    )
    parser.add_argument("--ylabel", default="MoE")
    # nargs, not a comma-separated string: display names legitimately contain
    # commas, and splitting on them silently produced the wrong label count.
    parser.add_argument(
        "--order",
        nargs="+",
        help="policy_name values, in the order to plot (default: as they appear)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="display names, one per policy, in the same order as --order",
    )
    parser.add_argument(
        "--color-by-type",
        action="store_true",
        help=(
            "colour boxes by policy_type - trained models take the validated "
            "categorical slots, non-learned baselines share one neutral. Needed "
            "beyond 4 policies, where the categorical palette runs out."
        ),
    )
    parser.add_argument(
        "--scenario",
        help=(
            "keep only rows whose scenario_name matches, so a results CSV "
            "covering several maps yields one figure per map"
        ),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    if args.scenario:
        sel = frame["scenario_name"] == args.scenario
        if not sel.any():
            raise SystemExit(
                f"--scenario {args.scenario!r} matched nothing; present: "
                f"{sorted(set(frame['scenario_name']))}"
            )
        frame = frame[sel]
    if args.order:
        names = [n.strip() for n in args.order]
        missing = [n for n in names if n not in set(frame["policy_name"])]
        if missing:
            raise SystemExit(
                f"policies not in CSV: {missing}. "
                f"Present: {sorted(set(frame['policy_name']))}"
            )
    else:
        names = list(dict.fromkeys(frame["policy_name"]))
    display = [n.strip() for n in args.labels] if args.labels else names
    if len(display) != len(names):
        raise SystemExit(
            f"--labels has {len(display)} entries but there are {len(names)} "
            f"policies: {names}"
        )

    groups = [
        pd.to_numeric(
            frame.loc[frame["policy_name"] == name, args.metric], errors="coerce"
        ).dropna().to_numpy()
        for name in names
    ]
    escapes = [
        (
            pd.to_numeric(
                frame.loc[frame["policy_name"] == name, ESCAPE_COL], errors="coerce"
            ) == 1.0
        ).mean() * 100.0
        if ESCAPE_COL in frame else float("nan")
        for name in names
    ]

    # zip() against HUES silently drops boxes past the 4th - the strip points
    # and mean line for a 5th policy would just vanish. Either colour by kind
    # (baseline vs learned) or refuse.
    if args.color_by_type and "policy_type" in frame:
        kinds = [
            frame.loc[frame["policy_name"] == n, "policy_type"].iloc[0] for n in names
        ]
        model_hues = iter(HUES)
        hues = [
            next(model_hues) if k == "ppo_model" else BASELINE_HUE for k in kinds
        ]
    elif len(groups) > len(HUES):
        raise SystemExit(
            f"{len(groups)} policies but only {len(HUES)} validated categorical "
            "slots. Pass --color-by-type to colour by policy kind instead, or "
            "split the figure."
        )
    else:
        hues = list(HUES)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=10)

    positions = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.5,
        showfliers=False,      # the strip below already shows every run
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 2.0},
        whiskerprops={"color": INK_MUTED, "linewidth": 1.2},
        capprops={"color": INK_MUTED, "linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )
    for patch, hue in zip(bp["boxes"], hues):
        patch.set_facecolor(hue)
        patch.set_alpha(0.20)
        patch.set_edgecolor(hue)

    rng = np.random.default_rng(0)
    for pos, values, hue in zip(positions, groups, hues):
        jitter = rng.uniform(-0.13, 0.13, size=values.size)
        ax.scatter(
            pos + jitter, values, s=16, color=hue, alpha=0.55,
            linewidths=0.5, edgecolors=SURFACE, zorder=3,
        )
        # Mean as a distinct mark: the median sits in the empty middle when the
        # distribution is two-clustered, so it alone would misrepresent the run.
        # Drawn over a surface-coloured underlay so the red stays legible where
        # it crosses the strip points and the box fill.
        ax.hlines(
            values.mean(), pos - 0.25, pos + 0.25,
            color=SURFACE, linewidth=4.4, zorder=5,
        )
        ax.hlines(
            values.mean(), pos - 0.25, pos + 0.25,
            color=MEAN_HUE, linewidth=2.4, zorder=6,
        )

    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(display, fontsize=11, color=INK)
    ax.set_ylabel(args.ylabel, color=INK, fontsize=11)

    # Values under each box: identity and magnitude never depend on the fill.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.16 * (hi - lo), hi)
    for pos, values, escaped in zip(positions, groups, escapes):
        ax.annotate(
            f"mean {values.mean():+.3f}\nmedian {np.median(values):+.3f}\n"
            f"escaped {escaped:.0f}%",
            xy=(pos, ax.get_ylim()[0]), xytext=(0, 8),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=9, color=INK_MUTED,
        )

    n_runs = min(len(g) for g in groups)
    title = args.title or "Policy comparison, end-of-simulation MoE"
    height = fig.get_figheight()
    fig.suptitle(title, x=0.008, y=1 - 0.28 / height, ha="left",
                 fontsize=13, color=INK, weight="bold")
    fig.text(
        0.008, 1 - 0.56 / height,
        f"{n_runs} simulations per policy, paired by ignition seed  ·  "
        f"red line = mean, black bar = median  ·  source {Path(args.results).name}",
        ha="left", fontsize=9, color=INK_MUTED,
    )
    # The note gets its own line rather than being appended: subtitles that
    # explain a short axis label are long, and one line runs off the canvas.
    if args.note:
        fig.text(0.008, 1 - 0.76 / height, args.note,
                 ha="left", fontsize=9, color=INK_MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_inches = 1.05 if args.note else 0.85
    fig.tight_layout(rect=(0, 0, 1, 1 - header_inches / height))
    fig.savefig(out, dpi=130, facecolor=SURFACE)

    # Table view - the relief the contrast WARN obliges, and the paired stats.
    width = max(len(label) for label in display) + 2
    print(f"\n{'policy':<{width}}{'n':>5}{'mean':>10}{'median':>10}{'escaped':>10}")
    for name, values, escaped in zip(display, groups, escapes):
        print(f"{name:<{width}}{values.size:>5}{values.mean():>+10.4f}"
              f"{np.median(values):>+10.4f}{escaped:>9.0f}%")

    # Paired against the LAST policy, which is the one under test when the
    # baselines are listed first. Sign convention: positive = the baseline beat
    # the reference on that seed, so a negative mean means the reference wins.
    pivot = frame.pivot_table(index="seed", columns="policy_name", values=args.metric)
    pivot = pivot.dropna(subset=[n for n in names if n in pivot])
    if len(names) > 1 and len(pivot) > 1:
        ref, ref_label = names[-1], display[-1]
        print(f"\npaired differences vs {ref_label} (n={len(pivot)} shared seeds):")
        for name, label in zip(names[:-1], display[:-1]):
            delta = pivot[name].to_numpy() - pivot[ref].to_numpy()
            se = delta.std(ddof=1) / np.sqrt(delta.size)
            sigma = delta.mean() / se if se > 0 else float("nan")
            wins = (delta > 0).mean() * 100.0
            print(f"  {label:<{width}}{delta.mean():>+9.4f} +/- {se:.4f}  "
                  f"({sigma:>+5.1f} sigma)   beat {ref_label} on {wins:.0f}% of seeds")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
