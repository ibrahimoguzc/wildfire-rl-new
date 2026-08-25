#!/usr/bin/env python
"""Show how a multi-scenario run's episodes divide between maps over time.

    python scripts/plot_scenario_mix.py <run-dir> [--out <png>]

For a concurrent (--switch-scenarios) run the workers are split evenly, but
episodes are not: whichever map's episodes finish faster accumulates more of
them. This plots what actually happened - cumulative incidents per map, and the
running share - so the achieved split can be reported instead of the nominal
one.
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def latest_summary(folder: Path) -> Path:
    files = glob.glob(os.path.join(folder, "results_*_summary.csv"))
    if not files:
        fb = folder / "training_episode_summaries.csv"
        if fb.exists():
            return fb
        raise SystemExit(f"no summary CSV in {folder}")
    return Path(max(files, key=lambda p: int(re.search(r"_(\d+)_summary\.csv$", p).group(1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="verification_plots/scenario_mix.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--window", type=int, default=4000,
                    help="window for the running-share panel")
    args = ap.parse_args()

    folder = Path(args.run_dir)
    csv = latest_summary(folder)
    frame = pd.read_csv(csv)
    if "scenario_name" not in frame:
        raise SystemExit(f"{csv.name} has no scenario_name column")
    names = frame["scenario_name"].to_numpy()
    order = list(dict.fromkeys(names))
    n = names.size
    x = np.arange(1, n + 1)

    fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                     height_ratios=[1.2, 1])
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_c, ax_s):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    # Perfectly even reference: n/len(order) each.
    ax_c.plot(x, x / len(order), color=INK_MUTED, linewidth=1.0,
              linestyle=(0, (4, 3)), alpha=0.7, zorder=2, label="even split")
    finals = []
    for hue, name in zip(HUES, order):
        cum = np.cumsum(names == name)
        ax_c.plot(x, cum, color=hue, linewidth=2.0, zorder=3,
                  label=name.replace("5sp5ev.json", ""))
        finals.append((cum[-1], name, hue))
        w = min(args.window, n)
        share = np.convolve((names == name).astype(float), np.ones(w) / w,
                            mode="valid") * 100.0
        ax_s.plot(np.arange(w - 1, w - 1 + share.size), share,
                  color=hue, linewidth=2.0, zorder=3)

    ax_s.axhline(100.0 / len(order), color=INK_MUTED, linewidth=1.0,
                 linestyle=(0, (4, 3)), alpha=0.7, zorder=2)
    ax_c.set_ylabel("cumulative incidents", color=INK, fontsize=10)
    ax_s.set_ylabel(f"share of incidents\n(%, {args.window}-episode window)",
                    color=INK, fontsize=10)
    ax_s.set_xlabel("Epochs (episodes completed, in completion order)",
                    color=INK, fontsize=10)
    leg = ax_c.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    for cum, name, hue in finals:
        ax_c.annotate(f"{name.replace('5sp5ev.json','')} {cum:,} ({cum/n:.1%})",
                      xy=(n, cum), xytext=(8, 0), textcoords="offset points",
                      va="center", ha="left", fontsize=9, color=INK,
                      annotation_clip=False)

    title = args.title or folder.name.replace("_", " ")
    h = fig.get_figheight()
    fig.suptitle(title, x=0.006, y=1 - 0.28 / h, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.text(0.006, 1 - 0.56 / h,
             f"{n:,} incidents total  ·  dashed line = a perfectly even split  "
             f"·  source {csv.name}",
             ha="left", fontsize=9, color=INK_MUTED)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.88, 1 - 0.85 / h))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"{n:,} incidents")
    for cum, name, _ in finals:
        print(f"  {name:<24} {cum:>7,}  ({cum/n:.1%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
