#!/usr/bin/env python
"""Per-map MoE for a multi-scenario run, one panel per map.

    python scripts/plot_scenario_moe.py <run-dir> [--out <png>]

A concurrent (--switch-scenarios) run interleaves maps, so its pooled MoE curve
is a mixture of both and means nothing on its own. This splits the episodes by
scenario and gives each map its own panel and its own y scale - Palisades sits
near +0.96 and Pyrenees near 0, so a shared axis would flatten one of them.

x is the run-wide episode index, so the two panels share a timeline and can be
read against each other: at any x, both maps are being trained simultaneously.
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
    ap.add_argument("--out", default="verification_plots/scenario_moe.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--xaxis", choices=("run", "own"), default="run",
                    help="'run' = shared run-wide episode index (default); "
                         "'own' = each map indexed by its own episode count")
    args = ap.parse_args()

    folder = Path(args.run_dir)
    csv = latest_summary(folder)
    f = pd.read_csv(csv)
    r = pd.to_numeric(f["moe_cumulative_reward"], errors="coerce").to_numpy()
    e = (pd.to_numeric(f["propagation_factor"], errors="coerce").to_numpy() == 1.0)
    sc = f["scenario_name"].to_numpy()
    k = ~np.isnan(r)
    r, e, sc = r[k], e[k], sc[k]
    idx = np.arange(1, r.size + 1)
    order = list(dict.fromkeys(sc))

    fig, axes = plt.subplots(len(order), 1, figsize=(13, 4.0 * len(order)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    w = args.window
    for ax, hue, name in zip(axes, HUES, order):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        m = sc == name
        vals, xs = r[m], (idx[m] if args.xaxis == "run" else np.arange(1, m.sum() + 1))
        ww = min(w, vals.size)
        ma = np.convolve(vals, np.ones(ww) / ww, mode="valid")
        xm = xs[ww - 1:]
        ax.plot(xm, ma, color=hue, linewidth=2.0, zorder=3)
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=2)
        short = name.replace("5sp5ev.json", "")
        ax.set_ylabel("MoE", color=INK, fontsize=10)
        esc = e[m]
        ax.annotate(
            f"{short}  ·  {int(m.sum()):,} incidents  ·  "
            f"MoE {ma[0]:+.3f} → {ma[-1]:+.3f}  ·  escaped "
            f"{esc[:ww].mean():.1%} → {esc[-ww:].mean():.1%}",
            xy=(0.006, 0.94), xycoords="axes fraction", ha="left", va="top",
            fontsize=10, color=INK, weight="bold")
    axes[-1].set_xlabel(
        "Epochs (run-wide episode index)" if args.xaxis == "run"
        else "Epochs (each map's own episode count)", color=INK, fontsize=10)

    title = args.title or folder.name.replace("_", " ")
    h = fig.get_figheight()
    fig.suptitle(title, x=0.006, y=1 - 0.28 / h, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.text(0.006, 1 - 0.54 / h,
             f"{r.size:,} incidents  ·  {w}-episode moving average within each "
             f"map  ·  separate y scales  ·  source {csv.name}",
             ha="left", fontsize=9, color=INK_MUTED)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.80 / h))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    for name in order:
        m = sc == name
        ww = min(w, int(m.sum()))
        print(f"  {name:<24} {int(m.sum()):>7,}  MoE {r[m][-ww:].mean():+.4f}  "
              f"escaped {e[m][-ww:].mean():.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
