#!/usr/bin/env python
"""Three side-by-side panels: overall, Palisades-only, Pyrenees-only.

    python scripts/plot_arms_triptych.py DIR=LABEL [DIR=LABEL ...] --out <png>

Panel 1 plots every incident in run order - the raw training curve, which for a
multi-mission run is a two-map mixture and shows curriculum STRUCTURE, not
performance. Panels 2 and 3 keep only the incidents flown on that map and
re-index from zero, so arms are compared on like with like.

An arm that never flew a map is simply absent from that panel: a Palisades-only
run has no Pyrenees curve to draw, and saying so by omission is more honest than
extrapolating one.

Each panel carries its own y scale on purpose. Palisades sits near +0.96 and
Pyrenees near 0; a shared axis would flatten whichever map the reader cares
about.
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
REWARD, ESCAPE = "moe_cumulative_reward", "propagation_factor"


def latest(folder: Path) -> Path:
    fs = glob.glob(os.path.join(folder, "results_*_summary.csv"))
    if not fs:
        fb = folder / "training_episode_summaries.csv"
        if fb.exists():
            return fb
        raise SystemExit(f"no summary in {folder}")
    return Path(max(fs, key=lambda p: int(re.search(r"_(\d+)_summary\.csv$", p).group(1))))


def ma(v, w):
    w = min(w, max(1, v.size))
    return np.convolve(v, np.ones(w) / w, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", metavar="DIR=LABEL")
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--out", default="verification_plots/arms_triptych.png")
    ap.add_argument("--title", default="Curriculum arms")
    ap.add_argument("--metric", choices=("moe", "escape"), default="moe")
    ap.add_argument("--slots", help="1-based hue slot per run, e.g. '1,1,2,3,4'")
    ap.add_argument("--linestyles", nargs="+")
    ap.add_argument("--palisades", default="Palisades5sp5ev.json")
    ap.add_argument("--pyrenees", default="Pyrenees5sp5ev.json")
    args = ap.parse_args()

    if args.slots:
        hues = [HUES[int(v) - 1] for v in args.slots.split(",")]
    else:
        if len(args.runs) > len(HUES):
            raise SystemExit("pass --slots when there are more runs than hues")
        hues = HUES[: len(args.runs)]
    styles = args.linestyles or ["solid"] * len(args.runs)

    series = []
    for spec in args.runs:
        path, _, label = spec.partition("=")
        folder = Path(path)
        f = pd.read_csv(latest(folder))
        r = pd.to_numeric(f[REWARD], errors="coerce").to_numpy()
        e = (pd.to_numeric(f[ESCAPE], errors="coerce").to_numpy() == 1.0)
        sc = f["scenario_name"].to_numpy() if "scenario_name" in f else None
        k = ~np.isnan(r)
        r, e = r[k], e[k]
        sc = sc[k] if sc is not None else np.array(["?"] * r.size)
        series.append({"label": label or folder.name, "r": r, "e": e, "sc": sc})

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    fig.patch.set_facecolor(SURFACE)
    panels = [("Overall (all incidents, run order)", None),
              ("Palisades incidents only", args.palisades),
              ("Pyrenees incidents only", args.pyrenees)]

    for ax, (ptitle, want) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        drawn = 0
        lo, hi = np.inf, -np.inf
        for hue, style, s in zip(hues, styles, series):
            if want is None:
                vals = s["r"] if args.metric == "moe" else s["e"].astype(float) * 100
            else:
                m = s["sc"] == want
                if not m.any():
                    continue          # arm never flew this map - omit it
                vals = (s["r"][m] if args.metric == "moe"
                        else s["e"][m].astype(float) * 100)
            y = ma(vals, args.window)
            x = np.arange(args.window - 1, args.window - 1 + y.size)
            ax.plot(x, y, color=hue, linewidth=1.9, linestyle=style,
                    label=s["label"], zorder=3)
            lo = min(lo, float(y.min())); hi = max(hi, float(y.max()))
            drawn += 1
        # The zero line forces the axis to span 0. On Palisades every curve sits
        # near +0.96, so including 0 compresses the whole panel into a sliver.
        # Draw it only where the curves actually straddle zero.
        if args.metric == "moe" and lo < 0.0 < hi:
            ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.45, zorder=2)
        if drawn and np.isfinite(lo) and hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(ptitle, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel("Epochs" if want is None else f"{ptitle.split()[0]}-phase epochs",
                      color=INK, fontsize=10)
        if drawn == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=INK_MUTED)
    axes[0].set_ylabel("MoE" if args.metric == "moe" else "fire escaped the map (%)",
                       color=INK, fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    for t in leg.get_texts():
        t.set_color(INK)

    h = fig.get_figheight()
    fig.suptitle(args.title, x=0.005, y=1 - 0.26 / h, ha="left", fontsize=13,
                 color=INK, weight="bold")
    fig.text(0.005, 1 - 0.52 / h,
             f"{args.window}-incident moving average  ·  each panel has its own y "
             f"scale  ·  an arm absent from a panel never flew that map",
             ha="left", fontsize=9, color=INK_MUTED)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.055, 1, 1 - 0.80 / h))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    for s in series:
        pal = int((s["sc"] == args.palisades).sum()); pyr = int((s["sc"] == args.pyrenees).sum())
        print(f"  {s['label']:<28} total={s['r'].size:>7,}  Palisades={pal:>7,}  Pyrenees={pyr:>7,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
