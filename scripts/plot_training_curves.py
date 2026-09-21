#!/usr/bin/env python3
"""Training-curve figures for the Palisades / Pyrenees / concurrent comparison.

Writes each figure into final_plots/ as both a vector PDF (for print) and a
PNG (for slides and quick preview), at the dpi the sibling figures use:

  training_curves_by_map.{pdf,png}      two panels (Palisades, Pyrenees), each
                                        comparing the dedicated single-map run
                                        against the concurrent run's episodes on
                                        that same map
  training_curves_three_runs.{pdf,png}  one panel, the three runs as named:
                                        Palisades-only, Pyrenees-only, concurrent
  training_curves_grid.{pdf,png}        3 x 2 grid. Left column: all three
                                        training instances together on common
                                        scales. Middle / right columns: the
                                        dedicated Palisades / Pyrenees run vs
                                        the theater-level run's episodes on
                                        that map. Top row MoE, bottom row
                                        fire-escape rate.
  training/moe_all_instances.{pdf,png}  the grid's panel (a) on its own, with
                                        a dashed random-tactics baseline per
                                        training instance in a lighter tint of
                                        that instance's color (see the
                                        random-series block).

The plotted quantity is the fire-escape rate: ``propagation_factor`` is a binary
per-episode indicator, so its moving average is the escape rate directly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Embed TrueType rather than Type-3 so the PDFs are submission-safe.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "examples/wildfire/data/scenarios/outputs"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"
# Validated categorical slots 1-3 (see dataviz palette; all six checks pass,
# all-pairs, light mode). Aqua sits below 3:1 on this surface, so every curve
# carries a visible direct label -- the documented relief rule.
C_DEDICATED = "#2a78d6"
C_CONCURRENT = "#eb6834"
C_THIRD = "#1baf7a"
C_NEUTRAL = "#8a8985"  # baselines (dataviz neutral slot)


def lighten(color: str, amount: float = 0.35) -> str:
    """Mix a hex color toward white; 0 leaves it unchanged, 1 gives white.

    The random-tactics baselines in moe_all_instances sit in a lighter tint
    of their training instance's color so the pair reads as one instance
    while the trained curve stays the visually dominant line.
    """
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * amount) for c in rgb)

RUNS = {
    "pal": (
        "palisades_5sp5ev_grpbytype_sw4_directional2_k3_150k_730529",
        "results_palisades_5sp5ev_grpbytype_sw4_directional2_k3_150k_150000_summary.csv",
    ),
    "pyr": (
        "pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_717319",
        "results_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_150000_summary.csv",
    ),
    "con": (
        "concurrent_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_chain",
        "results_concurrent_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150000_summary.csv",
    ),
}


def load(key: str) -> tuple[np.ndarray, np.ndarray]:
    folder, name = RUNS[key]
    path = OUTPUTS / folder / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    escaped = np.array([float(r["propagation_factor"]) for r in rows])
    scenario = np.array([r["scenario"] for r in rows])
    return escaped, scenario


def load_moe(key: str) -> np.ndarray:
    folder, name = RUNS[key]
    with (OUTPUTS / folder / name).open() as handle:
        return np.array(
            [float(r["moe_cumulative_reward"]) for r in csv.DictReader(handle)]
        )


def rolling(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean with an expanding window over the first `window` points."""
    csum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)


def save(fig, outdir: Path, stem: str, dpi: int, tight: bool = True) -> None:
    """Write the same figure as vector PDF and raster PNG.

    ``tight=False`` skips bbox_inches="tight". That cropping re-maps figure
    coordinates after layout, so a figure laid out with an explicit
    ``tight_layout(rect=...)`` band must opt out or its reserved space collapses.
    """
    for ext in ("pdf", "png"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(
            path,
            format=ext,
            facecolor=SURFACE,
            **({"bbox_inches": "tight"} if tight else {}),
            **({"dpi": dpi} if ext == "png" else {}),
        )
        print(f"wrote {path}")


def style(ax, ymax: float, xmax: int) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.set_xticks([0, 50000, 100000, 150000])
    ax.set_xticklabels(["0", "50k", "100k", "150k"])


def end_labels(ax, items, ymax, fmt="{:.1f}%"):
    """Direct-label each curve end, nudging apart any that would collide.

    The labels are the relief for the sub-3:1 contrast of the aqua slot, so they
    have to stay legible: when two ends sit closer than 5% of the axis range they
    are pushed symmetrically apart about their midpoint. Only the label moves --
    the printed value is still the true endpoint.
    """
    placed = sorted([list(it) for it in items], key=lambda it: it[0])
    gap = 0.05 * ymax
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < gap:
            mid = (placed[i][0] + placed[i - 1][0]) / 2.0
            placed[i - 1][0] = mid - gap / 2.0
            placed[i][0] = mid + gap / 2.0
    for ypos, xend, value, color in placed:
        ax.annotate(
            fmt.format(value),
            xy=(xend, ypos),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
            color=color,
            annotation_clip=False,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=2000, help="moving-average window (episodes)")
    ap.add_argument("--outdir", type=Path, default=REPO / "final_plots")
    ap.add_argument("--png-dpi", type=int, default=130,
                    help="raster dpi for the PNG copies (matches the sibling figures)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    w = args.window

    pal, _ = load("pal")
    pyr, _ = load("pyr")
    con, con_scen = load("con")
    gi = np.arange(1, len(con) + 1)
    is_pal = con_scen == "Palisades"
    is_pyr = con_scen == "Pyrenees"
    xmax = 150000

    def curve(y, idx):
        m = rolling(y, w)
        keep = idx >= w  # drop the warm-up where the window is not yet full
        return idx[keep], m[keep] * 100.0

    def final(y, n=20000):
        """Mean over the last n episodes of this series.

        The endpoint of a 2,000-episode moving average carries roughly +/-1
        point of noise on Pyrenees -- enough to flip the ordering between the
        two runs -- so curves are labelled with this larger, stabler window.
        """
        return float(y[-n:].mean() * 100.0)

    # ---- Figure 1: per-map, dedicated vs concurrent --------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)
    panels = [
        ("Palisades", pal, np.arange(1, len(pal) + 1), con[is_pal], gi[is_pal], 16, int(is_pal.sum())),
        ("Pyrenees", pyr, np.arange(1, len(pyr) + 1), con[is_pyr], gi[is_pyr], 100, int(is_pyr.sum())),
    ]
    for ax, (title, ded, di, cc, ci, ymax, n_con) in zip(axes, panels):
        dx, dy = curve(ded, di)
        cx, cy = curve(cc, ci)
        ax.plot(dx, dy, color=C_DEDICATED, lw=1.8, zorder=3, label="Dedicated (single map)")
        ax.plot(cx, cy, color=C_CONCURRENT, lw=1.8, zorder=3, label="Concurrent (both maps)")
        style(ax, ymax, xmax)
        end_labels(ax, [
            (dy[-1], dx[-1], final(ded), C_DEDICATED),
            (cy[-1], cx[-1], final(cc), C_CONCURRENT),
        ], ymax)
        ax.set_title(
            f"{title}   ({n_con:,} of 150,000 episodes concurrent)",
            fontsize=10.5, color=INK, weight="bold", loc="left", pad=8,
        )
        ax.set_xlabel("simulations", fontsize=9.5, color=INK_MUTED)
    axes[0].set_ylabel("fire-escape rate", fontsize=9.5, color=INK_MUTED)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_MUTED)
    for ax in axes:
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    fig.suptitle(
        "Concurrent vs. dedicated training at a matched 150,000-simulation budget",
        fontsize=12.5, color=INK, weight="bold", x=0.008, ha="left", y=0.99,
    )
    fig.text(
        0.008, 0.005,
        f"{w:,}-episode moving average of the binary escape indicator; labels give the mean over each run's final 20,000 episodes. "
        "5 seaplanes + 5 eVTOLs,\nswitch-ignition-4, directional-2 (K=3). Lower is better. Palisades: the two are indistinguishable. "
        "Pyrenees: concurrent is 1.0 point worse (z = 2.4).",
        fontsize=8.5, color=INK_MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    save(fig, args.outdir, "training_curves_by_map", args.png_dpi)
    plt.close(fig)

    # ---- Figure 2: the three runs as named -----------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=SURFACE)
    ends = []
    for y, idx, color, label in (
        (pyr, np.arange(1, len(pyr) + 1), C_CONCURRENT, "Pyrenees only"),
        (con, gi, C_THIRD, "Concurrent (both maps)"),
        (pal, np.arange(1, len(pal) + 1), C_DEDICATED, "Palisades only"),
    ):
        x, yy = curve(y, idx)
        ax.plot(x, yy, color=color, lw=1.8, zorder=3, label=label)
        ends.append((yy[-1], x[-1], final(y), color))
    style(ax, 100, xmax)
    end_labels(ax, ends, 100)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0f}%")
    ax.set_xlabel("simulations", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("fire-escape rate", fontsize=9.5, color=INK_MUTED)
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.46), frameon=False,
              fontsize=9, labelcolor=INK_MUTED)
    ax.set_title(
        "Training progress by run", fontsize=12.5, color=INK, weight="bold",
        loc="left", pad=10,
    )
    fig.text(
        0.008, 0.005,
        f"{w:,}-episode moving average of the binary escape indicator, matched 150,000-simulation budget; labels give the mean over the final 20,000 episodes.\n"
        "The concurrent curve blends two maps of very different difficulty and so sits between them by construction; "
        "see training_curves_by_map.pdf for the per-map comparison.",
        fontsize=8.5, color=INK_MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, args.outdir, "training_curves_three_runs", args.png_dpi)
    plt.close(fig)

    # ---- Figure 3: 3 x 2 grid ------------------------------------------
    # Columns: (1) every training instance on common scales; (2) Palisades --
    # dedicated run vs the theater-level run's Palisades episodes; (3) same
    # for Pyrenees. Top row MoE, bottom row escape rate. Color follows the
    # training instance everywhere: blue = Palisades only, orange = Pyrenees
    # only, aqua = theater level, so the same policy keeps one color whether
    # it appears whole (left) or as its per-map share (middle/right).
    moe = {k: load_moe(k) for k in ("con", "pal", "pyr")}
    esc = {"con": con, "pal": pal, "pyr": pyr}
    fidx = {k: np.arange(1, len(esc[k]) + 1) for k in esc}
    C_PAL, C_PYR, C_TH = C_DEDICATED, C_CONCURRENT, C_THIRD

    idx_map = {
        "pal": fidx["pal"], "pyr": fidx["pyr"], "con": fidx["con"],
        "con-pal": gi[is_pal], "con-pyr": gi[is_pyr],
    }
    esc_map = {
        "pal": esc["pal"], "pyr": esc["pyr"], "con": esc["con"],
        "con-pal": esc["con"][is_pal], "con-pyr": esc["con"][is_pyr],
    }
    moe_map = {
        "pal": moe["pal"], "pyr": moe["pyr"], "con": moe["con"],
        "con-pal": moe["con"][is_pal], "con-pyr": moe["con"][is_pyr],
    }
    color_map = {"pal": C_PAL, "pyr": C_PYR, "con": C_TH,
                 "con-pal": C_TH, "con-pyr": C_TH}
    cols = [
        ("All training instances", ["pal", "pyr", "con"]),
        ("Palisades", ["pal", "con-pal"]),
        ("Pyrenees", ["pyr", "con-pyr"]),
    ]

    def fmean(y, scale=1.0, n=20000):
        return float(y[-n:].mean()) * scale

    fig, axes = plt.subplots(
        2, 3, figsize=(13.2, 6.8), facecolor=SURFACE, sharex=True,
    )
    for j, (title, keys) in enumerate(cols):
        # -- top row: MoE ------------------------------------------------
        ax = axes[0][j]
        plotted = []
        drawn = []
        for k in keys:
            x, y = curve(moe_map[k], idx_map[k])
            y = y / 100.0  # curve() returns percent; MoE is a plain ratio
            drawn.append(y)
            ax.plot(x, y, color=color_map[k], lw=1.7, zorder=3)
            m = fmean(moe_map[k])
            plotted.append((m, x[-1], m, color_map[k]))
        if j == 0:
            lo, hi = -0.22, 1.06
        else:
            vals = np.concatenate(drawn)
            pad = 0.12 * float(vals.max() - vals.min())
            lo, hi = float(vals.min()) - pad, float(vals.max()) + pad
        if lo < 0.0 < hi:
            ax.axhline(0.0, color=GRID, lw=0.9, zorder=1)
        style(ax, hi, xmax)
        ax.set_ylim(lo, hi)
        if j == 0:
            ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        end_labels(ax, plotted, hi - lo, fmt="{:.3f}")
        ax.set_title(f"({'abc'[j]}) {title}", fontsize=11, color=INK,
                     weight="bold", loc="left", pad=7)

        # -- bottom row: escape rate ------------------------------------
        ax = axes[1][j]
        plotted = []
        drawn = []
        for k in keys:
            x, y = curve(esc_map[k], idx_map[k])
            drawn.append(y)
            ax.plot(x, y, color=color_map[k], lw=1.7, zorder=3)
            m = fmean(esc_map[k], 100.0)
            plotted.append((m, x[-1], m, color_map[k]))
        if j == 0:
            lo, hi = 0.0, 100.0
        else:
            vals = np.concatenate(drawn)
            pad = 0.12 * float(vals.max() - vals.min())
            lo, hi = max(0.0, float(vals.min()) - pad), float(vals.max()) + pad
        style(ax, hi, xmax)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_formatter(lambda v, _p: f"{round(v, 1):g}%")
        end_labels(ax, plotted, hi - lo, fmt="{:.1f}%")
        ax.set_title(f"({'def'[j]})", fontsize=11, color=INK,
                     weight="bold", loc="left", pad=7)

    axes[0][0].set_ylabel("MoE", fontsize=10, color=INK_MUTED)
    axes[1][0].set_ylabel("Fire Escape Rate", fontsize=10, color=INK_MUTED)
    axes[1][1].set_xlabel("Simulations", fontsize=10, color=INK_MUTED, labelpad=6)

    handles = [Line2D([], [], color=c, lw=2.2) for c in (C_PAL, C_PYR, C_TH)]
    fig.legend(
        handles,
        ["Palisades only", "Pyrenees only", "Theater level (both maps)"],
        loc="upper left", bbox_to_anchor=(0.065, 0.995), ncol=3, frameon=False,
        fontsize=9.5, labelcolor=INK_MUTED, handlelength=1.6, columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.07, right=0.958, top=0.90, bottom=0.095,
        hspace=0.34, wspace=0.36,
    )
    save(fig, args.outdir, "training_curves_grid", args.png_dpi, tight=False)
    plt.close(fig)

    # ---- The grid's six panels as stand-alone figures ------------------
    # Same data, colors, axis-limit rules and 20k-episode end labels as the
    # grid; each figure carries its own legend since there is no shared one.
    singles = args.outdir / "training"
    singles.mkdir(parents=True, exist_ok=True)
    run_label = {
        "pal": "Palisades only",
        "pyr": "Pyrenees only",
        "con": "Theater level (both maps)",
        "con-pal": "Theater level (both maps)",
        "con-pyr": "Theater level (both maps)",
    }
    col_stem = {
        "All training instances": "all_instances",
        "Palisades": "palisades",
        "Pyrenees": "pyrenees",
    }
    def single_panel(row, data_map, scale, j, title, keys, stem,
                     baseline_series=None, dashed_baselines=None):
        """One stand-alone panel; baseline_series = (per-episode array,
        label) draws a neutral moving-average curve through those episodes,
        same window and treatment as the training runs. dashed_baselines =
        [(per-episode array, label, color), ...] draws each as a dashed
        moving-average curve in the given color, so a baseline can sit next
        to the training instance it belongs to."""
        fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=SURFACE)
        plotted = []
        drawn = []
        for k in keys:
            x, y = curve(data_map[k], idx_map[k])
            if row == "moe":
                y = y / 100.0
            drawn.append(y)
            ax.plot(x, y, color=color_map[k], lw=1.7, zorder=3,
                    label=run_label[k])
            m = fmean(data_map[k], scale)
            plotted.append((m, x[-1], m, color_map[k]))
        if baseline_series is not None:
            arr, blabel = baseline_series
            bx, by = curve(arr, np.arange(1, arr.size + 1))
            if row == "moe":
                by = by / 100.0
            drawn.append(by)
            ax.plot(bx, by, color=C_NEUTRAL, lw=1.7, zorder=2, label=blabel)
            m = fmean(arr, scale)
            plotted.append((m, bx[-1], m, C_NEUTRAL))
        for arr, blabel, bcolor in (dashed_baselines or ()):
            bx, by = curve(arr, np.arange(1, arr.size + 1))
            if row == "moe":
                by = by / 100.0
            drawn.append(by)
            # Solid like the trained curves; identity rides on the light
            # tint plus the legend and end label (no dash needed, so no
            # decimation either).
            ax.plot(bx, by, color=bcolor, lw=1.7, zorder=2, label=blabel)
            m = fmean(arr, scale)
            plotted.append((m, bx[-1], m, bcolor))
        if j == 0:
            lo, hi = (-0.22, 1.06) if row == "moe" else (0.0, 100.0)
        else:
            vals = np.concatenate(drawn)
            pad = 0.12 * float(vals.max() - vals.min())
            lo, hi = float(vals.min()) - pad, float(vals.max()) + pad
            if row == "escape":
                lo = max(0.0, lo)
        if row == "moe" and lo < 0.0 < hi:
            ax.axhline(0.0, color=GRID, lw=0.9, zorder=1)
        style(ax, hi, xmax)
        ax.set_ylim(lo, hi)
        if row == "moe":
            if j == 0:
                ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            end_labels(ax, plotted, hi - lo, fmt="{:.3f}")
            ax.set_ylabel("MoE", fontsize=10, color=INK_MUTED)
        else:
            ax.yaxis.set_major_formatter(lambda v, _p: f"{round(v, 1):g}%")
            end_labels(ax, plotted, hi - lo, fmt="{:.1f}%")
            ax.set_ylabel("Fire Escape Rate", fontsize=10, color=INK_MUTED)
        ax.set_xlabel("Simulations", fontsize=10, color=INK_MUTED)
        # The three-run panels get an anchored legend in the empty band
        # between curves; "best" lands it on the theater-level curve there.
        if j == 0 and row == "moe" and dashed_baselines:
            # Six entries, solid column then dashed column, in the empty
            # band between the Palisades random line and the theater curve.
            leg = dict(loc="center right", bbox_to_anchor=(0.98, 0.69),
                       ncol=2, columnspacing=1.2)
        elif j == 0 and row == "moe":
            leg = dict(loc="center right", bbox_to_anchor=(0.98, 0.30))
        elif j == 0:
            leg = dict(loc="center left", bbox_to_anchor=(0.35, 0.30))
        else:
            leg = dict(loc="best")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, **leg)
        fig.tight_layout()
        save(fig, singles, stem, args.png_dpi)
        plt.close(fig)

    # ---- Random-tactics baseline series ---------------------------------
    # Used by the "_new" per-map variants and by moe_all_instances.
    # Moving average over the matched-regime random run's episodes (job
    # 855198, 2,000 sims per map under the training env settings), then
    # CONTINUED by bootstrap: episodes beyond the recorded 2,000 are drawn
    # with replacement from the recorded ones, so the curve keeps the run's
    # own episode-to-episode noise. Random tactics are stationary, so the
    # resample is distribution-preserving; the same resampled episode
    # indices feed both metrics, keeping the MoE and escape figures
    # consistent (episode k is one simulation in both). Targets: Palisades
    # to 100k, Pyrenees to 75k.
    # Palisades: the paired seed replay (job 864343) - 51,400 real episodes
    # on the exact fires of the mw training run, rows sorted back into
    # episode order (verified: sorted seeds == the training seed list).
    # Pyrenees: still the 2,000-sim matched-regime run; swap to the replay
    # (job 870546) when it completes.
    rand_src = {
        "palisades": (OUTPUTS / "random_tactics_mw_palisades_seed_replay_864343"
                      / "results.csv", "run"),
        "pyrenees": (OUTPUTS / "random_tactics_training_maps_855198"
                     / "results_pyrenees.csv", None),
    }
    rand_target = {"palisades": 150_000, "pyrenees": 150_000}
    rand_rng = np.random.default_rng(855198)
    rand_series: dict[str, dict[str, np.ndarray]] = {}
    rand_n_real: dict[str, int] = {}
    # Persist the full per-episode series (real + bootstrap continuation)
    # for downstream use; the fixed rng seed above makes it reproducible.
    series_dir = OUTPUTS / "random_tactics_extended_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for map_key, target in rand_target.items():
        path, sort_col = rand_src[map_key]
        with path.open() as handle:
            rrows = list(csv.DictReader(handle))
        if sort_col is not None:
            rrows.sort(key=lambda r: int(r[sort_col]))
        moe_vals = np.array([float(r["moe_cumulative_reward"]) for r in rrows])
        esc_vals = np.array(
            [float(r["propagation_factor"]) == 1.0 for r in rrows], dtype=float
        )
        n_real = moe_vals.size
        rand_n_real[map_key] = n_real
        extra = rand_rng.integers(0, n_real, size=max(0, target - n_real))
        rand_series[map_key] = {
            "moe": np.concatenate([moe_vals, moe_vals[extra]]),
            "escape": np.concatenate([esc_vals, esc_vals[extra]]),
        }
        out_path = series_dir / f"{map_key}_random_150k.csv"
        with out_path.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["episode", "moe", "escaped", "source"])
            for i, (mv, ev) in enumerate(
                zip(rand_series[map_key]["moe"],
                    rand_series[map_key]["escape"]), start=1):
                writer.writerow([i, f"{mv:.6f}", int(ev),
                                 "real" if i <= n_real else "bootstrap"])
        print(f"wrote {out_path} ({n_real:,} real + {target - n_real:,} bootstrap)")
    # Theater-level random baseline: STATIONARY, since random tactics do not
    # learn. Episode k's map is drawn i.i.d. with the Palisades share random
    # tactics would themselves realise under the 64/64 concurrent worker
    # split - episodes complete in inverse proportion to their duration, and
    # job 855198 measured Palisades 2,616 s / Pyrenees 1,976 s per 2,000
    # episodes -> share 1976/(1976+2616) = 0.430. The outcome is then drawn
    # from that map's RECORDED random episodes (real rows only, no bootstrap
    # continuation), so the curve oscillates flat around the mix-weighted
    # level 0.430*0.869 + 0.570*(-0.110) ~= +0.31 with the recorded runs'
    # own episode noise. (An earlier variant followed the TRAINED run's
    # per-episode map sequence; its upward drift was the trained policy's
    # changing episode-length mix, not anything random tactics do.)
    # Separate rng so the per-map series above stay bit-identical.
    TH_SHARE_PAL = 1976.0 / (1976.0 + 2616.0)
    th_rng = np.random.default_rng(870545)
    th_is_pal = th_rng.random(con.size) < TH_SHARE_PAL
    th_series = {
        "moe": np.empty(con.size, dtype=float),
        "escape": np.empty(con.size, dtype=float),
    }
    for map_key, mask in (("palisades", th_is_pal), ("pyrenees", ~th_is_pal)):
        pick = th_rng.integers(0, rand_n_real[map_key], size=int(mask.sum()))
        for rowkey in ("moe", "escape"):
            th_series[rowkey][mask] = rand_series[map_key][rowkey][pick]
    th_path = series_dir / "theater_random_150k.csv"
    with th_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["episode", "map", "moe", "escaped", "source"])
        for i, (pal, mv, ev) in enumerate(
            zip(th_is_pal, th_series["moe"], th_series["escape"]), start=1
        ):
            writer.writerow([i, "Palisades" if pal else "Pyrenees",
                             f"{mv:.6f}", int(ev), "resampled"])
    real_mean = {
        m: float(rand_series[m]["moe"][: rand_n_real[m]].mean())
        for m in ("palisades", "pyrenees")
    }
    print(
        f"wrote {th_path}: random MoE means Palisades {real_mean['palisades']:.4f} "
        f"({rand_n_real['palisades']:,} real), Pyrenees {real_mean['pyrenees']:.4f} "
        f"({rand_n_real['pyrenees']:,} real); stationary Palisades share "
        f"{TH_SHARE_PAL:.4f} -> mix-weighted level "
        f"{TH_SHARE_PAL * real_mean['palisades'] + (1 - TH_SHARE_PAL) * real_mean['pyrenees']:.4f}, "
        f"series mean {th_series['moe'].mean():.4f}"
    )

    # ---- Draw the six stand-alone panels --------------------------------
    # moe_all_instances additionally carries the random-tactics baseline of
    # each training instance as a dashed line in that instance's color.
    for row, data_map, scale in (("moe", moe_map, 1.0), ("escape", esc_map, 100.0)):
        for j, (title, keys) in enumerate(cols):
            dashed = None
            if row == "moe" and j == 0:
                dashed = [
                    (rand_series["palisades"]["moe"], "Random tactics, Palisades", lighten(C_PAL, 0.45)),
                    (rand_series["pyrenees"]["moe"], "Random tactics, Pyrenees", lighten(C_PYR, 0.45)),
                    (th_series["moe"], "Random tactics, theater level", lighten(C_TH, 0.45)),
                ]
            single_panel(row, data_map, scale, j, title, keys,
                         f"{row}_{col_stem[title]}", dashed_baselines=dashed)

    # ---- "_new" per-map variants with the neutral random baseline --------
    for row, data_map, scale in (("moe", moe_map, 1.0), ("escape", esc_map, 100.0)):
        for j, (title, keys) in enumerate(cols):
            if j == 0:
                continue
            map_key = col_stem[title]
            single_panel(row, data_map, scale, j, title, keys,
                         f"{row}_{map_key}_new",
                         baseline_series=(rand_series[map_key][row],
                                          "Random tactics"))


if __name__ == "__main__":
    main()
