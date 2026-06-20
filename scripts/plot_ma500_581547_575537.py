#!/usr/bin/env python
"""500-pt MA of moe_cumulative_reward for 581547 and 575537 (individual + overlay).

Both are live, so the newest *_summary.csv may be a partial mid-write; fall
back to the latest readable file. Different fleet configs (9p g3 vs 4p ind),
so the overlay is a reward comparison, not a like-for-like ablation.
"""
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "examples/wildfire/data/scenarios/outputs"
WIN = 500
COL = "moe_cumulative_reward"

RUNS = [
    ("581547", "pyrenees_9p_g3_old_581547",
     "581547  Pyrenees 9p g3  old  (1M, running)", "#1f77b4"),
    ("575537", "pyrenees_4p_ind_old_2p5M_575537",
     "575537  Pyrenees 4p ind  old  (2.5M, running)", "#d62728"),
]


def load_latest_series(folder):
    """Return (csv_path, series) for the highest-numbered summary that parses."""
    files = glob.glob(os.path.join(OUT, folder, "*_summary.csv"))
    def ck(f):
        m = re.search(r"_(\d+)_summary\.csv$", f)
        return int(m.group(1)) if m else -1
    for csv in sorted(files, key=ck, reverse=True):
        try:
            s = pd.to_numeric(
                pd.read_csv(csv, usecols=[COL])[COL], errors="coerce"
            ).to_numpy()
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            continue
        s = s[~np.isnan(s)]
        if s.size:
            return csv, s
    return None, None


def ma(a, w):
    if a.size < w:
        w = max(1, a.size)
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


series = []
for jid, folder, label, color in RUNS:
    csv, s = load_latest_series(folder)
    if csv is None:
        print(f"{jid}: NO readable summary csv in {folder}")
        continue
    m = ma(s, WIN)
    x = np.arange(WIN - 1, WIN - 1 + m.size)
    series.append((jid, label, color, s, m, x))
    print(f"{jid}: {os.path.basename(csv)}  {s.size} episodes, final MA={m[-1]:.4f}, "
          f"peak MA={m.max():.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(s.size), s, color=color, alpha=0.15, lw=0.5)
    ax.plot(x, m, color=color, lw=2.0,
            label=f"{label}  (n={s.size}, fin={m[-1]:.3f})")
    # Autoscale to the MA range so the learning trend is readable despite
    # raw-episode spikes at the reward cap.
    pad = 0.1 * (m.max() - m.min() + 1e-9)
    ax.set_ylim(m.min() - pad, m.max() + pad)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{COL}  ({WIN}-pt moving avg)")
    ax.set_title(f"{label} — 500-pt MA of {COL}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save = f"verification_plots/moe_ma500_{jid}.png"
    fig.savefig(save, dpi=130)
    plt.close(fig)
    print(f"saved -> {save}")

if series:
    fig, ax = plt.subplots(figsize=(13, 7))
    for jid, label, color, s, m, x in series:
        ax.plot(x, m, color=color, lw=2.0,
                label=f"{label}  (n={s.size}, fin={m[-1]:.3f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{COL}  ({WIN}-pt moving avg)")
    ax.set_title("581547 (9p g3) vs 575537 (4p ind 2.5M) — 500-pt MA of "
                 "moe_cumulative_reward")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save = "verification_plots/moe_ma500_compare_581547_575537.png"
    fig.savefig(save, dpi=130)
    print(f"saved -> {save}")
