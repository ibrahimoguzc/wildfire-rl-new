#!/usr/bin/env python
"""500-pt MA of moe_cumulative_reward for 583702 / 587423 / 589156 (Palisades, fleet-size comparison)."""
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

# (job id, output folder, label, color)
RUNS = [
    ("583702", "pyrenees_6p_g2_old_583702",
     "583702  Palisades 5p  old state, fixed ign", "#1f77b4"),
    ("587423", "palisades_lr00075_587423",
     "587423  Palisades 6p  old state, sw1", "#2ca02c"),
    ("589156", "palisades_4p_ind_directional_k3_fixed_589156",
     "589156  Palisades 4p  dir state, fixed ign", "#d62728"),
]


def latest_csv(folder):
    # newer runs write *_NNNN_summary.csv; older runs (583702) accumulate into
    # a single training_episode_summaries.csv
    files = glob.glob(os.path.join(OUT, folder, "*_summary.csv"))
    if files:
        def ck(f):
            m = re.search(r"_(\d+)_summary\.csv$", f)
            return int(m.group(1)) if m else -1
        return max(files, key=ck)
    legacy = os.path.join(OUT, folder, "training_episode_summaries.csv")
    return legacy if os.path.exists(legacy) else None


def ma(a, w):
    if a.size < w:
        w = max(1, a.size)
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


series = []
for jid, folder, label, color in RUNS:
    csv = latest_csv(folder)
    if not csv:
        print(f"{jid}: NO summary csv in {folder}")
        continue
    s = pd.to_numeric(pd.read_csv(csv, usecols=[COL])[COL], errors="coerce").to_numpy()
    s = s[~np.isnan(s)]
    if s.size == 0:
        print(f"{jid}: column empty")
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

fig, ax = plt.subplots(figsize=(13, 7))
for jid, label, color, s, m, x in series:
    ax.plot(x, m, color=color, lw=2.0,
            label=f"{label}  (n={s.size}, fin={m[-1]:.3f})")
ax.set_xlabel("Episode")
ax.set_ylabel(f"{COL}  ({WIN}-pt moving avg)")
ax.set_title("Palisades fleet-size comparison — 500-pt MA of moe_cumulative_reward\n"
              "583702=5p fixed/old state, 587423=6p sw1/old state, 589156=4p fixed/directional state")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
save = "verification_plots/moe_ma500_compare_583702_587423_589156.png"
fig.savefig(save, dpi=130)
print(f"saved -> {save}")
