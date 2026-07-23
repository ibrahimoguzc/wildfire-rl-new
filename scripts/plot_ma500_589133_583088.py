#!/usr/bin/env python
"""500-pt MA of moe_cumulative_reward for 589133 and 583088 (individual + overlay)."""
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
    ("589133", "pyrenees_6p_ind_directional_k3_589133",
     "589133  Pyrenees 6p ind  directional k3", "#1f77b4"),
    ("583088", "pyrenees_6p_g2_old_583088",
     "583088  Pyrenees 6p g2  old state", "#d62728"),
]


def latest_csv(folder):
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
ax.set_title("589133 (directional state) vs 583088 (old state) — 500-pt MA of moe_cumulative_reward\n"
             "different state-space versions, reward scale not directly comparable")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
save = "verification_plots/moe_ma500_compare_589133_583088.png"
fig.savefig(save, dpi=130)
print(f"saved -> {save}")
