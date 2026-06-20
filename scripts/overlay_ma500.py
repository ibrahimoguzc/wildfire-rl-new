#!/usr/bin/env python
"""Overlay 500-pt moving averages of moe_cumulative_reward for several runs."""
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
    ("568021", "pyrenees_6p_g2_old_568021",
     "568021  Pyrenees 6p g2  old 42f", "#7f7f7f"),
    ("568022", "pyrenees_6p_g2_directional_k2_568022",
     "568022  Pyrenees 6p g2  dir 54f k2", "#1f77b4"),
    ("569076", "pyrenees_6p_g2_directional_k2_sw1_569076",
     "569076  Pyrenees 6p g2  dir 54f k2 sw1", "#2ca02c"),
    ("569078", "pyrenees_6p_g2_directional_k4_569078",
     "569078  Pyrenees 6p g2  dir 66f k4", "#d62728"),
    ("569075", "palisades_directional_k4_sw1_569075",
     "569075  Palisades 4p  dir 62f k4 sw1", "#9467bd"),
]


def latest_summary(folder):
    files = glob.glob(os.path.join(OUT, folder, "*_summary.csv"))
    def ck(f):
        m = re.search(r"_(\d+)_summary\.csv$", f)
        return int(m.group(1)) if m else -1
    return max(files, key=ck) if files else None


def ma(a, w):
    if a.size < w:
        w = max(1, a.size)
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


fig, ax = plt.subplots(figsize=(13, 7))
for jid, folder, label, color in RUNS:
    csv = latest_summary(folder)
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
    ax.plot(x, m, color=color, lw=2.0, label=f"{label}  (n={s.size}, fin={m[-1]:.3f})")
    print(f"{jid}: {s.size} episodes, final MA={m[-1]:.4f}")

ax.set_xlabel("Episode")
ax.set_ylabel(f"{COL}  ({WIN}-pt moving avg)")
ax.set_title("PPO run comparison — 500-pt MA of moe_cumulative_reward")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
save = "verification_plots/moe_ma500_compare_569078-76-75_568022-21.png"
fig.savefig(save, dpi=130)
print(f"\nsaved -> {save}")
