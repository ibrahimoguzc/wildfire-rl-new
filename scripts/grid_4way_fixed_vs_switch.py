#!/usr/bin/env python
"""2x2 grid: Palisades vs Pyrenees, fixed vs switch ignition.

Each subplot shows the raw moe_cumulative_reward (light) and its 500-pt
moving average. Columns = ignition (fixed | switch), rows = scenario.
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

# grid position (row, col) -> (job id, folder, title, color)
RUNS = {
    (0, 0): ("563523", "palisades_lr00075_563523",
             "Palisades 4p — fixed (563523)", "#1f77b4"),
    (0, 1): ("563524", "palisades_lr00075_563524",
             "Palisades 4p — switch (563524)", "#1f77b4"),
    (1, 0): ("568021", "pyrenees_6p_g2_old_568021",
             "Pyrenees 6p — fixed (568021)", "#d62728"),
    (1, 1): ("571284", "pyrenees_6p_g2_directional_k2_sw1_571284",
             "Pyrenees 6p — switch (571284)", "#d62728"),
}


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


fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
for (r, c), (jid, folder, title, color) in RUNS.items():
    ax = axes[r][c]
    csv = latest_summary(folder)
    if not csv:
        ax.set_title(f"{title}\n(no data)")
        continue
    s = pd.to_numeric(pd.read_csv(csv, usecols=[COL])[COL], errors="coerce").to_numpy()
    s = s[~np.isnan(s)]
    ep = np.arange(s.size)
    m = ma(s, WIN)
    x = np.arange(WIN - 1, WIN - 1 + m.size)
    ax.plot(ep, s, color="0.82", lw=0.6, label=f"{COL} (raw)")
    ax.plot(x, m, color=color, lw=2.0, label=f"{WIN}-pt MA")
    ax.set_title(f"{title}   n={s.size}, final MA={m[-1]:.3f}", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    if r == 1:
        ax.set_xlabel("Episode")
    if c == 0:
        ax.set_ylabel(COL)
    print(f"{jid}: {s.size} episodes, final MA={m[-1]:.4f}")

axes[0][0].set_ylim(-0.4, 1.05)
fig.suptitle("Fixed vs switch ignition — Palisades (4p) & Pyrenees (6p)\n"
             "500-pt moving average of moe_cumulative_reward",
             fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save = "verification_plots/moe_ma500_4grid_fixed_vs_switch.png"
fig.savefig(save, dpi=130)
print(f"\nsaved -> {save}")
