#!/usr/bin/env python
"""4-way overlay: Palisades vs Pyrenees, fixed vs switch ignition.

500-pt moving average of moe_cumulative_reward. Color encodes scenario,
linestyle encodes ignition (solid = fixed, dashed = switch).
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

# (job id, output folder, label, color, linestyle)
RUNS = [
    ("563523", "palisades_lr00075_563523",
     "563523  Palisades 4p  fixed", "#1f77b4", "-"),
    ("563524", "palisades_lr00075_563524",
     "563524  Palisades 4p  switch", "#1f77b4", "--"),
    ("568021", "pyrenees_6p_g2_old_568021",
     "568021  Pyrenees 6p  fixed", "#d62728", "-"),
    ("571284", "pyrenees_6p_g2_directional_k2_sw1_571284",
     "571284  Pyrenees 6p  switch", "#d62728", "--"),
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
for jid, folder, label, color, ls in RUNS:
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
    ax.plot(x, m, color=color, ls=ls, lw=2.0,
            label=f"{label}  (n={s.size}, fin={m[-1]:.3f})")
    print(f"{jid}: {s.size} episodes, final MA={m[-1]:.4f}")

ax.set_xlabel("Episode")
ax.set_ylabel(f"{COL}  ({WIN}-pt moving avg)")
ax.set_title("Fixed vs switch ignition — Palisades (4p) & Pyrenees (6p)\n"
             "500-pt MA of moe_cumulative_reward")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)
fig.tight_layout()
save = "verification_plots/moe_ma500_4way_fixed_vs_switch.png"
fig.savefig(save, dpi=130)
print(f"\nsaved -> {save}")
