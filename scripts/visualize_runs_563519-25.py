#!/usr/bin/env python
"""Compare PPO training runs 563519/563523/563524/563525.

All four: Palisades, lr=0.0005, 'old' 38-feat state, 4 planes, waterset1.
They differ by ignition randomization mode (see RUNS below).

Sources per run:
  * latest *_summary.csv  -> per-episode reward (moe_cumulative_reward) + final_metrics
  * slurm-<id>.out        -> per-episode "Mission Completed/Failed" + SB3 train/ tables
"""
import ast
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEW = "examples/wildfire/data/scenarios/outputs"
NEWLOG = "logs"
OLD = "/home/ibrahimoguz/FirefightingSoS/rl-5-30/SoSID_X-Challenge-git"

RUNS = {
    "563519": dict(label="563519  fixed ign. (baseline)", color="#7f7f7f",
                   out=f"{OLD}/examples/wildfire/data/scenarios/outputs/palisades_lr00075_563519",
                   log=f"{OLD}/logs/slurm-563519.out"),
    "563523": dict(label="563523  fixed ignition", color="#1f77b4",
                   out=f"{NEW}/palisades_lr00075_563523",
                   log=f"{NEWLOG}/slurm-563523.out"),
    "563524": dict(label="563524  switch-ign-1 (local box)", color="#2ca02c",
                   out=f"{NEW}/palisades_lr00075_563524",
                   log=f"{NEWLOG}/slurm-563524.out"),
    "563525": dict(label="563525  switch-ign-2 (map box)", color="#d62728",
                   out=f"{NEW}/palisades_lr00075_563525",
                   log=f"{NEWLOG}/slurm-563525.out"),
}

WIN = 500  # rolling-window (episodes)


def latest_summary(outdir):
    files = glob.glob(os.path.join(outdir, "results_*_summary.csv"))
    # checkpoint number is the integer before "_summary.csv"
    def ck(f):
        m = re.search(r"_(\d+)_summary\.csv$", f)
        return int(m.group(1)) if m else -1
    return max(files, key=ck) if files else None


def parse_metric(s, key):
    try:
        return ast.literal_eval(s).get(key, np.nan)
    except Exception:
        m = re.search(rf"'{key}':\s*([-\d.eE]+)", str(s))
        return float(m.group(1)) if m else np.nan


def load_episodes(outdir):
    f = latest_summary(outdir)
    if not f:
        return None
    df = pd.read_csv(f, usecols=["moe_cumulative_reward", "final_metrics"])
    out = pd.DataFrame()
    out["reward"] = pd.to_numeric(df["moe_cumulative_reward"], errors="coerce")
    out["casualties"] = df["final_metrics"].map(lambda s: parse_metric(s, "casualties"))
    out["burnt"] = df["final_metrics"].map(lambda s: parse_metric(s, "burnt_area_m2"))
    return out


def load_success(logpath):
    """Binary per-episode success from 'Mission Completed/Failed' lines, in order."""
    succ = []
    with open(logpath, errors="ignore") as fh:
        for line in fh:
            if "Mission Completed" in line:
                succ.append(1)
            elif "Mission Failed" in line:
                succ.append(0)
    return np.array(succ, dtype=float)


def load_train_tables(logpath):
    """Parse SB3 stdout tables -> dict of metric -> (timesteps, values)."""
    keys = ["total_timesteps", "value_loss", "policy_gradient_loss",
            "entropy_loss", "explained_variance", "approx_kl", "clip_fraction"]
    rows = []
    cur = {}
    pat = re.compile(r"\|\s*([a-z_]+)\s*\|\s*([-\d.eE+]+)\s*\|")
    with open(logpath, errors="ignore") as fh:
        for line in fh:
            if set(line.strip()) == {"-"} and line.strip():   # table border
                if "total_timesteps" in cur:
                    rows.append(cur)
                    cur = {}
                continue
            m = pat.match(line.strip())
            if m and m.group(1) in keys:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    if "total_timesteps" in cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def roll(a, w=WIN):
    s = pd.Series(a)
    return s.rolling(w, min_periods=max(10, w // 10)).mean().to_numpy()


# ---- load -------------------------------------------------------------------
data = {}
for rid, cfg in RUNS.items():
    ep = load_episodes(cfg["out"])
    succ = load_success(cfg["log"]) if os.path.exists(cfg["log"]) else np.array([])
    tt = load_train_tables(cfg["log"]) if os.path.exists(cfg["log"]) else pd.DataFrame()
    data[rid] = dict(ep=ep, succ=succ, tt=tt)
    n = 0 if ep is None else len(ep)
    print(f"{rid}: {n:6d} episodes | {len(succ):6d} mission-results | "
          f"{len(tt):4d} train-tables | success={np.mean(succ)*100 if len(succ) else 0:4.1f}%")

# ---- plot -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("PPO run comparison — Palisades, lr=5e-4, 4 planes, 'old' 38-feat state\n"
             f"(rolling window = {WIN} episodes)", fontsize=14, fontweight="bold")

axR, axS, axC, axV = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

for rid, cfg in RUNS.items():
    d = data[rid]
    c, lab = cfg["color"], cfg["label"]
    if d["ep"] is not None:
        r = d["ep"]["reward"].to_numpy()
        axR.plot(roll(r), color=c, label=lab, lw=1.8)
        axC.plot(roll(d["ep"]["casualties"].to_numpy()), color=c, label=lab, lw=1.8)
    if len(d["succ"]):
        axS.plot(roll(d["succ"]) * 100, color=c, label=lab, lw=1.8)
    if len(d["tt"]) and "explained_variance" in d["tt"]:
        tt = d["tt"]
        axV.plot(tt["total_timesteps"] / 1e6, tt["explained_variance"],
                 color=c, label=lab, lw=1.8)

axR.set_title("Episode reward (moe_cumulative_reward)")
axR.set_xlabel("episode"); axR.set_ylabel("reward"); axR.legend(fontsize=9)
axR.grid(alpha=0.3)

axS.set_title("Mission success rate")
axS.set_xlabel("episode"); axS.set_ylabel("% completed"); axS.legend(fontsize=9)
axS.grid(alpha=0.3); axS.set_ylim(-2, 102)

axC.set_title("Casualties per episode")
axC.set_xlabel("episode"); axC.set_ylabel("casualties"); axC.legend(fontsize=9)
axC.grid(alpha=0.3)

axV.set_title("Explained variance (value-fn fit)")
axV.set_xlabel("timesteps (millions)"); axV.set_ylabel("explained_variance")
axV.legend(fontsize=9); axV.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("graphs", exist_ok=True)
outpng = "graphs/run_comparison_563519-25.png"
fig.savefig(outpng, dpi=130)
print(f"\nsaved -> {outpng}")
