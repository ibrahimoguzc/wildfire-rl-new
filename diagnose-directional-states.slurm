#!/usr/bin/env bash
#SBATCH -J diag-dir-states
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=32G
#SBATCH --qos=owl_normal_short

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Measure how the directional state channels behave across Pyrenees and
# Palisades (3 episodes each) and flag channels that are noise -- i.e. stuck
# near 0 or near 1 for the whole episode (no variance, so the agent cannot
# learn anything from them). --cell-size code exercises the corrected flank
# geometry (2c/2e/2f). Pyrenees.json defines 9 aircraft (6 controlled here),
# Palisades.json defines 4 (4 controlled).
OUTDIR=examples/wildfire/data/scenarios/outputs/directional_state_diag_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

K=3
EPISODES=3
STEPS=36

# 1) Pyrenees -- 6 planes in 3 groups of 2.
python -u scripts/diagnose_directional_state.py \
  --scenario "Pyrenees.json" \
  --state-fire-fronts "$K" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --controlled-agent-count 6 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --water-set 1 \
  --action-mode random \
  --cell-size code \
  --output-dir "$OUTDIR"

# 2) Palisades -- 4 planes, individual tactics.
python -u scripts/diagnose_directional_state.py \
  --scenario "Palisades.json" \
  --state-fire-fronts "$K" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --controlled-agent-count 4 \
  --tactic-distribution individual \
  --water-set 1 \
  --action-mode random \
  --cell-size code \
  --output-dir "$OUTDIR"

# === Noise verdict ===
# Classify each channel (aggregated over all K slots, both scenarios) from the
# per-channel min/max/mean/std/nonzero_rate the diagnostic already wrote.
python - "$OUTDIR" <<'PY'
import csv, glob, os, sys
outdir = sys.argv[1]
rows = []
for csv_path in sorted(glob.glob(os.path.join(outdir, "*_summary.csv"))):
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("slot") == "all":          # channel aggregated over slots
                rows.append(r)

def verdict(mn, mx, mean, std, nzr):
    if mx < 0.02:
        return "NOISE (stuck ~0)"
    if mn > 0.90 and std < 0.03:
        return "NOISE (stuck ~1)"
    if nzr < 0.10:
        return "NEAR-NOISE (rarely nonzero)"
    if std < 0.02:
        return "NEAR-NOISE (near-constant)"
    return "informative"

print("\n================ DIRECTIONAL STATE NOISE VERDICT ================")
hdr = f"{'scenario':10s} {'channel':20s} {'min':>6} {'max':>6} {'mean':>6} {'std':>6} {'nz%':>5}  verdict"
print(hdr); print("-" * len(hdr))
for r in rows:
    mn, mx = float(r["min"]), float(r["max"])
    mean, std, nzr = float(r["mean"]), float(r["std"]), float(r["nonzero_rate"])
    v = verdict(mn, mx, mean, std, nzr)
    print(f"{r['scenario']:10s} {r['feature']:20s} {mn:6.3f} {mx:6.3f} "
          f"{mean:6.3f} {std:6.3f} {nzr*100:5.1f}  {v}")
print("================================================================\n")
PY

echo "Diagnostics + verdict written under ${OUTDIR}"
