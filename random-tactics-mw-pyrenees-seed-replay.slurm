#!/usr/bin/env bash
#SBATCH -J rand-mw-pyr-replay
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 72:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_base
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail
mkdir -p logs
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Pyrenees sibling of the Palisades seed replay (864343): random tactics on
# the EXACT fire instances of the COMPLETED multiweight Pyrenees 150k
# training run (job 769428). Its final summary records the sim_seed of all
# 150,000 episodes in logged order; ppo_tester --seed-file replays them
# verbatim, so episode i here is the same fire as training episode i - a
# fully paired random baseline aligned on the episode index.
#
# Scenario: Pyrenees5sp5ev_multiweight.json, the file 769428 trained on
# (the per-type firefront cost weights shape tactic execution even under
# random tactic selection).
#
# TIME: Pyrenees random tactics measured 2,000 eps / 1,976 s on 64 workers
# (job 855198) ~= 3,640/h -> 150k ~= 41 h; 72 h wall covers a slow node.
SRC=examples/wildfire/data/scenarios/outputs/pyrenees_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k_769428
OUTDIR=examples/wildfire/data/scenarios/outputs/random_tactics_mw_pyrenees_seed_replay_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

LATEST=$(ls -t "$SRC"/results_*_summary.csv | head -1)
echo "seed source: $LATEST"
python - "$LATEST" "$OUTDIR/seeds.txt" <<'PYEOF'
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, on_bad_lines="skip")
seeds = pd.to_numeric(df["sim_seed"], errors="coerce").dropna().astype(int)
seeds.to_csv(dst, index=False, header=False)
print(f"{len(seeds)} seeds extracted (episode order preserved)")
PYEOF

python -u -m examples.wildfire.ppo_tester \
  --include-random-tactics \
  --scenarios Pyrenees5sp5ev_multiweight.json \
  --seed-file "$OUTDIR/seeds.txt" \
  --device cpu --num-workers 64 \
  --state-space directional-2 --state-fire-fronts 3 \
  --water-set 2 --switch-ignition 4 \
  --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"

python - "$OUTDIR" <<'PYEOF'
import sys, pandas as pd
out = sys.argv[1]
df = pd.read_csv(f"{out}/results.csv")
esc = (pd.to_numeric(df["propagation_factor"]) == 1.0).mean() * 100.0
moe = pd.to_numeric(df["moe_cumulative_reward"]).mean()
print(f"paired random baseline: n={len(df)}  mean MoE {moe:+.4f}  escape {esc:.1f}%")
PYEOF
echo "=== done: $OUTDIR ==="
