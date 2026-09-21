#!/usr/bin/env bash
#SBATCH -J rand-mw-replay
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

# Random tactics on the EXACT fire instances of the multiweight Palisades
# 150k training run (job 855200): its summary CSV records the sim_seed of
# every episode in logged order, the env initialises the whole simulation
# from that seed, and ppo_tester --seed-file (added for this job) replays a
# seed list verbatim. Episode i of this evaluation is therefore the same
# fire as episode i of the training run - a fully paired random baseline
# for the training curve, aligned on the episode index.
#
# The seed list is snapshotted AT JOB START from the newest summary the
# training run has written (it rotates with --summary-retention 3); if
# 855200 has finished by then this replays all 150,000 episodes, otherwise
# the prefix recorded so far. The log prints the count.
#
# Scenario: Palisades5sp5ev_multiweight.json - the same file 855200 trains
# on. The per-type firefront cost weights matter even under random tactics
# (they score firefront choices inside the tactic pieces), so the
# multiweight scenario keeps the environment identical to training.
#
# TIME: Palisades random tactics measured 2,000 eps / 2,616 s on 64 workers
# (job 855198) ~= 2,750/h. Worst case 150k eps ~= 55 h; 72 h wall covers a
# slow node.
SRC=examples/wildfire/data/scenarios/outputs/palisades_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k_855200
OUTDIR=examples/wildfire/data/scenarios/outputs/random_tactics_mw_palisades_seed_replay_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

LATEST=$(ls -t "$SRC"/results_*_summary.csv | head -1)
echo "seed source: $LATEST"
python - "$LATEST" "$OUTDIR/seeds.txt" <<'PYEOF'
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
# The newest summary may still be mid-write; skip a torn trailing line.
df = pd.read_csv(src, on_bad_lines="skip")
seeds = pd.to_numeric(df["sim_seed"], errors="coerce").dropna().astype(int)
seeds.to_csv(dst, index=False, header=False)
print(f"{len(seeds)} seeds extracted (episode order preserved)")
PYEOF

python -u -m examples.wildfire.ppo_tester \
  --include-random-tactics \
  --scenarios Palisades5sp5ev_multiweight.json \
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
