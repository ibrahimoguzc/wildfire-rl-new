#!/usr/bin/env bash
#
#SBATCH -J rl-eval
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -t 1:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mem=200G
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

MODEL=examples/wildfire/data/scenarios/outputs/switch_ignition_1m/switch_ignition_1m.zip

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_palisades_on_pyrenees

mkdir -p "$OUTDIR"

python -u -m examples.wildfire.ppo_tester \
  --model-path "$MODEL" \
  --include-random-tactics \
  --scenarios "Pyrenees_comparison.json" \
  --runs-per-scenario 200 \
  --seed 1 \
  --random-seeds \
  --device cpu \
  --num-workers 24 \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"
