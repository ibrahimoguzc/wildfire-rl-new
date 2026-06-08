#!/usr/bin/env bash
#SBATCH -J pal-dt1-probe
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=250G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Palisades copy.json" \
  --num-envs 36 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_dt1_probe.zip \
  --progress-file-tag palisades_dt1_probe \
  --vec-start-method auto \
  --switch-ignition-2 
