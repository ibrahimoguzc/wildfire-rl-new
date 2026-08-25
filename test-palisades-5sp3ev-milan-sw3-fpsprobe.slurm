#!/usr/bin/env bash
#SBATCH -J pal5sp3ev-milan-fpsprobe
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --constraint=milan
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_5sp3ev_milan_fpsprobe_sw3_${SLURM_JOB_ID}

# ===================== THROUGHPUT PROBE, NOT A TRAINING RUN =================
# Purpose: separate the two confounded causes of the 638920 (11 fps) vs 639121
# (18-19 fps) gap. Those two jobs differed in BOTH
#   (a) node generation: owl-hm001 amd/milan (no AVX-512) vs owl030 amd/genoa
#       (avx512), and
#   (b) sosid version: the editable install was repointed from
#       rl-5-30/SoSID_X-Challenge-local to wildfire-rl-new/src at
#       2026-07-23 10:11:13, i.e. after 638920 started (08:08) but before
#       639121 started (14:16). The only behavioural difference between those
#       two sosid trees is trajectory.get_state_duration_segements returning
#       list(zip(...)) instead of a one-shot zip -- the eVTOL propellant fix.
#
# This job pins --constraint=milan while running CURRENT sosid. Its pair is
# job 647822 (palisades 5sp3ev sw3), identical configuration, unconstrained
# node. Comparing the two isolates hardware, because sosid is now the same for
# both:
#   * this job ~= 647822's fps  -> the node generation did NOT drive the gap,
#                                 so the sosid trajectory fix did.
#   * this job << 647822's fps  -> milan vs genoa is the dominant factor.
# Confirm 647822 actually landed on a genoa node before drawing a conclusion
# (sacct -j 647822 --format=NodeList).
#
# Sized to finish inside the wall rather than to train: 92160 timesteps = 5 PPO
# iterations at n_steps=144 x 128 envs. fps stabilises by iteration 2-3, so
# this is ample. At the 11 fps worst case it needs ~2.3 h of the 4 h wall.
# Rollout geometry is identical to 647822 (n_steps=144, batch 18432), so the
# per-step simulation cost is directly comparable; only the LR-decay
# denominator differs, which does not affect throughput.
# All model/scenario flags are byte-identical to 647822.
# ===========================================================================
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades5sp3ev.json" \
  --num-envs 128 \
  --timesteps 92160 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_5sp3ev_milan_fpsprobe_sw3.zip \
  --progress-file-tag palisades_5sp3ev_milan_fpsprobe_sw3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 5 3 \
  --controlled-agent-count 8 \
  --switch-ignition-3 \
  --water-set 2 \
  --state-space directional \
  --cell-size code
