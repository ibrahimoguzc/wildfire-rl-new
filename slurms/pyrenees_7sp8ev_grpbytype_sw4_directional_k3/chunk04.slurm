#!/usr/bin/env bash
#SBATCH -J pyr7sp8ev-c4
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=BEGIN,END
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

# Chunk 4 of 5: simulations 90001-120000 of 150000.
# Pyrenees, 7 example_seaplane_1 + 8 example_aircraft_1 (eVTOL) = 15 aircraft.
# --group-sizes 7 8: one tactic decision for all 7 seaplanes and one for all 8
# eVTOLs (2 decisions/step). Seaplanes are listed first in the scenario, which
# is what the grouping relies on.
# --switch-ignition-4: 400x400 cell box (2.0x2.0 km ground) around the original
# ignition, relaxed 150 m urban keep-out. water set 2, directional state (K=3),
# --cell-size code for the corrected flank geometry.
#
# Every chunk writes into ONE folder, so the summary CSVs form a single
# continuous training curve across all five jobs:
OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_7sp8ev_grpbytype_sw4_directional_k3

# --chain-total-simulations 150000: the LR starts at --learning-rate and
# decays over the WHOLE 150000-simulation run, not over this chunk, so the
# schedule is continuous across the five jobs. This chunk resumes at 2.633e-04 (90000 simulations done).
# --simulations 30000: this chunk's budget.
# --max-train-hours 22.5: clean stop + save if 30000 simulations do not fit
# in the 24 h wall, rather than being killed by SLURM. If that happens this
# chunk saves fewer than 120000 simulations; the log and chain_progress.json
# carry the true count and the next chunk still resumes correctly.
# Queue chunk 5 now, held until this job ends, so training resumes right away
# instead of waiting for someone to submit it. afterany: it also runs if SLURM
# kills this job at the wall limit, which is a resumable state. Set
# NO_AUTO_NEXT=1 in the environment (sbatch --export=ALL,NO_AUTO_NEXT=1) to
# submit chunk 5 by hand instead.
if [[ -z "${NO_AUTO_NEXT:-}" ]] && ! squeue -u "$USER" -h -o "%j" | grep -qx "pyr7sp8ev-c5"; then
  sbatch --dependency=afterany:$SLURM_JOB_ID slurms/pyrenees_7sp8ev_grpbytype_sw4_directional_k3/chunk05.slurm \
    && echo "queued chunk 5 (starts when job $SLURM_JOB_ID ends)" \
    || echo "WARNING: could not queue chunk 5; submit it by hand" >&2
fi

/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees7sp8ev.json" \
  --output-dir "$OUTDIR" \
  --load-model "$OUTDIR/pyrenees_7sp8ev_grpbytype_sw4_directional_k3_sims090000.zip" \
  --resume-summary auto \
  --save-model pyrenees_7sp8ev_grpbytype_sw4_directional_k3_sims120000.zip \
  --progress-file-tag pyrenees_7sp8ev_grpbytype_sw4_directional_k3 \
  --chain-total-simulations 150000 \
  --simulations 30000 \
  --max-train-hours 22.5 \
  --num-envs 128 \
  --device cpu \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 7 8 \
  --controlled-agent-count 15 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional \
  --cell-size code \
  --summary-retention 3 \
  --checkpoint-interval 252000

echo
echo "=== chunk 4/5 done: pyrenees_7sp8ev_grpbytype_sw4_directional_k3_sims120000.zip ==="
echo "Chunk 5 was queued at start; check: squeue -u $USER"
echo "If it is missing: sbatch slurms/pyrenees_7sp8ev_grpbytype_sw4_directional_k3/chunk05.slurm"
