#!/usr/bin/env bash
# Generic chunk of a chained training run. Not submitted directly: use
#   scripts/chain_submit.sh chains/<run>.env [chunk]
# which fills in --job-name/-t/--qos from the config and exports CHAIN_CONFIG
# and CHAIN_CHUNK.
#
# One chunk = one <=24h job. It resumes the previous chunk's policy, trains its
# own budget, saves, and prints the command for the next chunk.
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

PROJECT_DIR=/home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
cd "$PROJECT_DIR"
mkdir -p logs

: "${CHAIN_CONFIG:?CHAIN_CONFIG not set - submit with scripts/chain_submit.sh}"
: "${CHAIN_CHUNK:?CHAIN_CHUNK not set - submit with scripts/chain_submit.sh}"

# shellcheck source=/dev/null
source "$CHAIN_CONFIG"

: "${RUN_NAME:?RUN_NAME missing from $CHAIN_CONFIG}"
: "${SCENARIO:?SCENARIO missing from $CHAIN_CONFIG}"
if ! declare -p TRAIN_ARGS >/dev/null 2>&1 || [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: TRAIN_ARGS array missing or empty in $CHAIN_CONFIG" >&2
  exit 2
fi

OUTDIR="examples/wildfire/data/scenarios/outputs/${RUN_NAME}_chain"
LATEST="$OUTDIR/latest.zip"
CHUNK_TAG=$(printf "chunk%02d" "$CHAIN_CHUNK")
mkdir -p "$OUTDIR"

if [[ -f "$OUTDIR/CHAIN_COMPLETE" ]]; then
  echo "CHAIN COMPLETE marker already present in $OUTDIR; nothing to do."
  cat "$OUTDIR/CHAIN_COMPLETE"
  exit 0
fi

# Manual stop (scripts/chain_stop.sh). Checked before any work so a queued
# successor that was already submitted turns into an instant no-op.
if [[ -f "$OUTDIR/STOP" ]]; then
  echo "STOP marker present in $OUTDIR; not training."
  cat "$OUTDIR/STOP"
  echo "Delete it and resubmit to continue the chain from its last save."
  exit 0
fi

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

echo "=== chain $RUN_NAME chunk $CHAIN_CHUNK (job $SLURM_JOB_ID) ==="
echo "config:  $CHAIN_CONFIG"
echo "outdir:  $OUTDIR"

CHAIN_FLAGS=()
if [[ -n "${CHAIN_TOTAL_TIMESTEPS:-}" ]]; then
  CHAIN_FLAGS+=(--chain-total-timesteps "$CHAIN_TOTAL_TIMESTEPS")
elif [[ -n "${CHAIN_TOTAL_SIMULATIONS:-}" ]]; then
  CHAIN_FLAGS+=(--chain-total-simulations "$CHAIN_TOTAL_SIMULATIONS")
else
  echo "ERROR: set CHAIN_TOTAL_TIMESTEPS or CHAIN_TOTAL_SIMULATIONS in $CHAIN_CONFIG" >&2
  exit 2
fi

# Per-chunk budget. Timesteps and simulations are mutually exclusive in the
# runner; either may be omitted to let MAX_TRAIN_HOURS bound the chunk.
BUDGET_FLAGS=()
if [[ -n "${CHUNK_TIMESTEPS:-}" ]]; then
  BUDGET_FLAGS+=(--timesteps "$CHUNK_TIMESTEPS")
elif [[ -n "${CHUNK_SIMULATIONS:-}" ]]; then
  BUDGET_FLAGS+=(--simulations "$CHUNK_SIMULATIONS")
fi
if [[ -n "${MAX_TRAIN_HOURS:-}" ]]; then
  BUDGET_FLAGS+=(--max-train-hours "$MAX_TRAIN_HOURS")
fi

# Resume from the previous chunk when there is one. --resume-summary auto picks
# the highest-N summary already in $OUTDIR, so episode numbering and the
# training curve continue instead of restarting.
RESUME_FLAGS=()
if [[ -f "$LATEST" ]]; then
  RESUME_FLAGS+=(--load-model "$LATEST" --resume-summary auto)
  echo "resuming from $LATEST"
  # Only needed when adopting a model whose stored timestep counter does not
  # reflect true chain progress; the runner errors out and says so if it does.
  if [[ -n "${CHAIN_ELAPSED_TIMESTEPS:-}" ]]; then
    RESUME_FLAGS+=(--chain-elapsed-timesteps "$CHAIN_ELAPSED_TIMESTEPS")
  fi
elif [[ "$CHAIN_CHUNK" -gt 1 ]]; then
  echo "ERROR: chunk $CHAIN_CHUNK requested but $LATEST does not exist." >&2
  echo "The previous chunk did not save. Check its log before resubmitting." >&2
  exit 3
else
  echo "no previous model; starting the chain from scratch"
fi

# Queue the next chunk NOW, held on this job finishing, so it is already in
# line rather than waiting for someone to notice this one ended. afterany (not
# afterok) means it also runs when SLURM kills this job at the wall limit,
# which is a state we resume from cleanly. If this chunk saves nothing the
# successor hits the latest.zip guard above and exits without training.
if [[ -z "${AUTO_SUBMIT_NEXT:-1}" || "${AUTO_SUBMIT_NEXT:-1}" == "0" ]]; then
  echo "auto-submit disabled; submit the next chunk by hand"
else
  NEXT=$((CHAIN_CHUNK + 1))
  # Hard stop so a misconfigured chain cannot submit forever. The runner's
  # CHAIN_COMPLETE marker normally ends it well before this.
  #
  # The per-chunk budget is an upper bound: MAX_TRAIN_HOURS routinely ends a
  # chunk early, so the real number of chunks can be several times the budget
  # estimate. Deriving the cap straight from the budget once stalled three
  # chains at chunk 12 when they needed ~17, so allow generous headroom - this
  # is a runaway guard, not a schedule. Set MAX_CHUNKS explicitly to override.
  if [[ -z "${MAX_CHUNKS:-}" ]]; then
    if [[ -n "${CHAIN_TOTAL_SIMULATIONS:-}" && -n "${CHUNK_SIMULATIONS:-}" ]]; then
      MAX_CHUNKS=$(( 4 * ((CHAIN_TOTAL_SIMULATIONS + CHUNK_SIMULATIONS - 1) / CHUNK_SIMULATIONS) + 4 ))
    elif [[ -n "${CHAIN_TOTAL_TIMESTEPS:-}" && -n "${CHUNK_TIMESTEPS:-}" ]]; then
      MAX_CHUNKS=$(( 4 * ((CHAIN_TOTAL_TIMESTEPS + CHUNK_TIMESTEPS - 1) / CHUNK_TIMESTEPS) + 4 ))
    else
      MAX_CHUNKS=80
    fi
  fi
  BASE_JOB_NAME=$(printf "%.13s" "${JOB_NAME:-$RUN_NAME}")
  NEXT_JOB_NAME=$(printf "%s-c%02d" "$BASE_JOB_NAME" "$NEXT")
  if [[ "$NEXT" -gt "$MAX_CHUNKS" ]]; then
    echo "not queueing chunk $NEXT: MAX_CHUNKS=$MAX_CHUNKS reached"
  elif squeue -u "$USER" -h -o "%j" | grep -qx -- "$NEXT_JOB_NAME"; then
    echo "not queueing chunk $NEXT: $NEXT_JOB_NAME is already in the queue"
  else
    SUBMIT_ARGS=(
      --job-name "$NEXT_JOB_NAME"
      --time "${SLURM_TIME:-24:00:00}"
      --qos "${SLURM_QOS:-owl_normal_short}"
      --cpus-per-task "${SLURM_CPUS:-96}"
      --dependency "afterany:$SLURM_JOB_ID"
      --export "ALL,CHAIN_CONFIG=$CHAIN_CONFIG,CHAIN_CHUNK=$NEXT"
    )
    [[ -n "${MAIL_USER:-}" ]] && SUBMIT_ARGS+=(--mail-type BEGIN,END --mail-user "$MAIL_USER")
    sbatch "${SUBMIT_ARGS[@]}" slurms/chain_chunk.slurm \
      && echo "queued chunk $NEXT (starts when job $SLURM_JOB_ID ends)" \
      || echo "WARNING: could not queue chunk $NEXT; submit it by hand" >&2
  fi
fi

set +e
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "$SCENARIO" \
  --output-dir "$OUTDIR" \
  --save-model latest.zip \
  --progress-file-tag "$RUN_NAME" \
  "${CHAIN_FLAGS[@]}" \
  "${BUDGET_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}" \
  "${TRAIN_ARGS[@]}"
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "chunk $CHAIN_CHUNK exited with status $STATUS; not advancing the chain." >&2
  exit $STATUS
fi

# Archive this chunk's endpoint alongside the rolling latest.zip.
if [[ -f "$LATEST" ]]; then
  cp -p "$LATEST" "$OUTDIR/${CHUNK_TAG}.zip"
fi

cat > "$OUTDIR/chain_state.env" <<EOF
LAST_CHUNK=$CHAIN_CHUNK
LAST_JOB_ID=$SLURM_JOB_ID
LAST_FINISHED=$(date -Is)
EOF

echo
echo "=== chunk $CHAIN_CHUNK done ==="
if [[ -f "$OUTDIR/CHAIN_COMPLETE" ]]; then
  echo "CHAIN COMPLETE - no further chunks needed."
  cat "$OUTDIR/CHAIN_COMPLETE"
  # One email for the whole chain, not one per chunk. Configs that want this
  # set FINAL_MAIL_USER and leave MAIL_USER unset, so SLURM sends no per-chunk
  # BEGIN/END mail and only this fires. Optional: chains without
  # FINAL_MAIL_USER behave exactly as before.
  if [[ -n "${FINAL_MAIL_USER:-}" ]]; then
    {
      echo "Chain $RUN_NAME finished."
      echo
      cat "$OUTDIR/CHAIN_COMPLETE"
      echo
      echo "last chunk: $CHAIN_CHUNK (job $SLURM_JOB_ID)"
      echo "outdir:     $OUTDIR"
    } | mail -s "[chain complete] $RUN_NAME" "$FINAL_MAIL_USER" \
      && echo "final notification sent to $FINAL_MAIL_USER" \
      || echo "WARNING: could not send final notification" >&2
  fi
  # The successor was queued before training started; it would run, see the
  # marker and exit immediately, so cancel it rather than burn a slot.
  NEXT_JOB_NAME=$(printf "%.13s-c%02d" "${JOB_NAME:-$RUN_NAME}" "$((CHAIN_CHUNK + 1))")
  PENDING=$(squeue -u "$USER" -h -o "%i %j" | awk -v n="$NEXT_JOB_NAME" '$2==n {print $1}')
  if [[ -n "$PENDING" ]]; then
    scancel $PENDING && echo "cancelled queued chunk $((CHAIN_CHUNK + 1)) (job $PENDING)"
  fi
else
  echo "The next chunk was queued when this job started; check with:"
  echo "  squeue -u $USER"
  echo "If it is not there, submit it with:"
  echo "  cd $PROJECT_DIR && scripts/chain_submit.sh $CHAIN_CONFIG"
fi
