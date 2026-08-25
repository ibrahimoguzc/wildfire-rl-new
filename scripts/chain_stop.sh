#!/usr/bin/env bash
# Stop a self-chaining training run.
#
#   scripts/chain_stop.sh chains/<run>.env          # let the running chunk finish
#   scripts/chain_stop.sh chains/<run>.env --now    # also cancel the running chunk
#   scripts/chain_stop.sh chains/<run>.env --resume # clear the stop and continue
#
# Writes a STOP marker in the run's output folder, which every chunk checks
# before it trains, and cancels chunks already sitting in the queue. Whatever
# the run has learned is kept: each chunk saves at its end, and cancelling a
# running chunk raises SIGTERM, which the runner catches to write a rescue save
# plus chain_progress.json before exiting.

set -euo pipefail

PROJECT_DIR=/home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
cd "$PROJECT_DIR"

CONFIG=${1:-}
MODE=${2:-}
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "usage: scripts/chain_stop.sh chains/<run>.env [--now|--resume]" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG"
: "${RUN_NAME:?RUN_NAME missing from $CONFIG}"
OUTDIR="examples/wildfire/data/scenarios/outputs/${RUN_NAME}_chain"
BASE_JOB_NAME=$(printf "%.13s" "${JOB_NAME:-$RUN_NAME}")

if [[ "$MODE" == "--resume" ]]; then
  rm -f "$OUTDIR/STOP"
  echo "cleared $OUTDIR/STOP"
  echo "continue with: scripts/chain_submit.sh $CONFIG"
  exit 0
fi

mkdir -p "$OUTDIR"
cat > "$OUTDIR/STOP" <<EOF
stopped_at=$(date -Is)
stopped_by=$USER
config=$CONFIG
EOF
echo "wrote $OUTDIR/STOP"

mapfile -t JOBS < <(squeue -u "$USER" -h -o "%i %j %T" | awk -v n="$BASE_JOB_NAME" '$2 ~ "^"n"-c" {print}')
if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "no queued or running chunks for $BASE_JOB_NAME"
else
  for entry in "${JOBS[@]}"; do
    read -r id name state <<<"$entry"
    if [[ "$state" == "RUNNING" && "$MODE" != "--now" ]]; then
      echo "leaving $id ($name) running; it will save and stop, and nothing follows it"
      echo "  (use --now to cancel it immediately - it still saves via the SIGTERM handler)"
    else
      scancel "$id" && echo "cancelled $id ($name, was $state)"
    fi
  done
fi

if [[ -f "$OUTDIR/chain_progress.json" ]]; then
  echo -n "progress so far: "; tr -d '\n ' < "$OUTDIR/chain_progress.json"; echo
fi
echo "resume later with: scripts/chain_stop.sh $CONFIG --resume"
