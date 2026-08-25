#!/usr/bin/env bash
# Submit one chunk of a chained training run.
#
#   scripts/chain_submit.sh chains/<run>.env [chunk]
#
# With no chunk number it submits the next one: (number of archived chunks) + 1.
# Refuses to submit when the chain is already complete, and warns when a chunk
# for this run is still queued or running.

set -euo pipefail

PROJECT_DIR=/home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
cd "$PROJECT_DIR"

CONFIG=${1:-}
if [[ -z "$CONFIG" ]]; then
  echo "usage: scripts/chain_submit.sh chains/<run>.env [chunk]" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG"
: "${RUN_NAME:?RUN_NAME missing from $CONFIG}"

OUTDIR="examples/wildfire/data/scenarios/outputs/${RUN_NAME}_chain"
# Created up front so counting archived chunks below works on chunk 1 too.
mkdir -p "$OUTDIR"

if [[ -f "$OUTDIR/CHAIN_COMPLETE" ]]; then
  echo "chain '$RUN_NAME' is already complete:"
  cat "$OUTDIR/CHAIN_COMPLETE"
  echo "(delete $OUTDIR/CHAIN_COMPLETE to extend it, after raising the chain total)"
  exit 0
fi

if [[ -f "$OUTDIR/STOP" ]]; then
  echo "chain '$RUN_NAME' is stopped:"
  cat "$OUTDIR/STOP"
  echo "clear it with: scripts/chain_stop.sh $CONFIG --resume"
  exit 0
fi

# Next chunk = archived chunks + 1. An interrupted chunk leaves no archive, so
# its index is reused, which is what we want.
if [[ -n "${2:-}" ]]; then
  CHUNK=$2
else
  ARCHIVED=$(find "$OUTDIR" -maxdepth 1 -name 'chunk*.zip' 2>/dev/null | wc -l)
  CHUNK=$((ARCHIVED + 1))
fi

BASE_JOB_NAME=$(printf "%.13s" "${JOB_NAME:-$RUN_NAME}")
CHUNK_JOB_NAME=$(printf "%s-c%02d" "$BASE_JOB_NAME" "$CHUNK")

# An explicit chunk number is the override: you have said which chunk you mean,
# so the "something is already queued" guard does not apply.
if [[ -z "${2:-}" ]] && squeue -u "$USER" -h -o "%j" | grep -q -- "^${BASE_JOB_NAME}-c"; then
  echo "warning: a chunk of this run is already queued or running:"
  squeue -u "$USER" -o "%.10i %.20j %.10T %.10M" | grep -- "${BASE_JOB_NAME}-c" || true
  if [[ -t 0 ]]; then
    read -r -p "submit anyway? [y/N] " reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || exit 0
  else
    echo "not submitting; pass the chunk number explicitly to override." >&2
    exit 1
  fi
fi

SBATCH_ARGS=(
  --job-name "$CHUNK_JOB_NAME"
  --time "${SLURM_TIME:-24:00:00}"
  --qos "${SLURM_QOS:-owl_normal_short}"
  --cpus-per-task "${SLURM_CPUS:-96}"
  --export "ALL,CHAIN_CONFIG=$CONFIG,CHAIN_CHUNK=$CHUNK"
)
if [[ -n "${MAIL_USER:-}" ]]; then
  SBATCH_ARGS+=(--mail-type BEGIN,END --mail-user "$MAIL_USER")
fi

echo "run:    $RUN_NAME"
echo "chunk:  $CHUNK"
echo "outdir: $OUTDIR"
if [[ -n "${CHAIN_DRY_RUN:-}" ]]; then
  echo "dry run, would submit:"
  echo "  sbatch ${SBATCH_ARGS[*]} slurms/chain_chunk.slurm"
  exit 0
fi
sbatch "${SBATCH_ARGS[@]}" slurms/chain_chunk.slurm
