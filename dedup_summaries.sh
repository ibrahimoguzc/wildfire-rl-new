#!/usr/bin/env bash
# Deduplicate cumulative results_*_summary.csv snapshots in PPO run output dirs.
# Each snapshot at step N is a strict superset's subset: rows 1..N. The keeper
# (training_episode_summaries.csv if present, else the highest-step snapshot) is
# a superset of every file we delete, so no episode data is lost.
#
# Usage: dedup_summaries.sh [dry|apply]   (default: dry)
set -euo pipefail
OUT="$HOME/FirefightingSoS/wildfire-rl-new/examples/wildfire/data/scenarios/outputs"
cd "$OUT"
mode="${1:-dry}"

total_del_bytes=0
total_del_files=0
declare -a skipped

for d in */; do
  d="${d%/}"
  shopt -s nullglob
  summaries=("$d"/results_*_summary.csv)
  shopt -u nullglob
  [ ${#summaries[@]} -eq 0 ] && continue

  master="$d/training_episode_summaries.csv"

  # highest-step snapshot
  best_step=-1; best_file=""
  for f in "${summaries[@]}"; do
    step="$(basename "$f" | sed -E 's/.*_([0-9]+)_summary\.csv$/\1/')"
    case "$step" in ''|*[!0-9]*) step=-1 ;; esac
    if [ "$step" -gt "$best_step" ]; then best_step="$step"; best_file="$f"; fi
  done

  if [ -f "$master" ]; then
    keeper="$master"
    # SAFETY: master must have >= rows than the highest snapshot, else skip dir
    mrows=$(wc -l < "$master"); srows=$(wc -l < "$best_file")
    if [ "$mrows" -lt "$srows" ]; then
      skipped+=("$d (master $mrows rows < snapshot $srows rows)"); continue
    fi
  else
    keeper="$best_file"   # the highest-step snapshot is the superset we keep
    [ -z "$keeper" ] && { skipped+=("$d (no valid keeper)"); continue; }
  fi

  for f in "${summaries[@]}"; do
    [ "$f" = "$keeper" ] && continue
    sz=$(stat -c%s "$f")
    total_del_bytes=$((total_del_bytes+sz)); total_del_files=$((total_del_files+1))
    [ "$mode" = "apply" ] && rm -f "$f"
  done
done

echo "mode: $mode"
echo "files to delete: $total_del_files"
awk "BEGIN{printf \"reclaim: %.1f GB\n\", $total_del_bytes/1073741824}"
set +u
if [ "${#skipped[@]}" -gt 0 ]; then
  echo "SKIPPED dirs (safety check, left untouched):"
  printf '  %s\n' "${skipped[@]}"
else
  echo "skipped dirs: none"
fi
