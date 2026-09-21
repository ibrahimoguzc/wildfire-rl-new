#!/usr/bin/env bash
#SBATCH -J eval-stats-5arm
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail
mkdir -p logs
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Fires on afterok:745463 - the 5-arm / 500-simulation evaluation. Two CPUs and
# 30 minutes: this is pandas and scipy over a 5,000-row CSV, seconds of work.
# It is a separate job rather than a tail on the eval script so that a failure
# in the analysis cannot cost the 5,000 episodes, and so it can be re-run
# against the same CSV without recomputing anything.
EVAL_JOB=745463
OUTDIR=$(ls -d examples/wildfire/data/scenarios/outputs/eval_5arm_100k_${EVAL_JOB} 2>/dev/null)
CSV="$OUTDIR/results.csv"
if [[ ! -f "$CSV" ]]; then
  echo "ERROR: $CSV not found - did job $EVAL_JOB actually write results?" >&2
  exit 3
fi

REPORT="$OUTDIR/statistical_comparison.txt"
{
  echo "Statistical comparison of the five curriculum arms"
  echo "generated $(date -Is) from job ${EVAL_JOB}"
  echo
  # All pairs within each map: Wilson CIs on containment, exact McNemar on the
  # paired binary outcome, paired t and Wilcoxon on MoE, Holm across the family.
  python -u scripts/eval_stats.py "$CSV"
  echo
  echo "###############################################################"
  echo "# Same comparison, every arm against the Pyrenees-only control #"
  echo "###############################################################"
  python -u scripts/eval_stats.py "$CSV" --reference arm2_pyrenees_only_100k
} | tee "$REPORT"

echo
echo "report written to $REPORT"
