#!/usr/bin/env bash
#
# A/B test: variant_a (shared 512->256 trunk + ReLU, ~half the params)
# vs default (independent pi=[512,256,128] / vf=[512,256,128], Tanh).
#
# Same scenario set, same timestep budget, same num_envs, same seed
# initialisation pattern (PPO uses its own randomness; both runs are
# initialised the same way by sb3). Differences in MoE reward / per-step
# wall / final value-loss / explained_variance are attributable to the
# architecture change.
#
# Switch-scenario (all 3 maps round-robin at 12 planes) is used because
# that's the realistic training distribution. 6912 timesteps = 4 rollouts
# at n_steps=144 * num_envs=12; enough to see learning signal stabilise
# but not so much that one variant eats the whole wall budget.
#
#SBATCH -J ab-arch-12p
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 06:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=160G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

NUM_ENVS=12
TIMESTEPS=6912   # 12 envs * 144 n_steps * 4 rollouts

OUTDIR_BASE=examples/wildfire/data/scenarios/outputs/ab_arch_$SLURM_JOB_ID
mkdir -p "$OUTDIR_BASE"

run_variant () {
  local arch="$1"
  local tag="$2"
  echo "============================================================"
  echo ">>> A/B PHASE: policy-arch=$arch (12-plane switch-scenario)"
  echo "============================================================"
  /usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
    --switch-scenario \
    --switch-scenarios "Palisades_12p.json" "Pyrenees_12p.json" "Salamis_12p.json" \
    --switch-ignition-1 \
    --num-envs "$NUM_ENVS" \
    --timesteps "$TIMESTEPS" \
    --device cpu \
    --policy-arch "$arch" \
    --profile-timings \
    --gc-collect-on-reset \
    --output-dir "$OUTDIR_BASE/${tag}" \
    --progress-file-tag "ab_${tag}" \
    --vec-start-method auto
}

# Run default first (matches prior baseline 501353/switch_all), then
# variant_a back-to-back so both share the same node / cache state.
run_variant "default"   "default"
run_variant "variant_a" "variant_a"

echo "============================================================"
echo ">>> DONE. Compare:"
echo "    $OUTDIR_BASE/default/   vs   $OUTDIR_BASE/variant_a/"
echo "    Diff their results_*_summary.csv (cumulative_moe_reward) and"
echo "    rollout time/value_loss/explained_variance printouts above."
echo "============================================================"
ls -la "$OUTDIR_BASE"
