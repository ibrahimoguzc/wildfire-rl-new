"""Per-scenario timing breakdown of the wildfire RL environment.

Measures where wall-time goes, separately for each scenario, so we can decide
what to optimize. Times these phases:

  reset / total            : full env.reset() (sim rebuild + warm-up burn)
    - sim_build            : WildfireSimulation(...) construction
    - warmup_burn          : _advance_to_decision_start (detection-delay steps)
  step / total             : full env.step() under a random policy
    - apply_actions        : assign tactics to agents
    - advance_window        : the ~decision-interval sim steps (the hot loop)
        - fire_kernel       : wildfire.step() only
        - firefighter_abm   : firefighters/agents schedule step (advance - fire)
    - compute_metrics
    - compute_reward
    - compute_state

The advance_window split (fire vs ABM) is obtained by monkey-patching the two
model step() methods with timers for the duration of the run, then restored.

Usage:
  python -m examples.wildfire.profile_sim_timing \
      [--scenarios "Palisades copy.json" Pyrenees.json Salamis.json] \
      [--reset-samples 6] [--step-samples 60] [--seed 0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    WildfireHourlyEnv,
    _scenario_agent_count,
)

INPUTS = Path("examples/wildfire/data/scenarios/inputs")

# Per-scenario controlled-agent count: as many as we plan to use, capped by fleet.
DESIRED_CONTROLLED = {
    "Palisades copy.json": 6,
    "Pyrenees.json": 12,
    "Salamis.json": 3,
}


class _Timer:
    """Accumulates time and call count under a label."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def add(self, dt):
        self.total += dt
        self.calls += 1


def _patch_model_timers(env):
    """Wrap wildfire.step and firefighters.step to accumulate their own time.

    Returns (fire_timer, abm_timer, restore_fn). Must be called after the sim
    exists (i.e. after a reset).
    """
    sim = env.sim
    wildfire = sim.wildfire
    firefighters = sim.firefighters
    fire_t = _Timer()
    abm_t = _Timer()

    orig_fire = wildfire.step
    orig_abm = firefighters.step

    def timed_fire(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_fire(*a, **k)
        finally:
            fire_t.add(time.perf_counter() - t0)

    def timed_abm(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_abm(*a, **k)
        finally:
            abm_t.add(time.perf_counter() - t0)

    wildfire.step = timed_fire
    firefighters.step = timed_abm

    def restore():
        wildfire.step = orig_fire
        firefighters.step = orig_abm

    return fire_t, abm_t, restore


def profile_scenario(name, reset_samples, step_samples, seed):
    fleet = _scenario_agent_count(INPUTS / name)
    controlled = min(DESIRED_CONTROLLED.get(name, 6), fleet)
    group_size = 4 if controlled >= 4 else 1

    env = WildfireHourlyEnv(
        INPUTS / name,
        decision_interval_minutes=10,
        fire_detection_delay_minutes=30,
        switch_ignition_mode=2,
        state_space="mixed",
        gc_collect_on_reset=True,
        tactic_distribution="group",
        aircraft_group_size=group_size,
        controlled_agent_count=controlled,
    )

    # Warm up JIT / caches once (not counted).
    env.reset(seed=seed)

    # ---- RESET timing (sim build + warm-up burn) ----
    reset_t, build_t, warmup_t = [], [], []
    for i in range(reset_samples):
        t0 = time.perf_counter()
        env.reset(seed=seed + 100 + i)
        reset_t.append(time.perf_counter() - t0)
    # sub-phase split: measure sim build and warm-up burn directly
    for i in range(reset_samples):
        # build only
        from examples.wildfire.simulation import WildfireSimulation
        # use the env's already-selected parameters
        params = env.parameters
        t0 = time.perf_counter()
        sim = WildfireSimulation(parameters=params, seed=seed + 200 + i)
        sim.wildfire.ignite(sim.ignition_centers)
        build_t.append(time.perf_counter() - t0)
        # warm-up burn: step until detection delay (mission_runtime is a timedelta)
        delay_s = env.fire_detection_delay_seconds
        t0 = time.perf_counter()
        while (sim.timer.mission_runtime.total_seconds() < delay_s
               and sim.is_running):
            sim.step(force=True)
        warmup_t.append(time.perf_counter() - t0)

    # ---- STEP timing under random policy ----
    nA = int(env.action_space.nvec[0])
    ndec = env.action_space.shape[0]
    rng = np.random.default_rng(seed)

    # Persistent fire/abm accumulators. Each reset rebuilds the sim, so we
    # re-patch the fresh model objects and feed them the SAME accumulators.
    fire_total = _Timer()
    abm_total = _Timer()

    def patch_into(env, fire_acc, abm_acc):
        wildfire = env.sim.wildfire
        firefighters = env.sim.firefighters
        orig_fire = wildfire.step
        orig_abm = firefighters.step

        def timed_fire(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig_fire(*a, **k)
            finally:
                fire_acc.add(time.perf_counter() - t0)

        def timed_abm(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig_abm(*a, **k)
            finally:
                abm_acc.add(time.perf_counter() - t0)

        wildfire.step = timed_fire
        firefighters.step = timed_abm

    env.reset(seed=seed + 7)
    patch_into(env, fire_total, abm_total)

    phase = {k: _Timer() for k in
             ["apply", "advance", "metrics", "reward", "state", "full"]}
    eplens = []
    cur = 0
    for i in range(step_samples):
        a = rng.integers(0, nA, size=ndec)
        t_full = time.perf_counter()
        t = time.perf_counter(); env._apply_actions(a); phase["apply"].add(time.perf_counter() - t)
        t = time.perf_counter(); env._advance_time_window(); phase["advance"].add(time.perf_counter() - t)
        t = time.perf_counter(); metrics = env._compute_metrics(); phase["metrics"].add(time.perf_counter() - t)
        t = time.perf_counter(); env._compute_reward_and_info(metrics); phase["reward"].add(time.perf_counter() - t)
        env.current_step += 1
        t = time.perf_counter(); env._compute_state(); phase["state"].add(time.perf_counter() - t)
        phase["full"].add(time.perf_counter() - t_full)
        cur += 1
        done = (env.done or env.current_step >= env.max_steps
                or env._mission_complete())
        if done:
            eplens.append(cur); cur = 0
            env.reset(seed=seed + 1000 + i)   # rebuilds sim (fresh model objs)
            patch_into(env, fire_total, abm_total)  # re-wrap, same accumulators

    fire_t, abm_t = fire_total, abm_total

    return {
        "name": name, "fleet": fleet, "controlled": controlled,
        "reset_mean": np.mean(reset_t), "reset_std": np.std(reset_t),
        "build_mean": np.mean(build_t), "warmup_mean": np.mean(warmup_t),
        "phase": {k: (v.total / max(v.calls, 1)) for k, v in phase.items()},
        "fire_per_step": fire_t.total / max(phase["advance"].calls, 1),
        "abm_per_step": abm_t.total / max(phase["advance"].calls, 1),
        "fire_calls": fire_t.calls, "abm_calls": abm_t.calls,
        "advance_calls": phase["advance"].calls,
        "eplen_mean": np.mean(eplens) if eplens else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+",
                    default=["Palisades copy.json", "Pyrenees.json", "Salamis.json"])
    ap.add_argument("--reset-samples", type=int, default=6)
    ap.add_argument("--step-samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = []
    for name in args.scenarios:
        print(f"\n[profiling] {name} ...", flush=True)
        results.append(profile_scenario(name, args.reset_samples,
                                        args.step_samples, args.seed))

    def ms(x):
        return f"{x*1000:8.1f}ms"

    print("\n" + "=" * 78)
    print("RESET breakdown (per reset)")
    print("=" * 78)
    print(f"{'scenario':22s}{'fleet/ctrl':>11}{'reset':>12}{'sim_build':>12}{'warmup_burn':>13}")
    for r in results:
        fc = f"{r['fleet']}/{r['controlled']}"
        print(f"{r['name']:22s}{fc:>11}"
              f"{ms(r['reset_mean'])}{ms(r['build_mean'])}{ms(r['warmup_mean'])}")

    print("\n" + "=" * 78)
    print("STEP breakdown (per RL decision, random policy)")
    print("=" * 78)
    print(f"{'scenario':22s}{'full':>11}{'advance':>11}{'apply':>9}{'metrics':>9}{'reward':>9}{'state':>9}")
    for r in results:
        p = r["phase"]
        print(f"{r['name']:22s}{ms(p['full'])}{ms(p['advance'])}"
              f"{ms(p['apply'])}{ms(p['metrics'])}{ms(p['reward'])}{ms(p['state'])}")

    print("\n" + "=" * 78)
    print("ADVANCE-WINDOW split: fire kernel vs firefighter ABM (per RL decision)")
    print("=" * 78)
    print(f"{'scenario':22s}{'advance':>11}{'fire_kernel':>13}{'ffighter_abm':>14}{'fire%':>7}{'abm%':>7}{'eplen':>7}")
    for r in results:
        adv = r["phase"]["advance"]
        fire = r["fire_per_step"]; abm = r["abm_per_step"]
        tot = max(fire + abm, 1e-9)
        print(f"{r['name']:22s}{ms(adv)}{ms(fire)}{ms(abm)}"
              f"{100*fire/tot:6.0f}%{100*abm/tot:6.0f}%{r['eplen_mean']:7.1f}")
    print()


if __name__ == "__main__":
    main()
