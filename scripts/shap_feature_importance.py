#!/usr/bin/env python
"""SHAP feature-importance analysis for a trained PPO wildfire policy.

Explains BOTH heads of the actor-critic:
  * value head    V(s)            -- what drives the critic's assessment
  * policy head   pi(a|s)         -- what drives the chosen tactic per agent

Pipeline
--------
1. Load the SB3 PPO checkpoint.
2. Build a WildfireHourlyEnv whose observation layout matches the model
   (env flags must match training; --include-scenario-features is auto-tried).
3. Roll out the *trained* policy to collect on-policy states (the SHAP
   background + explain sets must come from the policy's own state
   distribution -- zeros/random baselines give misleading attributions).
4. Wrap value / per-head action-logit as differentiable torch modules and run
   shap.GradientExplainer (expected gradients -- robust for arbitrary MLPs).
5. Save mean(|SHAP|) rankings (CSV) + bar charts, and a value-head beeswarm.

Install first (not bundled in rl-env):
    ~/.conda/envs/rl-env/bin/python -m pip install shap

Example
-------
    ~/.conda/envs/rl-env/bin/python scripts/shap_feature_importance.py \
        --model-path runs/job_557251/model.zip \
        --scenario "Palisades copy.json" --switch-ignition-mode 1 \
        --state-space updated --controlled-agent-count 5 \
        --episodes 40 --explain-size 800 --background-size 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from examples.wildfire.ppo_runnerv2 import (  # noqa: E402
    WildfireHourlyEnv,
    _resolve_scenario,
)

DEFAULT_OUTPUT_DIR = REPO / "examples" / "wildfire" / "snapshots" / "shap"


# --------------------------------------------------------------------------- #
# Environment construction matched to the checkpoint's observation layout
# --------------------------------------------------------------------------- #
def build_matching_env(args, model: PPO) -> WildfireHourlyEnv:
    """Build an env whose obs dim equals the model's, auto-trying the
    scenario-feature toggle (the most common layout ambiguity)."""
    model_dim = int(model.observation_space.shape[0])
    scenario_path = _resolve_scenario(args.scenario)

    base_kwargs = dict(
        scenario_path=scenario_path,
        decision_interval_minutes=args.decision_interval_minutes,
        switch_ignition_mode=args.switch_ignition_mode,
        state_fire_fronts=args.state_fire_fronts,
        state_space=args.state_space,
        controlled_agent_count=args.controlled_agent_count,
    )

    # Try the explicit toggle first, then the alternatives.
    if args.include_scenario_features == "auto":
        toggles = [None, True, False]
    else:
        toggles = [args.include_scenario_features == "true"]

    tried = []
    for toggle in toggles:
        env = WildfireHourlyEnv(include_scenario_features=toggle, **base_kwargs)
        env_dim = int(env.observation_space.shape[0])
        tried.append((toggle, env_dim))
        if env_dim == model_dim:
            print(f"env matched model obs dim={model_dim} "
                  f"(include_scenario_features={toggle})")
            return env
        env.close()

    raise SystemExit(
        f"Could not match model obs dim={model_dim}. Tried "
        f"{tried}. Adjust --state-space / --controlled-agent-count / "
        f"--state-fire-fronts / --include-scenario-features to match how "
        f"this checkpoint was trained."
    )


# --------------------------------------------------------------------------- #
# On-policy state collection
# --------------------------------------------------------------------------- #
def collect_states(env, model: PPO, episodes: int, seed: int, max_states: int,
                   save_path: Path | None = None,
                   initial: np.ndarray | None = None) -> np.ndarray:
    import time
    states: list[np.ndarray] = (
        [row for row in initial] if initial is not None and len(initial) else [])
    have = len(states)
    print(f"rollout: starting with {have} cached states, target {max_states}",
          flush=True)
    t0 = time.perf_counter()
    for ep in range(episodes):
        if len(states) >= max_states:
            break
        obs, _ = env.reset(seed=seed + have + ep,
                           options={"sim_seed": seed + have + ep})
        terminated = truncated = False
        ep_steps = 0
        while not (terminated or truncated):
            states.append(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_steps += 1
        print(f"  episode {ep + 1}/{episodes}: +{ep_steps} steps  "
              f"total={len(states)}  elapsed={time.perf_counter() - t0:.0f}s",
              flush=True)
        # Incremental save so a kill mid-rollout never wastes prior episodes.
        if save_path is not None:
            np.save(save_path, np.asarray(states, dtype=np.float32))
    arr = np.asarray(states, dtype=np.float32)
    print(f"collected {len(arr)} on-policy states", flush=True)
    return arr


# --------------------------------------------------------------------------- #
# Differentiable scalar wrappers around the SB3 policy
# --------------------------------------------------------------------------- #
def make_modules(model: PPO):
    import torch

    policy = model.policy
    policy.set_training_mode(False)

    class ValueHead(torch.nn.Module):
        def forward(self, obs):
            return policy.predict_values(obs)  # (N, 1)

    class PolicyHead(torch.nn.Module):
        """Logits of a single MultiDiscrete head (one agent decision)."""
        def __init__(self, head: int):
            super().__init__()
            self.head = head

        def forward(self, obs):
            dist = policy.get_distribution(obs).distribution  # list[Categorical]
            return dist[self.head].logits  # (N, n_actions)

    return ValueHead(), PolicyHead


# --------------------------------------------------------------------------- #
# SHAP runners
# --------------------------------------------------------------------------- #
def _to_tensor(arr, device):
    import torch
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def shap_value_head(value_module, background, explain, device, nsamples):
    import shap
    bg = _to_tensor(background, device)
    X = _to_tensor(explain, device)
    explainer = shap.GradientExplainer(value_module, bg)
    sv = explainer.shap_values(X, nsamples=nsamples)
    if isinstance(sv, list):
        sv = sv[0]
    return np.asarray(sv).reshape(len(explain), -1)  # (N, F)


def shap_policy_heads(policy_head_cls, n_heads, background, explain,
                      device, nsamples):
    """For each head, explain the greedy (top-logit) action's attribution."""
    import shap
    bg = _to_tensor(background, device)
    X = _to_tensor(explain, device)
    per_head = []
    for h in range(n_heads):
        module = policy_head_cls(h)
        explainer = shap.GradientExplainer(module, bg)
        sv, _idx = explainer.shap_values(X, ranked_outputs=1, nsamples=nsamples)
        if isinstance(sv, list):
            sv = sv[0]
        per_head.append(np.asarray(sv).reshape(len(explain), -1))  # (N, F)
        print(f"  policy head {h}: SHAP done")
    return per_head  # list of (N, F)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def save_ranking_csv(path: Path, names, importance):
    order = np.argsort(importance)[::-1]
    lines = ["rank,feature,mean_abs_shap"]
    for r, i in enumerate(order, 1):
        lines.append(f"{r},{names[i]},{importance[i]:.6e}")
    path.write_text("\n".join(lines) + "\n")
    print(f"saved -> {path}")
    return order


def bar_plot(path: Path, names, importance, title, top=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(importance)[::-1][:top]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(order))))
    ax.barh(y, importance[order], color="#3b7dd8")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def beeswarm(path: Path, shap_vals, explain, names, title, top=20):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return
    order = np.argsort(np.abs(shap_vals).mean(0))[::-1][:top]
    expl = shap.Explanation(
        values=shap_vals[:, order],
        data=explain[:, order],
        feature_names=[names[i] for i in order],
    )
    plt.figure()
    shap.plots.beeswarm(expl, max_display=top, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True,
                    help="Path to the trained PPO .zip checkpoint (run ID).")
    ap.add_argument("--scenario", default="Palisades copy.json")
    ap.add_argument("--switch-ignition-mode", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--state-space", default="old")
    ap.add_argument("--state-fire-fronts", type=int, default=5)
    ap.add_argument("--controlled-agent-count", type=int, default=5)
    ap.add_argument("--include-scenario-features", default="auto",
                    choices=("auto", "true", "false"))
    ap.add_argument("--decision-interval-minutes", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=40,
                    help="Rollout episodes for state collection.")
    ap.add_argument("--explain-size", type=int, default=800)
    ap.add_argument("--background-size", type=int, default=100)
    ap.add_argument("--nsamples", type=int, default=64,
                    help="GradientExplainer local samples (speed/accuracy).")
    ap.add_argument("--targets", nargs="+", default=["value", "policy"],
                    choices=("value", "policy"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--states-cache", default=None,
                    help="Path to cache/reuse collected on-policy states "
                         "(.npy). Defaults to <output-dir>/<tag>_states_sw<m>.npy. "
                         "Lets re-runs skip the expensive rollout.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit(
            "shap is not installed in this env. Install with:\n"
            "  ~/.conda/envs/rl-env/bin/python -m pip install shap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model_path).stem

    print(f"loading model: {args.model_path}")
    model = PPO.load(args.model_path, device=args.device, print_system_info=False)
    env = build_matching_env(args, model)
    names = list(env.state_feature_names)
    assert len(names) == int(model.observation_space.shape[0]), (
        "feature-name count != model obs dim; env flags still mismatched")

    needed = args.explain_size + args.background_size
    cache_path = (Path(args.states_cache) if args.states_cache
                  else args.output_dir / f"{tag}_states_sw{args.switch_ignition_mode}.npy")
    cached = None
    if cache_path.exists():
        cached = np.load(cache_path)
        print(f"loaded {len(cached)} cached states from {cache_path}")
        if cached.shape[1] != len(names):
            raise SystemExit(f"cached states feature dim {cached.shape[1]} != "
                             f"{len(names)}; delete {cache_path} and rerun.")
    if cached is not None and len(cached) >= needed:
        states = cached
    else:
        # Rollout is the expensive part (ABM sim). Collect just enough + margin,
        # resuming from / saving to the cache so re-runs skip finished episodes.
        states = collect_states(env, model, args.episodes, args.seed,
                                max_states=needed + 30, save_path=cache_path,
                                initial=cached)
    env.close()
    if len(states) < args.background_size + 50:
        raise SystemExit(f"Too few states collected ({len(states)}). "
                         f"Increase --episodes.")

    # Use whatever was collected, even if short of the requested explain size.
    explain_size = min(args.explain_size, len(states) - args.background_size)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(states))
    bg = states[idx[:args.background_size]]
    pool = idx[args.background_size:]
    explain = states[pool[:explain_size]]
    print(f"background={len(bg)}  explain={len(explain)}  features={len(names)}")

    value_module, policy_head_cls = make_modules(model)
    n_heads = int(len(model.action_space.nvec))

    if "value" in args.targets:
        print("== SHAP: value head V(s) ==")
        sv = shap_value_head(value_module, bg, explain, args.device, args.nsamples)
        imp = np.abs(sv).mean(0)
        save_ranking_csv(args.output_dir / f"{tag}_value_importance.csv", names, imp)
        bar_plot(args.output_dir / f"{tag}_value_importance.png", names, imp,
                 f"{tag}: value-head feature importance  mean(|SHAP|)")
        beeswarm(args.output_dir / f"{tag}_value_beeswarm.png", sv, explain, names,
                 f"{tag}: value-head SHAP")

    if "policy" in args.targets:
        print(f"== SHAP: policy head (greedy action), {n_heads} agent heads ==")
        per_head = shap_policy_heads(policy_head_cls, n_heads, bg, explain,
                                     args.device, args.nsamples)
        # Per-head importance, plus overall = mean over heads.
        head_imps = np.stack([np.abs(h).mean(0) for h in per_head])  # (K, F)
        overall = head_imps.mean(0)
        save_ranking_csv(args.output_dir / f"{tag}_policy_importance.csv",
                         names, overall)
        bar_plot(args.output_dir / f"{tag}_policy_importance.png", names, overall,
                 f"{tag}: policy feature importance (mean over {n_heads} heads)")
        # Per-head CSV for asymmetry inspection.
        hdr = "feature," + ",".join(f"head_{h}" for h in range(n_heads))
        rows = [hdr]
        for i, nm in enumerate(names):
            rows.append(nm + "," + ",".join(f"{head_imps[h, i]:.6e}"
                                            for h in range(n_heads)))
        (args.output_dir / f"{tag}_policy_importance_per_head.csv").write_text(
            "\n".join(rows) + "\n")
        print(f"saved -> {args.output_dir / f'{tag}_policy_importance_per_head.csv'}")

    print("done.")


if __name__ == "__main__":
    main()
