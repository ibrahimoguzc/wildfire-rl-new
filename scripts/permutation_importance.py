#!/usr/bin/env python
"""Permutation feature importance for a trained PPO wildfire policy.

    python scripts/permutation_importance.py --model-path <zip> --states <npy>

Complements SHAP rather than repeating it. SHAP attributes a prediction among
inputs; with correlated inputs it splits credit arbitrarily, so a feature can
look unimportant only because a neighbour absorbed the attribution. Permutation
asks a different question - "if this input carried no information, how much
would the output move?" - and can be applied to a WHOLE BLOCK at once, which is
the only way to price a correlated cluster honestly. Permuting the 8 fire
bounding-box corners together cannot be masked by the corners covering for each
other.

Two metrics, matching the two SHAP heads:
  policy  mean KL( pi(.|s) || pi(.|s_permuted) ), summed over action heads
  value   mean |V(s) - V(s_permuted)|, as a % of the spread of V(s)

Both are computed on the cached on-policy states the SHAP run already saved, so
this needs no rollouts and runs in seconds.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
from stable_baselines3 import PPO  # noqa: E402
from examples.wildfire.ppo_runnerv2 import _state_feature_names  # noqa: E402


def blocks_for(names):
    """Group features the way the state vector is actually constructed."""
    g = {}
    for f in names:
        if f.startswith("agent_"):
            b = "per-agent x/y"
        elif f.startswith("scenario_is_"):
            b = "scenario one-hot"
        elif re.match(r".*_urgency_\d$", f):
            b = "per-front urgency"
        elif re.match(r"^front_severity_\d$", f):
            b = "per-front severity"
        elif re.match(r"^water_access_\d$", f):
            b = "per-front water access"
        elif f in {"leftmost_x","leftmost_y","rightmost_x","rightmost_y","uppermost_x",
                   "uppermost_y","lowermost_x","lowermost_y"}:
            b = "fire bounding box"
        elif f in {"spread_ray_hit_x","spread_ray_hit_y","spread_angle_deg"}:
            b = "spread geometry"
        elif f in {"fire_center_x","fire_center_y","ignition_x","ignition_y"}:
            b = "fire/ignition centre"
        elif f.startswith("distance_"):
            b = "distances to POI"
        else:
            b = "weather/time"
        g.setdefault(b, []).append(f)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--controlled-agent-count", type=int, default=10)
    ap.add_argument("--state-space", default="directional-2")
    ap.add_argument("--state-fire-fronts", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    obs = np.load(args.states).astype(np.float32)
    model = PPO.load(args.model_path, device="cpu", print_system_info=False)
    n_feat = obs.shape[1]
    names = list(_state_feature_names(
        include_scenario_flag=(n_feat > _base_width(args)),
        controlled_agent_count=args.controlled_agent_count,
        state_space=args.state_space, state_fire_fronts=args.state_fire_fronts))
    if len(names) != n_feat:
        raise SystemExit(f"name/width mismatch: {len(names)} names vs {n_feat} columns")

    pol = model.policy
    t = torch.as_tensor(obs)

    def forward(x):
        with torch.no_grad():
            feats = pol.extract_features(x)
            if pol.share_features_extractor:
                lat_pi, lat_vf = pol.mlp_extractor(feats)
            else:
                lat_pi, lat_vf = pol.mlp_extractor(feats[0]), pol.mlp_extractor(feats[1])[1]
            logits = pol.action_net(lat_pi)
            value = pol.value_net(lat_vf).squeeze(-1)
        return logits, value

    nvec = list(model.action_space.nvec)
    base_logits, base_v = forward(t)

    def split(logits):
        out, i = [], 0
        for k in nvec:
            out.append(torch.log_softmax(logits[:, i:i + k], dim=-1)); i += k
        return out

    base_logp = split(base_logits)
    v_spread = float(base_v.max() - base_v.min()) or 1.0
    rng = np.random.default_rng(args.seed)

    def score(cols):
        kls, dvs = [], []
        for _ in range(args.repeats):
            x = obs.copy()
            perm = rng.permutation(x.shape[0])
            for c in cols:
                x[:, c] = x[perm, c]
            lg, v = forward(torch.as_tensor(x))
            lp = split(lg)
            kl = sum(float((b.exp() * (b - p)).sum(-1).mean()) for b, p in zip(base_logp, lp))
            kls.append(kl); dvs.append(float((base_v - v).abs().mean()))
        return float(np.mean(kls)), float(np.mean(dvs)) / v_spread * 100.0

    idx = {f: i for i, f in enumerate(names)}
    print(f"model      : {Path(args.model_path).name}")
    print(f"states     : {obs.shape[0]} on-policy states x {n_feat} features")
    print(f"repeats    : {args.repeats}   (policy = KL nats, value = % of V spread)\n")

    grp = blocks_for(names)
    print(f"{'BLOCK (permuted together)':<28}{'n':>4}{'policy KL':>11}{'value %':>10}")
    print("-" * 53)
    rows = [(b, len(fs), *score([idx[f] for f in fs])) for b, fs in grp.items()]
    for b, n, kl, dv in sorted(rows, key=lambda r: -r[2]):
        print(f"{b:<28}{n:>4}{kl:>11.4f}{dv:>10.2f}")

    print(f"\n{'FEATURE (permuted alone)':<38}{'policy KL':>11}{'value %':>10}")
    print("-" * 59)
    single = [(f, *score([idx[f]])) for f in names]
    for f, kl, dv in sorted(single, key=lambda r: -r[1]):
        print(f"{f:<38}{kl:>11.4f}{dv:>10.2f}")

    if args.out:
        import pandas as pd
        pd.DataFrame(single, columns=["feature", "policy_kl", "value_pct"]).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


def _base_width(args):
    return len(_state_feature_names(
        include_scenario_flag=False,
        controlled_agent_count=args.controlled_agent_count,
        state_space=args.state_space, state_fire_fronts=args.state_fire_fronts))


if __name__ == "__main__":
    main()
