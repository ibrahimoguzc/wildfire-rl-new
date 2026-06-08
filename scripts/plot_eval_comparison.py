"""Compare PPO model vs random-tactics baseline from a ppo_tester results CSV.

Each row in the input is one evaluation episode. Rows are paired across
policies by `seed` (the tester runs every policy with the same seed for
each run_idx).

Usage:
    python3 scripts/plot_eval_comparison.py
    python3 scripts/plot_eval_comparison.py --results <path>
    python3 scripts/plot_eval_comparison.py --metric burnt_area_m2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = (
    REPO_ROOT
    / "examples/wildfire/data/scenarios/outputs/eval_palisades_on_pyrenees/results.csv"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="Per-run CSV from ppo_tester.")
    p.add_argument("--metric", default="moe_cumulative_reward",
                   help="Per-row column to compare (default: moe_cumulative_reward).")
    p.add_argument("--out", default=None, help="Output PNG path (default: alongside the CSV).")
    args = p.parse_args()

    results_path = Path(args.results)
    df = pd.read_csv(results_path)

    policies = sorted(df["policy_name"].unique())
    if "random_tactics" not in policies:
        raise SystemExit("results CSV has no 'random_tactics' policy — was --include-random-tactics set?")
    model_policies = [p for p in policies if p != "random_tactics"]
    if not model_policies:
        raise SystemExit("results CSV has no model policy alongside 'random_tactics'.")
    if len(model_policies) > 1:
        print(f"Multiple model policies present: {model_policies}. Using first: {model_policies[0]}.")
    model_name = model_policies[0]

    pivot = df.pivot_table(index="seed", columns="policy_name", values=args.metric)
    pivot = pivot.dropna(subset=[model_name, "random_tactics"])
    model_vals = pivot[model_name].to_numpy()
    rand_vals = pivot["random_tactics"].to_numpy()
    delta = model_vals - rand_vals

    pf = df.pivot_table(index="seed", columns="policy_name", values="propagation_factor")
    pf_model_pct = (pf[model_name] == 1.0).mean() * 100 if model_name in pf else float("nan")
    pf_rand_pct = (pf["random_tactics"] == 1.0).mean() * 100 if "random_tactics" in pf else float("nan")

    print(f"\n=== Paired comparison ({args.metric}, n={len(model_vals)} seeds) ===")
    table = pd.DataFrame({
        "policy": [model_name, "random_tactics", "model − random (paired delta)"],
        "mean":   [model_vals.mean(), rand_vals.mean(), delta.mean()],
        "std":    [model_vals.std(ddof=1), rand_vals.std(ddof=1), delta.std(ddof=1)],
        "min":    [model_vals.min(), rand_vals.min(), delta.min()],
        "max":    [model_vals.max(), rand_vals.max(), delta.max()],
        "pf=1 %": [pf_model_pct, pf_rand_pct, float("nan")],
    })
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    wins = (delta > 0).sum()
    ties = (delta == 0).sum()
    losses = (delta < 0).sum()
    print(f"\nModel vs random per-seed: wins={wins}  ties={ties}  losses={losses}  "
          f"(win rate: {wins / len(delta):.1%})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.boxplot(
        [model_vals, rand_vals],
        labels=[model_name, "random_tactics"],
        showmeans=True,
    )
    ax.set_title(f"Distribution of {args.metric}")
    ax.set_ylabel(args.metric)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(rand_vals, model_vals, alpha=0.5, s=20)
    lo = min(rand_vals.min(), model_vals.min())
    hi = max(rand_vals.max(), model_vals.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax.set_xlabel(f"random_tactics  {args.metric}")
    ax.set_ylabel(f"{model_name}  {args.metric}")
    ax.set_title(f"Per-seed paired ({len(model_vals)} seeds)\nabove diagonal = model wins")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.hist(delta, bins=30, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(delta.mean(), color="red", linestyle="-", linewidth=1.5,
               label=f"mean = {delta.mean():+.4f}")
    ax.set_xlabel(f"Δ ({model_name} − random_tactics)")
    ax.set_ylabel("count")
    ax.set_title(f"Paired delta distribution\nwin rate: {wins / len(delta):.1%}")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"PPO model vs random-tactics on {df['scenario_name'].iloc[0]}",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.out) if args.out else results_path.with_name(
        f"compare_{args.metric}.png"
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
