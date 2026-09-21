#!/usr/bin/env python
"""Paired statistical comparison of policies from a ppo_tester results CSV.

    python scripts/eval_stats.py <results.csv> [--reference <policy_name>]

ppo_tester gives every policy the SAME ignition seed for a given run index, so
the arms are paired, not independent. Paired tests are used throughout - they
are far more powerful here, because between-fire variance dwarfs between-policy
variance (an escaped fire scores ~-0.15 whatever the policy, a contained one
~+1.0).

Two outcomes, two tests:
  containment (binary, per fire)  McNemar exact on the discordant pairs. The
                                  right test for paired binary data: it looks
                                  only at fires where the two policies
                                  disagreed and ignores the rest.
  MoE (continuous, per fire)      paired t on the per-seed differences, plus
                                  Wilcoxon signed-rank, since MoE is strongly
                                  bimodal and not normal per-episode (the mean
                                  difference is still fine by CLT at n>=100,
                                  but reporting both is honest).

Holm-Bonferroni corrects across the pairwise family within each map.

CAVEAT the script prints and you should keep: every p-value here is
EPISODE-level. It quantifies "given these two trained policies, do they differ
on these fires". It says nothing about run-to-run variation from retraining
with a different seed, which in this project measured ~0.7 sigma on a matched
pair. Differences smaller than that should not be called real however small the
p-value.
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ESCAPE, MOE = "propagation_factor", "moe_cumulative_reward"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--reference", default=None,
                    help="compare every arm against this one (default: all pairs)")
    ap.add_argument("--noise-floor", type=float, default=0.7,
                    help="run-to-run sigma measured on a matched replicate pair")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    print(f"source: {args.results}")
    print(f"{len(df):,} rows · {df['policy_name'].nunique()} policies · "
          f"{df['scenario_name'].nunique()} scenarios\n")

    for scen, g in df.groupby("scenario_name"):
        names = list(dict.fromkeys(g["policy_name"]))
        cont = {}   # policy -> contained indicator, indexed by seed
        moe = {}
        for n in names:
            s = g[g.policy_name == n].set_index("seed")
            cont[n] = (s[ESCAPE] != 1.0).astype(int)
            moe[n] = s[MOE]
        seeds = set.intersection(*[set(v.index) for v in cont.values()])
        seeds = sorted(seeds)
        if len(seeds) < 2:
            print(f"{scen}: no shared seeds, skipping"); continue

        print("=" * 78)
        print(f"{scen}   ({len(seeds)} shared ignition seeds, fully paired)")
        print("=" * 78)
        print(f"{'policy':<32}{'contained':>11}{'95% CI':>17}{'mean MoE':>11}{'SE':>9}")
        for n in names:
            c = cont[n].loc[seeds].to_numpy(); m = moe[n].loc[seeds].to_numpy()
            lo, hi = wilson(int(c.sum()), len(c))
            print(f"{n[:32]:<32}{c.mean()*100:>10.1f}%"
                  f"{f'[{lo*100:.1f}, {hi*100:.1f}]':>17}"
                  f"{m.mean():>+11.4f}{m.std(ddof=1)/np.sqrt(len(m)):>9.4f}")

        pairs = ([(args.reference, n) for n in names if n != args.reference]
                 if args.reference else list(itertools.combinations(names, 2)))
        rows = []
        for a, b in pairs:
            ca, cb = cont[a].loc[seeds].to_numpy(), cont[b].loc[seeds].to_numpy()
            n01 = int(((ca == 0) & (cb == 1)).sum())   # b contained, a did not
            n10 = int(((ca == 1) & (cb == 0)).sum())
            # exact McNemar = two-sided binomial on the discordant pairs
            p_mc = (stats.binomtest(n10, n10 + n01, 0.5).pvalue
                    if (n10 + n01) else 1.0)
            d = moe[a].loc[seeds].to_numpy() - moe[b].loc[seeds].to_numpy()
            se = d.std(ddof=1) / np.sqrt(d.size)
            p_t = stats.ttest_rel(moe[a].loc[seeds], moe[b].loc[seeds]).pvalue
            p_w = (stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
            rows.append(dict(a=a, b=b, dc=(ca.mean() - cb.mean()) * 100,
                             n10=n10, n01=n01, p_mc=p_mc,
                             dm=d.mean(), se=se, sig=d.mean() / se if se else np.nan,
                             p_t=p_t, p_w=p_w))
        for key, lab in (("p_mc", "p_mcnemar"), ("p_t", "p_paired_t"), ("p_w", "p_wilcoxon")):
            adj = holm([r[key] for r in rows])
            for r, v in zip(rows, adj):
                r[lab + "_holm"] = v

        print(f"\n{'A vs B':<52}{'Δcontain':>10}{'McNemar':>10}{'ΔMoE':>10}{'σ':>7}{'t(Holm)':>10}")
        for r in sorted(rows, key=lambda r: -abs(r["dm"])):
            flag = "" if r["p_paired_t_holm"] < 0.05 else "  ns"
            note = "" if abs(r["sig"]) > args.noise_floor else "  <floor"
            print(f"{(r['a'][:24]+' vs '+r['b'][:24]):<52}"
                  f"{r['dc']:>+9.1f}%{r['p_mcnemar_holm']:>10.3g}"
                  f"{r['dm']:>+10.4f}{r['sig']:>+7.1f}{r['p_paired_t_holm']:>10.3g}{flag}{note}")
        print(f"\n  Δ = A − B. σ is the paired episode-level effect size.")
        print(f"  'ns'    = not significant after Holm correction across "
              f"{len(rows)} pairwise tests.")
        print(f"  '<floor'= |σ| below the {args.noise_floor} run-to-run replicate "
              f"floor; treat as indistinguishable regardless of p.\n")


if __name__ == "__main__":
    main()
