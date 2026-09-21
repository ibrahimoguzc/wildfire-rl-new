#!/usr/bin/env python
"""Statistical tables for the theater-level evaluation figures.

Companion to eval_stats.py, using the same tests on the same pairing
structure, restricted to the five arms shown in the eval_theater_* figures and
extended with a pooled "Overall" evaluation (pairing key: scenario + seed).

Per evaluation (Overall, Palisades, Pyrenees, Salamis) and per policy pair:
  MoE          paired t on per-seed differences + Wilcoxon signed-rank
  containment  exact McNemar (two-sided binomial on discordant pairs)
Holm-Bonferroni is applied within each evaluation across all 10 pairwise
tests, separately per test statistic. Every p-value is EPISODE-level: it says
nothing about run-to-run retraining variance (~0.7 sigma on a matched
replicate pair in this project); |sigma| below that floor should be treated as
indistinguishable regardless of p.

Writes final_plots/eval_theater_stats.csv (all 40 pairwise rows) and prints
one LaTeX table per evaluation (the four vs-theater rows of that CSV).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS = (REPO / "examples/wildfire/data/scenarios/outputs"
           / "eval_7way_3maps_754407/results.csv")
OUT_CSV = REPO / "final_plots/eval_theater_stats.csv"

ESCAPE, MOE = "propagation_factor", "moe_cumulative_reward"
THEATER = "arm5_concurrent_100k"
# Reference arm for the LaTeX tables' delta and p columns (the CSV always
# holds every pairwise comparison, so changing this re-slices, not re-tests).
REFERENCE = "random_tactics"
REFERENCE_LABEL = "Random Tactics"
ARMS = {
    "arm1_palisades_only_93k8": "Palisades only",
    "arm2_pyrenees_only_100k": "Pyrenees only",
    THEATER: "Theater level",
    "random_tactics": "Random Tactics",
    "fixed_Pyrenees5sp5ev_heuristic": "Fixed Tactics",
}
EVALS = [("Overall", None), ("Palisades", "Palisades"),
         ("Pyrenees", "Pyrenees"), ("Salamis", "Salamis")]

# Anchors recomputed independently earlier in the analysis; a mismatch means
# the data or the selection changed under us.
ANCHORS = {
    ("Overall", THEATER): 0.647530,
    ("Palisades", THEATER): 0.981491,
    ("Pyrenees", "arm2_pyrenees_only_100k"): 0.010946,
    ("Salamis", "fixed_Pyrenees5sp5ev_heuristic"): 0.954765,
}


def holm(pvals: list[float]) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def fmt_p(p: float) -> str:
    return r"$<\!0.001$" if p < 0.001 else f"${p:.3f}$"


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["policy_name"].isin(ARMS)].copy()
    df["pair_key"] = df["scenario"] + "|" + df["seed"].astype(str)

    # -- pairing integrity ------------------------------------------------
    for (arm, scen), g in df.groupby(["policy_name", "scenario"]):
        assert len(g) == 500, (arm, scen, len(g))
        assert g["seed"].is_unique, (arm, scen, "duplicate seeds")
    for scen, g in df.groupby("scenario"):
        sets = {a: frozenset(x["seed"]) for a, x in g.groupby("policy_name")}
        assert len(set(sets.values())) == 1, (scen, "seed sets differ")

    csv_rows = []
    tables = []
    for eval_name, scen in EVALS:
        g = df if scen is None else df[df["scenario"] == scen]
        n_expected = 1500 if scen is None else 500
        moe, cont = {}, {}
        for arm in ARMS:
            s = g[g.policy_name == arm].set_index("pair_key").sort_index()
            assert len(s) == n_expected, (eval_name, arm, len(s))
            moe[arm] = s[MOE]
            cont[arm] = (s[ESCAPE] != 1.0).astype(int)
        keys = moe[THEATER].index
        for arm in ARMS:
            assert moe[arm].index.equals(keys), (eval_name, arm, "key mismatch")
        for (an, arm), want in ANCHORS.items():
            if an == eval_name:
                got = float(moe[arm].mean())
                assert abs(got - want) < 1e-6, (an, arm, got, want)

        pairs = list(itertools.combinations(ARMS, 2))
        rows = []
        for a, b in pairs:
            d = moe[a].to_numpy() - moe[b].to_numpy()
            se = d.std(ddof=1) / np.sqrt(d.size)
            ca, cb = cont[a].to_numpy(), cont[b].to_numpy()
            n10 = int(((ca == 1) & (cb == 0)).sum())
            n01 = int(((ca == 0) & (cb == 1)).sum())
            rows.append(dict(
                evaluation=eval_name, policy_a=a, policy_b=b, n_pairs=d.size,
                mean_a=moe[a].mean(), mean_b=moe[b].mean(),
                escaped_a_pct=(1 - ca.mean()) * 100,
                escaped_b_pct=(1 - cb.mean()) * 100,
                moe_diff_a_minus_b=d.mean(), se_diff=se,
                sigma=d.mean() / se if se > 0 else np.nan,
                p_paired_t=stats.ttest_rel(moe[a], moe[b]).pvalue,
                p_wilcoxon=(stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0),
                n10_a_only_contained=n10, n01_b_only_contained=n01,
                p_mcnemar=(stats.binomtest(n10, n10 + n01, 0.5).pvalue
                           if (n10 + n01) else 1.0),
            ))
        for key in ("p_paired_t", "p_wilcoxon", "p_mcnemar"):
            adj = holm([r[key] for r in rows])
            for r, v in zip(rows, adj):
                r[key + "_holm"] = v
        csv_rows.extend(rows)

        # -- LaTeX table: each policy vs the reference arm ----------------
        by_pair = {(r["policy_a"], r["policy_b"]): r for r in rows}
        stats_rows = []
        for arm in ARMS:
            if arm == REFERENCE:
                stats_rows.append((arm, moe[arm].mean(),
                                   moe[arm].std(ddof=1) / np.sqrt(len(moe[arm])),
                                   (1 - cont[arm].mean()) * 100, None))
                continue
            r = by_pair.get((arm, REFERENCE)) or by_pair.get((REFERENCE, arm))
            sign = 1.0 if r["policy_a"] == arm else -1.0
            stats_rows.append((arm, moe[arm].mean(),
                               moe[arm].std(ddof=1) / np.sqrt(len(moe[arm])),
                               (1 - cont[arm].mean()) * 100,
                               dict(d=sign * r["moe_diff_a_minus_b"],
                                    sig=sign * r["sigma"],
                                    pt=r["p_paired_t_holm"],
                                    pm=r["p_mcnemar_holm"])))
        stats_rows.sort(key=lambda x: -x[1])
        lines = [
            r"\begin{table}[t]",
            r"  \centering",
            rf"  \caption{{Paired evaluation statistics, "
            + ("overall (pooled across the three maps, "
               if eval_name == "Overall" else rf"{eval_name} (")
            + rf"{n_expected:,} paired ignitions per policy)."
            rf" $\Delta$MoE is the policy's mean paired MoE difference against"
            rf" the {REFERENCE_LABEL} baseline; $p_t$ is the paired $t$-test and"
            r" $p_{\mathrm{McN}}$ the exact McNemar test on containment, both"
            r" Holm-corrected within this evaluation across all ten pairwise"
            r" comparisons.}",
            rf"  \label{{tab:eval-{eval_name.lower()}}}",
            r"  \begin{tabular}{lrrrrr}",
            r"    \toprule",
            r"    Policy & Mean MoE & Escaped & $\Delta$MoE & $p_t$ &"
            r" $p_{\mathrm{McN}}$ \\",
            r"    \midrule",
        ]
        for arm, m, se, esc, cmp in stats_rows:
            name = ARMS[arm]
            if cmp is None:
                lines.append(
                    rf"    {name} & ${m:+.3f}$ & {esc:.0f}\% &"
                    r" --- & --- & --- \\")
            else:
                lines.append(
                    rf"    {name} & ${m:+.3f}$ & {esc:.0f}\% &"
                    rf" ${cmp['d']:+.3f}$ & {fmt_p(cmp['pt'])} &"
                    rf" {fmt_p(cmp['pm'])} \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
        tables.append("\n".join(lines))

    out = pd.DataFrame(csv_rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"wrote {OUT_CSV} ({len(out)} rows)\n")
    print("\n\n".join(tables))


if __name__ == "__main__":
    main()
