#!/usr/bin/env python3
"""Paired statistical tests for the eval_2 figure set -> a markdown report.

Data: the FINAL EVALUATION (job 756122, final_evaluation/results.csv),
restricted to the five headline policies of final_plots/eval_2. Three
sections - Palisades (250 paired ignitions), Pyrenees (250), Combined
(500; Salamis excluded as saturated) - each with:

  * per-policy summary: containment + Wilson 95% CI, mean/median MoE, SE
  * ALL 10 pairwise comparisons: containment delta with exact McNemar,
    MoE delta with paired t (sigma = paired effect size) and Wilcoxon,
    Holm correction across the section's 10 tests
  * a highlights block for the mission-native specialist vs the other
    trained agents

|sigma| below 0.7 is flagged: run-to-run replicate noise of retraining the
same configuration was measured at ~0.7 sigma, so smaller paired effects
are indistinguishable from retraining noise regardless of p.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "final_evaluation/results.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "final_plots/eval_2/statistical_tests.md"
NOISE_FLOOR = 0.7

NAME = {
    "random_tactics": "Random tactics",
    "fixed_Pyrenees5sp5ev_heuristic": "Fixed tactics",
    "arm1_palisades_only": "Palisades only",
    "arm2_pyrenees_only": "Pyrenees only",
    "arm5_concurrent": "Theater level (concurrent)",
}

SECTIONS = [
    ("Palisades", ["Palisades5sp5ev.json"], "arm1_palisades_only"),
    ("Pyrenees", ["Pyrenees5sp5ev.json"], "arm2_pyrenees_only"),
    ("Combined (Palisades + Pyrenees)",
     ["Palisades5sp5ev.json", "Pyrenees5sp5ev.json"], None),
]


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = [0.0] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        adj[i] = min(1.0, max(prev, (len(pvals) - rank) * pvals[i]))
        prev = adj[i]
    return adj


def section(df: pd.DataFrame, title: str, scenarios: list[str],
            native: str | None, lines: list[str]) -> None:
    d = df[df["scenario_name"].isin(scenarios)]
    piv_m = d.pivot_table(index=["scenario_name", "seed"],
                          columns="policy_name", values="moe")
    piv_c = d.pivot_table(index=["scenario_name", "seed"],
                          columns="policy_name", values="contained").astype(bool)
    n = len(piv_m)
    assert piv_m.notna().all().all()

    lines.append(f"\n## {title}  ({n} shared ignitions, fully paired)\n")
    lines.append("| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |")
    lines.append("|---|---|---|---|---|---|")
    for p, label in NAME.items():
        c, m = piv_c[p], piv_m[p]
        lo, hi = stats.binomtest(int(c.sum()), n).proportion_ci(
            0.95, method="wilson")
        lines.append(
            f"| {label} | {c.mean()*100:.1f}% | [{lo*100:.1f}, {hi*100:.1f}] |"
            f" {m.mean():+.4f} | {m.median():+.4f} | {m.sem():.4f} |")

    rows = []
    for a, b in itertools.combinations(NAME, 2):
        diff = piv_m[a] - piv_m[b]
        sigma = diff.mean() / diff.sem()
        p_t = stats.ttest_rel(piv_m[a], piv_m[b]).pvalue
        p_w = stats.wilcoxon(piv_m[a], piv_m[b]).pvalue
        n01 = int(((~piv_c[a]) & piv_c[b]).sum())
        n10 = int((piv_c[a] & (~piv_c[b])).sum())
        p_mc = (stats.binomtest(min(n10, n01), n10 + n01).pvalue
                if n10 + n01 else 1.0)
        rows.append([a, b, (piv_c[a].mean() - piv_c[b].mean()) * 100,
                     p_mc, diff.mean(), sigma, p_t, p_w])
    adj = holm([r[6] for r in rows])

    lines.append("")
    lines.append("| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for (a, b, dc, mc, dm, sg, pt, pw), h in sorted(
            zip(rows, adj), key=lambda rh: -abs(rh[0][5])):
        if abs(sg) < NOISE_FLOOR:
            verdict = "indistinguishable (below noise floor)"
        elif h < 0.05:
            verdict = "**significant**"
        else:
            verdict = "ns"
        verdicts[(a, b)] = (dc, dm, sg, h, verdict)
        verdicts[(b, a)] = (-dc, -dm, -sg, h, verdict)
        lines.append(
            f"| {NAME[a]} vs {NAME[b]} | {dc:+.1f}% | {mc:.2g} | {dm:+.4f} |"
            f" {sg:+.1f} | {pt:.2g} | {h:.2g} | {pw:.2g} | {verdict} |")

    lines.append("")
    if native:
        others = [("arm5_concurrent", "the theater-level agent"),
                  ("arm2_pyrenees_only", "Pyrenees only"),
                  ("arm1_palisades_only", "Palisades only")]
        lines.append(f"**Mission-native specialist here: {NAME[native]}.**\n")
        for other, desc in others:
            if other == native:
                continue
            dc, dm, sg, h, verdict = verdicts[(native, other)]
            lines.append(
                f"- {NAME[native]} vs {desc}: {dc:+.1f}% containment,"
                f" {dm:+.4f} MoE ({sg:+.1f} sigma, Holm p = {h:.2g}) - {verdict}")
    else:
        for a, b in (("arm5_concurrent", "arm2_pyrenees_only"),
                     ("arm5_concurrent", "arm1_palisades_only"),
                     ("arm2_pyrenees_only", "arm1_palisades_only")):
            dc, dm, sg, h, verdict = verdicts[(a, b)]
            lines.append(
                f"- {NAME[a]} vs {NAME[b]}: {dc:+.1f}% containment,"
                f" {dm:+.4f} MoE ({sg:+.1f} sigma, Holm p = {h:.2g}) - {verdict}")
    lines.append("")


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["policy_name"].isin(NAME)].copy()
    df["contained"] = pd.to_numeric(df["propagation_factor"]) != 1.0
    df["moe"] = pd.to_numeric(df["moe_cumulative_reward"])

    lines = [
        "# Statistical comparison - eval_2 figure set",
        "",
        f"Data: `{RESULTS.relative_to(REPO) if RESULTS.is_relative_to(REPO) else RESULTS}` -",
        "ignitions drawn once per map with `--seed 1 --random-seeds` and presented",
        "identically to every policy, so all comparisons are fully paired",
        "(each section header states its n).",
        "Salamis is excluded throughout (100% containment for every policy,",
        "including random - it saturates and dilutes pooled contrasts).",
        "",
        "Methods: containment compared with an exact McNemar test on the",
        "discordant pairs; MoE compared with a paired t test (sigma = paired",
        "episode-level effect size, mean difference / SE) and a Wilcoxon",
        "signed-rank test as a distribution-free check; Holm-Bonferroni",
        "correction across the 10 pairwise tests within each section.",
        f"Paired effects below {NOISE_FLOOR} sigma are flagged as below the",
        "run-to-run replicate noise floor (retraining the same configuration",
        "moves results by about that much), and should be treated as",
        "indistinguishable regardless of p-value.",
    ]
    for title, scenarios, native in SECTIONS:
        section(df, title, scenarios, native, lines)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
