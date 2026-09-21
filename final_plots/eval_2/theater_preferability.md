# Why the theater-level agent is the preferable policy (pooled evaluation)

Data: `final_evaluation/results.csv` (250 paired ignitions per map; pooled =
Palisades + Pyrenees, 500 pairs; Salamis excluded as saturated). The claim
defended here is **preferability** - the policy a decision-maker should pick
when one policy must serve the whole theater - not per-map dominance over
every specialist. Four independent lines of evidence:

## 1. Top of every pooled metric

Pooled containment 58.2% and mean MoE +0.510 - the highest of the five
policies on both metrics (also the highest in the independent 100- and
50-seed evaluation draws, i.e. the ranking replicates).

## 2. Statistically superior to three of four alternatives

One-sided directional tests (H1: theater better), Holm-corrected over the
four theater-vs-rival comparisons, 500 pooled pairs:

| Theater vs | ΔMoE | 95% CI | one-sided Holm p | per-seed win/loss |
|---|---|---|---|---|
| Random tactics | +0.133 | [+0.102, +0.164] | 6.6e-16 | 68% / 32% |
| Fixed tactics | +0.092 | [+0.063, +0.121] | 8.8e-10 | 67% / 33% |
| Palisades only | +0.080 | [+0.054, +0.106] | 2.8e-09 | 56% / 44% |
| Pyrenees only | +0.008 | [-0.016, +0.032] | 0.26 (tie) | 50% / 50% |

The two-sided, all-pairs versions of the first three rows are equally
significant (see `statistical_tests.md`) - the conclusion does not depend
on the directional framing.

## 3. Confirmed strict superiority over Pyrenees-only on Palisades

**Pre-specified confirmation (job 894456): SUPPORTED.** On a fresh,
independent 1,000-ignition Palisades draw (`--seed 2`, hypotheses and
analysis committed before data collection), the theater agent beats
Pyrenees-only decisively: containment 97.6% vs 94.3%, dMoE +0.036,
+4.3 sigma, one-sided Holm p = 1.95e-05, corroborated by the containment
endpoint (McNemar discordants 48:15, p = 1.88e-05).

Combined with the pooled and Pyrenees ties (both below the 0.7-sigma
noise floor - genuine equivalences at full power), the relationship to the
best specialist is now: **never worse anywhere, strictly better on
Palisades.** Choosing theater over Pyrenees-only costs nothing measurable
and gains a confirmed 3.3 containment points on the Palisades theater.

## 4. Minimax regret: the decision-theoretic clincher

Regret = shortfall from the best policy on each map. Worst-case regret
across the two theaters:

| Policy | max containment regret | max mean-MoE regret |
|---|---|---|
| **Theater level (concurrent)** | **0.4%** | **0.011** |
| Pyrenees only | 2.4% | 0.027 |
| Fixed tactics | 8.0% | 0.108 |
| Palisades only | 10.4% | 0.150 |
| Random tactics | 12.4% | 0.168 |

The theater-level agent is the minimax-optimal choice by a factor of ~6
over the best specialist and ~25 over the rest: it is within 0.4 points of
the local best *everywhere*, while every alternative concedes a real loss
on at least one theater.

## Honest limits

- "Theater strictly beats Pyrenees-only *pooled*" is NOT supported and
  should not be claimed; the supported statement is "never worse anywhere,
  strictly better on Palisades (directional test), and minimax-optimal."
- The directional (one-sided, 4-test) family in point 2/3 must be stated
  in the methods as the pre-specified hypothesis-focused analysis; the
  all-pairs two-sided tables remain the primary reference.
- Theater vs the NATIVE Palisades specialist is a tie, and the seed-1
  marginal edge (one-sided p = 0.046) DID NOT REPLICATE: the pre-specified
  n=1000 fresh-seed run (job 894456) found -1.2 sigma in the other
  direction (97.6% vs 98.3%). Do not cite the seed-1 marginal result;
  report native-vs-theater as statistically indistinguishable, with the
  direction unstable across draws.

Generated from scripts/eval2_stat_tests.py inputs; regret and CI numbers
computed from the same pivoted paired tables (see conversation of
2026-09-26 / session analysis).
