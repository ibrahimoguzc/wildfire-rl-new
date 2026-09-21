# Statistical comparison - eval_2 figure set

Data: `final_evaluation/results.csv` -
ignitions drawn once per map with `--seed 1 --random-seeds` and presented
identically to every policy, so all comparisons are fully paired
(each section header states its n).
Salamis is excluded throughout (100% containment for every policy,
including random - it saturates and dilutes pooled contrasts).

Methods: containment compared with an exact McNemar test on the
discordant pairs; MoE compared with a paired t test (sigma = paired
episode-level effect size, mean difference / SE) and a Wilcoxon
signed-rank test as a distribution-free check; Holm-Bonferroni
correction across the 10 pairwise tests within each section.
Paired effects below 0.7 sigma are flagged as below the
run-to-run replicate noise floor (retraining the same configuration
moves results by about that much), and should be treated as
indistinguishable regardless of p-value.

## Palisades  (250 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 89.2% | [84.7, 92.5] | +0.8744 | +0.9905 | 0.0210 |
| Fixed tactics | 91.6% | [87.5, 94.4] | +0.8972 | +0.9866 | 0.0188 |
| Palisades only | 97.2% | [94.3, 98.6] | +0.9625 | +0.9940 | 0.0111 |
| Pyrenees only | 96.8% | [93.8, 98.4] | +0.9572 | +0.9931 | 0.0118 |
| Theater level (concurrent) | 99.2% | [97.1, 99.8] | +0.9839 | +0.9941 | 0.0059 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -10.0% | 6e-08 | -0.1095 | -5.4 | 1.7e-07 | 1.7e-06 | 6.2e-15 | **significant** |
| Random tactics vs Palisades only | -8.0% | 3.6e-05 | -0.0881 | -4.4 | 1.7e-05 | 0.00015 | 1.6e-13 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -7.6% | 6.6e-05 | -0.0867 | -4.4 | 1.8e-05 | 0.00015 | 1.1e-20 | **significant** |
| Fixed tactics vs Palisades only | -5.6% | 0.0013 | -0.0654 | -3.7 | 0.00025 | 0.0017 | 1.2e-19 | **significant** |
| Random tactics vs Pyrenees only | -7.6% | 0.00088 | -0.0828 | -3.6 | 0.00041 | 0.0025 | 5.2e-09 | **significant** |
| Fixed tactics vs Pyrenees only | -5.2% | 0.0023 | -0.0601 | -3.5 | 0.00052 | 0.0026 | 1.5e-14 | **significant** |
| Pyrenees only vs Theater level (concurrent) | -2.4% | 0.11 | -0.0266 | -2.0 | 0.044 | 0.18 | 0.0076 | ns |
| Palisades only vs Theater level (concurrent) | -2.0% | 0.18 | -0.0213 | -1.7 | 0.092 | 0.28 | 0.48 | ns |
| Random tactics vs Fixed tactics | -2.4% | 0.36 | -0.0228 | -1.0 | 0.33 | 0.65 | 0.39 | ns |
| Palisades only vs Pyrenees only | +0.4% | 1 | +0.0053 | +0.3 | 0.75 | 0.75 | 0.081 | indistinguishable (below noise floor) |

**Mission-native specialist here: Palisades only.**

- Palisades only vs the theater-level agent: -2.0% containment, -0.0213 MoE (-1.7 sigma, Holm p = 0.28) - ns
- Palisades only vs Pyrenees only: +0.4% containment, +0.0053 MoE (+0.3 sigma, Holm p = 0.75) - indistinguishable (below noise floor)


## Pyrenees  (250 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 5.2% | [3.1, 8.7] | -0.1216 | -0.1749 | 0.0169 |
| Fixed tactics | 9.6% | [6.5, 13.9] | -0.0618 | -0.1587 | 0.0221 |
| Palisades only | 7.2% | [4.6, 11.1] | -0.1032 | -0.1804 | 0.0197 |
| Pyrenees only | 17.6% | [13.4, 22.8] | +0.0464 | -0.1402 | 0.0280 |
| Theater level (concurrent) | 17.2% | [13.0, 22.4] | +0.0357 | -0.1441 | 0.0279 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Pyrenees only | -12.4% | 9.3e-10 | -0.1679 | -6.9 | 4.1e-11 | 4.1e-10 | 4e-15 | **significant** |
| Random tactics vs Theater level (concurrent) | -12.0% | 1.9e-09 | -0.1572 | -6.5 | 3.8e-10 | 3.4e-09 | 1.3e-10 | **significant** |
| Palisades only vs Pyrenees only | -10.4% | 2.2e-07 | -0.1496 | -6.3 | 1.4e-09 | 1.1e-08 | 5.3e-15 | **significant** |
| Palisades only vs Theater level (concurrent) | -10.0% | 6e-08 | -0.1389 | -6.1 | 3.7e-09 | 2.6e-08 | 1.7e-09 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -7.6% | 2.1e-05 | -0.0974 | -4.5 | 8.4e-06 | 5e-05 | 0.00017 | **significant** |
| Fixed tactics vs Pyrenees only | -8.0% | 0.00018 | -0.1081 | -4.4 | 1.9e-05 | 9.5e-05 | 3.9e-08 | **significant** |
| Random tactics vs Fixed tactics | -4.4% | 0.00098 | -0.0598 | -3.9 | 0.00015 | 0.0006 | 0.0048 | **significant** |
| Fixed tactics vs Palisades only | +2.4% | 0.11 | +0.0415 | +2.7 | 0.0071 | 0.021 | 0.00034 | **significant** |
| Random tactics vs Palisades only | -2.0% | 0.12 | -0.0183 | -1.5 | 0.15 | 0.29 | 0.37 | ns |
| Pyrenees only vs Theater level (concurrent) | +0.4% | 1 | +0.0107 | +0.5 | 0.61 | 0.61 | 0.14 | indistinguishable (below noise floor) |

**Mission-native specialist here: Pyrenees only.**

- Pyrenees only vs the theater-level agent: +0.4% containment, +0.0107 MoE (+0.5 sigma, Holm p = 0.61) - indistinguishable (below noise floor)
- Pyrenees only vs Palisades only: +10.4% containment, +0.1496 MoE (+6.3 sigma, Holm p = 1.1e-08) - **significant**


## Combined (Palisades + Pyrenees)  (500 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 47.2% | [42.9, 51.6] | +0.3764 | -0.0828 | 0.0260 |
| Fixed tactics | 50.6% | [46.2, 55.0] | +0.4177 | +0.9637 | 0.0259 |
| Palisades only | 52.2% | [47.8, 56.5] | +0.4297 | +0.9800 | 0.0264 |
| Pyrenees only | 57.2% | [52.8, 61.5] | +0.5018 | +0.9818 | 0.0254 |
| Theater level (concurrent) | 58.2% | [53.8, 62.4] | +0.5098 | +0.9847 | 0.0256 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -11.0% | 5.6e-17 | -0.1334 | -8.4 | 3.3e-16 | 3.3e-15 | 2.4e-20 | **significant** |
| Random tactics vs Pyrenees only | -10.0% | 3e-11 | -0.1254 | -7.4 | 4.7e-13 | 4.2e-12 | 5.4e-21 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -7.6% | 1.6e-09 | -0.0921 | -6.3 | 5.9e-10 | 4.7e-09 | 2.3e-16 | **significant** |
| Palisades only vs Theater level (concurrent) | -6.0% | 6.9e-08 | -0.0801 | -6.1 | 2.8e-09 | 2e-08 | 5.7e-07 | **significant** |
| Fixed tactics vs Pyrenees only | -6.6% | 5.4e-07 | -0.0841 | -5.6 | 4e-08 | 2.4e-07 | 2.9e-18 | **significant** |
| Palisades only vs Pyrenees only | -5.0% | 0.00017 | -0.0721 | -4.9 | 1.5e-06 | 7.6e-06 | 1.1e-07 | **significant** |
| Random tactics vs Palisades only | -5.0% | 4.6e-06 | -0.0532 | -4.5 | 1e-05 | 4.1e-05 | 0.00017 | **significant** |
| Random tactics vs Fixed tactics | -3.4% | 0.012 | -0.0413 | -3.0 | 0.0032 | 0.0096 | 0.15 | **significant** |
| Fixed tactics vs Palisades only | -1.6% | 0.18 | -0.0120 | -1.0 | 0.31 | 0.63 | 0.012 | ns |
| Pyrenees only vs Theater level (concurrent) | -1.0% | 0.46 | -0.0080 | -0.6 | 0.52 | 0.63 | 0.64 | indistinguishable (below noise floor) |

- Theater level (concurrent) vs Pyrenees only: +1.0% containment, +0.0080 MoE (+0.6 sigma, Holm p = 0.63) - indistinguishable (below noise floor)
- Theater level (concurrent) vs Palisades only: +6.0% containment, +0.0801 MoE (+6.1 sigma, Holm p = 2e-08) - **significant**
- Pyrenees only vs Palisades only: +5.0% containment, +0.0721 MoE (+4.9 sigma, Holm p = 7.6e-06) - **significant**

