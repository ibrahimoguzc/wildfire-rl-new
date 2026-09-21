# Statistical comparison - eval_2 figure set

Data: `examples/wildfire/data/scenarios/outputs/eval2_5way_3maps_100sims_886047/results.csv` -
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

## Palisades  (100 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 88.0% | [80.2, 93.0] | +0.8604 | +0.9902 | 0.0353 |
| Fixed tactics | 91.0% | [83.8, 95.2] | +0.8878 | +0.9850 | 0.0314 |
| Palisades only | 96.0% | [90.2, 98.4] | +0.9481 | +0.9922 | 0.0213 |
| Pyrenees only | 96.0% | [90.2, 98.4] | +0.9489 | +0.9919 | 0.0208 |
| Theater level (concurrent) | 100.0% | [96.3, 100.0] | +0.9913 | +0.9930 | 0.0007 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -12.0% | 0.00049 | -0.1309 | -3.7 | 0.00032 | 0.0032 | 2.3e-06 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -9.0% | 0.0039 | -0.1035 | -3.3 | 0.0013 | 0.012 | 7.8e-09 | **significant** |
| Random tactics vs Palisades only | -8.0% | 0.021 | -0.0877 | -2.7 | 0.0088 | 0.071 | 3.4e-05 | ns |
| Fixed tactics vs Palisades only | -5.0% | 0.062 | -0.0604 | -2.5 | 0.012 | 0.087 | 9.3e-08 | ns |
| Random tactics vs Pyrenees only | -8.0% | 0.077 | -0.0884 | -2.1 | 0.039 | 0.23 | 0.0009 | ns |
| Pyrenees only vs Theater level (concurrent) | -4.0% | 0.12 | -0.0424 | -2.1 | 0.043 | 0.23 | 0.1 | ns |
| Palisades only vs Theater level (concurrent) | -4.0% | 0.12 | -0.0432 | -2.0 | 0.044 | 0.23 | 0.33 | ns |
| Fixed tactics vs Pyrenees only | -5.0% | 0.18 | -0.0611 | -1.9 | 0.057 | 0.23 | 7.4e-07 | ns |
| Random tactics vs Fixed tactics | -3.0% | 0.58 | -0.0274 | -0.7 | 0.48 | 0.96 | 0.4 | ns |
| Palisades only vs Pyrenees only | +0.0% | 1 | -0.0007 | -0.0 | 0.98 | 0.98 | 0.83 | indistinguishable (below noise floor) |

**Mission-native specialist here: Palisades only.**

- Palisades only vs the theater-level agent: -4.0% containment, -0.0432 MoE (-2.0 sigma, Holm p = 0.23) - ns
- Palisades only vs Pyrenees only: +0.0% containment, -0.0007 MoE (-0.0 sigma, Holm p = 0.98) - indistinguishable (below noise floor)


## Pyrenees  (100 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 6.0% | [2.8, 12.5] | -0.1120 | -0.1723 | 0.0286 |
| Fixed tactics | 9.0% | [4.8, 16.2] | -0.0727 | -0.1661 | 0.0342 |
| Palisades only | 7.0% | [3.4, 13.7] | -0.1088 | -0.1847 | 0.0308 |
| Pyrenees only | 19.0% | [12.5, 27.8] | +0.0618 | -0.1398 | 0.0457 |
| Theater level (concurrent) | 17.0% | [10.9, 25.5] | +0.0362 | -0.1432 | 0.0439 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Palisades only vs Pyrenees only | -12.0% | 0.00049 | -0.1706 | -4.5 | 2.2e-05 | 0.00022 | 1.5e-07 | **significant** |
| Random tactics vs Pyrenees only | -13.0% | 0.00024 | -0.1738 | -4.4 | 2.3e-05 | 0.00022 | 5e-06 | **significant** |
| Palisades only vs Theater level (concurrent) | -10.0% | 0.002 | -0.1450 | -4.0 | 0.00011 | 0.00091 | 1.5e-05 | **significant** |
| Random tactics vs Theater level (concurrent) | -11.0% | 0.00098 | -0.1483 | -4.0 | 0.00012 | 0.00091 | 0.00025 | **significant** |
| Fixed tactics vs Pyrenees only | -10.0% | 0.0063 | -0.1345 | -3.4 | 0.0011 | 0.0065 | 2.5e-05 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -8.0% | 0.0078 | -0.1090 | -3.3 | 0.0013 | 0.0067 | 0.00026 | **significant** |
| Random tactics vs Fixed tactics | -3.0% | 0.25 | -0.0393 | -2.0 | 0.053 | 0.21 | 0.65 | ns |
| Fixed tactics vs Palisades only | +2.0% | 0.62 | +0.0361 | +1.5 | 0.14 | 0.41 | 0.072 | ns |
| Pyrenees only vs Theater level (concurrent) | +2.0% | 0.69 | +0.0256 | +0.9 | 0.38 | 0.76 | 0.18 | ns |
| Random tactics vs Palisades only | -1.0% | 1 | -0.0032 | -0.2 | 0.87 | 0.87 | 0.12 | indistinguishable (below noise floor) |

**Mission-native specialist here: Pyrenees only.**

- Pyrenees only vs the theater-level agent: +2.0% containment, +0.0256 MoE (+0.9 sigma, Holm p = 0.76) - ns
- Pyrenees only vs Palisades only: +12.0% containment, +0.1706 MoE (+4.5 sigma, Holm p = 0.00022) - **significant**


## Combined (Palisades + Pyrenees)  (200 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 47.0% | [40.2, 53.9] | +0.3742 | -0.0847 | 0.0412 |
| Fixed tactics | 50.0% | [43.1, 56.9] | +0.4075 | +0.4389 | 0.0412 |
| Palisades only | 51.5% | [44.6, 58.3] | +0.4197 | +0.9781 | 0.0419 |
| Pyrenees only | 57.5% | [50.6, 64.1] | +0.5053 | +0.9834 | 0.0402 |
| Theater level (concurrent) | 58.5% | [51.6, 65.1] | +0.5138 | +0.9848 | 0.0403 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -11.5% | 2.4e-07 | -0.1396 | -5.5 | 1.2e-07 | 1.2e-06 | 1.3e-07 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -8.5% | 1.5e-05 | -0.1062 | -4.7 | 5.2e-06 | 4.7e-05 | 5.6e-10 | **significant** |
| Random tactics vs Pyrenees only | -10.5% | 0.0001 | -0.1311 | -4.5 | 9.4e-06 | 7.5e-05 | 1.4e-07 | **significant** |
| Palisades only vs Theater level (concurrent) | -7.0% | 0.00012 | -0.0941 | -4.4 | 1.5e-05 | 0.0001 | 7.7e-05 | **significant** |
| Fixed tactics vs Pyrenees only | -7.5% | 0.0015 | -0.0978 | -3.8 | 0.00017 | 0.001 | 4.8e-10 | **significant** |
| Palisades only vs Pyrenees only | -6.0% | 0.012 | -0.0857 | -3.4 | 0.00076 | 0.0038 | 2.2e-05 | **significant** |
| Random tactics vs Palisades only | -4.5% | 0.022 | -0.0455 | -2.3 | 0.021 | 0.083 | 0.25 | ns |
| Random tactics vs Fixed tactics | -3.0% | 0.21 | -0.0333 | -1.5 | 0.13 | 0.38 | 0.66 | ns |
| Fixed tactics vs Palisades only | -1.5% | 0.51 | -0.0121 | -0.7 | 0.48 | 0.96 | 0.042 | ns |
| Pyrenees only vs Theater level (concurrent) | -1.0% | 0.75 | -0.0084 | -0.5 | 0.64 | 0.96 | 0.82 | indistinguishable (below noise floor) |

- Theater level (concurrent) vs Pyrenees only: +1.0% containment, +0.0084 MoE (+0.5 sigma, Holm p = 0.96) - indistinguishable (below noise floor)
- Theater level (concurrent) vs Palisades only: +7.0% containment, +0.0941 MoE (+4.4 sigma, Holm p = 0.0001) - **significant**
- Pyrenees only vs Palisades only: +6.0% containment, +0.0857 MoE (+3.4 sigma, Holm p = 0.0038) - **significant**

