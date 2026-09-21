# Statistical comparison - eval_2 figure set

Data: `examples/wildfire/data/scenarios/outputs/eval2_5way_3maps_50sims_890854/results.csv` -
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

## Palisades  (50 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 94.0% | [83.8, 97.9] | +0.9258 | +0.9902 | 0.0361 |
| Fixed tactics | 94.0% | [83.8, 97.9] | +0.9224 | +0.9860 | 0.0368 |
| Palisades only | 100.0% | [92.9, 100.0] | +0.9916 | +0.9931 | 0.0010 |
| Pyrenees only | 96.0% | [86.5, 98.9] | +0.9489 | +0.9905 | 0.0297 |
| Theater level (concurrent) | 100.0% | [92.9, 100.0] | +0.9910 | +0.9920 | 0.0010 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Fixed tactics vs Palisades only | -6.0% | 0.25 | -0.0692 | -1.9 | 0.066 | 0.66 | 0.0018 | ns |
| Fixed tactics vs Theater level (concurrent) | -6.0% | 0.25 | -0.0686 | -1.9 | 0.068 | 0.66 | 0.0017 | ns |
| Random tactics vs Palisades only | -6.0% | 0.25 | -0.0658 | -1.8 | 0.073 | 0.66 | 0.0032 | ns |
| Random tactics vs Theater level (concurrent) | -6.0% | 0.25 | -0.0652 | -1.8 | 0.076 | 0.66 | 0.0065 | ns |
| Palisades only vs Pyrenees only | +4.0% | 0.5 | +0.0427 | +1.5 | 0.15 | 0.92 | 0.15 | ns |
| Pyrenees only vs Theater level (concurrent) | -4.0% | 0.5 | -0.0421 | -1.4 | 0.16 | 0.92 | 0.41 | ns |
| Palisades only vs Theater level (concurrent) | +0.0% | 1 | +0.0006 | +1.1 | 0.29 | 1 | 0.86 | ns |
| Fixed tactics vs Pyrenees only | -2.0% | 1 | -0.0265 | -0.7 | 0.47 | 1 | 0.0025 | ns |
| Random tactics vs Pyrenees only | -2.0% | 1 | -0.0231 | -0.5 | 0.63 | 1 | 0.017 | indistinguishable (below noise floor) |
| Random tactics vs Fixed tactics | +0.0% | 1 | +0.0034 | +0.1 | 0.94 | 1 | 0.62 | indistinguishable (below noise floor) |

**Mission-native specialist here: Palisades only.**

- Palisades only vs the theater-level agent: +0.0% containment, +0.0006 MoE (+1.1 sigma, Holm p = 1) - ns
- Palisades only vs Pyrenees only: +4.0% containment, +0.0427 MoE (+1.5 sigma, Holm p = 0.92) - ns


## Pyrenees  (50 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 6.0% | [2.1, 16.2] | -0.1095 | -0.1660 | 0.0405 |
| Fixed tactics | 10.0% | [4.3, 21.4] | -0.0618 | -0.1667 | 0.0509 |
| Palisades only | 8.0% | [3.2, 18.8] | -0.0966 | -0.1929 | 0.0465 |
| Pyrenees only | 16.0% | [8.3, 28.5] | +0.0276 | -0.1418 | 0.0608 |
| Theater level (concurrent) | 16.0% | [8.3, 28.5] | +0.0302 | -0.1426 | 0.0606 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -10.0% | 0.062 | -0.1396 | -2.8 | 0.0065 | 0.065 | 0.0038 | ns |
| Random tactics vs Pyrenees only | -10.0% | 0.062 | -0.1370 | -2.8 | 0.0079 | 0.071 | 0.0065 | ns |
| Palisades only vs Theater level (concurrent) | -8.0% | 0.12 | -0.1268 | -2.7 | 0.0086 | 0.071 | 0.0012 | ns |
| Palisades only vs Pyrenees only | -8.0% | 0.12 | -0.1242 | -2.7 | 0.01 | 0.072 | 0.00037 | ns |
| Fixed tactics vs Theater level (concurrent) | -6.0% | 0.25 | -0.0919 | -2.2 | 0.031 | 0.19 | 0.0029 | ns |
| Fixed tactics vs Pyrenees only | -6.0% | 0.38 | -0.0893 | -1.7 | 0.11 | 0.53 | 0.0074 | ns |
| Random tactics vs Fixed tactics | -4.0% | 0.5 | -0.0477 | -1.5 | 0.14 | 0.57 | 0.82 | ns |
| Fixed tactics vs Palisades only | +2.0% | 1 | +0.0349 | +0.8 | 0.41 | 1 | 0.49 | ns |
| Random tactics vs Palisades only | -2.0% | 1 | -0.0128 | -0.3 | 0.75 | 1 | 0.35 | indistinguishable (below noise floor) |
| Pyrenees only vs Theater level (concurrent) | +0.0% | 1 | -0.0026 | -0.1 | 0.94 | 1 | 0.6 | indistinguishable (below noise floor) |

**Mission-native specialist here: Pyrenees only.**

- Pyrenees only vs the theater-level agent: +0.0% containment, -0.0026 MoE (-0.1 sigma, Holm p = 1) - indistinguishable (below noise floor)
- Pyrenees only vs Palisades only: +8.0% containment, +0.1242 MoE (+2.7 sigma, Holm p = 0.072) - ns


## Combined (Palisades + Pyrenees)  (100 shared ignitions, fully paired)

| Policy | Contained | Wilson 95% CI | Mean MoE | Median MoE | SE |
|---|---|---|---|---|---|
| Random tactics | 50.0% | [40.4, 59.6] | +0.4082 | +0.4518 | 0.0586 |
| Fixed tactics | 52.0% | [42.3, 61.5] | +0.4303 | +0.9659 | 0.0585 |
| Palisades only | 54.0% | [44.3, 63.4] | +0.4475 | +0.9798 | 0.0594 |
| Pyrenees only | 56.0% | [46.2, 65.3] | +0.4882 | +0.9834 | 0.0572 |
| Theater level (concurrent) | 58.0% | [48.2, 67.2] | +0.5106 | +0.9838 | 0.0569 |

| A vs B | ΔContain | McNemar p | ΔMoE | σ | t p (raw) | t p (Holm) | Wilcoxon p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random tactics vs Theater level (concurrent) | -8.0% | 0.0078 | -0.1024 | -3.4 | 0.0011 | 0.011 | 0.00041 | **significant** |
| Fixed tactics vs Theater level (concurrent) | -6.0% | 0.031 | -0.0803 | -2.9 | 0.0044 | 0.04 | 3.5e-05 | **significant** |
| Palisades only vs Theater level (concurrent) | -4.0% | 0.12 | -0.0631 | -2.6 | 0.0097 | 0.078 | 0.021 | ns |
| Random tactics vs Pyrenees only | -6.0% | 0.11 | -0.0801 | -2.3 | 0.023 | 0.16 | 0.0018 | ns |
| Fixed tactics vs Pyrenees only | -4.0% | 0.29 | -0.0579 | -1.8 | 0.079 | 0.48 | 0.00014 | ns |
| Random tactics vs Palisades only | -4.0% | 0.22 | -0.0393 | -1.5 | 0.15 | 0.75 | 0.31 | ns |
| Palisades only vs Pyrenees only | -2.0% | 0.69 | -0.0407 | -1.4 | 0.16 | 0.75 | 0.023 | ns |
| Pyrenees only vs Theater level (concurrent) | -2.0% | 0.62 | -0.0224 | -1.0 | 0.33 | 0.99 | 0.99 | ns |
| Random tactics vs Fixed tactics | -2.0% | 0.69 | -0.0222 | -0.8 | 0.41 | 0.99 | 0.3 | ns |
| Fixed tactics vs Palisades only | -2.0% | 0.69 | -0.0172 | -0.6 | 0.55 | 0.99 | 0.1 | indistinguishable (below noise floor) |

- Theater level (concurrent) vs Pyrenees only: +2.0% containment, +0.0224 MoE (+1.0 sigma, Holm p = 0.99) - ns
- Theater level (concurrent) vs Palisades only: +4.0% containment, +0.0631 MoE (+2.6 sigma, Holm p = 0.078) - ns
- Pyrenees only vs Palisades only: +2.0% containment, +0.0407 MoE (+1.4 sigma, Holm p = 0.75) - ns

