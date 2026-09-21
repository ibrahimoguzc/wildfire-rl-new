# final_plots/eval_2 - data provenance

Every unsuffixed file in this folder is computed from the SAME dataset as
the original evaluation boxplots in `final_plots/` (`eval_palisades.png`,
`eval_pyrenees.png`, `eval_salamis.png`, `eval_overall.png`):

> **`final_evaluation/results.csv` - the canonical final evaluation,
> SLURM job 756122: 8 policies x 3 maps x 250 ignitions, one seed draw
> per map (`--seed 1 --random-seeds`) shared by every policy.**

The eval_2 files differ from the `final_plots/` originals only in *view*,
never in data:

- restricted to the 5 headline policies (arm1 Palisades-only, arm2
  Pyrenees-only, arm5 concurrent, fixed heuristic, random tactics) out of
  the 8 in the CSV;
- Salamis rows are excluded from every pooled view (saturated: 100%
  containment for all policies including random);
- horizontal layout, entity-fixed colors matching the training figures.

| File(s) | Data | Generator |
|---|---|---|
| `eval_palisades.{png,pdf}`, `eval_pyrenees.{png,pdf}` | final_evaluation/results.csv (n=250/map) | `scripts/plot_eval2_horizontal.py` (no args) |
| `eval_pooled.{png,pdf}` | same, Palisades+Pyrenees only (500 pairs) | same |
| `statistical_tests.md` | same | `scripts/eval2_stat_tests.py` (no args) |
| `theater_preferability.md` | same | derived analysis (see conversation notes) |
| `regret.{png,pdf}` | same | `scripts/plot_regret.py` |
| `*_100sims.*` | job 886047: 5 policies x 3 maps x 100, seed-1 draw of 100 | same scripts, args: `<csv> _100sims` |
| `eval_palisades_1000sims.*` | job 894456: pre-specified confirmation - 3 policies, Palisades only, 1000 ignitions, FRESH `--seed 2` draw | `plot_eval2_horizontal.py`, args: `<csv> _1000sims` |
| `eval_pyrenees_1000sims.*` | job 896245: Pyrenees sibling of 894456 (same design; H1 theater > Palisades-only SUPPORTED +10.7 sigma; theater vs Pyrenees-only TIE, CI [-0.015, +0.023]) | same, rendered in-job |
| `*_50sims.*` | job 890854: 5 policies x 3 maps x 50, seed-1 draw of 50 | same scripts, args: `<csv> _50sims` |

Caveat on the three sample sizes: the 250/100/50 seed sets are nested
prefixes of the same seed-1 RNG stream per map (the 100-set's Palisades
fires are the first 100 of the 250-set's), so the smaller runs are partial
replications of the larger one, NOT independent samples - do not pool them.
The independent fresh draw is the pre-specified Palisades confirmation run
(job 894456, `--seed 2`, 1000 ignitions), reported separately.
