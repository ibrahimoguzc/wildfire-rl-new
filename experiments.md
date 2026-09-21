# Experiment log

Paper experiment registry: job IDs, models, data and figure locations.
Last updated: 2026-09-20. All training runs: 5 seaplanes + 5 eVTOLs grouped
by type (`--group-sizes 5 5`), switch-ignition-4, water set 2, directional-2
state space (K=3 fire fronts), `--cell-size code`, PPO 128 envs, LR 5e-4 on
the episode-driven schedule. `OUT` below = `examples/wildfire/data/scenarios/outputs`.

## The five presentation arms + multiweight pilot (all complete)

150,000 simulations each (arm 6: 25,000). Frozen finals are symlinked in
`final_evaluation/models/`.

| Arm | Curriculum | Job(s) | Output dir (under `OUT/`) | Final model |
|---|---|---|---|---|
| 1 | Palisades only | 730529 | `palisades_5sp5ev_grpbytype_sw4_directional2_k3_150k_730529` | `palisades_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip` |
| 2 | Pyrenees only | 717319 | `pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_717319` | `pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip` |
| 3 | Even split 75k/75k (`--missions`) | chained chunks | `missions_pal75k_pyr75k_5sp5ev_grpbytype_sw4_directional2_k3_chain` | `latest.zip` |
| 4 | Palisades 25k -> Pyrenees 125k (`--missions` + weights) | 717320 | `missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_717320` | `missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip` |
| 5 | Concurrent round-robin 64/64 (`--switch-scenarios`) | chained chunks | `concurrent_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_chain` | `latest.zip` (= `chunk23.zip`); realised split 65,569 Palisades / 84,431 Pyrenees |
| 6 | Palisades multiweight pilot, 25k | 745083 | `palisades_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_745083` | `palisades_multiweight_5sp5ev_directional2_sims25000.zip` (trained with 0.5 eVTOL weights - the scenario JSON was edited afterwards; never re-evaluate it in a live-file env) |

Arms 3-5 observe 69 features (3-way scenario one-hot); arms 1, 2, 6 observe 66.
Training curves per arm: `results_*_summary.csv` in each output dir
(`moe_cumulative_reward`, `propagation_factor`, `sim_seed` per episode).

## Final evaluation (complete)

- **Job 756122**, script `final-evaluation-8way-3maps-250sims.slurm`.
- 8 policies (arms 1-6 + fixed heuristic + random tactics) x 3 maps
  (Palisades / Pyrenees / Salamis) x 250 ignitions, seed-paired within map
  (`--seed 1 --random-seeds`), evaluated under the standard uniform-weight
  scenarios.
- Data: `final_evaluation/results.csv` (6,000 rows, merged + pairing
  verified), per-map `results_*.csv` / `summary_*.csv`,
  `statistical_comparison.txt` (Wilson CIs, McNemar, paired t / Wilcoxon,
  Holm), frozen model symlinks in `final_evaluation/models/`.

## Multiweight study (150k, per-type firefront cost weights)

Scenarios `Palisades5sp5ev_multiweight.json` / `Pyrenees5sp5ev_multiweight.json`
(`examples/wildfire/data/scenarios/inputs/`): seaplanes 1.0 on all five cost
terms, eVTOLs 0.75 on vegetation/topography. Runner: `ppo_runnerv3`
(= v2 + firefront-cost-weight logging only).

| Run | Job | State | Output dir (under `OUT/`) | Final model |
|---|---|---|---|---|
| Palisades mw 150k | **855200** (predecessor 769427 hung on owl-hm002 file-table exhaustion at 12.7k, cancelled; 815726 cancelled while pending) | COMPLETE 2026-09-20, 150,006 sims | `palisades_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k_855200` | `palisades_multiweight_5sp5ev_directional2_sims150000.zip` |
| Pyrenees mw 150k | **769428** | COMPLETE 2026-09-11, 150,000 sims | `pyrenees_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k_769428` | `pyrenees_multiweight_5sp5ev_directional2_sims150000.zip` |
| Concurrent mw 150k (64/64 workers, both mw scenarios) | **870545** | RUNNING (48k/150k on 2026-09-20; ETA ~Sep 22) | `concurrent_multiweight_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_870545` | `concurrent_multiweight_5sp5ev_directional2_sims150000.zip` (on completion) |

Slurm scripts: `150k-{palisades,pyrenees}-multiweight-5sp5ev-sw4-ws2-directional2.slurm`,
`150k-concurrent-multiweight-5sp5ev-sw4-ws2-directional2.slurm`.

## Random-tactics baselines

| Purpose | Job | State | Data |
|---|---|---|---|
| Matched-regime baseline for the training figures: 2,000 sims per map, standard scenarios | **855198** | COMPLETE | `OUT/random_tactics_training_maps_855198/results_{palisades,pyrenees}.csv`. Palisades: MoE +0.866, escape 11.7%. Pyrenees: MoE -0.110, escape 93.5% |
| Paired seed replay of pal-mw (855200): its recorded `sim_seed`s replayed via `ppo_tester --seed-file` on the mw scenario | **864343** | COMPLETE - covers episodes 1-51,400 only (snapshot taken mid-training; top-up for 51,401-150,006 not yet run) | `OUT/random_tactics_mw_palisades_seed_replay_864343/{seeds.txt,results.csv}`. MoE +0.869, escape 11.4% |
| Paired seed replay of pyr-mw (769428): all 150,000 recorded seeds | **870546** | RUNNING (ETA ~Sep 21; CSV written only at job end) | `OUT/random_tactics_mw_pyrenees_seed_replay_870546/` |

Slurm scripts: `random-tactics-training-maps-2000sims.slurm`,
`random-tactics-mw-{palisades,pyrenees}-seed-replay.slurm`.
Replay validity: the env initialises the whole simulation from `sim_seed`,
so replayed episode i is the same fire as training episode i.

## Figures (already produced)

- `final_plots/` - paper set from the five-arm study:
  - `fig1_all_arms.png`, `fig2_pyrenees_only.png`, `fig3_palisades_only.png`
    (training overlays, `scripts/plot_chain_compare.py`)
  - `eval_{palisades,pyrenees,salamis,overall}.png` (boxplots from
    `final_evaluation/results.csv`, `scripts/plot_eval_boxplots.py`)
  - `eval_theater_*.{png,pdf}`, `eval_theater_stats.csv`,
    `results_section.tex` (theater comparison + LaTeX)
  - `scenario_maps_sw4.png` (`scripts/plot_scenario_maps_sw4.py`)
  - `training_curves_by_map`, `training_curves_three_runs`,
    `training_curves_grid` (`scripts/plot_training_curves.py`; grid = arms
    1, 2, 5)
- `final_plots/training/` - the grid's six panels as stand-alone figures
  (`{moe,escape}_{all_instances,palisades,pyrenees}.{pdf,png}`), plus
  `*_new` variants with random tactics as a moving-average curve over its
  episodes, bootstrap-continued to 150k on both maps. Real-data share:
  Palisades 51,400 episodes (seed replay 864343), Pyrenees 2,000 (855198;
  swap to replay 870546 when it completes). The full per-episode series
  (moe, escaped, real/bootstrap marker) are persisted at
  `OUT/random_tactics_extended_series/{palisades,pyrenees}_random_150k.csv`
  and regenerate deterministically (fixed rng seed) with the same
  generator script.
- `final_mw_plots/` - multiweight training curves, one figure per map
  (`palisades_mw.png` snapshot at 101.7k - refresh now that 855200 is
  complete; `pyrenees_mw.png` full 150k), via `plot_chain_compare.py`.

## Follow-up evaluations (eval_2 series)

- **886047** (100 sims/map) and **890854** (50 sims/map): 5 policies
  (arm1/arm2/arm5 finals + fixed + random), seed-1 draws (nested prefixes
  of the 250-seed sets - partial replications, do not pool). Data under
  `OUT/eval2_5way_3maps_{100,50}sims_<job>/`; figures + stats in
  `final_plots/eval_2/` (see its README for provenance).
- **894456** - PRE-SPECIFIED confirmation, Palisades, 1,000 ignitions,
  fresh `--seed 2` draw, theater vs both specialists, hypotheses and
  analysis committed in the slurm before data collection. Verdicts
  (`OUT/eval_theater_palisades_1000sims_894456/prespecified_analysis.txt`):
  theater > Pyrenees-only **SUPPORTED** (97.6% vs 94.3%, +4.3 sigma,
  Holm p = 2e-05, McNemar 48:15 p = 2e-05); theater > Palisades-only
  **not supported** (97.6% vs 98.3%, -1.2 sigma - the marginal seed-1
  edge did not replicate; report native-vs-theater as a tie).

## Known caveats

- `Palisades5sp5ev_multiweight.json` was edited 2026-08-28 (eVTOL 0.5 ->
  0.75): arm 6 (745083) trained on 0.5 weights, the 150k mw runs on 0.75.
- The 8-way final evaluation ran arm 6 under standard uniform weights
  (common-environment design, deliberate).
- Superseded/failed eval jobs: 753124 (OOM at 96 workers), 754407 (7-way
  500-sim predecessor of the final evaluation).
- owl-hm[001-003] nodes are ~30% slower; 150k does not fit their 96 h wall.
