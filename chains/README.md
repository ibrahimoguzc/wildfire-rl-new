# Chained training runs

Run a long training job as a sequence of `owl_normal_short` (<= 24 h) jobs
instead of one long job in a slower queue. Each chunk loads the previous
chunk's policy, keeps training, saves, and prints the exact `sbatch` line for
the next chunk. You submit each chunk yourself.

## Layout

- `chains/<run>.env` — one config per experiment: scenario, python args,
  chain total, per-chunk budget.
- `slurms/chain_chunk.slurm` — the generic chunk job. Same for every run.
- `scripts/chain_submit.sh <config> [chunk]` — submits a chunk (defaults to
  the next unfinished one) and prints what it did.

Everything for one run lives in a stable directory that does not change
between chunks:

```
examples/wildfire/data/scenarios/outputs/<RUN_NAME>_chain/
  latest.zip                      <- resume point, overwritten by each chunk
  chunk01.zip, chunk02.zip, ...   <- per-chunk archive
  checkpoints/                    <- CheckpointCallback, globally numbered
  results_<TAG>_<N>_summary.csv   <- one continuous series across all chunks
  CHAIN_COMPLETE                  <- written when the chain total is reached
  chain_state.env                 <- last chunk index / job id
```

## Usage

```bash
# 1. Copy a config and edit it
cp chains/palisades_4sp6ev_grpbytype_sw3.env chains/my_run.env

# 2. Submit chunk 1
scripts/chain_submit.sh chains/my_run.env

# 3. When it finishes, submit the next chunk (the finished job's log ends
#    with this exact line)
scripts/chain_submit.sh chains/my_run.env
```

`chain_submit.sh` with no chunk number submits `(archived chunks) + 1`, so step
3 is the same command every time. An interrupted chunk leaves no archive, so
resubmitting reuses its number and picks up from whatever it managed to save.
The script refuses to submit when `CHAIN_COMPLETE` exists and warns when a
chunk of the same run is still queued. `CHAIN_DRY_RUN=1` prints the `sbatch`
line instead of submitting; a chunk number can be forced as the 2nd argument.

## How continuity is preserved

- **Learning rate.** The schedule is the usual
  `lr(p) = --learning-rate * p^--lr-decay-exponent`; what changes is where `p`
  comes from, and it follows the budget you chose:

  | chain budget | `p` | LR state lives in |
  |---|---|---|
  | `CHAIN_TOTAL_TIMESTEPS` | `1 - timesteps_done / chain_total` | the model zip (`num_timesteps`) |
  | `CHAIN_TOTAL_SIMULATIONS` | `1 - simulations_done / chain_total` | `chain_progress.json` beside the model |

  Both start each chunk exactly where the previous one left off, so the LR
  curve is the one a single long job would have produced and does not depend
  on where the chunk boundaries fall. A chunk that dies early costs time, not
  schedule position.

  Timestep chains work by handing `model.learn()` the chain's *remaining*
  budget with `reset_num_timesteps=False`, so SB3's own `progress_remaining`
  spans the chain; the per-chunk budget is enforced by a callback instead.
  Without that, every chunk would restart the LR at its initial value and
  decay to ~0 — a sawtooth.

  Simulation chains cannot use SB3's `progress_remaining` at all: the timestep
  ceiling is `simulations x 144` (the longest possible episode) while episodes
  really average ~48 steps on Palisades 4sp6ev, so a timestep-driven LR would
  still sit at 75% of initial at the end of the chain. The schedule reads the
  episode counter directly instead. That makes the episode count training
  state rather than logging, which is why it is saved next to the weights; if
  a resumed chunk cannot recover it, the run aborts instead of silently
  restarting the decay at the initial LR.
- **Training curves.** `--resume-summary auto` reloads the previous chunk's
  `*_summary.csv`, so episode numbering continues and every progress file
  holds the full chain history. The plotting scripts, which take the
  highest-N summary in a run directory, keep working with no changes.
- **Optimizer state.** SB3 stores `policy.optimizer` in the zip, so Adam
  moments survive the restart; nothing extra is needed.
- **Episode sequence.** The env base seed is drawn fresh per process, so a
  chunk does not replay the previous chunk's ignitions.
- **Wall limit.** `--max-train-hours` stops training cleanly with time to
  spare, and a SIGTERM handler saves the model plus the episode summary if
  SLURM kills the job first.

## Disk

Every `results_<tag>_<N>_summary.csv` snapshot holds the full episode history,
so keeping all of them costs O(episodes^2): a 42,800-episode run spends 3.4 GB
on 428 snapshots, and a 150,000-episode chain would spend ~43 GB. The configs
pass `--summary-retention 3`, which deletes a snapshot once its (strictly
larger) replacement is written and brings that back to ~175 MB. The newest file
still holds everything, so the plotting scripts and `--resume-summary auto` are
unaffected. Omit the flag to keep every snapshot, as single jobs always have.

Per-step and decision-step CSVs are flushed and cleared every 1000 episodes, so
they grow linearly — budget roughly 1 GB per 40k episodes for those.

## Adopting a run that is already in flight

A single-job run that is about to hit its wall limit can be turned into a
chain. Its checkpoints and summaries are exactly what a chunk would have
produced, so they only need to be moved into a chain directory:

```bash
SRC=examples/wildfire/data/scenarios/outputs/palisades_4sp6ev_grpbytype_sw3_directional_k3_646949
DST=examples/wildfire/data/scenarios/outputs/palisades_4sp6ev_grpbytype_sw3_directional_k3_chain
mkdir -p "$DST"

# newest checkpoint becomes the resume point
cp "$SRC/checkpoints/palisades_4sp6ev_grpbytype_sw3_directional_k3_1007872_steps.zip" \
   "$DST/latest.zip"

# episode history: take the summary written closest to that checkpoint, so the
# curve does not include episodes whose weight updates were rolled back
cp "$SRC/results_..._<N>_summary.csv" "$DST/"

scripts/chain_submit.sh chains/palisades_4sp6ev_grpbytype_sw3.env 2
```

The `RUN_NAME` in the config must match the `--progress-file-tag` the original
job used, otherwise `--resume-summary auto` will not recognise the copied
summary. Pass the chunk number explicitly (`2` above) since the adopted work
left no `chunk01.zip`.

Adopted models sometimes carry a timestep counter that understates their real
progress — a policy continued with `--load-model` before chaining existed
restarted its counter at 0 on every job. The runner refuses to resume such a
model rather than restart the LR decay; tell it where the run actually stands
by adding to the config:

```bash
CHAIN_ELAPSED_TIMESTEPS=1007872   # true cumulative timesteps of the adopted run
```

It applies to the first chunk after adoption only — from then on the counter
advances normally, so remove it before submitting the chunk after that. It is
not needed for a chain started with `chain_submit.sh`, and the same situation
in a simulation-budgeted chain is handled by `--resume-summary` instead.

## Sizing a chunk

Measured on 128 envs / 96 CPUs, one job per node (2026-07-29/30):

| run | fleet | simulations/h | timesteps/h | steps/episode |
|---|---|---|---|---|
| Palisades 4sp6ev sw3 | 4 sp + 6 ev | ~900 | 44,000 | 48.2 |
| Pyrenees 7p sw4 | 7 sp | 2,190–2,350 | 38,300 | 17.5 |
| Pyrenees 10p sw4 | 10 sp | 1,810–1,970 | 31,800 | 15.8 |
| Pyrenees 7sp5ev sw4 | 7 sp + 5 ev | 1,560–1,610 | — | 15.9 |
| Pyrenees 6sp6ev sw3 | 6 sp + 6 ev | 1,650 | 26,400 | 16.3 |
| Pyrenees 4sp10ev sw3 | 4 sp + 10 ev | 1,790 | 27,500 | 15.7 |
| Pyrenees 4sp12ev sw3 | 4 sp + 12 ev | 1,790 | 27,200 | 15.2 |

Seaplane count drives cost more than total aircraft: 4sp12ev runs no slower
than 4sp10ev, while going from 7 seaplanes to 7 seaplanes + 5 eVTOLs costs a
third of the throughput. Palisades episodes are ~3x longer than Pyrenees ones,
so its simulations/hour is far lower at a similar timestep rate.

Set `CHUNK_TIMESTEPS` or `CHUNK_SIMULATIONS` to a round number that comfortably
fits 22.5 h at the pessimistic end of the range and leave `MAX_TRAIN_HOURS` as
the backstop. Take the real rate from the first chunk's log and adjust the
later chunks — since chunks are submitted by hand, changing the per-chunk
budget mid-chain is free and does not disturb the LR schedule.
