# PPO Runner Reconstruction Guide (LLM-Focused)

This document is a reconstruction spec for `examples/wildfire/ppo_runner.py`.
Use it when you have the wildfire simulation stack but **do not** have the PPO runner file.

The goal is deterministic re-creation of runner behavior, interfaces, state/action/reward definitions, and outputs.

## 1) Purpose and Scope

`ppo_runner.py` trains a Stable-Baselines3 PPO policy over a custom Gymnasium environment (`WildfireHourlyEnv`) that:

- wraps `examples/wildfire/simulation.py` (`WildfireSimulation`, `WildfireParameters`),
- applies tactic tuples to controlled aircraft every decision interval,
- advances wildfire simulation until next decision boundary,
- returns a normalized observation vector in `[0, 1]`,
- uses **delta cumulative MoE** as reward.

It supports:

- single-scenario training,
- switch-scenario training (across multiple scenario JSONs),
- ignition randomization modes,
- multi-process vectorized environments,
- periodic CSV progress export and final CSV/XLSX summaries,
- model save/load and checkpointing.

## 2) Module Dependencies and Imports

Reconstruct with these functional dependencies:

- Core: `argparse`, `csv`, `json`, `math`, `time`, `dataclasses`, `pathlib.Path`, `datetime.timedelta`, `typing`.
- Numeric/science: `numpy`, `scipy.ndimage`, `scipy.spatial.cKDTree`, `scipy.spatial.distance.cdist`.
- RL: `gymnasium`, `stable_baselines3.PPO`, callbacks (`BaseCallback`, `CallbackList`, `CheckpointCallback`), vec env (`DummyVecEnv`, `SubprocVecEnv`).
- Torch: `torch.nn as nn` (for optional custom shared-trunk extractor).
- Simulation API:
  - `WildfireSimulation`, `WildfireParameters`, `IgnitionCenterInput`, `ProtectionLocationInput`, `_TerrainParametersCache`
  - path constant `SCENARIOS_DIR`
  - tactic tables/enums:
    - `SELECT_POI_TABLE`, `SelectPOIType`
    - `TRACK_POI_TABLE`, `TrackPOIType`
    - `SUPPRESS_TABLE`, `SuppressType`
  - transform helpers:
    - `gps_to_mercator`, `gps_to_pos`, `index_to_pos`, `pos_to_gps`
  - terrain typing:
    - `TerrainTypes`, `FEATURES_COLOR_TABLE`

Optional dependency:

- `psutil` for RSS profiling (runner must still work without it).

## 3) Top-Level Constants (Current Behavior)

Use these constants (or equivalent configurable defaults):

- `MOE_WEIGHT = 0.25`
- `MAX_SPREAD_RATE_NORM_MPM = 30.0`
- `CONTROLLED_AGENT_COUNT = 3`
- `AGENT_FEATURE_COUNT = 2`  (x,y per controlled aircraft)
- `DEFAULT_DECISION_INTERVAL_MINUTES = 10`
- `ROLLOUT_STEPS_PER_ENV = 144`
- `NUM_MINIBATCH = 12`
- `LOG_INTERVAL_SUMMARY_EPISODES = 100`
- `LOG_INTERVAL_STEPS_EPISODES = 1000`
- `LR_DECAY_EXPONENT = 0.60`
- `POI_CANDIDATE_QUANTILE = 0.95`
- `MAX_POI_CANDIDATES = 4000`
- ignition constants:
  - `IGNITION_BOUNDARY_MARGIN_RATIO = 0.3`
  - `IGNITION_BOX_HALF_SIZE = 50`
  - `IGNITION_URBAN_BUFFER_M = 2000.0`
  - `IGNITION_V2_MARGIN_RATIO = 0.25`
- default switch set:
  - `("Palisades copy.json", "Pyrenees.json", "Salamis.json")`

### Scenario MoE Normalization Dictionary

Keep key matching by stem substring (`palisades`, `pyrenees`, `salamis`):

- Salamis:
  - `burnt_area_norm = 4146 * 10_000`
  - `cost_norm = 13_993 * 1_000_000`
  - `emission_norm = 714_009`
  - `casualty_norm = 6_000`
- Pyrenees:
  - `burnt_area_norm = 9938 * 10_000`
  - `cost_norm = 17_509 * 1_000_000`
  - `emission_norm = 2_364_064`
  - `casualty_norm = 10_000`
- Palisades:
  - `burnt_area_norm = 9087 * 10_000`
  - `cost_norm = 191_106 * 1_000_000`
  - `emission_norm = 131_224`
  - `casualty_norm = 300_000`

## 4) Action Space Design

### Primitive tactic options

- `select_poi`: `water`, `vip`, `vegetation`, `topography`, `indirect`
- `track_poi`: `indirect`, `follow_firefront`
- `suppress`: `direct`, `indirect`

### Combination rule

Generate Cartesian product then filter:

- If `select_poi == indirect`, force:
  - `track_poi == indirect`
  - `suppress == indirect`
- Otherwise allow all track/suppress combinations.

Resulting action list length is **17**.

Indices in current order:

1. (`water`, `indirect`, `direct`)
2. (`water`, `indirect`, `indirect`)
3. (`water`, `follow_firefront`, `direct`)
4. (`water`, `follow_firefront`, `indirect`)
5. (`vip`, `indirect`, `direct`)
6. (`vip`, `indirect`, `indirect`)
7. (`vip`, `follow_firefront`, `direct`)
8. (`vip`, `follow_firefront`, `indirect`)
9. (`vegetation`, `indirect`, `direct`)
10. (`vegetation`, `indirect`, `indirect`)
11. (`vegetation`, `follow_firefront`, `direct`)
12. (`vegetation`, `follow_firefront`, `indirect`)
13. (`topography`, `indirect`, `direct`)
14. (`topography`, `indirect`, `indirect`)
15. (`topography`, `follow_firefront`, `direct`)
16. (`topography`, `follow_firefront`, `indirect`)
17. (`indirect`, `indirect`, `indirect`)

### Gym action space

- `spaces.MultiDiscrete([17, 17, 17])` for 3 controlled aircraft.
- For each controlled agent, decode index -> tactic tuple and assign to:
  - `agent.tactic.select_poi = SELECT_POI_TABLE[select]()`
  - `agent.tactic.track_poi = TRACK_POI_TABLE[track]()`
  - `agent.tactic.suppress = SUPPRESS_TABLE[suppress]()`
  - `agent.force_tactic_swap = True`

## 5) Observation Space (State) Design

Observation is always normalized and clipped to `[0,1]`.
Shape is:

- 36 global features + 6 aircraft-position features = **42**.

### Ordered feature list

1. `scenario_is_palisades`
2. `scenario_is_pyrenees`
3. `scenario_is_salamis`
4. `time_since_detection_min`
5. `temperature_c`
6. `humidity_pct`
7. `wind_speed_ms`
8. `wind_direction_deg`
9. `time_to_sunset_min`
10. `distance_to_fire_line_m`
11. `distance_to_water_m`
12. `distance_fire_boundary_to_water`
13. `distance_fire_boundary_to_vip`
14. `distance_fire_boundary_to_vegetation`
15. `distance_fire_boundary_to_topography`
16. `distance_fire_boundary_to_indirect`
17. `ignition_x`
18. `ignition_y`
19. `fire_center_x`
20. `fire_center_y`
21. `leftmost_x`
22. `leftmost_y`
23. `rightmost_x`
24. `rightmost_y`
25. `uppermost_x`
26. `uppermost_y`
27. `lowermost_x`
28. `lowermost_y`
29. `spread_angle_deg`
30. `spread_ray_hit_x`
31. `spread_ray_hit_y`
32. `max_spread_rate_norm`
33. `distance_left_boundary`
34. `distance_right_boundary`
35. `distance_bottom_boundary`
36. `distance_top_boundary`
37. `agent_0_x`
38. `agent_0_y`
39. `agent_1_x`
40. `agent_1_y`
41. `agent_2_x`
42. `agent_2_y`

### Scaling rules

All features pass through:

- `_scale_to_unit(value, min, max)` then `_clip01`
- `NaN/inf` replaced by finite values (`nan->0`, `+inf->1`, `-inf->0`)
- final `np.clip(obs, 0, 1)`

Key scale anchors:

- Time features (`time_since_detection_min`, `time_to_sunset_min`): divide by 24h window (`0..1440 min`).
- Wind direction: `0..360`.
- Max spread rate: `0..MAX_SPREAD_RATE_NORM_MPM`.
- Coordinates: normalized by terrain bounding positions from active map.
- Distances to boundaries: divide by map width/height.
- Distances to POI classes from fire boundary points: divide by map diagonal.
- Scenario indicators: one-hot values in `{0,1}`.

## 6) Reward Function (Current)

Reward at step `t` is **delta cumulative MoE**:

- `reward_t = total_moe_t - total_moe_(t-1)`

where:

- `base_moe = 1 - 0.25*(burnt/b_norm + cost/c_norm + emissions/e_norm + casualties/cas_norm)`
- `propagation_penalty = 1 if wildfire out-of-bounds else 0`
- `total_moe = base_moe - propagation_penalty`

Important:

- penalty is part of MoE itself, not an extra additive reward term.
- runner stores `prev_total_moe`, initialized at reset from post-delay initial state.

## 7) Environment Lifecycle

### reset()

1. Select scenario:
   - single scenario: fixed scenario path.
   - switch-scenario:
     - `num_envs == 1`: stochastic reset sampling (optionally timestep-balanced).
     - `num_envs > 1`: worker assignment is round-robin at vec build.
2. Merge urban locations into protection locations (dedupe by gps/pos key).
3. Sample ignition center (if ignition-switch mode enabled), otherwise keep scenario ignition.
4. Create `WildfireSimulation(parameters, seed)`.
5. Ignite wildfire with `self.sim.wildfire.ignite(self.sim.ignition_centers)`.
6. Build static POI caches:
   - water, VIP, topography, vegetation.
7. Cache map scaling values and map diagonal.
8. Advance simulation to first decision time:
   - if CLI override provided: use that,
   - else use scenario `response_time` seconds.
9. Compute initial metrics and initialize cumulative MoE bookkeeping.
10. Return normalized observation + info dict.

### step(action)

1. Decode and apply per-agent tactics from `TACTIC_COMBINATIONS`.
2. Advance sim until next decision boundary (`decision_interval_minutes`).
3. Recompute metrics, reward, info.
4. Compute new observation.
5. If done, attach `episode_summary` including `moe_cumulative_reward`.

Done conditions:

- mission complete (fire out or sim stopped),
- max scenario runtime reached,
- decision-step cap reached.

## 8) Scenario Switching and Ignition Switching

### Scenario switching

- CLI `--switch-scenario` with optional `--switch-scenarios`.
- Validates equal agent counts across switch set.
- Scenario one-hot is always included in observation in both single and switch modes.

### Ignition switching

- `--switch-ignition-1`:
  - sample from ignitable cells inside 100x100 box around original ignition.
- `--switch-ignition-2`:
  - sample from ignitable cells in centered inner box (margin ratio from boundaries).
- Excludes water + urban + non-combustible cells, and enforces urban distance buffer.
- Converts selected grid index back to GPS and validates in-bounds; falls back to original ignition on conversion drift.

## 9) Training Loop + PPO Config

### Vec env

- Single env: direct `WildfireHourlyEnv`.
- Multi env: `SubprocVecEnv`; start method:
  - `--vec-start-method auto` => `fork` on Linux else `spawn`.

### Rollout/minibatch logic

- `rollout_steps = min(ROLLOUT_STEPS_PER_ENV, total_timesteps // num_envs)`
- `effective_batch = rollout_steps * num_envs`
- `dynamic_minibatches = min(NUM_MINIBATCH, effective_batch)`
- `batch_size = max(2, effective_batch // dynamic_minibatches)`

### PPO defaults (current)

- `learning_rate=0.0005` with schedule `lr(p)=initial * p^(lr_decay_exponent)`
- `lr_decay_exponent=0.60`
- `n_epochs=5`
- `gamma=0.99`
- `target_kl=None` by default

Policy choices:

- `default`: independent MLP heads (`pi=[512,256,128]`, `vf=[512,256,128]`, SB3 default activation).
- `variant_a`: custom shared trunk (`512->256`, ReLU) then `pi=[128]`, `vf=[128]`.

### Resume training

- `--load-model` loads PPO zip and overwrites key runtime hyperparams (`n_steps`, `batch_size`, LR schedule, `n_epochs`, `target_kl`, `gamma`) before learning continues.

## 10) Logging and Output Files

`TrainingLogger` stores:

- per-step records,
- per-episode summaries.

Periodic exports:

- every 100 episodes:
  - `results_<tag>_<N>_summary.csv`
- every 1000 episodes:
  - `results_<tag>_<N>_steps.csv`

`<tag>` source:

- CLI `--progress-file-tag`, else:
  - `switch_scenarios` for switch mode,
  - `single_<scenario_stem>` for single mode.

Final outputs:

- detailed step CSV:
  - default: `sparse_rewards_results.csv`
  - or CLI `--log-output`
- episode summary:
  - default: `training_episode_summaries.csv`
  - and auto `.xlsx` duplicate if `--summary-output` not provided.
- model:
  - `--save-model` path (resolved relative to `--output-dir` when not absolute)
- checkpoints:
  - enabled by `--checkpoint-interval`, written under `<output-dir>/checkpoints/`.

## 11) CLI Contract (Rebuild Checklist)

Rebuild parser with these arguments:

- scenario/training:
  - `--scenario`
  - `--timesteps`
  - `--decision-interval-minutes`
  - `--fire-detection-delay-minutes`
  - `--learning-rate`
  - `--gamma`
  - `--n-epochs`
  - `--target-kl`
  - `--lr-decay-exponent`
- switching:
  - `--switch-scenario`
  - `--switch-scenarios`
  - mutually exclusive:
    - `--switch-ignition-1`
    - `--switch-ignition-2`
  - `--ignition-inputs` (parsed; currently informational)
- execution:
  - `--num-envs`
  - `--vec-start-method {auto,fork,spawn,forkserver}`
  - `--device`
  - `--policy-arch {default,variant_a}`
- outputs/model:
  - `--output-dir`
  - `--log-output`
  - `--summary-output`
  - `--progress-file-tag`
  - `--save-model`
  - `--load-model`
  - `--checkpoint-interval`
- diagnostics:
  - `--profile-timings`
  - `--gc-collect-on-reset`

## 12) Required Simulation Interfaces (Contract Summary)

`WildfireHourlyEnv` assumes the simulation layer exposes:

- construction: `WildfireSimulation(parameters, seed)`
- ignition: `sim.wildfire.ignite(sim.ignition_centers)`
- stepping: `sim.step(force=True)`
- atmosphere:
  - `sim.atmosphere.temperature`
  - `sim.atmosphere.relative_humidity`
  - `sim.atmosphere.wind_speed`
  - `sim.atmosphere.wind_aspect`
  - `sim.atmosphere.next_sunset`
- timing:
  - `sim.timer.mission_time`
  - `sim.timer.mission_runtime`
- wildfire fields:
  - `burnt_area`, `fire_in_bounds`, `burning_indices`, `fire_positions`, `fire_states`
  - `get_spread_rates(indices)`
- firefighters fields:
  - `firefighters` list (agents with `pos`, `tactic`, `force_tactic_swap`)
  - `water_sources` with `pos`
  - `protection_locations` with `pos`
  - `fire_block_indices`
  - `current_block_index`
- global metrics:
  - `sim.total_fire_cost`
  - `sim.total_casualties`
  - `sim.total_fire_emissions`
- map geometry:
  - `sim.environment.dimensions`
  - `sim.environment.terrain.bounding_positions`
  - terrain `grid_description`, `origin`
  - priority map and elevation arrays
- stop signals:
  - `sim.is_stopped.is_set()`

## 13) Recommended Reconstruction Order for an LLM Agent

1. Recreate constants + helper pure functions (`_clip01`, `_scale_to_unit`, MoE functions).
2. Recreate tactic options and filtered combination generator.
3. Recreate `WildfireHourlyEnv` in this order:
   - init and spaces,
   - parameter loading / scenario selection / ignition selection,
   - static POI extraction and scaler caching,
   - `_compute_state`,
   - reward/info,
   - `reset`, `step`.
4. Recreate vector-env factories.
5. Recreate `TrainingLogger` callback and periodic export logic.
6. Recreate CLI `main()` and training orchestration.
7. Validate with one-episode smoke test and state-range check.

## 14) Minimal Validation Checklist

After reconstruction, verify:

- observation shape is `(42,)`.
- every observation value is in `[0,1]`.
- action space is `MultiDiscrete([17,17,17])`.
- reward changes when:
  - metrics change,
  - fire switches in/out of bounds.
- switch-scenario mode:
  - one-hot scenario flags match selected scenario.
- progress exports appear:
  - summary every 100 episodes,
  - steps every 1000 episodes.

## 15) Notes on Known Fragility

- Scenario JSON validation can fail if coordinates exceed map bounds.
- `_TerrainParametersCache.metadata` must be cleared before per-scenario validation/loading to avoid cross-scenario contamination.
- For high parallelism, start method and memory profile matter (`fork` on Linux typically lower overhead).

