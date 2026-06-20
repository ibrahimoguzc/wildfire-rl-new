#!/usr/bin/env python3
"""Verify switch-ignition (modes 1 and 2) seed fires only on valid cells.

For each (scenario, mode) we:
  * build the candidate set the env would use,
  * sample N ignitions through the real _sample_ignition_centers path,
  * round-trip each sampled GPS through the sim's grid (gps_to_pos ->
    pos_to_index) to find the cell the sim will ACTUALLY ignite, and
  * assert that cell is ignitable (not water/urban/rock, beyond the urban
    keep-out) and inside the mode's intended box.

Reports fallback rate (sample returned the scenario default), terrain
composition of the ignited cells, and box coverage.
"""
from __future__ import annotations

import collections
import numpy as np

from sosid.environment.terrain import TerrainTypes
from sosid.model.transform import gps_to_mercator, gps_to_pos, pos_to_index

from examples.wildfire.ppo_runnerv2 import (
    IGNITION_URBAN_BUFFER_M,
    IGNITION_V2_MARGIN_RATIO,
    WildfireHourlyEnv,
    _resolve_scenario,
)

N_SAMPLES = 2000
SCENARIOS = ("Palisades6p.json", "Pyrenees6p.json")
MODES = (1, 2)


def terrain_type_at(feature_data, fr, fc):
    """Return TerrainTypes for a fire-raster cell (argmax over the type axis)."""
    cell = feature_data[fr, fc]
    if np.ndim(cell) == 0:
        return TerrainTypes(int(cell))
    return TerrainTypes(int(np.argmax(cell)))


def run_case(scenario_name: str, mode: int) -> None:
    path = _resolve_scenario(scenario_name)
    env = WildfireHourlyEnv(
        scenario_path=path,
        switch_ignition_mode=mode,
        controlled_agent_count=6,
        state_space="old",
    )
    env.np_random, _ = __import__(
        "gymnasium"
    ).utils.seeding.np_random(12345)

    params = env.parameters
    ti = params.terrain_inputs
    feature_data = np.asarray(np.load(ti.features_file, allow_pickle=False))
    grid_desc = env._grid_description(params)
    grid_shape = (int(ti.grid_shape[0]), int(ti.grid_shape[1]))
    tl_merc = gps_to_mercator(ti.fire_map_coordinates[0])
    tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))

    # Ground-truth ignitable mask (same rule the env enforces).
    ignitable = env._build_ignitable_mask(feature_data, float(params.cell_size))
    candidates = env._scenario_ignition_candidates.get(path)

    # Intended box for mode 2 (central region); mode 1 box is around the
    # scenario ignition center, so we report candidate bbox instead.
    rows, cols = ignitable.shape
    print(f"\n===== {scenario_name}  mode {mode} "
          f"(--switch-ignition-{mode}) =====")
    print(f"  feature raster shape : {feature_data.shape[:2]}   "
          f"grid_shape: {grid_shape}   "
          f"{'MATCH' if feature_data.shape[:2] == grid_shape else 'MISMATCH!!'}")
    print(f"  ignitable cells      : {int(ignitable.sum()):,} / {rows*cols:,} "
          f"({100*ignitable.mean():.1f}%)")
    print(f"  candidate cells      : "
          f"{0 if candidates is None else len(candidates):,}")
    if candidates is None or len(candidates) == 0:
        print("  !! no candidates — switch-ignition would fall back to default")
        return
    cand_rows = candidates[:, 0]
    cand_cols = candidates[:, 1]
    print(f"  candidate row range  : [{cand_rows.min()}, {cand_rows.max()}] "
          f"of {rows}")
    print(f"  candidate col range  : [{cand_cols.min()}, {cand_cols.max()}] "
          f"of {cols}")
    if mode == 2:
        m = IGNITION_V2_MARGIN_RATIO
        exp = (int(round(rows*m)), int(round(rows*(1-m))),
               int(round(cols*m)), int(round(cols*(1-m))))
        in_box = ((cand_rows >= exp[0]) & (cand_rows < exp[1]) &
                  (cand_cols >= exp[2]) & (cand_cols < exp[3])).all()
        print(f"  expected v2 box      : rows[{exp[0]},{exp[1]}) "
              f"cols[{exp[2]},{exp[3]})   all candidates in box: {in_box}")

    # --- sample through the real path and validate the IGNITED cell ---
    fallback = 0
    bad_ignitable = 0
    out_of_box = 0
    drift_cells = []
    terr_counts = collections.Counter()
    default_gps = None
    if params.ignition_centers and params.ignition_centers[0].gps_coords:
        default_gps = tuple(params.ignition_centers[0].gps_coords)

    for _ in range(N_SAMPLES):
        sampled = env._sample_ignition_centers(path, params)
        gps = tuple(sampled[0].gps_coords)
        if default_gps is not None and gps == default_gps:
            fallback += 1
            continue
        px, py = gps_to_pos((float(gps[0]), float(gps[1])), tl_bounds)
        fr, fc = pos_to_index((float(px), float(py)), grid_desc)
        if not (0 <= fr < rows and 0 <= fc < cols):
            out_of_box += 1
            continue
        if not bool(ignitable[fr, fc]):
            bad_ignitable += 1
        terr_counts[terrain_type_at(feature_data, fr, fc).name] += 1
        # nearest candidate distance (drift check): the ignited cell should BE
        # a candidate cell, i.e. distance 0.
        d = np.min(np.abs(cand_rows - fr) + np.abs(cand_cols - fc))
        drift_cells.append(int(d))

    n_ok = N_SAMPLES - fallback
    print(f"  samples              : {N_SAMPLES}")
    print(f"  fell back to default : {fallback}  "
          f"({100*fallback/N_SAMPLES:.1f}%)")
    print(f"  ignited NON-ignitable: {bad_ignitable}   "
          f"<-- must be 0")
    print(f"  ignited out-of-grid  : {out_of_box}   <-- must be 0")
    if drift_cells:
        dc = np.asarray(drift_cells)
        print(f"  drift from candidate : max {dc.max()} cells, "
              f"mean {dc.mean():.2f}  <-- must be 0")
    print(f"  ignited terrain mix  : "
          f"{dict(terr_counts.most_common())}")
    status = "OK" if (bad_ignitable == 0 and out_of_box == 0
                      and (not drift_cells or max(drift_cells) == 0)) else "FAIL"
    print(f"  RESULT               : {status}")


def main() -> None:
    print(f"IGNITION_URBAN_BUFFER_M = {IGNITION_URBAN_BUFFER_M} m   "
          f"IGNITION_V2_MARGIN_RATIO = {IGNITION_V2_MARGIN_RATIO}")
    for scen in SCENARIOS:
        for mode in MODES:
            try:
                run_case(scen, mode)
            except Exception as exc:  # noqa: BLE001
                print(f"\n===== {scen} mode {mode}: ERROR {exc!r}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
