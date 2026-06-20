#!/usr/bin/env python
"""Verify --switch-ignition-1 seeds fires in a box centred on the TRUE
ignition, for each scenario, using the real (fixed) ppo_runnerv2 code path.

For each scenario it:
  1. reports the box centre the OLD code computed (int(y/cell_size)) vs the
     FIXED code (pos_to_index with the sim's grid) vs the true ignition cell,
  2. actually samples N ignitions via WildfireHourlyEnv._sample_ignition_centers
     and checks every sampled cell is (a) within the intended box of the true
     ignition and (b) ignitable (not water/urban/rock).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from examples.wildfire.ppo_runnerv2 import (
    WildfireHourlyEnv,
    IGNITION_BOX_HALF_SIZE,
)
from examples.wildfire.simulation import (
    WildfireParameters,
    _TerrainParametersCache,
)
from sosid.model.transform import (
    gps_to_mercator,
    gps_to_pos,
    pos_to_index,
)

INPUTS = Path("examples/wildfire/data/scenarios/inputs")
SCENARIOS = ["Pyrenees.json", "Palisades.json"]
N_SAMPLES = 2000


def true_ignition_index(params):
    ti = params.terrain_inputs
    center = params.ignition_centers[0]
    tl_merc = gps_to_mercator(ti.fire_map_coordinates[0])
    tl_bounds = (float(tl_merc[1]), float(tl_merc[0]))
    x, y = gps_to_pos(center.gps_coords, tl_bounds)
    gd = WildfireHourlyEnv._grid_description(params)
    r, c = pos_to_index((float(x), float(y)), gd)
    return (int(r), int(c)), (x, y), tl_bounds, gd


def make_env():
    env = object.__new__(WildfireHourlyEnv)
    env.switch_ignition_mode = 1
    env.switch_ignition = True
    env._scenario_ignition_candidates = {}
    env._scenario_ignitable_mask = {}
    env.np_random = np.random.default_rng(0)
    return env


for scen in SCENARIOS:
    path = INPUTS / scen
    # Terrain metadata cache is global; clear so each scenario validates
    # against its own map bounds (mirrors the runner).
    _TerrainParametersCache.metadata = {}
    params = WildfireParameters.model_validate_json(path.read_text())
    ti = params.terrain_inputs
    gs = ti.grid_shape
    md = ti.meta_data["mercator_dimensions"]
    cell = float(params.cell_size)
    real_cy, real_cx = md[1] / gs[0], md[0] / gs[1]

    (tr, tc), (x, y), tl_bounds, gd = true_ignition_index(params)
    old = (int(y / cell), int(x / cell))           # pre-fix box centre
    new = WildfireHourlyEnv._ignition_center_grid_pos(params)  # fixed

    print(f"\n===== {scen} =====")
    print(f"grid_shape {gs}  cell_size(json) {cell}  "
          f"real merc cell ~{real_cy:.2f} m  factor {real_cy/cell:.3f}")
    print(f"true ignition cell      : ({tr}, {tc})")
    print(f"OLD box centre (/cell)  : {old}   offset "
          f"~({(old[0]-tr)*real_cy:.0f}, {(old[1]-tc)*real_cx:.0f}) m")
    print(f"FIXED box centre        : {new}   offset "
          f"~({(new[0]-tr)*real_cy:.0f}, {(new[1]-tc)*real_cx:.0f}) m")

    # Exercise the real sampling path.
    env = make_env()
    rows, cols, bad_box, bad_ign = [], [], 0, 0
    ignitable = WildfireHourlyEnv._build_ignitable_mask(
        np.asarray(np.load(ti.features_file, allow_pickle=False)), cell
    )
    half = IGNITION_BOX_HALF_SIZE
    for _ in range(N_SAMPLES):
        (sampled,) = env._sample_ignition_centers(path, params)
        sx, sy = gps_to_pos(sampled.gps_coords, tl_bounds)
        sr, sc = pos_to_index((float(sx), float(sy)), gd)
        rows.append(sr)
        cols.append(sc)
        if abs(sr - new[0]) > half or abs(sc - new[1]) > half:
            bad_box += 1
        if not (0 <= sr < ignitable.shape[0] and 0 <= sc < ignitable.shape[1]
                and bool(ignitable[sr, sc])):
            bad_ign += 1

    rows, cols = np.array(rows), np.array(cols)
    dist_m = np.hypot((rows - tr) * real_cy, (cols - tc) * real_cx)
    print(f"sampled {N_SAMPLES}: row[{rows.min()}..{rows.max()}] "
          f"col[{cols.min()}..{cols.max()}]")
    print(f"  dist from true ignition: median {np.median(dist_m):.0f} m, "
          f"max {dist_m.max():.0f} m  (box half ~{half*real_cy:.0f} m)")
    print(f"  outside intended box : {bad_box}/{N_SAMPLES}")
    print(f"  on water/urban/rock  : {bad_ign}/{N_SAMPLES}")
    ok = (bad_box == 0 and bad_ign == 0
          and abs(new[0] - tr) <= 1 and abs(new[1] - tc) <= 1)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
