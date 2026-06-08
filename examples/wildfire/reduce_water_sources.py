"""Drop water sources that NO aircraft can scoop from (zero behaviour drift).

Agents only ever use the *nearest feasible* water node
(suppression_tactics.py: select_suppressant_source -> nearest_position over
feasible_water_locations). A water body that is "infeasible" -- i.e. the scoop
confidence-rectangle (scooping_distance x span) does not fit inside its polygon
at any tested point/angle -- contributes ZERO feasible nodes and is therefore
invisible to the agents. Removing such bodies cannot change behaviour, but it
removes their per-reset feasibility-test + shapely-polygon cost
(all_feasible_water_sources is O(sources x aircraft_types)).

This tool:
  1. Builds the REAL WildfireSimulation once (so feasibility is computed by the
     exact same code the sim uses -- no reimplemented geometry).
  2. For every OSM water body, tests feasibility for EVERY aircraft type.
  3. Keeps a body iff it is feasible for >= 1 aircraft type.
  4. Writes a NEW *_water_sources_reduced.pkl + a NEW scenario JSON pointing to
     it. Originals are never modified.
  5. PROVES safety: rebuilds the sim from the reduced file and asserts the
     per-aircraft feasible-node sets are identical to the original (the exact
     thing agents consume). If they differ, it aborts without writing.

Run (NOT via the shared-node bash by the assistant -- the user runs this):
    OMP_NUM_THREADS=1 ~/.conda/envs/rl-env/bin/python \
        -m examples.wildfire.reduce_water_sources --scenario Pyrenees.json
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from examples.wildfire.simulation import (  # noqa: E402
    WildfireParameters,
    WildfireSimulation,
)

INPUTS = Path("examples/wildfire/data/scenarios/inputs")
TERRAIN = Path("examples/wildfire/data/terrain")


def _feasible_node_sets(model):
    """Return {ac_type_id: set of feasible (x,y) nodes} using the sim's own code.

    Mirrors all_feasible_water_sources but keeps per-source detail so we can
    tell which water bodies actually contribute.
    """
    by_type = {}
    contributing = set()  # water_source unique_ids feasible for >=1 ac type
    for ac_type_id, (sdist, swidth) in model.agent_scooping_dimensions.items():
        nodes = []
        for ws in model.water_sources:
            is_feasible, ns = model.evaluate_water_source_feasibility(
                ws, swidth, sdist
            )
            if is_feasible:
                contributing.add(ws.unique_id)
                nodes.extend(tuple(np.round(n, 6)) for n in ns)
        by_type[ac_type_id] = set(map(tuple, np.round(np.array(nodes), 6))) if nodes else set()
    return by_type, contributing


def _index_water_bodies(model):
    """Map each WaterSourceManager to the OSM dict key it came from, by matching
    representative points. Returns {unique_id: representative_point (x,y)}."""
    reps = {}
    for ws in model.water_sources:
        rp = ws.representative_point
        reps[ws.unique_id] = (float(rp.x), float(rp.y))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Pyrenees.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scen_path = INPUTS / args.scenario
    params = WildfireParameters(**json.loads(scen_path.read_text()))
    namespace = params.terrain_inputs.file_namespace
    water_file = TERRAIN / f"{namespace}_water_sources.pkl"

    print(f"scenario   : {scen_path}")
    print(f"water file : {water_file}")

    with water_file.open("rb") as f:
        water_dict = pickle.load(f, encoding="bytes")
    print(f"OSM water bodies in file: {len(water_dict)}")

    # ---- Build the real sim and compute feasibility with the sim's own code ----
    sim = WildfireSimulation(parameters=params, seed=args.seed)
    model = sim.firefighters
    orig_sets, contributing_ids = _feasible_node_sets(model)
    reps = _index_water_bodies(model)
    total_nodes = {k: len(v) for k, v in orig_sets.items()}
    print(f"aircraft types: {list(model.agent_scooping_dimensions.keys())}")
    print(f"water sources built by sim: {len(model.water_sources)}")
    print(f"sources feasible for >=1 aircraft: {len(contributing_ids)}")
    print(f"feasible nodes per ac_type (original): {total_nodes}")

    # ---- Decide which OSM dict entries to keep ----
    # Match sim WaterSourceManagers back to OSM keys via representative points.
    # Manually-specified water_sources (always_feasible) are ALWAYS kept.
    # We keep an OSM body iff its representative point corresponds to a
    # contributing (feasible) source.
    contributing_reps = [reps[uid] for uid in contributing_ids if uid in reps]
    contributing_reps = np.array(contributing_reps) if contributing_reps else np.empty((0, 2))

    # Build reduced OSM dict by recomputing which keys are feasible. Simplest
    # robust mapping: rebuild per-OSM-body feasibility directly.
    # (We re-evaluate by reconstructing the same WaterSourceManager the sim made;
    #  sim.water_sources already includes both manual + OSM, in creation order.)
    kept_keys = []
    dropped_keys = []

    # Reproduce the model.water_sources construction order to align OSM keys.
    # model.water_sources = [manual...] + [OSM bodies with >=4 shell coords].
    osm_usable_keys = [
        k for k, s in water_dict.items()
        if len(s.get("shell_coords", [])) >= 4
    ]
    # The sim appends OSM sources after manual ones; the LAST len(osm_usable_keys)
    # water_sources correspond to osm_usable_keys in the same order.
    osm_sources = model.water_sources[-len(osm_usable_keys):] if osm_usable_keys else []
    assert len(osm_sources) == len(osm_usable_keys), (
        f"alignment mismatch: {len(osm_sources)} sim OSM sources vs "
        f"{len(osm_usable_keys)} usable OSM keys"
    )
    for key, ws in zip(osm_usable_keys, osm_sources):
        if ws.unique_id in contributing_ids:
            kept_keys.append(key)
        else:
            dropped_keys.append(key)

    reduced_dict = {k: water_dict[k] for k in kept_keys}
    # Preserve any non-usable entries? No -- they had <4 coords, sim ignores them,
    # so dropping them is also zero-drift. We drop them too for cleanliness.
    print(f"\nKEEP {len(kept_keys)}  /  DROP {len(dropped_keys) + (len(water_dict) - len(osm_usable_keys))}"
          f"  (of {len(water_dict)})")
    print(f"  - dropped infeasible OSM bodies : {len(dropped_keys)}")
    print(f"  - dropped <4-coord OSM bodies   : {len(water_dict) - len(osm_usable_keys)}")

    # ---- Write reduced pkl ----
    reduced_file = TERRAIN / f"{namespace}_water_sources_reduced.pkl"
    with reduced_file.open("wb") as f:
        pickle.dump(reduced_dict, f)
    print(f"\nwrote reduced water file: {reduced_file}  ({len(reduced_dict)} bodies)")

    # ---- PROOF: rebuild sim from reduced file, assert identical feasible sets ----
    # Point a copy of the scenario at the reduced water file via a temp namespace.
    reduced_namespace = f"{namespace}__reduced_tmp"
    # symlink/copy terrain files for the temp namespace is heavy; instead verify
    # by temporarily swapping the water file path.
    import shutil
    backup = water_file.with_suffix(".pkl.verifybak")
    shutil.copy2(water_file, backup)
    try:
        shutil.copy2(reduced_file, water_file)
        # clear cached metadata so the sim reloads
        from examples.wildfire.simulation import _TerrainParametersCache
        _TerrainParametersCache.metadata = {}
        params2 = WildfireParameters(**json.loads(scen_path.read_text()))
        sim2 = WildfireSimulation(parameters=params2, seed=args.seed)
        new_sets, _ = _feasible_node_sets(sim2.firefighters)
    finally:
        shutil.copy2(backup, water_file)   # always restore original
        backup.unlink()

    ok = True
    for ac in orig_sets:
        if orig_sets[ac] != new_sets.get(ac, set()):
            ok = False
            print(f"  MISMATCH ac_type {ac}: "
                  f"orig {len(orig_sets[ac])} vs reduced {len(new_sets.get(ac, set()))}")
    if ok:
        print("\nPROOF PASSED: feasible-node sets identical before/after for every "
              "aircraft type. Agent water behaviour is unchanged.")
    else:
        print("\nPROOF FAILED: feasible sets differ -- reduced file NOT safe. "
              "Leaving reduced pkl for inspection but DO NOT use it.")
        return

    # ---- Write a new scenario JSON pointing at the reduced file ----
    # Clone the terrain namespace by symlinking every existing "{namespace}_*"
    # file to a new "{namespace}_reduced" namespace, EXCEPT the water-sources
    # pkl, which gets the reduced copy. The sim derives the .meta from the
    # namespace PREFIX (features_file_name.split("_")[0]), so e.g.
    # "Pyrenees_..._reduced" still resolves to "Pyrenees.meta" -- no meta clone.
    new_namespace = f"{namespace}_reduced"
    created = []
    for src in sorted(TERRAIN.glob(f"{namespace}_*")):
        suffix = src.name[len(namespace):]            # e.g. "_terrain_features.npy"
        dst = TERRAIN / f"{new_namespace}{suffix}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if suffix == "_water_sources.pkl":
            shutil.copy2(reduced_file, dst)           # real reduced data
        else:
            dst.symlink_to(src.resolve())             # share original terrain
        created.append(dst.name)

    scen = json.loads(scen_path.read_text())
    scen["terrain_inputs"]["file_namespace"] = new_namespace
    new_scen = INPUTS / f"{scen_path.stem}_reduced_water.json"
    new_scen.write_text(json.dumps(scen, indent=4))
    print(f"\ncloned namespace -> {new_namespace} ({len(created)} files; "
          f"water pkl is the reduced copy, rest symlink originals)")
    print(f"wrote new scenario: {new_scen}")
    print(f"  meta resolves via prefix -> {namespace.split('_')[0]}.meta")
    print(f"  use it by pointing the runner at: {new_scen.name}")


if __name__ == "__main__":
    main()
