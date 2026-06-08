"""Generate two water-source subsets per scenario, selectable via --water-set.

Set 1 : water bodies the CURRENT fleet can scoop (its own scoop footprint).
Set 2 : water bodies example_aircraft_1 (scooping_distance=6, span=6) can scoop.

Because a smaller scoop footprint fits in strictly more (or equal) bodies,
Set 1 is a subset of Set 2 (Set 1 <= Set 2 in size), as requested.

Each set is written as a filtered copy of the OSM water_dict (same polygon
data, fewer entries) to:
    {namespace}_water_sources_set1.pkl
    {namespace}_water_sources_set2.pkl
The originals ({namespace}_water_sources.pkl) are never modified. At runtime,
pass --water-set 1|2 to the runner; TerrainParameters.water_sources_file_name
then resolves to the matching pkl.

Feasibility is computed with the sim's OWN evaluate_water_source_feasibility
(builds the real sim once), so the subsets match what agents would actually use.

Manually-specified (always_feasible) water sources are kept in BOTH sets (they
are scenario points, not OSM polygons, and are aircraft-independent).

Run (user runs it; one sim build per scenario, no training):
    OMP_NUM_THREADS=1 ~/.conda/envs/rl-env/bin/python \
        -m examples.wildfire.generate_water_sets --scenario Pyrenees.json
    # or all three:
    ... --scenario "Pyrenees.json" "Palisades copy.json" "Salamis.json"
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from examples.wildfire.simulation import (  # noqa: E402
    WildfireParameters,
    WildfireSimulation,
    _TerrainParametersCache,
)

INPUTS = Path("examples/wildfire/data/scenarios/inputs")
TERRAIN = Path("examples/wildfire/data/terrain")

SMALL_SCOOP = (6.0, 6.0)  # example_aircraft_1.json (scooping_distance, span)


def _feasible_osm_keys(model, osm_keys, osm_sources, sdist, swidth):
    """Return the subset of osm_keys whose body is feasible for (sdist,swidth)."""
    kept = []
    for key, ws in zip(osm_keys, osm_sources):
        is_feasible, _ = model.evaluate_water_source_feasibility(
            ws, swidth, sdist
        )
        if is_feasible:
            kept.append(key)
    return kept


def process(scenario, seed=0):
    _TerrainParametersCache.metadata = {}
    params = WildfireParameters(**json.loads((INPUTS / scenario).read_text()))
    namespace = params.terrain_inputs.file_namespace
    water_file = TERRAIN / f"{namespace}_water_sources.pkl"

    with water_file.open("rb") as f:
        water_dict = pickle.load(f, encoding="bytes")

    sim = WildfireSimulation(parameters=params, seed=seed)
    model = sim.firefighters

    # Align OSM dict keys with the sim's WaterSourceManagers (OSM bodies are
    # appended after manual ones, in the order of usable keys).
    osm_keys = [k for k, s in water_dict.items()
                if len(s.get("shell_coords", [])) >= 4]
    osm_sources = model.water_sources[-len(osm_keys):] if osm_keys else []
    assert len(osm_sources) == len(osm_keys), (
        f"alignment mismatch: {len(osm_sources)} vs {len(osm_keys)}"
    )

    # Current-fleet footprint: smallest-run aircraft in the fleet (most
    # permissive), matching how the agents' union of feasible nodes is built.
    dims = list(model.agent_scooping_dimensions.values())
    fleet_dist, fleet_width = (min(dims, key=lambda d: d[0] * d[1])
                               if dims else (200.0, 18.0))

    keys_set1 = _feasible_osm_keys(model, osm_keys, osm_sources,
                                   fleet_dist, fleet_width)
    keys_set2 = _feasible_osm_keys(model, osm_keys, osm_sources,
                                   SMALL_SCOOP[0], SMALL_SCOOP[1])

    # Sanity: set1 should be a subset of set2 (bigger footprint => fewer/equal).
    s1, s2 = set(keys_set1), set(keys_set2)
    subset_ok = s1.issubset(s2)

    dict_set1 = {k: water_dict[k] for k in keys_set1}
    dict_set2 = {k: water_dict[k] for k in keys_set2}

    out1 = TERRAIN / f"{namespace}_water_sources_set1.pkl"
    out2 = TERRAIN / f"{namespace}_water_sources_set2.pkl"
    with out1.open("wb") as f:
        pickle.dump(dict_set1, f)
    with out2.open("wb") as f:
        pickle.dump(dict_set2, f)

    print(f"\n=== {scenario} (namespace {namespace}) ===")
    print(f"  original OSM bodies            : {len(water_dict)}")
    print(f"  set1 (current fleet {fleet_dist:g}x{fleet_width:g}): "
          f"{len(dict_set1)} bodies -> {out1.name}")
    print(f"  set2 (aircraft_1 {SMALL_SCOOP[0]:g}x{SMALL_SCOOP[1]:g})    : "
          f"{len(dict_set2)} bodies -> {out2.name}")
    print(f"  set1 subset of set2           : {subset_ok}  "
          f"(set1<=set2 size: {len(dict_set1) <= len(dict_set2)})")
    if not subset_ok:
        print("  WARNING: set1 not a strict subset of set2 -- inspect before use.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", default=["Pyrenees.json"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for scen in args.scenario:
        process(scen, seed=args.seed)


if __name__ == "__main__":
    main()
