"""Visualize feasible water-scoop points on the Pyrenees map.

Builds the REAL WildfireSimulation once and reuses the sim's own
`evaluate_water_source_feasibility` (no reimplemented geometry), so the plotted
feasible points exactly match what the agents would use.

Two panels, side by side:
  Left  : CURRENT fleet scoop size (Pyrenees seaplane: scooping_distance=200,
          span=18) -> the 19 feasible bodies / 197 nodes.
  Right : SMALL scoop size (scooping_distance=6, span=6, as in
          example_aircraft_1.json) -> typically many more bodies become
          feasible because a 6x6 skim run fits in far smaller ponds.

Feasible scoop NODES (cyan points) are overlaid on the terrain colormap, with
the water polygons faintly outlined for context.

Output: maps/pyrenees_water_feasibility.png

Run (user runs it; this is plotting + one sim build, no training):
    OMP_NUM_THREADS=1 ~/.conda/envs/rl-env/bin/python \
        -m examples.wildfire.visualize_water_feasibility
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from examples.wildfire.simulation import (  # noqa: E402
    WildfireParameters,
    WildfireSimulation,
)

INPUTS = Path("examples/wildfire/data/scenarios/inputs")
TERRAIN = Path("examples/wildfire/data/terrain")
MAPS = Path("maps")

SEED = 0

# Small-aircraft scoop case to compare against the scenario's own fleet.
# (example_aircraft_1.json uses scooping_distance=6, span=6.)
SMALL_SCOOP = (6.0, 6.0)


def feasible_nodes_for(model, scooping_distance, scooping_width):
    """Return (feasible_node_positions[N,2], n_feasible_bodies) using sim code."""
    nodes = []
    n_bodies = 0
    for ws in model.water_sources:
        is_feasible, ns = model.evaluate_water_source_feasibility(
            ws, scooping_width, scooping_distance
        )
        if is_feasible:
            n_bodies += 1
            nodes.extend([(float(x), float(y)) for (x, y) in ns])
    arr = np.array(nodes, dtype=float) if nodes else np.empty((0, 2))
    return arr, n_bodies


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Pyrenees.json")
    args = ap.parse_args()
    scenario = args.scenario

    params = WildfireParameters(**json.loads((INPUTS / scenario).read_text()))
    namespace = params.terrain_inputs.file_namespace
    meta = json.loads((TERRAIN / f"{namespace.split('_')[0]}.meta").read_text())
    cmap = np.load(TERRAIN / f"{namespace}_terrain_colormap.npy")
    background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)

    sim = WildfireSimulation(parameters=params, seed=SEED)
    model = sim.firefighters
    terr = sim.environment.terrain

    # "Current fleet" scoop = the scenario's own aircraft (smallest scoop run,
    # so the most permissive among the fleet). agent_scooping_dimensions maps
    # ac_type_id -> (scooping_distance, scooping_width).
    dims = list(model.agent_scooping_dimensions.values())
    if dims:
        # use the aircraft that needs the SHORTEST run (most feasible) as the
        # representative "current fleet" footprint, matching how the union of
        # feasible nodes is built across types.
        fleet_dist, fleet_width = min(dims, key=lambda d: d[0] * d[1])
    else:
        fleet_dist, fleet_width = 200.0, 18.0
    scoop_cases = [
        (f"current fleet ({fleet_dist:g} x {fleet_width:g})",
         float(fleet_dist), float(fleet_width)),
        (f"example_aircraft_1 ({SMALL_SCOOP[0]:g} x {SMALL_SCOOP[1]:g})",
         SMALL_SCOOP[0], SMALL_SCOOP[1]),
    ]

    # Water sources live in the WATER-MAP frame, which is larger than and offset
    # from the fire map. The sim converts any position to a fire-map index with
    #   index = (pos - origin) / cell_size      (pos_to_index, origin=terrain.origin)
    # so we plot everything in FIRE-MAP POSITION (metre) space relative to the
    # origin, and place the terrain image at its true metre extent. This avoids
    # the clipping that plotting in raw pixel space caused (water bodies sit at
    # ~87000 m, far outside the 0..13660 m fire map).
    ox, oy = terr.origin
    merc_w, merc_h = meta["mercator_dimensions"]
    # fire-map raster spans [origin, origin + mercator_dimensions] in metres.
    extent = [ox, ox + merc_w, oy + merc_h, oy]  # left,right,bottom,top (y down)

    # Water polygons in metre space (water-map frame, same units as origin).
    water_outlines = []
    for ws in model.water_sources:
        try:
            xy = np.asarray(ws.polygon.exterior.coords, dtype=float)
            water_outlines.append((xy[:, 0], xy[:, 1]))
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    for ax, (label, sdist, swidth) in zip(axes, scoop_cases):
        nodes, n_bodies = feasible_nodes_for(model, sdist, swidth)
        ax.imshow(background, extent=extent)
        # water resources (polygons) in RED
        for px, py in water_outlines:
            ax.fill(px, py, facecolor="red", edgecolor="red", lw=1.0,
                    alpha=0.85, zorder=3)
        cents = np.array([[px.mean(), py.mean()] for px, py in water_outlines]) \
            if water_outlines else np.empty((0, 2))
        if len(cents):
            ax.scatter(cents[:, 0], cents[:, 1], s=18, marker="s",
                       facecolor="red", edgecolors="darkred", linewidths=0.4,
                       zorder=4)
        # feasible scoop NODES on top (same metre frame)
        if len(nodes):
            ax.scatter(nodes[:, 0], nodes[:, 1],
                       s=10, c="#00e5ff", edgecolors="black", linewidths=0.3,
                       zorder=5, label=f"{len(nodes)} feasible nodes")
        # Focus the view on the fire-map terrain raster (where the sim runs),
        # with a small margin. Water bodies outside this box exist in the larger
        # water-map frame but are off the operational terrain.
        mx = 0.06 * merc_w
        ax.set_xlim(ox - mx, ox + merc_w + mx)
        ax.set_ylim(oy + merc_h + mx, oy - mx)  # y inverted
        ax.set_xlabel("fire-map position x (m)")
        ax.set_ylabel("fire-map position y (m)")
        ax.set_title(f"Scoop {label}\n{n_bodies} feasible water bodies, "
                     f"{len(nodes)} scoop nodes", fontsize=12)
        ax.legend(handles=[
            Patch(facecolor="red", edgecolor="red", alpha=0.45,
                  label="water resources (bodies)"),
            Patch(facecolor="#00e5ff", edgecolor="black",
                  label="feasible scoop nodes"),
        ], loc="upper right", framealpha=0.9)

    map_label = namespace.split("_")[0]
    fig.suptitle(
        f"{map_label} — Feasible Water-Scoop Points by Aircraft Scoop Size",
        fontsize=16, fontweight="bold")

    MAPS.mkdir(exist_ok=True)
    out = MAPS / f"{map_label.lower()}_water_feasibility.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")
    for label, sdist, swidth in scoop_cases:
        nodes, n_bodies = feasible_nodes_for(model, sdist, swidth)
        print(f"  {label:30s}: {n_bodies:3d} bodies, {len(nodes):4d} nodes")


if __name__ == "__main__":
    main()
