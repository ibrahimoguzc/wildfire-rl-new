"""Visualize urban (RESIDENTIAL) areas on the Palisades fire map.

Panel 1: urban areas from the OSM-derived terrain feature map (RESIDENTIAL==9).
Panel 2: those urban areas + the 500 m keep-out buffer that --switch-ignition
         uses to exclude ignition seeds near urban cells.
Panel 3: urban areas as *determined in Palisades copy.json* -- the explicit
         `urban_locations` ellipses, rasterized exactly as
         WildfireModel.update_urban_areas() does (skimage.draw.ellipse, placed
         via fire_map_top_left_bounds, semi-axes = radius / cell_size,
         rotation = angle).

Urban cells are RESIDENTIAL == 9 in the indexed terrain feature map
(see TerrainTypes / Palisades.meta Terrain_Features_Keys).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage
from skimage.draw import ellipse

sys.path.insert(0, "src")
from sosid.model.transform import (  # noqa: E402
    gps_to_mercator,
    gps_to_pos,
    pos_to_index,
)
from sosid.typedef import GridDescriptor  # noqa: E402

HERE = Path(__file__).resolve().parent
TERRAIN = Path("examples/wildfire/data/terrain")
SCENARIO = Path("examples/wildfire/data/scenarios/inputs/Palisades copy.json")

RESIDENTIAL = 9
CELL_SIZE_M = 5.0
URBAN_BUFFER_M = 500.0  # IGNITION_URBAN_BUFFER_M

# Fire-map georeferencing (from Palisades.meta `coordinates`, grid_shape).
FIRE_TL_GPS = [34.11541141736174, -118.59902777777]
FIRE_DIM = (12087.689715657383, 12094.13309275778)  # mercator (w, h)
ROWS, COLS = 1996, 2004


def rasterize_scenario_urban():
    """Reproduce WildfireModel.update_urban_areas() for Palisades copy.json.

    Returns (urban_mask, n_total, n_on_map).
    """
    params = json.loads(SCENARIO.read_text())
    locations = params.get("urban_locations", [])

    m = gps_to_mercator(FIRE_TL_GPS)
    fire_tlb = (float(m[1]), float(m[0]))
    gd = GridDescriptor(shape=(ROWS, COLS), dimensions=FIRE_DIM)

    mask = np.zeros((ROWS, COLS), dtype=bool)
    n_on_map = 0
    for loc in locations:
        pos = gps_to_pos(loc["gps_coords"], fire_tlb)
        row, col = pos_to_index(pos, gd)
        if 0 <= row < ROWS and 0 <= col < COLS:
            n_on_map += 1
        # Same arg order as model.py: cell_x from radius[1], cell_y from radius[0].
        cell_x = round(loc["radius"][1] / CELL_SIZE_M)
        cell_y = round(loc["radius"][0] / CELL_SIZE_M)
        rr, cc = ellipse(row, col, cell_x, cell_y, (ROWS, COLS),
                         rotation=loc["angle"])
        mask[rr, cc] = True
    return mask, len(locations), n_on_map


def main():
    cmap = np.load(TERRAIN / "Palisades_2004x1996_5m_terrain_colormap.npy")
    background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)
    features = np.load(TERRAIN / "Palisades_2004x1996_5m_terrain_features.npy")

    urban = features == RESIDENTIAL
    buffer_cells = URBAN_BUFFER_M / CELL_SIZE_M
    dist = ndimage.distance_transform_edt(~urban)
    buffer_zone = (dist < buffer_cells) & ~urban

    scenario_urban, n_total, n_on_map = rasterize_scenario_urban()

    urban_cmap = ListedColormap([[0, 0, 0, 0], [0.85, 0.10, 0.55, 0.85]])
    buffer_cmap = ListedColormap([[0, 0, 0, 0], [1.0, 0.65, 0.0, 0.45]])
    scen_cmap = ListedColormap([[0, 0, 0, 0], [0.0, 0.75, 0.85, 0.85]])

    fig, axes = plt.subplots(1, 3, figsize=(26, 9), constrained_layout=True)
    for ax in axes:
        ax.imshow(background, extent=[0, COLS, ROWS, 0])
        ax.set_xlim(0, COLS)
        ax.set_ylim(ROWS, 0)
        ax.set_xlabel("fire-map column (5 m / px)")
        ax.set_ylabel("fire-map row (5 m / px)")

    urban_pct = 100 * urban.sum() / urban.size
    buf_pct = 100 * buffer_zone.sum() / buffer_zone.size
    scen_pct = 100 * scenario_urban.sum() / scenario_urban.size

    # Panel 1 - OSM urban.
    axes[0].imshow(urban.astype(int), cmap=urban_cmap, vmin=0, vmax=1,
                   extent=[0, COLS, ROWS, 0], interpolation="nearest")
    axes[0].set_title(f"OSM urban (RESIDENTIAL) — {urban_pct:.1f}% of map",
                      fontsize=12)
    axes[0].legend(handles=[Patch(facecolor=(0.85, 0.10, 0.55, 0.85),
                                  label="urban / residential")],
                   loc="upper right", framealpha=0.9)

    # Panel 2 - OSM urban + buffer.
    axes[1].imshow(buffer_zone.astype(int), cmap=buffer_cmap, vmin=0, vmax=1,
                   extent=[0, COLS, ROWS, 0], interpolation="nearest")
    axes[1].imshow(urban.astype(int), cmap=urban_cmap, vmin=0, vmax=1,
                   extent=[0, COLS, ROWS, 0], interpolation="nearest")
    axes[1].set_title(
        f"OSM urban + 500 m keep-out buffer ({buf_pct:.1f}% of map)",
        fontsize=12)
    axes[1].legend(handles=[
        Patch(facecolor=(0.85, 0.10, 0.55, 0.85), label="urban / residential"),
        Patch(facecolor=(1.0, 0.65, 0.0, 0.55), label="500 m keep-out buffer"),
    ], loc="upper right", framealpha=0.9)

    # Panel 3 - scenario urban_locations ellipses.
    axes[2].imshow(scenario_urban.astype(int), cmap=scen_cmap, vmin=0, vmax=1,
                   extent=[0, COLS, ROWS, 0], interpolation="nearest")
    axes[2].set_title(
        "Palisades copy.json `urban_locations`\n"
        f"{n_on_map}/{n_total} ellipses land on fire map "
        f"({scen_pct:.1f}% of map)",
        fontsize=12)
    axes[2].legend(handles=[Patch(facecolor=(0.0, 0.75, 0.85, 0.85),
                                  label="scenario urban_locations")],
                   loc="upper right", framealpha=0.9)

    fig.suptitle("Palisades — Urban Areas", fontsize=16, fontweight="bold")

    out = HERE / "urban_areas_palisades.png"
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    print(f"scenario urban_locations: {n_on_map}/{n_total} centers on fire map; "
          f"covers {scen_pct:.2f}% of fire map")


if __name__ == "__main__":
    main()
