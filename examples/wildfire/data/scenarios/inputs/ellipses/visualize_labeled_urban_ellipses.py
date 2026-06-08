"""Visualize the scenario `urban_locations`, labelling each ellipse by its order.

Reads the urban_locations straight from the (now updated) scenario JSONs, then
rasterizes each ellipse through the SAME pipeline WildfireModel.update_urban_areas
uses and annotates the centre with its 1-based index in the list. One PNG per map.
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
INPUTS = Path("examples/wildfire/data/scenarios/inputs")

# map name -> (scenario json, terrain namespace)
MAPS = {
    "Palisades": ("Palisades copy.json", "Palisades_2004x1996_5m"),
    "Pyrenees": ("Pyrenees.json", "Pyrenees_2000x1996_5m"),
    "Salamis": ("Salamis.json", "Salamis_2004x1996_5m"),
}
RESIDENTIAL = 9
CELL_SIZE_M = 5.0

URBAN_FILL = (0.85, 0.10, 0.55, 0.85)
ELLIPSE_EDGE = (0.0, 0.85, 0.95, 1.0)
ELLIPSE_FILL = (0.0, 0.75, 0.85, 0.35)


def fire_map_geo(meta):
    (lat_tl, lon_tl), (lat_br, lon_br) = meta["coordinates"]
    m_tl = gps_to_mercator([lat_tl, lon_tl])
    m_br = gps_to_mercator([lat_br, lon_br])
    left, top = float(m_tl[0]), float(m_tl[1])
    right, bottom = float(m_br[0]), float(m_br[1])
    return (top, left), (right - left, top - bottom)


def ellipse_pixels(entry, gd, tlb, rows, cols):
    """Return (center_row, center_col, rr, cc) for one urban_locations entry."""
    if entry.get("gps_coords") is not None:
        pos = gps_to_pos(entry["gps_coords"], tlb)
    else:
        pos = np.asarray(entry["pos"], dtype=float)
    r, c = pos_to_index(pos, gd)
    cell_x = round(entry["radius"][1] / CELL_SIZE_M)  # r_radius
    cell_y = round(entry["radius"][0] / CELL_SIZE_M)  # c_radius
    rr, cc = ellipse(r, c, cell_x, cell_y, (rows, cols),
                     rotation=entry.get("angle", 0))
    return r, c, rr, cc


def process_map(name, scenario_file, namespace):
    meta = json.loads((TERRAIN / f"{name}.meta").read_text())
    scenario = json.loads((INPUTS / scenario_file).read_text())
    features = np.load(TERRAIN / f"{namespace}_terrain_features.npy")
    cmap = np.load(TERRAIN / f"{namespace}_terrain_colormap.npy")
    background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)
    rows, cols = features.shape

    urban = features == RESIDENTIAL
    tlb, dims = fire_map_geo(meta)
    gd = GridDescriptor(shape=(rows, cols), dimensions=dims)
    locations = scenario["urban_locations"]

    fill = np.zeros((rows, cols), dtype=bool)
    centers = []
    for entry in locations:
        cr, cc_, rr, cc = ellipse_pixels(entry, gd, tlb, rows, cols)
        fill[rr, cc] = True
        centers.append((cr, cc_))
    edge = fill & ~ndimage.binary_erosion(fill, iterations=3)

    urban_cmap = ListedColormap([[0, 0, 0, 0], list(URBAN_FILL)])
    fill_cmap = ListedColormap([[0, 0, 0, 0], list(ELLIPSE_FILL)])
    edge_cmap = ListedColormap([[0, 0, 0, 0], list(ELLIPSE_EDGE)])

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.imshow(background, extent=[0, cols, rows, 0])
    ax.imshow(urban.astype(int), cmap=urban_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")
    ax.imshow(fill.astype(int), cmap=fill_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")
    ax.imshow(edge.astype(int), cmap=edge_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")

    for i, (cr, cc_) in enumerate(centers, start=1):
        ax.annotate(
            str(i), xy=(cc_, cr), ha="center", va="center",
            fontsize=14, fontweight="bold", color="black",
            bbox=dict(boxstyle="circle,pad=0.25", fc="yellow",
                      ec="black", lw=1.4, alpha=0.95),
            zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xlabel("fire-map column (5 m / px)")
    ax.set_ylabel("fire-map row (5 m / px)")
    ax.set_title(
        f"{name} — {len(locations)} urban_locations ellipses (numbered by order)\n"
        f"source: {scenario_file}",
        fontsize=13)
    ax.legend(handles=[
        Patch(facecolor=URBAN_FILL, label="real urban (RESIDENTIAL)"),
        Patch(facecolor=ELLIPSE_FILL, edgecolor=ELLIPSE_EDGE,
              label="urban_locations ellipses"),
    ], loc="upper right", framealpha=0.9)

    out = HERE / f"urban_ellipses_labeled_{name.lower()}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{name}: {len(locations)} ellipses labelled -> {out.name}")


def main():
    for name, (scenario_file, namespace) in MAPS.items():
        process_map(name, scenario_file, namespace)


if __name__ == "__main__":
    main()
