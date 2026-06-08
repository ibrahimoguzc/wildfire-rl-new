"""Visualize scenario `urban_locations` for every map, one PNG per map.

For each map (Palisades, Pyrenees, Salamis) this rasterizes the scenario's
`urban_locations` ellipses exactly as WildfireModel.update_urban_areas() does
(skimage.draw.ellipse, placed via fire_map_top_left_bounds, semi-axes =
radius / cell_size, rotation = angle) and overlays them on the terrain colormap.

Fire-map georeferencing (top-left bounds + mercator extent) is derived from each
terrain `.meta` file's `coordinates` corners, so it generalizes across maps.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
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

# map name -> (scenario json from the switch set, terrain file namespace)
MAPS = {
    "Palisades": ("Palisades copy.json", "Palisades_2004x1996_5m"),
    "Pyrenees": ("Pyrenees.json", "Pyrenees_2000x1996_5m"),
    "Salamis": ("Salamis.json", "Salamis_2004x1996_5m"),
}

URBAN_FILL = (0.0, 0.75, 0.85, 0.85)


def fire_map_geo(meta):
    """Top-left bounds (top, left) and mercator dimensions (w, h) from corners."""
    (lat_tl, lon_tl), (lat_br, lon_br) = meta["coordinates"]
    m_tl = gps_to_mercator([lat_tl, lon_tl])
    m_br = gps_to_mercator([lat_br, lon_br])
    left, top = float(m_tl[0]), float(m_tl[1])
    right, bottom = float(m_br[0]), float(m_br[1])
    return (top, left), (right - left, top - bottom)


def rasterize_urban(scenario, meta, rows, cols):
    """Reproduce update_urban_areas(); return (mask, n_total, n_on_map)."""
    locations = scenario.get("urban_locations", [])
    cell_size = float(scenario["terrain_inputs"]["cell_size"])
    tlb, dims = fire_map_geo(meta)
    gd = GridDescriptor(shape=(rows, cols), dimensions=dims)

    mask = np.zeros((rows, cols), dtype=bool)
    n_on_map = 0
    for loc in locations:
        # PositionInput accepts either gps_coords or pos; angle defaults to 0.
        if loc.get("gps_coords") is not None:
            pos = gps_to_pos(loc["gps_coords"], tlb)
        else:
            pos = np.asarray(loc["pos"], dtype=float)
        row, col = pos_to_index(pos, gd)
        if 0 <= row < rows and 0 <= col < cols:
            n_on_map += 1
        cell_x = round(loc["radius"][1] / cell_size)  # r_radius (rows)
        cell_y = round(loc["radius"][0] / cell_size)  # c_radius (cols)
        rr, cc = ellipse(row, col, cell_x, cell_y, (rows, cols),
                         rotation=loc.get("angle", 0))
        mask[rr, cc] = True
    return mask, len(locations), n_on_map


def main():
    overlay_cmap = ListedColormap([[0, 0, 0, 0], list(URBAN_FILL)])
    for name, (scenario_file, namespace) in MAPS.items():
        scenario = json.loads((INPUTS / scenario_file).read_text())
        meta = json.loads((TERRAIN / f"{name}.meta").read_text())
        cmap = np.load(TERRAIN / f"{namespace}_terrain_colormap.npy")
        background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)
        rows, cols = background.shape[:2]

        mask, n_total, n_on_map = rasterize_urban(scenario, meta, rows, cols)
        pct = 100 * mask.sum() / mask.size

        fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
        ax.imshow(background, extent=[0, cols, rows, 0])
        ax.imshow(mask.astype(int), cmap=overlay_cmap, vmin=0, vmax=1,
                  extent=[0, cols, rows, 0], interpolation="nearest")
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_xlabel("fire-map column (5 m / px)")
        ax.set_ylabel("fire-map row (5 m / px)")
        ax.set_title(
            f"{name} — {scenario_file} `urban_locations`\n"
            f"{n_on_map}/{n_total} ellipses land on fire map ({pct:.1f}% of map)",
            fontsize=13)
        ax.legend(handles=[Patch(facecolor=URBAN_FILL,
                                 label="scenario urban_locations")],
                  loc="upper right", framealpha=0.9)

        out = HERE / f"scenario_urban_{name.lower()}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"{name}: {n_on_map}/{n_total} on map, {pct:.2f}% -> {out}")


if __name__ == "__main__":
    main()
