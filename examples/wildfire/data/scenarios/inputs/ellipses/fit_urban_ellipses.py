"""Fit a per-map number of ellipses to cover the real (OSM) urban areas.

The existing scenario `urban_locations` barely cover the actual urban cells
(wrong coordinate frame + wrong angle units). This tool fits N ellipses per map
(N set per map in MAPS) that DO cover them, encoded so the unmodified
WildfireModel.update_urban_areas() pipeline reproduces them on the fire map.

For each map (Palisades, Pyrenees, Salamis):
  1. Read the OSM urban mask (RESIDENTIAL == 9) from the terrain feature map.
  2. KMeans-cluster the urban cells into N groups (scipy.cluster.vq.kmeans2;
     sklearn is not installed in the env).
  3. Fit an orientation-aware covariance ellipse per cluster, then size each
     ellipse INDEPENDENTLY to maximize its F-beta (local recall over the
     cluster's own urban cells, precision over all urban -> punishes spill).
  4. Encode each ellipse as a scenario `urban_locations` entry in the SAME
     convention WildfireModel.update_urban_areas() consumes:
        - gps_coords : center, via index_to_pos -> pos_to_gps(fire_map_tlb)
        - radius     : [c_radius_cells*cell, r_radius_cells*cell]
                       (radius[0] -> col semi-axis, radius[1] -> row semi-axis)
        - angle      : rotation in RADIANS (passed straight to skimage ellipse;
                       in [0,2pi) it also satisfies the AngleInDegree [0,360)
                       pydantic bound)
  5. Verify by rasterizing the encoded entries through that exact code path and
     report urban-cell coverage; render a verification PNG.

Outputs (no scenario files are modified):
  - urban_ellipses_<map>.json : 9 urban_locations entries, ready to paste.
  - urban_ellipses_<map>.png  : ellipses overlaid on real urban cells.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.draw import ellipse

sys.path.insert(0, "src")
from sosid.model.transform import (  # noqa: E402
    gps_to_mercator,
    gps_to_pos,
    index_to_pos,
    pos_to_gps,
    pos_to_index,
)
from sosid.typedef import GridDescriptor  # noqa: E402

HERE = Path(__file__).resolve().parent
TERRAIN = Path("examples/wildfire/data/terrain")

# map name -> (terrain namespace, number of ellipses to fit)
MAPS = {
    "Palisades": ("Palisades_2004x1996_5m", 11),
    "Pyrenees": ("Pyrenees_2000x1996_5m", 13),
    "Salamis": ("Salamis_2004x1996_5m", 5),
}
RESIDENTIAL = 9
CELL_SIZE_M = 5.0          # nominal cell size the model uses for radius<->cells
# Per-ellipse sigma multipliers searched independently.
SCALE_GRID = np.round(np.arange(1.0, 4.01, 0.25), 3)
FBETA = 2.0               # >1 weights recall (coverage) above precision
RNG_SEED = 0

URBAN_FILL = (0.85, 0.10, 0.55, 0.85)
ELLIPSE_EDGE = (0.0, 0.85, 0.95, 1.0)
ELLIPSE_FILL = (0.0, 0.75, 0.85, 0.40)


def fire_map_geo(meta):
    """fire_map_top_left_bounds (top, left) and mercator dims (w, h)."""
    (lat_tl, lon_tl), (lat_br, lon_br) = meta["coordinates"]
    m_tl = gps_to_mercator([lat_tl, lon_tl])
    m_br = gps_to_mercator([lat_br, lon_br])
    left, top = float(m_tl[0]), float(m_tl[1])
    right, bottom = float(m_br[0]), float(m_br[1])
    return (top, left), (right - left, top - bottom)


def fit_cluster_ellipse(points):
    """Covariance ellipse for cluster `points` (rows, cols).

    Returns (r_radius_cells, c_radius_cells, rotation_rad) at 1-sigma. rotation
    is the major-axis angle measured from the column (x) axis, consistent with
    skimage.draw.ellipse(r, c, r_radius, c_radius, rotation): the c_radius axis
    is rotated by `rotation`, so we put the major length on c_radius.
    """
    if len(points) < 3:
        return 3.0, 3.0, 0.0
    rows = points[:, 0].astype(float)
    cols = points[:, 1].astype(float)
    cov = np.cov(np.vstack([rows, cols]))  # axis order (row, col)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    major = eigvecs[:, order][:, 0]        # [row_comp, col_comp]
    rotation = float(np.arctan2(major[0], major[1]))
    c_radius = max(1.0, float(np.sqrt(eigvals[0])))   # major -> col axis
    r_radius = max(1.0, float(np.sqrt(eigvals[1])))   # minor -> row axis
    return r_radius, c_radius, rotation


def encode_entry(center_rc, r_radius, c_radius, rotation, gd, tlb):
    """Ellipse (pixel space) -> urban_locations JSON entry."""
    x, y = index_to_pos((int(center_rc[0]), int(center_rc[1])), gd)
    lat, lon = pos_to_gps((float(x), float(y)), tlb)
    return {
        "gps_coords": [float(lat), float(lon)],
        "radius": [
            round(float(c_radius) * CELL_SIZE_M, 2),  # radius[0] -> col axis
            round(float(r_radius) * CELL_SIZE_M, 2),  # radius[1] -> row axis
        ],
        # RADIANS in [0,2pi); see module docstring on the AngleInDegree quirk.
        "angle": round(float(rotation) % (2 * np.pi), 6),
    }


def rasterize(entries, gd, tlb, rows, cols):
    """Rasterize encoded urban_locations entries through the model's pipeline."""
    mask = np.zeros((rows, cols), dtype=bool)
    for e in entries:
        pos = gps_to_pos(e["gps_coords"], tlb)
        r, c = pos_to_index(pos, gd)
        cell_x = round(e["radius"][1] / CELL_SIZE_M)  # r_radius
        cell_y = round(e["radius"][0] / CELL_SIZE_M)  # c_radius
        rr, cc = ellipse(r, c, cell_x, cell_y, (rows, cols), rotation=e["angle"])
        mask[rr, cc] = True
    return mask


def process_map(name, namespace, n_ellipses):
    meta = json.loads((TERRAIN / f"{name}.meta").read_text())
    features = np.load(TERRAIN / f"{namespace}_terrain_features.npy")
    cmap = np.load(TERRAIN / f"{namespace}_terrain_colormap.npy")
    background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)
    rows, cols = features.shape

    urban = features == RESIDENTIAL
    urban_rc = np.argwhere(urban)
    tlb, dims = fire_map_geo(meta)
    gd = GridDescriptor(shape=(rows, cols), dimensions=dims)

    # Cluster on a subsample for speed.
    rng = np.random.default_rng(RNG_SEED)
    sample = urban_rc
    if len(urban_rc) > 40000:
        sample = urban_rc[rng.choice(len(urban_rc), 40000, replace=False)]
    centers, _ = kmeans2(sample.astype(float), n_ellipses, seed=RNG_SEED,
                         minit="++", missing="warn")

    # Assign EVERY urban cell to its nearest centre (memory-safe loop) so each
    # ellipse is scored against its own territory rather than the global count.
    d2 = np.empty((len(urban_rc), n_ellipses), dtype=float)
    for k in range(n_ellipses):
        diff = urban_rc - centers[k]
        d2[:, k] = (diff * diff).sum(1)
    assign = d2.argmin(1)

    raw = [fit_cluster_ellipse(urban_rc[assign == k]) for k in range(n_ellipses)]
    b2 = FBETA * FBETA

    entries = []
    for k in range(n_ellipses):
        r_rad, c_rad, rot = raw[k]
        local_rc = urban_rc[assign == k]
        local_mask = np.zeros((rows, cols), dtype=bool)
        if len(local_rc):
            local_mask[local_rc[:, 0], local_rc[:, 1]] = True
        local_n = max(int(local_mask.sum()), 1)
        best_k, best_f = None, -1.0
        for scale in SCALE_GRID:
            e = encode_entry(centers[k], r_rad * scale, c_rad * scale, rot,
                             gd, tlb)
            single = rasterize([e], gd, tlb, rows, cols)
            area = int(single.sum())
            if area == 0:
                continue
            prec = (single & urban).sum() / area
            rec = (single & local_mask).sum() / local_n
            denom = b2 * prec + rec
            f = (1 + b2) * prec * rec / denom if denom > 0 else 0.0
            if f > best_f:
                best_f, best_k = f, e
        entries.append(best_k if best_k is not None
                       else encode_entry(centers[k], r_rad, c_rad, rot, gd, tlb))

    covered = rasterize(entries, gd, tlb, rows, cols)
    tp = int((covered & urban).sum())
    recall = tp / max(int(urban.sum()), 1)
    precision = tp / max(int(covered.sum()), 1)

    # Verification PNG.
    urban_cmap = ListedColormap([[0, 0, 0, 0], list(URBAN_FILL)])
    edge_cmap = ListedColormap([[0, 0, 0, 0], list(ELLIPSE_EDGE)])
    fill_cmap = ListedColormap([[0, 0, 0, 0], list(ELLIPSE_FILL)])
    edge = covered & ~ndimage.binary_erosion(covered, iterations=3)

    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(background, extent=[0, cols, rows, 0])
    ax.imshow(urban.astype(int), cmap=urban_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")
    ax.imshow(covered.astype(int), cmap=fill_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")
    ax.imshow(edge.astype(int), cmap=edge_cmap, vmin=0, vmax=1,
              extent=[0, cols, rows, 0], interpolation="nearest")
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xlabel("fire-map column (5 m / px)")
    ax.set_ylabel("fire-map row (5 m / px)")
    ax.set_title(
        f"{name} — {n_ellipses} fitted urban ellipses "
        f"(per-ellipse F{FBETA:.0f} sizing)\n"
        f"urban coverage {recall*100:.1f}%  |  "
        f"ellipse precision {precision*100:.1f}%",
        fontsize=13)
    ax.legend(handles=[
        Patch(facecolor=URBAN_FILL, label="real urban (RESIDENTIAL)"),
        Patch(facecolor=ELLIPSE_FILL, edgecolor=ELLIPSE_EDGE,
              label=f"{n_ellipses} fitted ellipses"),
    ], loc="upper right", framealpha=0.9)

    png = HERE / f"urban_ellipses_{name.lower()}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    out_json = HERE / f"urban_ellipses_{name.lower()}.json"
    out_json.write_text(json.dumps({"urban_locations": entries}, indent=4))
    print(f"{name}: coverage {recall*100:.1f}% precision {precision*100:.1f}% "
          f"-> {png.name}, {out_json.name}")


def main():
    for name, (namespace, n_ellipses) in MAPS.items():
        process_map(name, namespace, n_ellipses)


if __name__ == "__main__":
    main()
