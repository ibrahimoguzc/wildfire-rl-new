"""Visualize ignition points for two Palisades result files side by side.

Left panel  : results_palisades_dt1_probe_9200_summary.csv  (broad probe sweep)
Right panel : switchignitionPalisades.xlsx                  (clustered switch test)

Ignition positions are stored in the *operational* coordinate frame (relative to
the operational top-left bounds, in EPSG:3857 mercator metres). They are
translated into high-resolution fire-map pixels and overlaid on the Palisades
terrain colormap.

The probe-sweep CSV was generated WITHOUT the ignitable-mask filter the real
switch-ignition code applies, so a fraction of its points fall on forbidden
cells (water / urban / rock). We replicate `_build_ignitable_mask`
(ppo_runner.py) here to (a) drop those invalid points, (b) write a cleaned CSV,
and (c) show in the plot which points were removed.
"""

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sosid.environment.terrain import TerrainTypes  # noqa: E402
from sosid.model.transform import gps_to_mercator  # noqa: E402

HERE = Path(__file__).resolve().parent
TERRAIN = Path("examples/wildfire/data/terrain")

# Defaults; override the probe CSV with a positional CLI argument.
CSV_FILE = HERE / "results_palisades_dt1_probe_9200_summary.csv"
XLSX_FILE = HERE / "switchignitionPalisades.xlsx"

# --- Coordinate frames (from Palisades.meta) --------------------------------
OP_TL_GPS = [34.59704151614417, -119.17968749999996]    # operational top-left
FIRE_TL_GPS = [34.11541141736174, -118.59902777777]     # fire-map top-left
FIRE_DIM = (12087.689715657383, 12094.13309275778)      # mercator (w, h)
ROWS, COLS = 1996, 2004                                 # fire-map grid shape
CELL_SIZE_M = 5.0
URBAN_BUFFER_M = 500.0  # IGNITION_URBAN_BUFFER_M (real mechanism keep-out)

_m_op = gps_to_mercator(OP_TL_GPS)
LEFT_OP, TOP_OP = _m_op[0], _m_op[1]
_m_f = gps_to_mercator(FIRE_TL_GPS)
LEFT_F, TOP_F = _m_f[0], _m_f[1]


def op_to_pixel(x_op, y_op):
    """Operational-frame position -> fire-map (col, row) pixel coordinates."""
    x_merc = x_op + LEFT_OP
    y_merc = TOP_OP - y_op
    fx = x_merc - LEFT_F
    fy = TOP_F - y_merc
    col = fx / FIRE_DIM[0] * COLS
    row = fy / FIRE_DIM[1] * ROWS
    return col, row


def build_ignitable_mask(features, with_buffer):
    """Cells permitted as ignition seeds (mirrors _build_ignitable_mask).

    Excludes water, urban (residential) and non-combustible (rock/bare). When
    `with_buffer`, also enforces the 500 m keep-out from urban cells.
    """
    from scipy import ndimage

    water = features == int(TerrainTypes.WATER)
    urban = features == int(TerrainTypes.RESIDENTIAL)
    rock = features == int(TerrainTypes.NON_COMBUSTIBLE)
    ignitable = ~(water | urban | rock)
    if with_buffer and urban.any():
        buffer_cells = URBAN_BUFFER_M / CELL_SIZE_M
        dist = ndimage.distance_transform_edt(~urban)
        ignitable &= dist >= buffer_cells
    return ignitable


def load_points(path):
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    valid_pos = df["ignition_pos"].dropna()
    pos = valid_pos.apply(ast.literal_eval)
    xs = np.array([t[0] for t in pos])
    ys = np.array([t[1] for t in pos])
    return df.loc[valid_pos.index], xs, ys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=str(CSV_FILE),
                        help="Probe summary CSV with an `ignition_pos` column.")
    args = parser.parse_args()
    csv_file = Path(args.csv)
    if not csv_file.is_absolute() and not csv_file.exists():
        csv_file = HERE / csv_file.name
    clean_csv = csv_file.with_name(csv_file.stem + "_clean.csv")
    out = csv_file.with_name(csv_file.stem.replace("_summary", "") + "_points.png")

    cmap = np.load(TERRAIN / "Palisades_2004x1996_5m_terrain_colormap.npy")
    background = np.clip(cmap[..., :3], 0, 255).astype(np.uint8)
    features = np.load(TERRAIN / "Palisades_2004x1996_5m_terrain_features.npy")

    # Hard mask = the complaint (no water/urban/rock). Buffer mask = full
    # parity with the real switch-ignition keep-out. We filter on the hard mask
    # and additionally report how many also violate the buffer.
    hard_mask = build_ignitable_mask(features, with_buffer=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)

    # ----- Left: probe CSV, filtered -----
    df_csv, xs, ys = load_points(csv_file)
    col, row = op_to_pixel(xs, ys)
    ci = np.clip(col.astype(int), 0, COLS - 1)
    ri = np.clip(row.astype(int), 0, ROWS - 1)
    valid = hard_mask[ri, ci]
    n_total, n_valid, n_drop = len(valid), int(valid.sum()), int((~valid).sum())

    # Write cleaned CSV (valid rows only).
    df_csv.loc[valid].to_csv(clean_csv, index=False)

    ax = axes[0]
    ax.imshow(background, extent=[0, COLS, ROWS, 0])
    ax.scatter(col[valid], row[valid], s=6, c="#ff2d2d", alpha=0.35,
               edgecolors="none", label=f"{n_valid} valid ignition points")
    if n_drop:
        ax.scatter(col[~valid], row[~valid], s=26, c="black", marker="x",
                   linewidths=1.1, label=f"{n_drop} removed (water/urban/rock)")
    ax.set_title(f"Probe sweep — filtered\n{csv_file.name}", fontsize=12)
    ax.set_xlim(0, COLS)
    ax.set_ylim(ROWS, 0)
    ax.set_xlabel("fire-map column (5 m / px)")
    ax.set_ylabel("fire-map row (5 m / px)")
    ax.legend(loc="upper right", framealpha=0.85)

    # ----- Right: switch XLSX (already clean) -----
    _, xs2, ys2 = load_points(XLSX_FILE)
    col2, row2 = op_to_pixel(xs2, ys2)
    ci2 = np.clip(col2.astype(int), 0, COLS - 1)
    ri2 = np.clip(row2.astype(int), 0, ROWS - 1)
    valid2 = hard_mask[ri2, ci2]

    ax = axes[1]
    ax.imshow(background, extent=[0, COLS, ROWS, 0])
    ax.scatter(col2[valid2], row2[valid2], s=6, c="#1f8fff", alpha=0.35,
               edgecolors="none", label=f"{int(valid2.sum())} valid ignition points")
    if (~valid2).any():
        ax.scatter(col2[~valid2], row2[~valid2], s=26, c="black", marker="x",
                   linewidths=1.1, label=f"{int((~valid2).sum())} removed")
    ax.set_title(f"Switch ignitions\n{XLSX_FILE.name}", fontsize=12)
    ax.set_xlim(0, COLS)
    ax.set_ylim(ROWS, 0)
    ax.set_xlabel("fire-map column (5 m / px)")
    ax.set_ylabel("fire-map row (5 m / px)")
    ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle("Palisades — Ignition Points (forbidden cells removed)",
                 fontsize=16, fontweight="bold")

    fig.savefig(out, dpi=150)
    print(f"CSV: {n_total} total -> {n_valid} valid, {n_drop} removed "
          f"({100*n_drop/n_total:.1f}%)")
    print(f"Cleaned CSV -> {clean_csv}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
