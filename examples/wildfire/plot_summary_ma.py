#!/usr/bin/env python
"""Plot a moving average of the latest results-summary file in an outputs folder.

Usage:
    python plot_summary_ma.py switch_ignition_1m
    python plot_summary_ma.py switch_ignition_1m --column delta_moe_total --window 200 --show

The folder argument is resolved against the standard outputs directory
(examples/wildfire/data/scenarios/outputs) unless it is already a valid path.
The "latest" summary is the one with the highest trailing episode count
(``results_<...>_<N>_summary.csv``).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUTS_DIR = (
    Path(__file__).resolve().parent / "data" / "scenarios" / "outputs"
)
SUMMARY_RE = re.compile(r"_(\d+)_summary\.csv$")


def find_latest_summary(folder: Path) -> Path:
    """Return the ``*_<N>_summary.csv`` file with the largest N in *folder*."""
    candidates: list[tuple[int, Path]] = []
    for path in folder.glob("*_summary.csv"):
        match = SUMMARY_RE.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No '*_<N>_summary.csv' files found in {folder}"
        )
    # Highest episode count wins; break ties on modification time.
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime))
    return candidates[-1][1]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple trailing moving average; returns len(values) - window + 1 points."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help="Folder name under the outputs dir (or a direct path).",
    )
    parser.add_argument(
        "--column",
        default="moe_cumulative_reward",
        help="Numeric column to plot (default: moe_cumulative_reward).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window size (default: 100).",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Base outputs directory used to resolve the folder name.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output image path (default: alongside the CSV).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        folder = Path(args.outputs_dir) / args.folder
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    csv_path = find_latest_summary(folder)
    frame = pd.read_csv(csv_path)
    if args.column not in frame.columns:
        raise SystemExit(
            f"Column {args.column!r} not in {csv_path.name}.\n"
            f"Available columns: {', '.join(frame.columns)}"
        )

    series = pd.to_numeric(frame[args.column], errors="coerce").to_numpy()
    series = series[~np.isnan(series)]
    if series.size == 0:
        raise SystemExit(f"Column {args.column!r} has no numeric values.")
    episodes = np.arange(series.size)

    window = max(1, min(args.window, series.size))
    moving = moving_average(series, window)
    moving_x = episodes[window - 1 :]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(episodes, series, color="0.8", lw=0.7, label=f"{args.column} (raw)")
    ax.plot(moving_x, moving, color="C0", lw=2.0, label=f"{window}-pt moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel(args.column)
    ax.set_title(
        f"{folder.name}  —  {csv_path.name}\n({series.size} episodes)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if args.save:
        save_path = Path(args.save)
    else:
        save_path = csv_path.with_name(
            f"{csv_path.stem}_{args.column}_ma{window}.png"
        )
    fig.savefig(save_path, dpi=130)

    print(f"Latest summary : {csv_path}")
    print(f"Episodes       : {series.size}")
    print(f"Column         : {args.column}")
    print(f"Window         : {window}")
    print(f"Final MA value : {moving[-1]:.6g}")
    print(f"Saved plot     : {save_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
