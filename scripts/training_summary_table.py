"""Print a summary table (N, reward mean/std, prop_factor=1 %) for the three
switch-ignition training runs.

Usage:
    python3 scripts/training_summary_table.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "examples" / "wildfire" / "data" / "scenarios" / "outputs"

RUNS = {
    "Palisades": OUTPUTS / "switch_ignition_1m",
    "Pyrenees":  OUTPUTS / "pyrenees_switch_ignition_1m",
    "Salamis":   OUTPUTS / "salamis_switch_ignition_1m",
}

REWARD_COL = "moe_cumulative_reward"


def latest_summary(d: Path) -> Path | None:
    files = list(d.glob("results_*_summary.csv"))
    if files:
        return max(files, key=lambda p: int(p.stem.rsplit("_", 2)[-2]))
    fallback = d / "training_episode_summaries.csv"
    return fallback if fallback.exists() else None


def main() -> None:
    rows = []
    sources = []
    for label, d in RUNS.items():
        f = latest_summary(d)
        if f is None:
            print(f"[warn] no summary CSV found in {d}")
            continue
        df = pd.read_csv(f)
        rows.append({
            "scenario": label,
            "N": len(df),
            "reward mean": df[REWARD_COL].mean(),
            "reward std": df[REWARD_COL].std(),
            "propagation_factor=1": (df["propagation_factor"] == 1.0).mean(),
        })
        sources.append((label, f.name))

    table = pd.DataFrame(rows).set_index("scenario")
    formatted = table.style.format({
        "N": "{:d}",
        "reward mean": "{:+.4f}",
        "reward std": "{:.4f}",
        "propagation_factor=1": "{:.1%}",
    })

    try:
        print(formatted.to_string())
    except AttributeError:
        # Older pandas: Styler has no to_string; format manually.
        disp = table.copy()
        disp["N"] = disp["N"].map("{:d}".format)
        disp["reward mean"] = disp["reward mean"].map("{:+.4f}".format)
        disp["reward std"] = disp["reward std"].map("{:.4f}".format)
        disp["propagation_factor=1"] = disp["propagation_factor=1"].map("{:.1%}".format)
        print(disp.to_string())

    print()
    for label, name in sources:
        print(f"  {label}: {name}")


if __name__ == "__main__":
    main()
