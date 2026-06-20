#!/usr/bin/env python
"""Visualize the K-sector flank decomposition of the "directional" state.

Runs a real WildfireHourlyEnv episode (state_space="directional") and, at
every decision step (default 10 simulated minutes), re-derives the per-sector
flank decomposition exactly the way ``_compute_directional_front_block`` does
-- same helpers, same order -- but keeps the intermediates so they can be
drawn. Each snapshot PNG shows:

  * the fire-state grid (fuel / burning / burnt / nonflammable / suppressed)
  * active front cells colored by their sector slot (K = --state-fire-fronts)
  * the burning-cell centroid and the K compass sector boundary rays
  * per sector: the representative (max-ROS) cell, its projected landing
    point, and the threat-search radius around the landing
  * aircraft, water sources, and urban/VIP objectives in view
  * the K x 6 directional feature block, annotated, as a side panel

The mirrored block is cross-checked against the env's own
``_compute_directional_front_block`` output every step, so a drift between
this script and ppo_runnerv2.py is reported instead of silently plotted.

Usage:
  ~/.conda/envs/rl-env/bin/python scripts/visualize_directional_flanks.py \
      --scenario Pyrenees.json --state-fire-fronts 4 --steps 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch

from examples.wildfire.fire_model.states import (
    BURNT,
    COMBUSTIBLE,
    EARLY_BURNING,
    EXTINGUISHING,
    FULL_BURNING,
    NONFLAMMABLE,
    SUPPRESSED,
)
from examples.wildfire.paths import TERRAIN_DIR
from examples.wildfire.ppo_runnerv2 import (
    DIRECTIONAL_ALIGN_MIN,
    DIRECTIONAL_LINE_SCALE_M,
    DIRECTIONAL_PER_FRONT_FEATURES,
    DIRECTIONAL_R_MAX_CELLS,
    DIRECTIONAL_R_MIN_CELLS,
    DIRECTIONAL_SEARCH_ALPHA,
    DIRECTIONAL_SPREAD_EPS,
    DIRECTIONAL_VIP_HORIZONS,
    DIRECTIONAL_WATER_SCALE_M,
    MAX_SPREAD_RATE_NORM_MPM,
    STATE_SPACE_DIRECTIONAL,
    TACTIC_COMBINATIONS,
    WildfireHourlyEnv,
    _resolve_scenario,
)

DEFAULT_OUTPUT_DIR = REPO / "examples" / "wildfire" / "snapshots"

# Fire-state grid colors (index == state integer value).
FIRE_STATE_CMAP = ListedColormap(
    [
        "#7ec8e3",  # 0 SUPPRESSED
        "#9e9e9e",  # 1 NONFLAMMABLE
        "#ede6c8",  # 2 COMBUSTIBLE
        "#ffb347",  # 3 EARLY_BURNING
        "#ff4500",  # 4 FULL_BURNING
        "#b22222",  # 5 EXTINGUISHING
        "#3b2f2f",  # 6 BURNT
    ]
)
FIRE_STATE_LABELS = {
    SUPPRESSED: "suppressed",
    NONFLAMMABLE: "nonflammable",
    COMBUSTIBLE: "combustible",
    EARLY_BURNING: "early burning",
    FULL_BURNING: "full burning",
    EXTINGUISHING: "extinguishing",
    BURNT: "burnt",
}
SECTOR_CMAP = plt.get_cmap("tab10")
FEATURE_SHORT_NAMES = ("sev", "vip", "veg", "topo", "ind", "h2o")


def decompose_flanks(env: WildfireHourlyEnv) -> dict:
    """Mirror _compute_directional_front_block, keeping the intermediates.

    Calls the same env helpers in the same order as ppo_runnerv2.py's
    ``_compute_directional_front_block`` so the plotted geometry matches the
    features the agent would observe.
    """
    assert env.sim is not None
    wildfire = env.sim.wildfire
    n_feat = len(DIRECTIONAL_PER_FRONT_FEATURES)
    fronts = int(env.state_fire_fronts)
    out: dict = {
        "fronts": fronts,
        "centroid": None,
        "burning": None,
        "active": None,
        "sectors": {},
        "block": np.zeros((fronts, n_feat), dtype=float),
    }

    burning_indices = wildfire.burning_indices
    burning = env._coerce_grid_indices(np.asarray(burning_indices))
    if burning.size == 0:
        return out

    fire_states = np.asarray(wildfire.fire_states)
    height, width = fire_states.shape
    bi = burning[:, 0].astype(np.int64)
    bj = burning[:, 1].astype(np.int64)
    out["burning"] = (bi, bj)
    ros = np.asarray(wildfire.get_spread_rates(burning), dtype=float).reshape(-1)
    aspect = np.asarray(wildfire.prop_aspect)[bi, bj].astype(float)

    active = (
        np.isfinite(ros)
        & (ros > DIRECTIONAL_SPREAD_EPS)
        & np.isfinite(aspect)
        & env._has_combustible_neighbor(bi, bj, fire_states, height, width)
    )
    centroid_i = float(bi.mean())
    centroid_j = float(bj.mean())
    out["centroid"] = (centroid_i, centroid_j)
    if not np.any(active):
        return out

    ai = bi[active]
    aj = bj[active]
    a_ros = ros[active]
    a_aspect = aspect[active]

    bearing = (
        np.degrees(np.arctan2(aj - centroid_j, -(ai - centroid_i))) + 360.0
    ) % 360.0
    sector_width = 360.0 / fronts
    sectors = np.clip((bearing / sector_width).astype(np.int64), 0, fronts - 1)
    out["active"] = (ai, aj, a_ros, sectors)

    cell_size = float(env._cell_size_m)
    map_diagonal = float(env._map_diagonal)
    horizon_min = float(env.decision_interval_minutes)
    combustibilities = env.sim.environment.terrain.features.combustibilities

    block = out["block"]
    for k in range(fronts):
        in_sector = sectors == k
        if not np.any(in_sector):
            continue
        k_ros = a_ros[in_sector]
        k_i = ai[in_sector]
        k_j = aj[in_sector]
        k_aspect_all = a_aspect[in_sector]
        k_bearing = bearing[in_sector]

        severity = float(
            min(max(float(k_ros.mean()) / MAX_SPREAD_RATE_NORM_MPM, 0.0), 1.0)
        )

        rep = int(np.argmax(k_ros))
        src_i = int(k_i[rep])
        src_j = int(k_j[rep])
        src_ros = float(k_ros[rep])
        # 2c direction repair (mirrors _compute_directional_front_block).
        src_aspect_raw = float(k_aspect_all[rep])
        src_outward = float(k_bearing[rep])
        align = math.cos(math.radians(src_aspect_raw - src_outward))
        src_dir = src_aspect_raw if align > DIRECTIONAL_ALIGN_MIN else src_outward

        land_i, land_j = env._project_flank_landing(
            src_i, src_j, src_dir, src_ros, horizon_min,
            cell_size, fire_states, height, width,
        )

        reach_cells = (src_ros * horizon_min) / cell_size
        radius_cells = float(
            min(
                max(DIRECTIONAL_SEARCH_ALPHA * reach_cells, DIRECTIONAL_R_MIN_CELLS),
                DIRECTIONAL_R_MAX_CELLS,
            )
        )
        radius_m = radius_cells * cell_size

        landing_pos = env._indices_to_positions(
            np.array([[land_i, land_j]], dtype=np.int64)
        )
        vip_scale_m = src_ros * DIRECTIONAL_VIP_HORIZONS * horizon_min
        vip_urgency = env._tree_proximity(env._vip_tree, landing_pos, vip_scale_m)
        water_access = env._tree_proximity(
            env._water_tree, landing_pos, DIRECTIONAL_WATER_SCALE_M
        )

        line_dist = env._front_distance_to_fire_line_m(land_i, land_j)
        indirect_urgency = (
            float(min(max(1.0 - line_dist / DIRECTIONAL_LINE_SCALE_M, 0.0), 1.0))
            if math.isfinite(line_dist)
            else 0.0
        )

        vegetation_urgency = env._vegetation_escalation(
            src_i, src_j, land_i, land_j, radius_cells,
            fire_states, combustibilities, height, width,
        )
        topography_urgency = env._topography_local(land_i, land_j, src_dir)

        block[k, :] = (
            severity,
            vip_urgency,
            vegetation_urgency,
            topography_urgency,
            indirect_urgency,
            water_access,
        )
        out["sectors"][k] = {
            "rep": (src_i, src_j),
            "rep_aspect": src_dir,
            "rep_ros": src_ros,
            "landing": (land_i, land_j),
            "radius_cells": radius_cells,
            "cell_count": int(np.count_nonzero(in_sector)),
        }

    # Guard against this mirror drifting from the env implementation.
    reference = np.asarray(
        env._compute_directional_front_block(np.asarray(burning_indices)),
        dtype=float,
    ).reshape(fronts, n_feat)
    if not np.allclose(block, reference, atol=1e-8):
        print(
            "WARNING: mirrored flank block diverges from "
            "_compute_directional_front_block; plot may not match the agent's "
            "observation. Max abs diff: "
            f"{np.max(np.abs(block - reference)):.3e}"
        )
    return out


def world_to_grid(env: WildfireHourlyEnv, positions: np.ndarray) -> np.ndarray:
    """World (x, y) -> fractional grid (i, j), inverse of _indices_to_positions."""
    arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    terrain = env.sim.environment.terrain
    gd = terrain.grid_description
    origin = terrain.origin
    cell_x = gd.dimensions[0] / gd.shape[1]
    cell_y = gd.dimensions[1] / gd.shape[0]
    jj = (arr[:, 0] - origin[0]) / cell_x
    ii = (arr[:, 1] - origin[1]) / cell_y
    return np.column_stack([ii, jj])


def _view_window(
    flanks: dict,
    height: int,
    width: int,
    full_map: bool,
) -> tuple[float, float, float, float]:
    """(i0, i1, j0, j1) view bounds around the fire, or the full grid."""
    if full_map or flanks["burning"] is None:
        return 0.0, float(height), 0.0, float(width)
    bi, bj = flanks["burning"]
    pts_i = [bi.min(), bi.max()]
    pts_j = [bj.min(), bj.max()]
    max_radius = 0.0
    for info in flanks["sectors"].values():
        li, lj = info["landing"]
        pts_i.extend([li - info["radius_cells"], li + info["radius_cells"]])
        pts_j.extend([lj - info["radius_cells"], lj + info["radius_cells"]])
        max_radius = max(max_radius, info["radius_cells"])
    margin = max(30.0, 1.5 * max_radius)
    i0 = max(0.0, float(min(pts_i)) - margin)
    i1 = min(float(height), float(max(pts_i)) + margin)
    j0 = max(0.0, float(min(pts_j)) - margin)
    j1 = min(float(width), float(max(pts_j)) + margin)
    return i0, i1, j0, j1


def save_snapshot(
    env: WildfireHourlyEnv,
    flanks: dict,
    step: int,
    minutes: float,
    out_path: Path,
    full_map: bool,
    dpi: int,
) -> None:
    fire_states = np.asarray(env.sim.wildfire.fire_states)
    height, width = fire_states.shape
    fronts = flanks["fronts"]
    sector_width = 360.0 / fronts

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[0, 1])

    ax.imshow(
        fire_states,
        cmap=FIRE_STATE_CMAP,
        vmin=-0.5,
        vmax=6.5,
        origin="upper",
        interpolation="nearest",
    )

    i0, i1, j0, j1 = _view_window(flanks, height, width, full_map)

    # Static objectives + aircraft (clipped to view by the axis limits).
    handles: list = []
    if env.water_positions.size:
        wg = world_to_grid(env, env.water_positions)
        ax.scatter(wg[:, 1], wg[:, 0], marker="s", s=14, c="#0050ff",
                   edgecolors="white", linewidths=0.3, zorder=4)
        handles.append(Line2D([], [], marker="s", ls="", mfc="#0050ff",
                              mec="white", label="water source"))
    if env.urban_positions.size:
        ug = world_to_grid(env, env.urban_positions)
        ax.scatter(ug[:, 1], ug[:, 0], marker="D", s=18, c="magenta",
                   edgecolors="black", linewidths=0.3, zorder=4)
        handles.append(Line2D([], [], marker="D", ls="", mfc="magenta",
                              mec="black", label="urban/VIP"))
    agents = env.sim.firefighters.firefighters
    if agents:
        ag = world_to_grid(env, np.array([a.pos for a in agents], dtype=float))
        controlled = ag[: env.controlled_agent_count]
        ax.scatter(controlled[:, 1], controlled[:, 0], marker="^", s=60,
                   c="black", edgecolors="white", linewidths=0.8, zorder=6)
        handles.append(Line2D([], [], marker="^", ls="", mfc="black",
                              mec="white", label="aircraft (controlled)"))

    if flanks["centroid"] is not None:
        ci, cj = flanks["centroid"]
        ax.plot(cj, ci, marker="+", ms=14, mew=2.5, c="black", zorder=7)
        handles.append(Line2D([], [], marker="+", ls="", mec="black",
                              label="burning centroid"))

        # Sector boundary rays + labels (compass bearings, 0=N up, 90=E).
        ray_len = max(
            25.0,
            0.35 * max(i1 - i0, j1 - j0),
        )
        if fronts > 1:
            for k in range(fronts):
                theta = math.radians(k * sector_width)
                di, dj = -math.cos(theta), math.sin(theta)
                ax.plot(
                    [cj, cj + ray_len * dj],
                    [ci, ci + ray_len * di],
                    c="black", lw=0.9, ls="--", alpha=0.65, zorder=5,
                )
        for k in range(fronts):
            mid = math.radians((k + 0.5) * sector_width)
            di, dj = -math.cos(mid), math.sin(mid)
            ax.text(
                cj + 0.85 * ray_len * dj,
                ci + 0.85 * ray_len * di,
                f"S{k}",
                color=SECTOR_CMAP(k % 10), fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(fc="white", alpha=0.65, ec="none", pad=1.5),
            )

    if flanks["active"] is not None:
        ai, aj, _a_ros, sectors = flanks["active"]
        for k in range(fronts):
            mask = sectors == k
            if not np.any(mask):
                continue
            ax.scatter(aj[mask], ai[mask], s=6, color=SECTOR_CMAP(k % 10),
                       marker="o", linewidths=0, zorder=5)

    for k, info in flanks["sectors"].items():
        color = SECTOR_CMAP(k % 10)
        ri, rj = info["rep"]
        li, lj = info["landing"]
        ax.plot([rj, lj], [ri, li], c=color, lw=1.4, ls="-", zorder=6)
        ax.plot(rj, ri, marker="*", ms=14, mfc=color, mec="black", mew=0.7,
                zorder=7)
        ax.plot(lj, li, marker="X", ms=10, mfc=color, mec="black", mew=0.7,
                zorder=7)
        ax.add_patch(Circle((lj, li), info["radius_cells"], fill=False,
                            ec=color, ls=":", lw=1.2, zorder=6))

    handles.extend(
        [
            Line2D([], [], marker="*", ls="", mfc="gray", mec="black",
                   label="rep cell (max ROS)"),
            Line2D([], [], marker="X", ls="", mfc="gray", mec="black",
                   label="projected landing"),
        ]
    )
    present_states = np.unique(fire_states[int(i0):int(i1), int(j0):int(j1)])
    handles.extend(
        Patch(fc=FIRE_STATE_CMAP(int(s)), label=FIRE_STATE_LABELS[int(s)])
        for s in present_states
        if int(s) in FIRE_STATE_LABELS
    )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    ax.set_xlim(j0, j1)
    ax.set_ylim(i1, i0)  # origin="upper": larger i is lower on screen
    ax.set_xlabel("grid j")
    ax.set_ylabel("grid i")

    atmosphere = env.sim.atmosphere
    wind_speed = float(getattr(atmosphere, "wind_speed", float("nan")))
    wind_aspect = float((atmosphere.wind_aspect + 360.0) % 360.0)
    n_burning = 0 if flanks["burning"] is None else flanks["burning"][0].size
    n_active = 0 if flanks["active"] is None else flanks["active"][0].size
    ax.set_title(
        f"{env.current_scenario_label}  K={fronts}  step {step}  "
        f"t={minutes:.0f} min\n"
        f"burning cells: {n_burning}  active front cells: {n_active}  "
        f"wind: {wind_speed:.1f} m/s @ {wind_aspect:.0f}\N{DEGREE SIGN} "
        "(0\N{DEGREE SIGN}=N)"
    )

    # Right panel: the K x 6 feature block exactly as the agent observes it.
    block = flanks["block"]
    im = ax_tab.imshow(block, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax_tab.set_xticks(range(len(FEATURE_SHORT_NAMES)))
    ax_tab.set_xticklabels(FEATURE_SHORT_NAMES, fontsize=9)
    ax_tab.set_yticks(range(fronts))
    ax_tab.set_yticklabels([f"S{k}" for k in range(fronts)], fontsize=10)
    for tick, k in zip(ax_tab.get_yticklabels(), range(fronts)):
        tick.set_color(SECTOR_CMAP(k % 10))
        tick.set_fontweight("bold")
    for k in range(fronts):
        for f in range(block.shape[1]):
            value = block[k, f]
            ax_tab.text(
                f, k, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value < 0.6 else "black",
            )
    ax_tab.set_title("directional feature block", fontsize=10)
    fig.colorbar(im, ax=ax_tab, fraction=0.04, pad=0.04)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_water_set(scenario_path: Path, water_set: str) -> int | None:
    if water_set == "off":
        return None
    selected = int(water_set)
    namespace = json.loads(scenario_path.read_text())["terrain_inputs"][
        "file_namespace"
    ]
    water_file = TERRAIN_DIR / f"{namespace}_water_sources_set{selected}.pkl"
    if not water_file.exists():
        print(
            f"WARNING: {water_file} missing; falling back to the scenario's "
            "default water sources (--water-set off)."
        )
        return None
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot the K-sector directional flank decomposition "
        "every decision step of a live simulation."
    )
    parser.add_argument("--scenario", default="Pyrenees.json")
    parser.add_argument("--state-fire-fronts", "-k", type=int, default=4,
                        help="K, the number of compass sectors (default 4)")
    parser.add_argument("--steps", type=int, default=24,
                        help="decision steps to simulate (default 24 = 4 h)")
    parser.add_argument("--decision-interval-minutes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for env reset and random actions")
    parser.add_argument("--sim-seed", type=int, default=None,
                        help="fixed simulation seed (default: derived from "
                        "--seed)")
    parser.add_argument("--fixed-action", type=int, default=None,
                        help="hold every decision at this TACTIC_COMBINATIONS "
                        f"index (0..{len(TACTIC_COMBINATIONS) - 1}); default "
                        "samples random actions")
    parser.add_argument("--controlled-agent-count", type=int, default=6)
    parser.add_argument("--tactic-distribution", default="group",
                        choices=["individual", "group"])
    parser.add_argument("--aircraft-group-size", type=int, default=2)
    parser.add_argument("--water-set", default="1", choices=["1", "2", "off"])
    parser.add_argument("--cell-size", default="code", choices=["json", "code"],
                        help="metres/cell for directional geometry: 'code' "
                        "(default) = actual projected spacing; 'json' = nominal "
                        "cell_size from the scenario file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-map", action="store_true",
                        help="show the whole grid instead of zooming on the "
                        "fire")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    scenario_path = _resolve_scenario(args.scenario)
    water_set = _resolve_water_set(scenario_path, args.water_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = WildfireHourlyEnv(
        scenario_path=scenario_path,
        decision_interval_minutes=args.decision_interval_minutes,
        state_space=STATE_SPACE_DIRECTIONAL,
        state_fire_fronts=args.state_fire_fronts,
        tactic_distribution=args.tactic_distribution,
        aircraft_group_size=args.aircraft_group_size,
        controlled_agent_count=args.controlled_agent_count,
        water_set=water_set,
        cell_size_source=args.cell_size,
    )
    env.action_space.seed(args.seed)
    options = {"sim_seed": args.sim_seed} if args.sim_seed is not None else None
    _obs, info = env.reset(seed=args.seed, options=options)
    print(
        f"scenario={env.current_scenario_name} sim_seed={info['sim_seed']} "
        f"K={args.state_fire_fronts} interval={args.decision_interval_minutes}min "
        f"detection_delay={env.fire_detection_delay_minutes:.0f}min"
    )

    stem = scenario_path.stem.split()[0].lower()
    for step in range(args.steps + 1):
        minutes = env._mission_time() / 60.0
        flanks = decompose_flanks(env)
        out_path = (
            args.output_dir
            / f"{stem}_k{args.state_fire_fronts}_step{step:03d}"
            f"_t{minutes:04.0f}min.png"
        )
        save_snapshot(env, flanks, step, minutes, out_path, args.full_map,
                      args.dpi)
        n_burning = 0 if flanks["burning"] is None else flanks["burning"][0].size
        n_active = 0 if flanks["active"] is None else flanks["active"][0].size
        occupied = sorted(flanks["sectors"])
        print(
            f"step {step:3d}  t={minutes:6.1f} min  burning={n_burning:5d}  "
            f"active={n_active:5d}  sectors={occupied}  -> {out_path.name}"
        )

        if step == args.steps:
            break
        if args.fixed_action is not None:
            action = np.full(
                env.action_decision_count, args.fixed_action, dtype=np.int64
            )
        else:
            action = env.action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            print(f"episode ended after step {step + 1}; stopping.")
            minutes = env._mission_time() / 60.0
            flanks = decompose_flanks(env)
            out_path = (
                args.output_dir
                / f"{stem}_k{args.state_fire_fronts}_step{step + 1:03d}"
                f"_t{minutes:04.0f}min.png"
            )
            save_snapshot(env, flanks, step + 1, minutes, out_path,
                          args.full_map, args.dpi)
            print(f"final snapshot -> {out_path.name}")
            break

    print(f"snapshots saved under {args.output_dir}")


if __name__ == "__main__":
    main()
