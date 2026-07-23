"""Deterministic behavioural fingerprint of the sosid simulation stack.

Used to prove that swapping the ``sosid`` install (rl-5-30 -> this repo's
``src/sosid``) does not change simulation behaviour. Run it before the swap
to capture a baseline, then after the swap and diff the two JSON files:

    python scripts/sosid_fingerprint.py --out baseline.json
    # ... switch the editable install ...
    python scripts/sosid_fingerprint.py --out candidate.json
    python scripts/sosid_fingerprint.py --compare baseline.json candidate.json

Any difference is an unreconciled semantic divergence between the two trees.
The fingerprint deliberately samples agent kinematics and energy mid-run --
not just end-of-mission totals -- so that differences which cancel out by the
end are still caught.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCENARIO_DIR = REPO / "examples/wildfire/data/scenarios/inputs"

# (scenario, seed) pairs. Kept small enough to run in minutes but wide
# enough to cover eVTOL + conventional fleets and both maps.
CASES = [
    ("Palisades4sp3ev.json", 0),
    ("Palisades4sp3ev.json", 1),
    ("Palisades4sp2ev.json", 0),
    ("Pyrenees4sp3ev.json", 0),
]

# Sim seconds between kinematic samples, and the cap per case.
SAMPLE_EVERY = 300
MAX_SECONDS = 7200


def _round(x, n=3):
    """Round for float-noise-tolerant comparison."""
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def fingerprint_case(scenario: str, seed: int) -> dict:
    import numpy as np

    from examples.wildfire.simulation import (
        WildfireParameters,
        WildfireSimulation,
        _TerrainParametersCache,
    )

    _TerrainParametersCache.metadata = {}
    data = json.loads((SCENARIO_DIR / scenario).read_text())
    data["run_headless"] = True
    sim = WildfireSimulation(
        parameters=WildfireParameters.model_validate(data), seed=seed
    )
    sim.wildfire.ignite(sim.ignition_centers)
    agents = list(sim.firefighters.firefighters)

    samples = []
    t = 0
    stopped_at = None
    crash = None
    while t < MAX_SECONDS:
        if sim.is_stopped.is_set():
            stopped_at = t
            break
        if t % SAMPLE_EVERY == 0:
            snap = []
            for a in agents:
                task = a.tasks.active_task
                snap.append(
                    {
                        "id": int(a.unique_id),
                        "task": task.task_method.__name__ if task else None,
                        "fs": int(getattr(a, "flight_state", 0)),
                        "alt": _round(a.altitude, 2),
                        "pos": [_round(p, 2) for p in np.asarray(a.pos, float)],
                        "prop": _round(
                            a.propulsion.mission_usable_propellant, 1
                        ),
                        "mass": _round(a.current_mass, 2),
                        "sup": int(a.total_suppressions),
                    }
                )
            samples.append({"t": t, "agents": snap})
        try:
            sim.step(force=True)
        except RuntimeError as err:
            if "not started cannot be stopped" in str(err):
                stopped_at = t
                break
            raise
        except ValueError as err:
            crash = str(err)
            break
        t += 1

    return {
        "scenario": scenario,
        "seed": seed,
        "steps": t,
        "stopped_at": stopped_at,
        "crash": crash,
        "n_agents": len(agents),
        "final": [
            {
                "id": int(a.unique_id),
                "sup": int(a.total_suppressions),
                "refills": int(a.propulsion.n_propellant_refills),
                "prop": _round(a.propulsion.mission_usable_propellant, 1),
                "dist_km": _round(a.distance_flown, 3),
            }
            for a in agents
        ],
        "samples": samples,
    }


def provenance() -> dict:
    import sosid

    return {
        "sosid_file": str(Path(sosid.__file__).resolve()),
        "sosid_version": getattr(sosid, "__version__", "?"),
        "python": sys.version.split()[0],
    }


def build(out: Path) -> None:
    cases = []
    for scenario, seed in CASES:
        if not (SCENARIO_DIR / scenario).exists():
            print(f"skip (missing): {scenario}", flush=True)
            continue
        print(f"running {scenario} seed={seed} ...", flush=True)
        cases.append(fingerprint_case(scenario, seed))
    payload = {"provenance": provenance(), "cases": cases}
    blob = json.dumps(payload, sort_keys=True, indent=2)
    out.write_text(blob + "\n")
    # digest excludes provenance so the same behaviour hashes equal
    digest = hashlib.sha256(
        json.dumps(payload["cases"], sort_keys=True).encode()
    ).hexdigest()
    print(f"\nwrote {out}")
    print(f"sosid   : {payload['provenance']['sosid_file']}")
    print(f"behaviour digest: {digest}")


def compare(a: Path, b: Path) -> int:
    pa, pb = json.loads(a.read_text()), json.loads(b.read_text())
    da = hashlib.sha256(
        json.dumps(pa["cases"], sort_keys=True).encode()
    ).hexdigest()
    db = hashlib.sha256(
        json.dumps(pb["cases"], sort_keys=True).encode()
    ).hexdigest()
    print(f"A sosid: {pa['provenance']['sosid_file']}\n  digest {da}")
    print(f"B sosid: {pb['provenance']['sosid_file']}\n  digest {db}")
    if da == db:
        print("\nIDENTICAL behaviour -- swap is safe.")
        return 0

    print("\nDIVERGENT behaviour. First differences:\n")
    shown = 0
    for ca, cb in zip(pa["cases"], pb["cases"], strict=False):
        tag = f"{ca['scenario']} seed={ca['seed']}"
        for key in ("steps", "stopped_at", "crash", "n_agents"):
            if ca.get(key) != cb.get(key):
                print(f"  [{tag}] {key}: {ca.get(key)!r} -> {cb.get(key)!r}")
                shown += 1
        for fa, fb in zip(ca["final"], cb["final"], strict=False):
            if fa != fb:
                print(f"  [{tag}] final agent {fa['id']}: {fa} -> {fb}")
                shown += 1
        for sa, sb in zip(ca["samples"], cb["samples"], strict=False):
            if sa != sb:
                for ga, gb in zip(sa["agents"], sb["agents"], strict=False):
                    if ga != gb:
                        print(
                            f"  [{tag}] t={sa['t']} agent {ga['id']}: "
                            f"{ga} -> {gb}"
                        )
                        shown += 1
                        break
                break
        if shown > 25:
            print("  ... (truncated)")
            break
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write a fingerprint here")
    ap.add_argument(
        "--compare", nargs=2, type=Path, metavar=("A", "B"),
        help="compare two fingerprint files",
    )
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.out:
        build(args.out)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
