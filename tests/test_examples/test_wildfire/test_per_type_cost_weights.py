"""Per-aircraft-type firefront-selection cost weights.

A scenario may carry a `weights` list, one entry per `agents` entry, matched by
position -- that index is the `ac_type_id` the firefighter model stamps on every
aircraft built from that definition. A scenario without the key keeps the
scenario-wide weights for every aircraft, and must behave exactly as it did
before the key existed; several tests below exist only to pin that down.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    _cost_weights,
    _select_topography_destination,
    _select_vegetation_destination,
    _select_vip_destination,
    _select_water_destination,
)
from examples.wildfire.ppo_runnerv3 import (
    COST_WEIGHT_NAMES,
    WildfireHourlyEnv,
    _resolve_scenario,
    _scenario_cost_weights,
)
from examples.wildfire.simulation import (
    FirefrontCostWeightsInput,
    WildfireParameters,
    _TerrainParametersCache,
)

PLAIN = "Palisades5sp5ev.json"
MULTI = "Palisades5sp5ev_multiweight.json"

SHORT = ("distance", "vip", "priority", "vegetation", "topography")

# Distinct, deliberately asymmetric weight sets injected via _mutate. These
# pin the per-type RESOLUTION independently of whatever the live scenario
# declares (its numbers are a modelling choice that has changed before and
# may change again; the tests must not drift with it).
INJECTED = {
    0: {"distance": 1.0, "vip": 2.0, "priority": 1.0,
        "vegetation": 1.0, "topography": 1.0},
    1: {"distance": 3.0, "vip": 0.5, "priority": 1.0,
        "vegetation": 2.0, "topography": 0.5},
}


def _declared(name):
    """What the scenario file itself declares, by ac_type_id.

    Read straight from the raw JSON, bypassing the model, so a test that
    compares resolved weights against this checks the resolver rather than
    restating the model's own parsing.
    """
    data = json.loads(_resolve_scenario(name).read_text(encoding="utf-8"))
    return {
        idx: {short: float(block[f"{short}_cost_weight"]) for short in SHORT}
        for idx, block in enumerate(data["weights"])
    }


def _inject(data):
    data["weights"] = [
        {f"{short}_cost_weight": value for short, value in INJECTED[idx].items()}
        for idx in range(len(data["agents"]))
    ]


def _load(name):
    _TerrainParametersCache.metadata = {}
    return WildfireParameters.model_validate_json(
        _resolve_scenario(name).read_text(encoding="utf-8")
    )


def _mutate(name, mutate):
    """Load a scenario with its raw JSON altered by `mutate`."""
    _TerrainParametersCache.metadata = {}
    data = json.loads(_resolve_scenario(name).read_text(encoding="utf-8"))
    mutate(data)
    return WildfireParameters.model_validate_json(json.dumps(data))


# -- Schema -----------------------------------------------------------------


def test_scenario_without_weights_is_unchanged():
    """The whole point: the new key is inert when absent."""
    params = _load(PLAIN)
    assert params.weights == ()
    # The resolver hands back the parameters object itself, so every aircraft
    # reads the same scenario-wide numbers it read before.
    for ac_type_id in range(len(params.agents)):
        assert params.firefront_cost_weights(ac_type_id) is params
    assert params.firefront_cost_weights(None) is params
    for name in COST_WEIGHT_NAMES:
        assert getattr(params, name) == 1.0


def test_multiweight_scenario_resolves_per_type():
    params = _load(MULTI)
    declared = _declared(MULTI)
    assert len(params.weights) == len(params.agents) == len(declared) == 2
    for ac_type_id, expected in declared.items():
        weights = params.firefront_cost_weights(ac_type_id)
        assert isinstance(weights, FirefrontCostWeightsInput)
        assert weights is not params
        for short, value in expected.items():
            assert getattr(weights, f"{short}_cost_weight") == value


def test_multiweight_scenario_declares_distinct_types():
    """A per-type file that gives both types identical weights is pointless;
    the live scenario must actually differentiate the two aircraft types."""
    declared = _declared(MULTI)
    assert declared[0] != declared[1]


def test_injected_weights_resolve_to_their_own_type():
    """File-independent pin on the resolver: each ac_type_id gets exactly its
    own entry, and the entries are not swapped, shared or averaged."""
    params = _mutate(MULTI, _inject)
    assert len(params.weights) == len(params.agents) == 2
    for ac_type_id, expected in INJECTED.items():
        weights = params.firefront_cost_weights(ac_type_id)
        assert weights is params.weights[ac_type_id]
        for short, value in expected.items():
            assert getattr(weights, f"{short}_cost_weight") == value
    first, second = params.firefront_cost_weights(0), params.firefront_cost_weights(1)
    assert first.distance_cost_weight != second.distance_cost_weight
    assert first.vip_cost_weight != second.vip_cost_weight


def test_length_mismatch_is_rejected():
    with pytest.raises(ValidationError, match="one weight set per entry"):
        _mutate(MULTI, lambda d: d["weights"].pop())
    with pytest.raises(ValidationError, match="one weight set per entry"):
        _mutate(MULTI, lambda d: d["weights"].append(dict(d["weights"][0])))


def test_misspelt_key_is_rejected():
    """`extra="ignore"` drops the typo, so the missing field must be fatal."""

    def typo(data):
        block = data["weights"][0]
        block["vip_weight"] = block.pop("vip_cost_weight")

    with pytest.raises(ValidationError, match="vip_cost_weight"):
        _mutate(MULTI, typo)


def test_scenario_wide_weights_still_load():
    """A scenario stating the five flat keys keeps using them for every type."""

    def flat(data):
        data.pop("weights", None)
        data["distance_cost_weight"] = 7.0
        data["vip_cost_weight"] = 9.0

    params = _mutate(MULTI, flat)
    assert params.weights == ()
    for ac_type_id in range(len(params.agents)):
        weights = params.firefront_cost_weights(ac_type_id)
        assert weights.distance_cost_weight == 7.0
        assert weights.vip_cost_weight == 9.0


def test_runner_reports_weights():
    plain_rows = _scenario_cost_weights(_load(PLAIN))
    assert len(plain_rows) == 2
    # Without the key both types report the same scenario-wide numbers.
    assert plain_rows[0][1] == plain_rows[1][1]

    multi_rows = _scenario_cost_weights(_load(MULTI))
    assert multi_rows[0][1] != multi_rows[1][1]
    # The log rows must be exactly what the file declares, per type.
    for ac_type_id, expected in _declared(MULTI).items():
        assert multi_rows[ac_type_id][1] == {
            f"{short}_cost_weight": value for short, value in expected.items()
        }

    injected_rows = _scenario_cost_weights(_mutate(MULTI, _inject))
    assert injected_rows[0][1]["vip_cost_weight"] == 2.0
    assert injected_rows[1][1]["distance_cost_weight"] == 3.0


# -- Integration ------------------------------------------------------------


def _env(scenario):
    return WildfireHourlyEnv(
        scenario_path=_resolve_scenario(scenario),
        max_steps=4,
        decision_interval_minutes=10,
        state_space="directional-2",
        state_fire_fronts=3,
        group_sizes=[5, 5],
        controlled_agent_count=10,
        water_set=2,
        cell_size_source="code",
        switch_ignition_mode=4,
    )


def _started(scenario, seed=11):
    env = _env(scenario)
    env.reset(seed=seed, options={"sim_seed": seed})
    return env


def test_agents_carry_their_type_weights():
    env = _started(MULTI)
    try:
        seen = {}
        for agent in env.sim.firefighters.firefighters:
            weights = _cost_weights(agent)
            seen.setdefault(agent.ac_type_id, weights)
            # One weight object per type, and it is that type's entry.
            assert weights is seen[agent.ac_type_id]
        assert set(seen) == {0, 1}
        for ac_type_id, expected in _declared(MULTI).items():
            for short, value in expected.items():
                assert getattr(seen[ac_type_id], f"{short}_cost_weight") == value
    finally:
        env.close()


def test_plain_scenario_agents_read_scenario_wide_weights():
    env = _started(PLAIN)
    try:
        for agent in env.sim.firefighters.firefighters:
            assert _cost_weights(agent) is agent.parameters
    finally:
        env.close()


@pytest.mark.parametrize(
    "select",
    [
        _select_water_destination,
        _select_vip_destination,
        _select_vegetation_destination,
        _select_topography_destination,
    ],
)
def test_selectors_run_under_per_type_weights(select):
    """Every selector still returns a usable destination for both types."""
    env = _started(MULTI)
    try:
        by_type = {}
        for agent in env.sim.firefighters.firefighters:
            by_type.setdefault(agent.ac_type_id, agent)
        assert set(by_type) == {0, 1}
        for agent in by_type.values():
            destination = select(agent)
            if destination is None:
                continue
            assert np.asarray(destination).shape == (2,)
            assert np.all(np.isfinite(destination))
    finally:
        env.close()


def _live_fire_env(seed=7, steps=4):
    """An episode advanced to a state with a wide, still-burning firefront.

    Selection only has something to choose between while fire is on the map, so
    a state with a single ignition cell (or none left) cannot show a weight
    changing the answer.
    """
    env = _started(MULTI, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        action = np.asarray(
            [rng.integers(0, int(h)) for h in env.action_space.nvec],
            dtype=np.int64,
        ).reshape(env.action_space.shape)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    return env


@pytest.mark.parametrize(
    "select,first,second",
    [
        (
            _select_water_destination,
            {"distance_cost_weight": 1.0, "vip_cost_weight": 0.0},
            {"distance_cost_weight": 0.0, "vip_cost_weight": 1.0},
        ),
        (
            _select_vip_destination,
            {"distance_cost_weight": 1.0, "vip_cost_weight": 0.0},
            {"distance_cost_weight": 0.0, "vip_cost_weight": 1.0},
        ),
        (
            _select_vegetation_destination,
            {"distance_cost_weight": 1.0, "vip_cost_weight": 0.0,
             "vegetation_cost_weight": 0.0},
            {"distance_cost_weight": 0.0, "vip_cost_weight": 0.0,
             "vegetation_cost_weight": 1.0},
        ),
        (
            _select_topography_destination,
            {"distance_cost_weight": 1.0, "vip_cost_weight": 0.0,
             "topography_cost_weight": 0.0},
            {"distance_cost_weight": 0.0, "vip_cost_weight": 0.0,
             "topography_cost_weight": 1.0},
        ),
    ],
    ids=["water", "vip", "vegetation", "topography"],
)
def test_every_selector_honours_its_weights(select, first, second):
    """Each selector's own term must be able to move its answer.

    Holds the aircraft and the fire fixed and moves only the weights, so a
    selector that ignored them (or read another type's entry) would return the
    same destination for both sets.
    """
    env = _live_fire_env()
    try:
        fire = env.sim.firefighters.wildfire.fire_positions
        if len(fire) < 50:
            pytest.skip(f"fire too small to discriminate ({len(fire)} fronts)")
        agent = env.sim.firefighters.firefighters[0]
        base = env.sim.parameters

        def destination(overrides):
            values = dict.fromkeys(COST_WEIGHT_NAMES, 1.0)
            values.update(overrides)
            weights = FirefrontCostWeightsInput(**values)
            env.sim.parameters = base.model_copy(
                update={"weights": (weights,) * len(base.agents)}
            )
            env.sim.firefighters.__cache__.pop("select_poi", None)
            return select(agent)

        try:
            one, two = destination(first), destination(second)
        finally:
            env.sim.parameters = base
        assert one is not None and two is not None
        assert not np.array_equal(np.asarray(one), np.asarray(two))
    finally:
        env.close()
