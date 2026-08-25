"""Tests for the "directional-2" observation/state space.

"directional-2" is "directional" with two changes:

* the four map-boundary distances are removed (each was an exact duplicate of a
  fire-extreme coordinate already present in the vector), and
* the episode's scaled ignition point is added in their place.

Everything else -- the per-flank block, the altitude-free agent block, and the
values of every shared feature -- must be untouched, and the other three state
spaces must be entirely unaffected.
"""

from pathlib import Path

import numpy as np
import pytest

from examples.wildfire.ppo_runnerv2 import (
    DIRECTIONAL2_STATE_FEATURES,
    DIRECTIONAL_PER_FRONT_FEATURES,
    OLD_STATE_FEATURES,
    STATE_SPACE_DIRECTIONAL,
    STATE_SPACE_DIRECTIONAL_2,
    STATE_SPACE_OLD,
    STATE_SPACE_UPDATED,
    SUPPORTED_STATE_SPACES,
    UPDATED_STATE_FEATURES,
    WildfireHourlyEnv,
    _agent_feature_count,
    _agent_feature_names,
    _normalize_state_space,
    _scale_to_unit,
    _state_feature_names,
    _resolve_scenario,
)

BOUNDARY_FEATURES = (
    "distance_left_boundary",
    "distance_right_boundary",
    "distance_bottom_boundary",
    "distance_top_boundary",
)
IGNITION_FEATURES = ("ignition_x", "ignition_y")


# -- Feature layout ---------------------------------------------------------#


def test_directional2_is_registered():
    assert STATE_SPACE_DIRECTIONAL_2 == "directional-2"
    assert STATE_SPACE_DIRECTIONAL_2 in SUPPORTED_STATE_SPACES
    assert _normalize_state_space("directional-2") == STATE_SPACE_DIRECTIONAL_2
    assert _normalize_state_space("  DIRECTIONAL-2  ") == STATE_SPACE_DIRECTIONAL_2


def test_unsupported_state_space_still_rejected():
    with pytest.raises(ValueError):
        _normalize_state_space("directional2")
    with pytest.raises(ValueError):
        _normalize_state_space("nonsense")


def test_core_is_old_minus_boundaries_plus_ignition():
    """The core differs from "old" by exactly the documented swap."""
    expected = [name for name in OLD_STATE_FEATURES if name not in BOUNDARY_FEATURES]
    insert_at = expected.index("fire_center_x")
    expected[insert_at:insert_at] = list(IGNITION_FEATURES)

    assert DIRECTIONAL2_STATE_FEATURES == expected
    assert len(DIRECTIONAL2_STATE_FEATURES) == len(OLD_STATE_FEATURES) - 4 + 2 == 28

    removed = set(OLD_STATE_FEATURES) - set(DIRECTIONAL2_STATE_FEATURES)
    added = set(DIRECTIONAL2_STATE_FEATURES) - set(OLD_STATE_FEATURES)
    assert removed == set(BOUNDARY_FEATURES)
    assert added == set(IGNITION_FEATURES)


@pytest.mark.parametrize("fronts", [1, 3, 5])
@pytest.mark.parametrize("agents", [1, 4])
@pytest.mark.parametrize("scenario_flag", [False, True])
def test_feature_names_and_dimension(fronts, agents, scenario_flag):
    names = _state_feature_names(
        include_scenario_flag=scenario_flag,
        controlled_agent_count=agents,
        state_space=STATE_SPACE_DIRECTIONAL_2,
        state_fire_fronts=fronts,
    )
    expected_len = (3 if scenario_flag else 0) + 28 + 6 * fronts + 2 * agents
    assert len(names) == expected_len
    assert len(set(names)) == len(names), "feature names must be unique"

    # No boundary distances, exactly one ignition pair.
    assert not set(names) & set(BOUNDARY_FEATURES)
    assert [n for n in names if n in IGNITION_FEATURES] == list(IGNITION_FEATURES)

    # The per-flank block is present in full and unchanged.
    for slot in range(fronts):
        for base in DIRECTIONAL_PER_FRONT_FEATURES:
            assert f"{base}_{slot}" in names

    # Agent block carries x/y only -- no altitude, same as "old"/"directional".
    assert _agent_feature_count(STATE_SPACE_DIRECTIONAL_2) == 2
    agent_names = _agent_feature_names(agents, STATE_SPACE_DIRECTIONAL_2)
    assert agent_names == tuple(
        part for idx in range(agents) for part in (f"agent_{idx}_x", f"agent_{idx}_y")
    )
    assert not any(name.endswith("_altitude") for name in names)


def test_shared_features_keep_directional_order():
    """Dropping/adding features must not reorder the ones both spaces share."""
    d = _state_feature_names(False, 4, STATE_SPACE_DIRECTIONAL, 3)
    d2 = _state_feature_names(False, 4, STATE_SPACE_DIRECTIONAL_2, 3)

    shared_d = [n for n in d if n not in BOUNDARY_FEATURES]
    shared_d2 = [n for n in d2 if n not in IGNITION_FEATURES]
    assert shared_d == shared_d2


# -- Regression: the other three state spaces are untouched -----------------#


def test_existing_state_spaces_unchanged():
    assert SUPPORTED_STATE_SPACES == ("old", "updated", "directional", "directional-2")
    assert len(OLD_STATE_FEATURES) == 30
    assert len(UPDATED_STATE_FEATURES) == 34
    assert OLD_STATE_FEATURES[-4:] == list(BOUNDARY_FEATURES)
    assert UPDATED_STATE_FEATURES[-4:] == list(BOUNDARY_FEATURES)

    # "old": 30 core + 2/agent, no altitude. "updated": 34 core + 3/agent.
    # "directional": 30 core + 6K + 2/agent.
    assert len(_state_feature_names(False, 3, STATE_SPACE_OLD, 5)) == 36
    assert len(_state_feature_names(False, 3, STATE_SPACE_UPDATED, 5)) == 43
    assert len(_state_feature_names(False, 3, STATE_SPACE_DIRECTIONAL, 3)) == 54

    assert _agent_feature_count(STATE_SPACE_OLD) == 2
    assert _agent_feature_count(STATE_SPACE_DIRECTIONAL) == 2
    assert _agent_feature_count(STATE_SPACE_UPDATED) == 3

    # Neither old nor updated gains an ignition channel.
    for space in (STATE_SPACE_OLD, STATE_SPACE_UPDATED, STATE_SPACE_DIRECTIONAL):
        names = _state_feature_names(False, 3, space, 3)
        assert not set(names) & set(IGNITION_FEATURES)
        assert set(names) >= set(BOUNDARY_FEATURES)


# -- Integration: real rollouts --------------------------------------------#


def _make_env(state_space: str, fronts: int = 3, agents: int = 4):
    return WildfireHourlyEnv(
        scenario_path=_resolve_scenario("Pyrenees.json"),
        max_steps=4,
        decision_interval_minutes=10,
        state_space=state_space,
        state_fire_fronts=fronts,
        tactic_distribution="group",
        aircraft_group_size=2,
        controlled_agent_count=agents,
        water_set=1,
        cell_size_source="code",
    )


def _rollout(env, steps: int, seed: int = 123):
    """Deterministic rollout; returns the observation matrix."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed, options={"sim_seed": seed})
    rows = [obs]
    for _ in range(steps):
        action = np.asarray(
            [rng.integers(0, int(high)) for high in env.action_space.nvec],
            dtype=np.int64,
        ).reshape(env.action_space.shape)
        obs, _reward, terminated, truncated, _info = env.step(action)
        rows.append(obs)
        if terminated or truncated:
            break
    return np.vstack(rows), env


@pytest.mark.slow
def test_directional2_matches_directional_on_shared_features():
    """Every feature both spaces share must be bit-identical on the same seed.

    This is the core "no other behaviour changed" check: the only differences
    between the two observation vectors are the removed boundary distances and
    the added ignition point.
    """
    steps = 3
    obs_d, env_d = _rollout(_make_env(STATE_SPACE_DIRECTIONAL), steps)
    obs_d2, env_d2 = _rollout(_make_env(STATE_SPACE_DIRECTIONAL_2), steps)

    names_d = list(env_d.state_feature_names)
    names_d2 = list(env_d2.state_feature_names)
    assert obs_d.shape[0] == obs_d2.shape[0], "rollouts diverged in length"
    assert obs_d.shape[1] == len(names_d)
    assert obs_d2.shape[1] == len(names_d2)
    assert obs_d2.shape[1] == obs_d.shape[1] - 4 + 2

    idx_d = {name: i for i, name in enumerate(names_d)}
    idx_d2 = {name: i for i, name in enumerate(names_d2)}

    for name in names_d2:
        if name in IGNITION_FEATURES:
            continue
        np.testing.assert_array_equal(
            obs_d2[:, idx_d2[name]],
            obs_d[:, idx_d[name]],
            err_msg=f"shared feature {name!r} changed value",
        )


@pytest.mark.slow
def test_ignition_features_are_correct_and_static():
    steps = 3
    obs, env = _rollout(_make_env(STATE_SPACE_DIRECTIONAL_2), steps)
    names = list(env.state_feature_names)
    ix = names.index("ignition_x")
    iy = names.index("ignition_y")

    assert env.current_ignition_pos is not None
    expected_x = _scale_to_unit(
        float(env.current_ignition_pos[0]), env._coord_x_min, env._coord_x_max
    )
    expected_y = _scale_to_unit(
        float(env.current_ignition_pos[1]), env._coord_y_min, env._coord_y_max
    )

    # Scaled correctly...
    assert obs[-1, ix] == pytest.approx(expected_x, abs=1e-6)
    assert obs[-1, iy] == pytest.approx(expected_y, abs=1e-6)
    # ...in range...
    assert 0.0 <= obs[:, ix].min() and obs[:, ix].max() <= 1.0
    assert 0.0 <= obs[:, iy].min() and obs[:, iy].max() <= 1.0
    # ...and constant for the whole episode.
    assert obs[:, ix].std() == 0.0
    assert obs[:, iy].std() == 0.0

    # The ignition point sits inside the fire it started: at the first decision
    # step the fire is still near its seed.
    fire = env.sim.wildfire.fire_positions
    centroid = fire.mean(axis=0)
    seed_dist = float(
        np.hypot(
            centroid[0] - env.current_ignition_pos[0],
            centroid[1] - env.current_ignition_pos[1],
        )
    )
    assert seed_dist < env._map_diagonal * 0.05


def test_scale_to_unit_normalizes_into_the_unit_interval():
    """The scaler used for the ignition point clamps to [0, 1] by construction."""
    lo, hi = 100.0, 300.0
    assert _scale_to_unit(100.0, lo, hi) == 0.0
    assert _scale_to_unit(200.0, lo, hi) == pytest.approx(0.5)
    assert _scale_to_unit(300.0, lo, hi) == 1.0
    # Out of range on both sides, including far outside, clamps rather than wraps.
    assert _scale_to_unit(-5_000.0, lo, hi) == 0.0
    assert _scale_to_unit(1e12, lo, hi) == 1.0
    # Degenerate extent cannot produce a divide-by-zero or a value outside [0, 1].
    assert _scale_to_unit(42.0, 10.0, 10.0) == 0.0


@pytest.mark.slow
def test_ignition_features_clamped_for_out_of_range_positions():
    """An ignition point outside the map extent still yields values in [0, 1]."""
    env = _make_env(STATE_SPACE_DIRECTIONAL_2)
    env.reset(seed=11, options={"sim_seed": 11})
    names = list(env.state_feature_names)
    ix, iy = names.index("ignition_x"), names.index("ignition_y")

    for pos in [
        (-1.0e9, 1.0e9),  # far below x-min, far above y-max
        (env._coord_x_max * 10.0, -1.0),
        (env._coord_x_min, env._coord_y_min),  # exactly on the lower corner
        (env._coord_x_max, env._coord_y_max),  # exactly on the upper corner
    ]:
        env.current_ignition_pos = pos
        obs = env._compute_state()
        assert 0.0 <= obs[ix] <= 1.0, f"ignition_x out of range for {pos}"
        assert 0.0 <= obs[iy] <= 1.0, f"ignition_y out of range for {pos}"
        assert np.all(np.isfinite(obs))
        assert obs.min() >= 0.0 and obs.max() <= 1.0

    # Explicit clamp directions for the extreme case.
    env.current_ignition_pos = (-1.0e9, 1.0e9)
    obs = env._compute_state()
    assert obs[ix] == 0.0
    assert obs[iy] == 1.0

    # A scenario without an ignition center falls back to 0.0 rather than NaN.
    env.current_ignition_pos = None
    obs = env._compute_state()
    assert obs[ix] == 0.0
    assert obs[iy] == 0.0
    assert np.all(np.isfinite(obs))


@pytest.mark.slow
def test_boundary_duplication_identity_still_holds_for_old():
    """The removed features were exact duplicates -- confirm that on live data.

    This both documents why "directional-2" drops them and guards the ordering
    of the boundary block that the other state spaces still emit.
    """
    obs, env = _rollout(_make_env(STATE_SPACE_OLD), 3)
    idx = {name: i for i, name in enumerate(env.state_feature_names)}

    np.testing.assert_array_equal(
        obs[:, idx["distance_left_boundary"]], obs[:, idx["leftmost_x"]]
    )
    np.testing.assert_array_equal(
        obs[:, idx["distance_bottom_boundary"]], obs[:, idx["lowermost_y"]]
    )
    np.testing.assert_allclose(
        obs[:, idx["distance_right_boundary"]],
        1.0 - obs[:, idx["rightmost_x"]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        obs[:, idx["distance_top_boundary"]],
        1.0 - obs[:, idx["uppermost_y"]],
        atol=1e-6,
    )


@pytest.mark.slow
def test_directional2_observation_is_well_formed():
    obs, env = _rollout(_make_env(STATE_SPACE_DIRECTIONAL_2), 3)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert env.observation_space.shape == (len(env.state_feature_names),)
    assert obs.shape[1] == env.observation_space.shape[0]


@pytest.mark.slow
@pytest.mark.parametrize("fronts", [1, 4])
def test_directional2_respects_state_fire_fronts(fronts):
    env = _make_env(STATE_SPACE_DIRECTIONAL_2, fronts=fronts)
    obs, _ = env.reset(seed=7, options={"sim_seed": 7})
    assert obs.shape[0] == 28 + 6 * fronts + 2 * 4
