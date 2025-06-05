"""Module testing path planning utilities."""

import numpy as np
import pytest

from examples.uam.planner.environment import Agent
from examples.uam.planner.utilities import (
    DistanceProfile,
    distance_based_profile,
)


@pytest.mark.parametrize(
    ("use_symmetric_agent", "distance", "start_altitude", "end_altitude"),
    [
        (True, 50_000.0, None, None),
        (True, 10_000.0, None, None),
        (True, 50_000.0, 100.0, 135.0),
        (False, 50_000.0, 100.0, 135.0),
        (False, 10_000.0, 35.0, 55.0),
    ],
)
def test_distance_based_profile(
    use_symmetric_agent: bool,
    distance: float,
    start_altitude: float | None,
    end_altitude: float | None,
) -> None:
    if use_symmetric_agent:
        agent = Agent(
            radius=50.0,
            cruise_climb_speeds=[40.0, 50.0],
            cruise_descent_speeds=[40.0, 50.0],
            cruise_speeds=[90.0, 100.0],
            takeoff_altitude=15.0,
            add_takeoff_duration=20.0,
            takeoff_ground_distance=1000.0,
            landing_altitude=15.0,
            landing_ground_distance=1000.0,
            cruise_altitude=465.0,
            cruise_climb_slope=0.05,
            cruise_descent_slope=0.05,
        )
    else:
        agent = Agent(
            radius=50.0,
            cruise_climb_speeds=[40.0, 50.0],
            cruise_descent_speeds=[50.0, 60.0],
            cruise_speeds=[90.0, 100.0],
            takeoff_altitude=15.0,
            add_takeoff_duration=20.0,
            takeoff_ground_distance=1000.0,
            landing_altitude=20.0,
            landing_ground_distance=1000.0,
            cruise_altitude=465.0,
            cruise_climb_slope=0.05,
            cruise_descent_slope=0.1,
        )
    profile = distance_based_profile(
        agent,
        distance,
        start_altitude=start_altitude,
        end_altitude=end_altitude,
    )
    # Validate general properties.
    assert isinstance(profile, DistanceProfile)
    assert profile.reaches_cruise == (
        max(profile.altitudes) == agent.cruise_altitude
    )
    assert len(profile.timestamps) == len(profile.distances)
    assert len(profile.timestamps) == len(profile.altitudes)
    if profile.reaches_cruise:
        assert len(profile.timestamps) == 4
    else:
        assert len(profile.timestamps) == 3

    d_timestamps = np.diff(profile.timestamps)
    d_distance = np.diff(profile.distances)
    if start_altitude is None:
        start_altitude = agent.takeoff_altitude
    if end_altitude is None:
        end_altitude = agent.landing_altitude

    # Validate start point.
    assert profile.timestamps[0] == 0.0
    assert profile.distances[0] == 0.0
    assert profile.altitudes[0] == start_altitude

    # Validate climb point.
    climb_distance = (
        profile.altitudes[1] - profile.altitudes[0]
    ) / agent.cruise_climb_slope
    np.testing.assert_almost_equal(profile.distances[1], climb_distance)
    climb_time = climb_distance / agent.cruise_climb_speeds[-1]
    np.testing.assert_almost_equal(profile.timestamps[1], climb_time)

    # Validate cruise point.
    if profile.reaches_cruise:
        cruise_distance = profile.distances[2] - profile.distances[1]
        cruise_time = cruise_distance / agent.cruise_speeds[-1]
        np.testing.assert_almost_equal(d_timestamps[1], cruise_time)
        np.testing.assert_almost_equal(d_distance[1], cruise_distance)
        np.testing.assert_almost_equal(
            profile.altitudes[2], agent.cruise_altitude
        )

    # Validate descent.
    descent_distance = (
        profile.altitudes[-2] - profile.altitudes[-1]
    ) / agent.cruise_descent_slope
    descent_time = descent_distance / agent.cruise_descent_speeds[-1]
    np.testing.assert_almost_equal(d_timestamps[-1], descent_time)
    np.testing.assert_almost_equal(d_distance[-1], descent_distance)
    np.testing.assert_almost_equal(profile.altitudes[-1], end_altitude)
