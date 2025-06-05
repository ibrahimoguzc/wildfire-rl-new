"""Module testing the implementation of the trajectory classes."""

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

import numpy as np
import pytest

from sosid.model.abm.trajectory import (
    AircraftProfileParameters,
    BaseTrajectory,
    FlightState,
    StraightTrajectory,
    WaypointTrajectory,
    generate_straight_trajectory,
)
from sosid.model.transform import GEODESIC

STRAIGHT_AIRSPACE_DATA = {
    "timestamps": [0.0, 7.5, 15.0, 115.0, 195.0, 295.0, 345.0],
    "horizontal_distances": [150.0, 150.0, 3000.0, 4000.0, 5000.0, 1500.0],
    "altitudes": [50.0, 60.0, 70.0, 100.0, 100.0, 100.0, 60.0],
    "flight_states": [
        FlightState.CRUISE_CLIMB,
        FlightState.CRUISE_CLIMB,
        FlightState.CRUISE_CLIMB,
        FlightState.CRUISE,
        FlightState.CRUISE,
        FlightState.CRUISE_DESCENT,
    ],
}
# Compute GPS coordinates as a straight line based on the distances.
STRAIGHT_AIRSPACE_DATA["gps"] = np.array(
    GEODESIC.fwd(
        *np.tile(
            [10.0, 53.0, 90.0], (len(STRAIGHT_AIRSPACE_DATA["timestamps"]), 1)
        ).T,
        np.cumsum(
            np.insert(STRAIGHT_AIRSPACE_DATA["horizontal_distances"], 0, 0.0)
        ),
    )[:2][::-1]
).T


class BaseTrajectoryTester(metaclass=ABCMeta):
    """Base class for testing trajectory classes."""

    supports_concatenation_with_time_gap: ClassVar[bool] = True
    immutable_arrays: ClassVar[str] = (
        "timestamps",
        "flight_states",
    )

    airspace_data: ClassVar[dict] = STRAIGHT_AIRSPACE_DATA
    takeoff_data: ClassVar[dict] = {
        "timestamps": [0.0, 30.0, 120.0, 150.0],
        "gps": [STRAIGHT_AIRSPACE_DATA["gps"][0]] * 4,
        "altitudes": [0.0, 0.0, 50.0, 50.0],
        "flight_states": [
            FlightState.TAXI_OUT,
            FlightState.TAKEOFF,
            FlightState.TRANSITION,
        ],
        "horizontal_distances": [0.0, 0.0, 0.0],
    }
    landing_data: ClassVar[dict] = {
        "timestamps": [0.0, 30.0, 120.0, 150.0],
        "gps": [STRAIGHT_AIRSPACE_DATA["gps"][-1]] * 4,
        "altitudes": [50.0, 50.0, 0.0, 0.0],
        "flight_states": [
            FlightState.TRANSITION,
            FlightState.LANDING,
            FlightState.TAXI_IN,
        ],
        "horizontal_distances": [0.0, 0.0, 0.0],
    }

    @abstractmethod
    def get_trajectory_from_dict(self, dct: dict) -> BaseTrajectory:
        raise NotImplementedError

    @abstractmethod
    def get_required_shapes(
        self, n_timestamps: int
    ) -> dict[str, tuple[int, ...]]:
        return {
            "timestamps": (n_timestamps,),
            "flight_states": (max(n_timestamps - 1, 0),),
        }

    def get_airspace_trajectory(
        self, start_time: datetime | None = None
    ) -> BaseTrajectory:
        return self.get_trajectory_from_dict(
            self.airspace_data | {"start_time": start_time}
        )

    def get_takeoff_trajectory(
        self, start_time: datetime | None = None
    ) -> BaseTrajectory:
        return self.get_trajectory_from_dict(
            self.takeoff_data | {"start_time": start_time}
        )

    def get_landing_trajectory(
        self, start_time: datetime | None = None
    ) -> BaseTrajectory:
        return self.get_trajectory_from_dict(
            self.landing_data | {"start_time": start_time}
        )

    def get_full_trajectory(
        self, start_time: datetime | None = None
    ) -> BaseTrajectory:
        takeoff = self.get_takeoff_trajectory(start_time=start_time)
        airspace = self.get_airspace_trajectory(start_time=start_time)
        landing = self.get_landing_trajectory(start_time=start_time)
        cls: type[BaseTrajectory] = airspace.__class__
        return cls.concatenate((takeoff, airspace, landing))

    def get_empty_trajectory(
        self, start_time: datetime | None = None
    ) -> BaseTrajectory:
        return self.get_trajectory_from_dict(
            {
                "start_time": start_time,
                "timestamps": [],
                "gps": [],
                "altitudes": [],
                "flight_states": [],
                "horizontal_distances": [],
            }
        )

    def validate_required_shapes(self, trajectory: BaseTrajectory) -> None:
        n = len(trajectory.timestamps)
        assert trajectory.flight_states.shape == (max(n - 1, 0),)
        for attr, shape in self.get_required_shapes(n).items():
            assert getattr(trajectory, attr).shape == shape

    @pytest.mark.parametrize(
        ("start_time", "end_time"),
        [
            (None, None),
            (
                datetime(2024, 11, 9, 12, 0, 0),
                datetime(2024, 11, 9, 12, 5, 45),
            ),
        ],
    )
    def test_init(self, start_time: datetime, end_time: datetime) -> None:
        traj = self.get_airspace_trajectory(start_time=start_time)
        assert traj.start_datetime == start_time
        assert traj.end_datetime == end_time
        assert traj.total_duration == 345.0
        assert len(traj.timestamps) == 7
        assert len(traj) == 7
        assert all(traj.state_start_indices == [0, 3, 5])
        assert all(
            traj.flight_states[traj.state_start_indices]
            == [
                FlightState.CRUISE_CLIMB,
                FlightState.CRUISE,
                FlightState.CRUISE_DESCENT,
            ]
        )
        self.validate_required_shapes(traj)
        if hasattr(traj, "horizontal_distances"):
            assert round(sum(traj.horizontal_distances)) == 13800

    def test_immutable(self) -> None:
        traj = self.get_airspace_trajectory()
        for attr in self.immutable_arrays:
            with pytest.raises(AttributeError):
                setattr(traj, attr, np.arange(7))
            with pytest.raises(ValueError, match="read-only"):
                getattr(traj, attr)[0] = 1.0

    def test_slice_by_idx(self) -> None:
        traj = self.get_airspace_trajectory(
            start_time=datetime(2024, 11, 9, 12, 0, 0)
        )
        sliced = traj.slice_by_idx(2, 5)
        assert len(sliced) == 3
        if hasattr(traj, "gps"):
            np.testing.assert_allclose(sliced.gps[0], traj.gps[2])
            np.testing.assert_allclose(sliced.gps[-1], traj.gps[4])
        assert sliced.start_datetime == datetime(2024, 11, 9, 12, 0, 15)

    def test_concatenate_no_time(self) -> None:
        airspace_traj = self.get_airspace_trajectory(start_time=None)
        takeoff_traj = self.get_takeoff_trajectory(start_time=None)
        cls: type[BaseTrajectory] = airspace_traj.__class__
        concatenated = cls.concatenate((takeoff_traj, airspace_traj))
        assert len(concatenated) == len(takeoff_traj) + len(airspace_traj) - 1
        assert concatenated.start_datetime is None
        assert concatenated.timestamps[0] == 0.0
        assert concatenated.timestamps[-1] == 495.0
        assert all(
            concatenated.flight_states[:5]
            == [
                FlightState.TAXI_OUT,
                FlightState.TAKEOFF,
                FlightState.TRANSITION,
                FlightState.CRUISE_CLIMB,
                FlightState.CRUISE_CLIMB,
            ]
        )
        self.validate_required_shapes(concatenated)

    def test_concatenate_same_time(self) -> None:
        airspace_traj = self.get_airspace_trajectory(
            start_time=datetime(2024, 11, 9, 12, 2, 30)
        )
        takeoff_traj = self.get_takeoff_trajectory(
            start_time=datetime(2024, 11, 9, 12, 0, 0)
        )
        cls: type[BaseTrajectory] = airspace_traj.__class__
        assert takeoff_traj.end_datetime == airspace_traj.start_datetime
        concatenated = cls.concatenate((takeoff_traj, airspace_traj))
        assert len(concatenated) == len(takeoff_traj) + len(airspace_traj) - 1
        assert concatenated.start_datetime == datetime(2024, 11, 9, 12, 0, 0)
        assert concatenated.end_datetime == datetime(2024, 11, 9, 12, 8, 15)
        assert concatenated.timestamps[0] == 0.0
        assert concatenated.timestamps[-1] == 495.0
        assert concatenated.flight_states[0] == FlightState.TAXI_OUT
        assert all(
            concatenated.flight_states[:5]
            == [
                FlightState.TAXI_OUT,
                FlightState.TAKEOFF,
                FlightState.TRANSITION,
                FlightState.CRUISE_CLIMB,
                FlightState.CRUISE_CLIMB,
            ]
        )
        self.validate_required_shapes(concatenated)

    def test_concatenate_different_time(self) -> None:
        airspace_traj = self.get_airspace_trajectory(
            start_time=datetime(2024, 11, 9, 12, 3, 30)
        )
        takeoff_traj = self.get_takeoff_trajectory(
            start_time=datetime(2024, 11, 9, 12, 0, 0)
        )
        cls: type[BaseTrajectory] = airspace_traj.__class__
        assert takeoff_traj.end_datetime < airspace_traj.start_datetime
        if not self.supports_concatenation_with_time_gap:
            with pytest.raises(ValueError, match="connected"):
                cls.concatenate((takeoff_traj, airspace_traj))
            return
        concatenated = cls.concatenate((takeoff_traj, airspace_traj))
        assert len(concatenated) == len(takeoff_traj) + len(airspace_traj)
        assert concatenated.start_datetime == datetime(2024, 11, 9, 12, 0, 0)
        assert concatenated.end_datetime == datetime(2024, 11, 9, 12, 9, 15)
        assert concatenated.timestamps[0] == 0.0
        assert concatenated.timestamps[-1] == 555.0
        assert concatenated.flight_states[0] == FlightState.TAXI_OUT
        assert all(
            concatenated.flight_states[:5]
            == [
                FlightState.TAXI_OUT,
                FlightState.TAKEOFF,
                FlightState.TRANSITION,
                FlightState.TRANSITION,
                FlightState.CRUISE_CLIMB,
            ]
        )

    def test_concatenate_wrong_order(self) -> None:
        airspace_traj = self.get_airspace_trajectory(
            start_time=datetime(2024, 11, 9, 12, 0, 0)
        )
        takeoff_traj = self.get_takeoff_trajectory(
            start_time=datetime(2024, 11, 9, 12, 5, 0)
        )
        assert takeoff_traj.end_datetime > airspace_traj.start_datetime
        with pytest.raises(ValueError, match="order|connected"):
            airspace_traj.__class__.concatenate((takeoff_traj, airspace_traj))

    def test_concatenate_overlapping_time(self) -> None:
        airspace_traj = self.get_airspace_trajectory(
            start_time=datetime(2024, 11, 9, 12, 1, 30)
        )
        takeoff_traj = self.get_takeoff_trajectory(
            start_time=datetime(2024, 11, 9, 12, 0, 0)
        )
        assert takeoff_traj.end_datetime > airspace_traj.start_datetime
        assert takeoff_traj.end_datetime < airspace_traj.end_datetime
        with pytest.raises(ValueError, match="overlap|connected"):
            airspace_traj.__class__.concatenate((takeoff_traj, airspace_traj))

    def test_concatenate_no_paths(self) -> None:
        cls = self.get_landing_trajectory().__class__
        with pytest.raises(ValueError, match="at least one"):
            cls.concatenate(())

    def test_get_state_duration_segements(self) -> None:
        airspace_traj = self.get_airspace_trajectory()
        segments = list(airspace_traj.get_state_duration_segements())
        assert segments == [
            (FlightState.CRUISE_CLIMB, 115.0),
            (FlightState.CRUISE, 180.0),
            (FlightState.CRUISE_DESCENT, 50.0),
        ]

    def test_state_start_end_indices(self) -> None:
        traj = self.get_full_trajectory()
        assert all(traj.state_start_indices == [0, 1, 2, 3, 6, 8, 9, 10, 11])
        assert all(traj.state_end_indices == [0, 1, 2, 5, 7, 8, 9, 10, 11])

    @pytest.mark.parametrize(
        ("flight_states", "expected"),
        [
            ((FlightState.CRUISE_CLIMB,), 3),
            ((FlightState.TAXI_OUT,), 0),
            ((FlightState.LOITER,), None),
            ((FlightState.CRUISE_CLIMB, FlightState.CRUISE), 3),
            ((FlightState.TAXI_IN, FlightState.CRUISE), 6),
        ],
    )
    def test_first_idx_of_state(
        self, flight_states: Sequence[FlightState], expected: int
    ) -> None:
        traj = self.get_full_trajectory()
        assert traj.first_idx_of_state(*flight_states) == expected

    @pytest.mark.parametrize(
        ("flight_states", "expected"),
        [
            ((FlightState.CRUISE_DESCENT,), 8),
            ((FlightState.TAXI_IN,), 11),
            ((FlightState.LOITER,), None),
            ((FlightState.CRUISE_CLIMB, FlightState.CRUISE), 7),
            ((FlightState.TAXI_IN, FlightState.CRUISE), 11),
        ],
    )
    def test_last_idx_of_state(
        self, flight_states: Sequence[FlightState], expected: int
    ) -> None:
        traj = self.get_full_trajectory()
        assert traj.last_idx_of_state(*flight_states) == expected


class TestStraightTrajectory(BaseTrajectoryTester):
    """Test the StraightTrajectory class."""

    supports_concatenation_with_time_gap = False
    immutable_arrays = (
        "altitudes",
        "flight_states",
        "horizontal_distances",
        "timestamps",
    )

    def get_trajectory_from_dict(self, dct: dict) -> StraightTrajectory:
        return StraightTrajectory(
            start_datetime=dct["start_time"],
            timestamps=dct["timestamps"],
            flight_states=dct["flight_states"],
            horizontal_distances=dct["horizontal_distances"],
            altitudes=dct["altitudes"],
            gps_start=dct["gps"][0],
            gps_end=dct["gps"][-1],
        )

    def get_required_shapes(
        self, n_timestamps: int
    ) -> dict[str, tuple[int, ...]]:
        return {
            **super().get_required_shapes(n_timestamps),
            "horizontal_distances": (max(n_timestamps - 1, 0),),
            "altitudes": (n_timestamps,),
            "gps_start": (2,),
            "gps_end": (2,),
        }


class TestWaypointTrajectory(BaseTrajectoryTester):
    """Test the WaypointTrajectory class."""

    immutable_arrays = (
        "altitudes",
        "flight_states",
        "gps",
        "timestamps",
    )

    def get_trajectory_from_dict(self, dct: dict) -> WaypointTrajectory:
        return WaypointTrajectory(
            start_datetime=dct["start_time"],
            timestamps=dct["timestamps"],
            flight_states=dct["flight_states"],
            altitudes=dct["altitudes"],
            gps=dct["gps"],
        )

    def get_required_shapes(
        self, n_timestamps: int
    ) -> dict[str, tuple[int, ...]]:
        return {
            **super().get_required_shapes(n_timestamps),
            "horizontal_distances": (max(n_timestamps - 1, 0),),
            "altitudes": (n_timestamps,),
            "gps": (n_timestamps, 2),
        }


BASE_PROFILE = AircraftProfileParameters.model_validate(
    {
        "taxi_out_duration": 30.0,
        "taxi_in_duration": 30.0,
        "transition_duration": 10.0,
        "retransition_duration": 10.0,
        "takeoff_altitude": 15.0,
        "landing_altitude": 12.0,
        "cruise_altitude": 215.0,
        "takeoff_climb_rate": 1.0,
        "landing_descent_rate": 2.0,
        "takeoff_ground_speed": 10.0,
        "landing_ground_speed": 15.0,
        "cruise_climb_rate": 5.0,
        "cruise_climb_ground_speed": 30.0,
        "cruise_descent_rate": 7.0,
        "cruise_descent_ground_speed": 40.0,
        "cruise_speed": 50.0,
        "loiter_speed": 10.0,
    }
)


def test_aircraft_profile_parameters() -> None:
    """Test the AircraftProfileParameters class."""
    profile = BASE_PROFILE
    assert profile.takeoff_duration == 15.0
    assert profile.takeoff_ground_distance == 150.0
    np.testing.assert_almost_equal(profile.takeoff_slope, 15 / 150)
    assert profile.cruise_climb_duration == 40.0
    assert profile.cruise_climb_distance == 1200.0
    np.testing.assert_almost_equal(profile.cruise_climb_slope, 200 / 1200)
    assert profile.cruise_descent_duration == 29.0
    assert profile.cruise_descent_distance == 1160.0
    np.testing.assert_almost_equal(profile.cruise_descent_slope, 203 / 1160)
    assert profile.landing_duration == 6.0
    assert profile.landing_ground_distance == 90.0
    np.testing.assert_almost_equal(profile.landing_slope, 12 / 90)
    assert profile.segment_horizontal_speed(FlightState.CRUISE_CLIMB) == 30.0
    assert profile.segment_horizontal_speed(FlightState.CRUISE_DESCENT) == 40.0
    assert profile.segment_vertical_speed(FlightState.CRUISE_CLIMB) == 5.0
    assert profile.segment_vertical_speed(FlightState.CRUISE_DESCENT) == -7.0
    assert profile.segment_vertical_speed(FlightState.TAKEOFF) == 1.0
    assert profile.segment_vertical_speed(FlightState.LANDING) == -2.0


@pytest.mark.parametrize(
    (
        "distance",
        "altitude_start",
        "altitude_end",
        "kwargs",
        "profile_updates",
        "expected",
    ),
    [
        (  # From takeoff altitude to cruise to landing altitude.
            3010.0,  # Distance
            15.0,  # Altitude start
            12.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 40.0, 53.0, 82.0],
                "horizontal_distances": [1200.0, 650.0, 1160.0],
                "flight_states": [
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE,
                    FlightState.CRUISE_DESCENT,
                ],
                "altitudes": [15.0, 215.0, 215.0, 12.0],
            },
        ),
        (  # Bit after takeoff doesn't reach cruise before descent.
            1000.0,  # Distance
            65.0,  # Altitude start -> 150 required to climb to cruise
            95.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 20.0, 30.0],
                "horizontal_distances": [600.0, 400.0],
                "flight_states": [
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE_DESCENT,
                ],
                "altitudes": [65.0, 165.0, 95.0],
            },
        ),
        (  # Only normal cruise climb.
            600.0,  # Distance
            50.0,  # Altitude start
            150.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 20.0],
                "horizontal_distances": [600.0],
                "flight_states": [FlightState.CRUISE_CLIMB],
                "altitudes": [50.0, 150.0],
            },
        ),
        (  # Circular cruise climb above cruise altitude.
            300.0,  # Distance
            150.0,  # Altitude start
            250.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 20.0],
                "horizontal_distances": [300.0],
                "flight_states": [FlightState.CRUISE_CLIMB],
                "altitudes": [150.0, 250.0],
            },
        ),
        (  # Only normal cruise descent.
            400.0,  # Distance
            200.0,  # Altitude start
            130.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 10.0],
                "horizontal_distances": [400.0],
                "flight_states": [FlightState.CRUISE_DESCENT],
                "altitudes": [200.0, 130.0],
            },
        ),
        (  # Circular cruise descent above cruise altitude.
            200.0,  # Distance
            300.0,  # Altitude start
            230.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 10.0],
                "horizontal_distances": [200.0],
                "flight_states": [FlightState.CRUISE_DESCENT],
                "altitudes": [300.0, 230.0],
            },
        ),
        (  # Climb and cruise above normal cruise altitude.
            1000.0,  # Distance
            200.0,  # Altitude start
            250.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 10.0, 24.0],
                "horizontal_distances": [300.0, 700.0],
                "flight_states": [
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE,
                ],
                "altitudes": [200.0, 250.0, 250.0],
            },
        ),
        (  # Full trajectory above cruise altitude.
            1000.0,  # Distance
            300.0,  # Altitude start
            350.0,  # Altitude end
            {},  # Kwargs
            {},  # Profile updates
            {
                "timestamps": [0.0, 10.0, 24.0],
                "horizontal_distances": [300.0, 700],
                "flight_states": [
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE,
                ],
                "altitudes": [300.0, 350.0, 350.0],
            },
        ),
        (  # Full trajectory with takeoff and landing.
            10_000.0,  # Distance
            0.0,  # Altitude start
            0.0,  # Altitude end
            {  # Get full trajectory.
                "start_datetime": datetime(2024, 11, 29, 12, 30, 25),
                "include_takeoff": True,
                "include_landing": True,
                "include_taxi_out": True,
                "include_taxi_in": True,
            },
            {},  # Profile updates.
            {
                "timestamps": [0, 30, 45, 55, 95, 243, 272, 282, 288, 318],
                "horizontal_distances": [
                    0,
                    150,
                    0,
                    1200,
                    7400,
                    1160,
                    0,
                    90,
                    0,
                ],
                "flight_states": [
                    FlightState.TAXI_OUT,
                    FlightState.TAKEOFF,
                    FlightState.TRANSITION,
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE,
                    FlightState.CRUISE_DESCENT,
                    FlightState.RETRANSITION,
                    FlightState.LANDING,
                    FlightState.TAXI_IN,
                ],
                "altitudes": [0, 0, 15, 15, 215, 215, 12, 12, 0, 0],
            },
        ),
        (  # End above cruise altitude with takeoff and landing
            # and exclude transitions.
            3020.0,  # Distance
            30.0,  # Altitude start
            288.0,  # Altitude end
            {  # Get full trajectory.
                "include_takeoff": True,
                "include_landing": True,
                "elevation_start": 30.0,
                "elevation_end": 288.0,
            },
            {  # Use aircraft without transitions.
                "transition_duration": 0.0,
                "retransition_duration": 0.0,
            },
            {
                "flight_states": [
                    FlightState.TAKEOFF,
                    FlightState.CRUISE_CLIMB,
                    FlightState.CRUISE,
                    FlightState.LANDING,
                ],
                "altitudes": [30.0, 45.0, 300.0, 300.0, 288.0],
                "horizontal_distances": [150.0, 1530.0, 1250.0, 90.0],
                "timestamps": [0.0, 15.0, 66.0, 91.0, 97.0],
            },
        ),
    ],
)
def test_straight_trajectories_cases(
    distance: float,
    altitude_start: float,
    altitude_end: float,
    kwargs: dict[str, object],
    profile_updates: dict[str, float],
    expected: dict[str, float],
) -> None:
    """Test different cases of straight trajectories."""
    gps_start = np.array([53.0, 10.0])
    gps_end = GEODESIC.fwd(*gps_start[::-1], 90.0, distance)[:2][::-1]
    trajectory: StraightTrajectory = generate_straight_trajectory(
        profile=BASE_PROFILE.model_copy(update=profile_updates),
        gps_start=gps_start,
        gps_end=gps_end,
        altitude_start=altitude_start,
        altitude_end=altitude_end,
        **kwargs,
    )
    assert trajectory.start_datetime == kwargs.get("start_datetime")
    assert all(trajectory.gps_start == gps_start)
    assert all(trajectory.gps_end == gps_end)
    np.testing.assert_almost_equal(
        sum(trajectory.horizontal_distances), distance
    )
    assert len(trajectory.flight_states) == len(set(trajectory.flight_states))
    for key, value in expected.items():
        ans = getattr(trajectory, key)
        msg = f"Expected {key} to be {value}, got {ans}"
        if key == "flight_states":
            assert all(ans == value), msg
        else:
            np.testing.assert_allclose(ans, value, err_msg=msg)


@pytest.mark.parametrize(
    ("distance", "include_takeoff", "include_landing"),
    [(50.0, True, False), (50.0, False, True), (120.0, True, True)],
)
def test_straight_trajectory_too_short(
    distance: float, include_takeoff: bool, include_landing: bool
) -> None:
    """Test paths that are too short to takeoff/land."""
    gps_start = np.array([53.0, 10.0])
    gps_end = GEODESIC.fwd(*gps_start[::-1], 90.0, distance)[:2][::-1]
    with pytest.raises(NotImplementedError):
        generate_straight_trajectory(
            profile=BASE_PROFILE,
            gps_start=gps_start,
            gps_end=gps_end,
            altitude_start=0.0,
            altitude_end=0.0,
            include_landing=include_landing,
            include_takeoff=include_takeoff,
        )
