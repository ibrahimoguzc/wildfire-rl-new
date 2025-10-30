# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import (
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from examples.wildfire.fire_model.model import CPUFireModel
from examples.wildfire.fire_model.states import COLOR_TABLE
from examples.wildfire.firefighter_model.model import FirefighterModel
from examples.wildfire.firefighter_model.tactic_pieces.change_tactic import (
    ChangeType,
)
from examples.wildfire.firefighter_model.tactic_pieces.select_poi import (
    SelectPOIType,
)
from examples.wildfire.firefighter_model.tactic_pieces.suppress import (
    SuppressType,
)
from examples.wildfire.firefighter_model.tactic_pieces.track_poi import (
    TrackPOIType,
)
from examples.wildfire.paths import DATA_DIR, FIGURE_DIR, TERRAIN_DIR
from sosid.environment import WildfireEnvironment
from sosid.environment.atmosphere import AtmosphereMathematical
from sosid.environment.environment import ExtendedWildfireEnvironment
from sosid.environment.terrain import (
    CASUALTIES_TABLE,
    COMBUSTIBILITY_TABLE,
    COSTS_TABLE,
    EMISSIONS_TABLE,
    TerrainTypes,
)
from sosid.model.abm.propulsion import (
    BasePropulsionInput,
    BatteryElectricPropulsionInput,
    ConventionalPropulsionInput,
    HybridElectricPropulsionInput,
)
from sosid.model.abm.special_agents import TakeoffLandingType
from sosid.model.abm.trajectory import AircraftProfileParameters
from sosid.model.places import deploy_airport_locations
from sosid.output import Output, TargetKey
from sosid.simulation import (
    BaseModel,
    Distribution,
    PositionInput,
    SimContext,
    Simulation,
    SimulationParameters,
)
from sosid.util.validation import (
    AngleInDegree,
    HourOfDay,
    Percentage,
    check_required_fields,
)


class AgentInput(BaseModel):
    """Define inputs and validators for `SuppressionUAV`."""

    payload: PositiveFloat
    icon: str
    flow_rate: PositiveFloat
    can_scoop: bool
    autonomous: bool
    
    mtom: PositiveFloat
    empty_mass: PositiveFloat
    takeoff_landing_type: TakeoffLandingType
    profile_parameters: AircraftProfileParameters
    propulsion_input: BasePropulsionInput
    # Scooping distance depends on type of aircraft:
    # ['fixed wing': 1341m (typical value)]
    # ['drone or helicopter': Largest length * 2]
    scooping_distance: PositiveFloat
    # Span depends on the type of aircraft
    # ['fixed wing': span * 2]
    # ['drones and helicopters': rotor diameter]
    span: PositiveFloat

    suppression_tactic: SuppressionTacticInput
    ac_type_id: PositiveInt | None = None

    # Required, but set to optional as agents_per_base input is provided
    #  from outside agent.json
    agents_per_base: tuple[int, ...] = ()

    file_name: str = ""
    output_id_name: str = ""

    @model_validator(mode="before")
    @classmethod
    def load_from_file(cls, values: dict) -> dict:
        """Load agent from file if path is supplied."""
        file_name = values.get("file_name")
        if file_name:
            with open(DATA_DIR / "aircraft" / file_name) as f:
                # Only keep values from file if not already specified.
                values = {**json.load(f), **values}
        return values

    @field_validator("output_id_name", mode="after")
    @classmethod
    def default_output_id_name(
        cls, output_id_name: str, info: ValidationInfo
    ) -> str:
        """Set default output id name if not provided."""
        if not output_id_name:
            file_name = info.data.get("file_name", "")
            if not file_name:
                raise ValueError(
                    "Output id name or file name must be provided."
                )
            output_id_name = ".".join(file_name.split(".")[:-1])
        return output_id_name

    @model_validator(mode="after")
    def check_masses(self) -> Self:
        """Check if mass values are consistent."""
        if self.mtom < self.empty_mass:
            msg = (
                f"Maximum takeoff mass ({self.mtom}) is less than empty mass "
                f"({self.empty_mass})."
            )
            raise ValueError(msg)
        match self.propulsion_input:
            case BatteryElectricPropulsionInput():
                propellant_mass = 0.0
            case ConventionalPropulsionInput():
                if (unit := self.propulsion_input.propellant_unit) != "kg":
                    msg = f"Unsupported propellant unit: {unit!r}"
                    raise ValueError(msg)
                propellant_mass = self.propulsion_input.total_propellant
            case HybridElectricPropulsionInput():
                if (
                    unit := self.propulsion_input.conventional.propellant_unit
                ) != "kg":
                    msg = f"Unsupported propellant unit: {unit!r}"
                    raise ValueError(msg)
                propellant_mass = (
                    self.propulsion_input.conventional.total_propellant
                )
            case _:
                msg = (
                    f"Unsupported propulsion input: {self.propulsion_input!r}"
                )
                raise ValueError(msg)
        if self.empty_mass + self.payload + propellant_mass > self.mtom:
            msg = (
                f"Empty mass ({self.empty_mass}), payload mass"
                f" ({self.payload}), and propellant mass ({propellant_mass})"
                f" exceed maximum takeoff mass ({self.mtom})."
            )
            raise ValueError(msg)
        return self


class TacticInput(BaseModel):
    select_poi: SelectPOIType = SelectPOIType.WATER
    track_poi: TrackPOIType = TrackPOIType.FOLLOW_FIREFRONT
    suppress: SuppressType = SuppressType.DIRECT


class AlternativeInput(BaseModel):
    # A numerical value used by the classes that inheret "ChangeType" in
    # order to parametrize the condition
    threshold: int | tuple[HourOfDay, HourOfDay] | None = None
    # The class that defines the condition of changing tactics
    change_condition: ChangeType = ChangeType.NO_CHANGE
    # The definition of the tactic to switch to
    alternative_tactic: TacticInput = TacticInput()


class SuppressionTacticInput(BaseModel):
    """The suppression tactic input will be composed of a main tactic,
    that the agetn starts with, and an alternative that can be changed
    based on some condition, during the mission.

    Args:
        BaseModel (_type_): _description_
    """

    main: TacticInput
    alternative: AlternativeInput | None = AlternativeInput()


class AirportInput(PositionInput):
    """Define inputs and validators for `AirTrafficManager`."""

    icon: str = "helipad_white.svg"
    takeoff_landing_types: (
        TakeoffLandingType | tuple[TakeoffLandingType, ...]
    ) = tuple(tp for tp in TakeoffLandingType)


class IgnitionCenterInput(PositionInput):
    """Define inputs and validators for `IgnitionCenter`."""


class ProtectionLocationInput(PositionInput):
    """Define inputs and validators for `ProtectionLocation`."""


class UrbanLocationInput(PositionInput):
    """Define inputs and validators for `UrbanLocation`."""

    radius: tuple[PositiveFloat, PositiveFloat]
    angle: AngleInDegree = 0


class WaterSourceInput(PositionInput):
    """Define inputs and validators for `WaterSource`."""


class AtmosphereParameters(BaseModel):
    """Base class for parameters classes regarding the atmosphere."""


class AtmosphereParametersMathematical(AtmosphereParameters):
    # Atmosphere user input
    # in °C (min/max) // min next day
    temperature_range: tuple[float, float, float]
    # times when temperature is lowest/highest
    temperature_times: tuple[HourOfDay, HourOfDay]
    # sun[0] = sunrise // sun[1] = sunset in hrs
    sun_times: tuple[HourOfDay, HourOfDay]
    # Time sun is at the highest peak in hrs
    time_of_max_solar_height: PositiveFloat  # Location specific
    # minimum and maximum relative humidity in %
    humidity_range: tuple[Percentage, Percentage]
    # total windrun of the day in km/d
    wind_run: PositiveFloat
    # overall tendency of winddirection
    general_winddirection: AngleInDegree
    # range in which the wind direction varies randomly
    range_winddirection: AngleInDegree
    # frequency with which to update the atmospheric parameteres
    update_frequency: PositiveInt  # SI seconds

    @field_validator("temperature_range")
    @classmethod
    def check_temperature_range(cls, value) -> tuple[float, float, float]:
        """Check if temperature range is valid."""
        min_temp, max_temp, min_temp_next_day = value
        assert min_temp <= max_temp, (
            f"Minimum temp input = {min_temp}, is"
            f"lower than Max temp = {max_temp}"
        )
        assert min_temp_next_day <= max_temp, (
            f"Min temp of following day ="
            f"{min_temp_next_day}, is less than Max temp = {max_temp}"
        )
        return value

    @field_validator("temperature_times", "sun_times")
    @classmethod
    def check_times(
        cls,
        value,
        info: ValidationInfo,
    ) -> tuple[PositiveFloat, PositiveFloat]:
        """Check if time range is valid."""
        lower, higher = value
        assert (  # noqa: S101
            lower < higher
        ), f"Max is lower than min value of {info.field_name}"
        return value

    @field_validator("humidity_range")
    @classmethod
    def check_humidity_range(
        cls,
        value: tuple[Percentage, Percentage],
    ) -> tuple[Percentage, Percentage]:
        """Check if humidity range is valid."""
        min_h, max_h = value
        assert (  # noqa: S101
            min_h <= max_h
        ), f"Max humidity ({max_h}) lower than Min humidity ({min_h})"
        return value


class _TerrainParametersCache:
    """Cache for terrain parameters metadata."""

    def __new__(cls) -> None:
        raise RuntimeError("This class should not be instantiated.")

    metadata: dict[str, Any] = {}


class TerrainParameters(BaseModel):
    """Define inputs and validators for `WildfireEnvironment`."""

    # Environment Variables
    # To use base file (import_osm=False) change the elevation file name
    file_namespace: str
    import_features_osm: bool  # Defined last to check if features is input
    # Dimension of a square cell (should match terrain_file); SI: meter
    cell_size: PositiveInt

    grid_shape: tuple[PositiveInt, PositiveInt] = ()  # Dynamic default
    height_scale_factor: (
        PositiveFloat  # Scale to multiply terrain elevation values with
    )
    # Sets the strength of the blend for priority map
    # A stronger sigma corresponds to a larger area
    priority_map_sigma: int

    @model_validator(mode="before")
    def warn_removed_filename_fields(cls, values: dict) -> dict:
        """Warn if deprecated fields are used."""
        for field in (
            "elevation_file_name",
            "features_file_name",
            "water_sources_file_name",
        ):
            if field in values:
                print(
                    f"'{field}' has been removed and is now specified based on"
                    f" the file_namespace.",
                )
        return values

    @field_validator("grid_shape", mode="before")
    @classmethod
    def default_grid_shape(
        cls,
        value: tuple[PositiveInt, PositiveInt],
        info: ValidationInfo,
    ) -> tuple[PositiveInt, PositiveInt]:
        """Set default grid shape."""
        check_required_fields("file_namespace", info.data)
        if not value:
            file_name = (
                TERRAIN_DIR / f"{info.data['file_namespace']}_elevation.npy"
            )
            with open(file_name, "rb") as f:
                np.lib.format.read_magic(f)  # condense file header
                value, _, _ = np.lib.format.read_array_header_1_0(f)
        return value

    @property
    def elevation_file_name(self) -> str:
        return f"{self.file_namespace}_elevation.npy"

    @property
    def features_file_name(self) -> str:
        return f"{self.file_namespace}_terrain_features.npy"

    @property
    def water_sources_file_name(self) -> str:
        return f"{self.file_namespace}_water_sources.pkl"

    @property
    def elevation_file(self) -> Path:  # noqa D102
        return TERRAIN_DIR / self.elevation_file_name

    @property
    def features_file(self) -> Path:  # noqa D102
        return TERRAIN_DIR / self.features_file_name

    @property
    def water_sources_file(self) -> Path:  # noqa D102
        return TERRAIN_DIR / self.water_sources_file_name

    @property
    def meta_file(self) -> Path:
        return TERRAIN_DIR / (self.features_file_name.split("_")[0] + ".meta")

    @cached_property
    def meta_data(self):
        if not _TerrainParametersCache.metadata:
            with open(self.meta_file) as file:
                _TerrainParametersCache.metadata = json.load(file)
        return _TerrainParametersCache.metadata

    @property
    def water_map_multiplier(self):
        return self.meta_data.get("water_map_multiplier")

    @property
    def coordinates(self):
        return self.fire_map_coordinates

    @cached_property
    def fire_map_coordinates(self):
        coordinates = np.array(self.meta_data["coordinates"])
        lats, lons = coordinates[:, 0], coordinates[:, 1]
        lats, lons = sorted(lats), sorted(lons)
        coordinates = [[lats[1], lons[0]], [lats[0], lons[1]]]
        return coordinates

    @property
    def water_map_coordinates(self):
        return self.meta_data["water_map_coordinates"]

    @property
    def water_map_grid_shape(self):
        """Grid shape where the map in map water sources are considered."""
        water_map_grid_shape_h = self.grid_shape[0] * self.water_map_multiplier
        water_map_grid_shape_w = self.grid_shape[1] * self.water_map_multiplier
        return (water_map_grid_shape_h, water_map_grid_shape_w)

    @property
    def water_map_dimensions(self):
        """Larger map dimensions with the prop transform of w and h."""
        water_map_dimensions_w = self.water_map_grid_shape[1] * self.cell_size
        water_map_dimensions_h = self.water_map_grid_shape[0] * self.cell_size
        return (water_map_dimensions_w, water_map_dimensions_h)

    @field_validator("grid_shape")
    @classmethod
    def check_width_is_multiple_of_four(
        cls, value
    ) -> tuple[PositiveInt, PositiveInt]:
        """Verify dim of input eval file to be multiples of four."""
        assert value[1] % 4 == 0, (
            f"Elevation file width {value[1]} is not divisible by 4, which is "
            f"incompatible with fire_models IndexedImageItem representation"
        )
        return value

    @model_validator(mode="after")
    def check_cell_size_matches_file(self) -> Self:
        """Verify input cell size to match with inputted stored file."""
        cell_size_from_file = int(
            self.elevation_file_name.split("_")[2].split("m")[0]
        )
        assert self.cell_size == cell_size_from_file, (
            f"Input cell size {self.cell_size} does not match with inputted "
            f"file value: {cell_size_from_file} \n"
        )
        return self


class ExtendedTerrainParameters(TerrainParameters):
    """Define inputs and validators for `ExtendedWildfireEnvironment`."""

    file_namespace_operational: str

    @model_validator(mode="before")
    def warn_removed_filename_fields(cls, values: dict) -> dict:
        super().warn_removed_filename_fields(values)
        for field in (
            "operational_elevation_file_name",
            "operational_image_file_name",
            "airport_locations_file_name",
        ):
            if field in values:
                print(
                    f"'{field}' has been removed and is now specified based on"
                    f" the file_namespace_operational.",
                )
        return values

    @property
    def operational_elevation_file_name(self) -> str:
        return f"{self.file_namespace_operational}_operational_grid.npy"

    @property
    def operational_image_file_name(self) -> str:
        return f"{self.file_namespace_operational}_operational_image.npy"

    @property
    def airport_locations_file_name(self) -> str:
        return f"{self.file_namespace_operational}_airport_locations.pkl"

    @cached_property
    def operational_elevation_file(self) -> Path:
        return TERRAIN_DIR / self.operational_elevation_file_name

    @cached_property
    def operational_image_file(self) -> Path:
        return TERRAIN_DIR / self.operational_image_file_name

    @cached_property
    def airport_locations_file(self) -> Path:
        return TERRAIN_DIR / self.airport_locations_file_name

    @cached_property
    def operational_map_coordinates(self):
        return self.meta_data["operational_map_retrieved_coordinates"]

    @cached_property
    def operational_terrain_cell_size(self):
        return self.meta_data["operational_terrain_cell_size"]

    @cached_property
    def operational_grid_shape(self):
        return self.meta_data["operational_grid_shape"]

    @cached_property
    def coordinates(self):
        return self.operational_map_coordinates


class WildfireParameters(SimulationParameters):
    """Define inputs and validators for `WildfireSimulation`."""

    # Simulation Parameters
    run_headless: bool  # Lets you run the simulation headless or with no gui
    name: str  # identifier of the simulation
    time_step: PositiveFloat
    mission_start: datetime
    max_runtime: PositiveInt
    output_sampling_time: PositiveInt

    # Map in Map Enabled
    map_in_map: bool
    export_img: bool

    # Environment Parameters
    terrain_inputs: TerrainParameters

    # Simulation Control Parameters
    reset_available: bool = True

    @field_validator("terrain_inputs", mode="before")
    @classmethod
    def initialize_terrain_inputs(
        cls,
        terrain_inputs: dict,
        info: ValidationInfo,
    ) -> TerrainParameters:
        """Validate Environment inputs."""
        check_required_fields("map_in_map", info.data)
        map_in_map = info.data["map_in_map"]
        if map_in_map:
            terrain_inputs = ExtendedTerrainParameters(**terrain_inputs)
        else:
            terrain_inputs = TerrainParameters(**terrain_inputs)
        return terrain_inputs

    max_elevation: float  # SI meter
    # weather API diversion and time frame variables
    run_api_for_atmosphere: bool  # True --> run api; False --> run model
    atmosphere_inputs: AtmosphereParameters

    @field_validator("atmosphere_inputs", mode="before")
    @classmethod
    def initialize_atmosphere_inputs(
        cls, atmosphere_inputs: dict, info: ValidationInfo
    ) -> AtmosphereParameters:
        """Validate Atmosphere inputs."""
        check_required_fields("run_api_for_atmosphere", info.data)
        run_api_for_atmosphere = info.data["run_api_for_atmosphere"]
        if run_api_for_atmosphere:
            raise NotImplementedError("API not allowed in X-Challenge")
        return AtmosphereParametersMathematical(**atmosphere_inputs)

    # Fire-Model Variables:
    enable_adaptive_time_step: bool = True
    correction_coefficient: float
    # IMP: Ignition center is defined in fire map coordinate system (relative
    # to origin fire map, not operational map)
    ignition_centers: tuple[IgnitionCenterInput, ...]

    # Infrastructure Systems
    deploy_osm_airport_locations: bool
    k_nearest_airports: int
    airports: tuple[AirportInput, ...]

    @field_validator("airports", mode="before")
    @classmethod
    def load_airports(
        cls,
        airports: tuple[AirportInput, ...],
        info: ValidationInfo,
    ) -> tuple[AirportInput, ...]:
        """Overwrite airports is overwrite is set to True."""
        check_required_fields(
            ["deploy_osm_airport_locations", "map_in_map"], info.data
        )
        if (
            info.data["deploy_osm_airport_locations"]
            and info.data["map_in_map"]
        ):
            airports = tuple(
                deploy_airport_locations(
                    info.data, list(airports), AirportInput
                )
            )
        return airports

    deploy_osm_waters: bool
    water_sources: tuple[WaterSourceInput, ...]
    protection_locations: tuple[ProtectionLocationInput, ...]
    urban_locations: tuple[UrbanLocationInput, ...]
    response_time: float | Distribution
    takeoff_interval: PositiveFloat  # SI second
    turnaround_time: PositiveFloat  # SI second

    suppression_altitude: PositiveFloat  # SI meters
    resupply_altitude: PositiveFloat  # SI meters

    # Flight Vehicle System:
    enable_nighttime_operations: bool
    agents: tuple[AgentInput, ...]

    scoop_time: PositiveFloat  # SI second

    # Firefront selection cost weights
    distance_cost_weight: float
    vip_cost_weight: float
    priority_cost_weight: float
    vegetation_cost_weight: float
    topography_cost_weight: float

    @property
    def cell_size(self):
        return self.terrain_inputs.cell_size

    @property
    def height_scale_factor(self):
        return self.terrain_inputs.height_scale_factor

    @field_validator("ignition_centers")
    @classmethod
    def check_ignition_center_in_bbox(
        cls,
        ignition_centers: tuple[IgnitionCenterInput, ...],
        info: ValidationInfo,
    ) -> tuple[IgnitionCenterInput, ...]:
        """Verify input ignition center is within map boundaries."""
        check_required_fields(
            ["cell_size", "elevation_file_name", "fire_map_coordinates"],
            info.data,
            "terrain_inputs",
        )
        coordinates = info.data["terrain_inputs"].fire_map_coordinates
        cell_size = info.data["terrain_inputs"].cell_size
        grid_shape = info.data["terrain_inputs"].grid_shape
        for ignition_center in ignition_centers:
            ignition_center.check_in_bbox(
                bbox_pos=[
                    [0, 0],
                    [grid_shape[0] * cell_size, grid_shape[1] * cell_size],
                ],
                bbox_gps=coordinates,
            )
        return ignition_centers

    @field_validator("airports")
    @classmethod
    def check_airports_in_bbox(
        cls,
        airports: tuple[AirportInput, ...],
        info: ValidationInfo,
    ) -> tuple[AirportInput, ...]:
        """Verify input airport locations are within map boundaries."""
        check_required_fields(
            ["cell_size", "grid_shape", "coordinates"],
            info.data,
            "terrain_inputs",
        )
        coordinates = info.data["terrain_inputs"].coordinates
        cell_size = info.data["terrain_inputs"].cell_size
        grid_shape = info.data["terrain_inputs"].grid_shape
        for airport in airports:
            airport.check_in_bbox(
                bbox_pos=[
                    [0, 0],
                    [grid_shape[0] * cell_size, grid_shape[1] * cell_size],
                ],
                bbox_gps=coordinates,
            )
        return airports

    @field_validator("water_sources")
    @classmethod
    def check_water_sources_in_bbox(
        cls,
        water_sources: tuple[WaterSourceInput, ...],
        info: ValidationInfo,
    ) -> tuple[WaterSourceInput, ...]:
        """Verify input water sources are within map boundaries."""
        check_required_fields("map_in_map", info.data)
        if info.data["map_in_map"]:
            # Use operational terrain
            check_required_fields(
                ["operational_grid_shape", "operational_terrain_cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data[
                "terrain_inputs"
            ].operational_terrain_cell_size
            grid_shape = info.data["terrain_inputs"].operational_grid_shape
        else:
            check_required_fields(
                ["grid_shape", "cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data["terrain_inputs"].cell_size
            grid_shape = info.data["terrain_inputs"].grid_shape
        check_required_fields("coordinates", info.data, "terrain_inputs")
        coordinates = info.data["terrain_inputs"].coordinates
        for water_source in water_sources:
            water_source.check_in_bbox(
                bbox_pos=[
                    [0, 0],
                    [grid_shape[0] * cell_size, grid_shape[1] * cell_size],
                ],
                bbox_gps=coordinates,
            )

        return water_sources

    @field_validator("protection_locations")
    @classmethod
    def check_protection_locations_in_bbox(
        cls,
        protection_locations: tuple[ProtectionLocationInput, ...],
        info: ValidationInfo,
    ) -> tuple[ProtectionLocationInput, ...]:
        """Verifies input protection locations are within map bounds."""
        check_required_fields("map_in_map", info.data)
        if info.data["map_in_map"]:
            # Use operational terrain
            check_required_fields(
                ["operational_grid_shape", "operational_terrain_cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data[
                "terrain_inputs"
            ].operational_terrain_cell_size
            grid_shape = info.data["terrain_inputs"].operational_grid_shape
        else:
            check_required_fields(
                ["grid_shape", "cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data["terrain_inputs"].cell_size
            grid_shape = info.data["terrain_inputs"].grid_shape

        check_required_fields("coordinates", info.data["terrain_inputs"])
        coordinates = info.data["terrain_inputs"].coordinates
        for protection_location in protection_locations:
            protection_location.check_in_bbox(
                bbox_pos=[
                    [0, 0],
                    [grid_shape[0] * cell_size, grid_shape[1] * cell_size],
                ],
                bbox_gps=coordinates,
            )
        return protection_locations

    @field_validator("urban_locations")
    @classmethod
    def check_urban_locations_in_bbox(
        cls,
        urban_locations: tuple[UrbanLocationInput, ...],
        info: ValidationInfo,
    ) -> tuple[ProtectionLocationInput, ...]:
        """Verifies input protection locations are within map bounds."""
        check_required_fields("map_in_map", info.data)
        if info.data["map_in_map"]:
            # Use operational terrain
            check_required_fields(
                ["operational_grid_shape", "operational_terrain_cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data[
                "terrain_inputs"
            ].operational_terrain_cell_size
            grid_shape = info.data["terrain_inputs"].operational_grid_shape
        else:
            check_required_fields(
                ["grid_shape", "cell_size"],
                info.data,
                "terrain_inputs",
            )
            cell_size = info.data["terrain_inputs"].cell_size
            grid_shape = info.data["terrain_inputs"].grid_shape

        check_required_fields("coordinates", info.data["terrain_inputs"])
        coordinates = info.data["terrain_inputs"].coordinates
        for location in urban_locations:
            location.check_in_bbox(
                bbox_pos=[
                    [0, 0],
                    [grid_shape[0] * cell_size, grid_shape[1] * cell_size],
                ],
                bbox_gps=coordinates,
            )
        return urban_locations

    @field_validator("agents")
    @classmethod
    def check_number_of_agents_per_base(
        cls,
        agents: tuple[AgentInput, ...],
        info: ValidationInfo,
    ) -> tuple[ValidationInfo, ...]:
        """Verify input length matches number of bases and agents."""
        check_required_fields("airports", info.data)
        n_airports = len(info.data["airports"])
        for i, agent in enumerate(agents):
            assert len(agent.agents_per_base) == n_airports, (
                f"Length of agents_per_base input "
                f"({len(agent.agents_per_base)}) for agent #{i} does not "
                f"the match number of bases ({n_airports})"
            )
        return agents


ACRE_IN_M2 = 4048.86
CLASS_1 = 0.25 * ACRE_IN_M2
CLASS_2 = 10 * ACRE_IN_M2
CLASS_3 = 100 * ACRE_IN_M2
CLASS_4 = 300 * ACRE_IN_M2
CLASS_5 = 1000 * ACRE_IN_M2
CLASS_6 = 5000 * ACRE_IN_M2

PEOPLE_PER_HOUSEHOLD = 2
M2_TO_HECTARS = 1 / 1e4
CO2_PER_KEROSENE_KG = 3.15


class WildfireSimulation(Simulation[WildfireParameters, SimContext]):
    """Combines all models for a wildfire simulation."""

    def __init__(
        self,
        parameters: WildfireParameters,
        seed: int | None = 0,
        context: SimContext | None = None,
    ):
        # Superclass simulation evaluates any probability
        # distributions within parameters on initialization
        super().__init__(parameters=parameters, seed=seed, context=context)
        self.init_states = locals().copy()
        self.init_states.pop("self")
        self.init_states.pop("__class__")
        self.atmosphere = AtmosphereMathematical(self)

        if self.parameters.map_in_map:
            self.environment = ExtendedWildfireEnvironment(
                simulation=self,
                terrain_parameters=self.parameters.terrain_inputs,
                atmosphere=self.atmosphere,
            )  # type:ignore
        else:
            self.environment = WildfireEnvironment(
                simulation=self,
                terrain_parameters=self.parameters.terrain_inputs,
                atmosphere=self.atmosphere,
            )  # type:ignore
        self.wildfire = CPUFireModel(simulation=self)
        self.firefighters = FirefighterModel(simulation=self)
        self.models: list = [
            self.wildfire,
            self.firefighters,
        ]
        self.ignition_centers = self.firefighters.ignition_centers

    def start(self) -> None:  # noqa D102
        self.wildfire.ignite(self.ignition_centers)
        super().start()

    def stop(self) -> None:
        """Stops the simulation and prints useful output info."""
        super().stop()
        print(
            "Total cost of burnt area : ",
            self.total_fire_cost,
            "Euros\nTotal emissions of area burned : ",
            self.total_fire_emissions,
            "tons of CO2\nTotal casualties for area burned : ",
            self.total_casualties,
            "people\nTotal burnt area :",
            self.wildfire.burnt_area,
            "m^2\nTime to complete mission (excluding response time):",
            self.timer.mission_runtime
            - timedelta(seconds=self.parameters.response_time),
            "\nResponse_time :",
            timedelta(seconds=self.parameters.response_time),
            "\nSimulation_runtime :",
            self.timer.runtime,
        )
        if self.parameters.export_img:
            self.export_env_image()

    def reset(self, seed=None) -> None:
        """Resets the model to its initial state."""
        # Reinitialize the environment
        self.environment.__init__(
            simulation=self,
            terrain_parameters=self.parameters.terrain_inputs,
            atmosphere=self.atmosphere,
        )
        super().reset()  # Reset base class components (timer, flags, etc.)

    def area_burnt_by_type(self, terrain_type: TerrainTypes):
        match terrain_type:
            case TerrainTypes.NEEDLE_LITTER:
                return self.litter_burnt_area
            case TerrainTypes.FALLEN_LEAVES:
                return self.leaves_burnt_area
            case TerrainTypes.GRASSES_WEEDS:
                return self.grass_burnt_area
            case TerrainTypes.CAREX_FORBS:
                return self.carex_forbs_burnt_area
            case TerrainTypes.PASTURE:
                return self.pasture_burnt_area
            case TerrainTypes.PINUS:
                return self.pinus_burnt_area
            case TerrainTypes.FIELD:
                return self.fields_burnt_area
            case TerrainTypes.RESIDENTIAL:
                return self.residential_burnt_area
            case _:
                return 0

    @property
    def agent_methods(self):  # noqa D102
        return self.firefighters.agents[0]

    @Output(target_key=TargetKey.SIMULATION)
    def effective_mission_time(self):  # noqa D102
        return (
            self.timer.mission_runtime.total_seconds()
            - self.parameters.response_time
        )

    @Output(target_key=TargetKey.SIMULATION)
    def mission_success(self):  # noqa D102
        return False if (self.wildfire.fire_positions.any()) else True

    @Output(target_key=TargetKey.SIMULATION)
    def burnt_area(self):  # noqa D102
        return self.wildfire.burnt_area

    @Output(target_key=TargetKey.SIMULATION)
    def residential_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.RESIDENTIAL]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def pinus_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.PINUS]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def litter_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.NEEDLE_LITTER]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def leaves_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.FALLEN_LEAVES]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def grass_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.GRASSES_WEEDS]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def carex_forbs_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.CAREX_FORBS]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def pasture_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.PASTURE]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def fields_burnt_area(self):
        return self.wildfire.burnt_area_for_combustibility(
            COMBUSTIBILITY_TABLE[TerrainTypes.FIELD]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def fire_size_class(self):  # noqa D102
        burnt_area = self.wildfire.burnt_area
        if burnt_area < CLASS_1:
            return 1
        if burnt_area < CLASS_2:
            return 2
        if burnt_area < CLASS_3:
            return 3
        if burnt_area < CLASS_4:
            return 4
        if burnt_area < CLASS_5:
            return 5
        if burnt_area < CLASS_6:
            return 6
        return 7

    @Output(target_key=TargetKey.SIMULATION)
    def sampled_times(self):  # noqa D102
        return self.wildfire.sampled_times

    @Output(target_key=TargetKey.SIMULATION)
    def burnt_area_samples(self):  # noqa D102
        return self.wildfire.burnt_area_samples

    @Output(target_key=TargetKey.SIMULATION)
    def total_burn_time(self):  # noqa D102
        return self.timer.mission_runtime.total_seconds()

    @Output(target_key=TargetKey.SIMULATION)
    def effective_mission_time(self):  # noqa D102
        return (
            self.timer.mission_runtime.total_seconds()
            - self.parameters.response_time
        )

    @Output(target_key=TargetKey.SIMULATION)
    def total_network_electric_energy(self) -> float:
        """Total electric energy consumed by all agents."""
        return sum(
            agent.total_electric_energy_consumed
            for agent in self.firefighters.firefighters
        )

    @Output(target_key=TargetKey.SIMULATION)
    def total_network_energy(self) -> float:
        """Total energy consumed by all agents in the network."""
        return sum(
            agent.total_energy_consumed
            for agent in self.firefighters.firefighters
        )

    @Output(target_key=TargetKey.SIMULATION)
    def total_network_fuel(self) -> float:
        """Total fuel mass consumed by all agents in the network."""
        return sum(
            agent.total_propellant_mass_consumed
            for agent in self.firefighters.firefighters
        )

    @Output(target_key=TargetKey.SIMULATION)
    def total_co2_fuel(self) -> float:
        """Total CO2 emission from network fuel in tons.

        Based on fuel burn of kerosene.
        """
        return self.total_network_fuel * CO2_PER_KEROSENE_KG / 1000

    @Output(target_key=TargetKey.SIMULATION)
    def n_bases(self):  # noqa D102
        return len(self.firefighters.air_traffic_managers)

    @Output(target_key=TargetKey.SIMULATION)
    def n_agents(self):  # noqa D102
        return len(self.firefighters.firefighters)

    @Output(target_key=TargetKey.SIMULATION)
    def n_water_sources(self):  # noqa D102
        return len(self.firefighters.water_sources)

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_average_reenergizations(self):  # noqa D102
        return self.fleet_total_propellant_refills / self.n_agents

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_average_suppressions(self):  # noqa D102
        return self.fleet_total_suppressions / self.n_agents

    @Output
    def fleet_total_propellant_refills(self):  # noqa D102
        return sum(
            agent.propulsion.n_propellant_refills
            for agent in self.firefighters.firefighters
        )

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_total_suppressions(self):  # noqa D102
        fleet_suppressions = 0
        for agent in self.firefighters.firefighters:
            fleet_suppressions += agent.total_suppressions
        return fleet_suppressions

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_average_distance_flown(self):  # noqa D102
        return self.fleet_total_distance_flown / self.n_agents

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_total_distance_flown(self):  # noqa D102
        fleet_distance_flown = 0
        for agent in self.firefighters.firefighters:
            fleet_distance_flown += agent.distance_flown
        return fleet_distance_flown

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_average_cumulative_flight_time(self) -> float:
        """Average cumulative flight time [s] per firefighter."""
        return self.fleet_cumulative_flight_time / self.n_agents

    @Output(target_key=TargetKey.SIMULATION)
    def fleet_cumulative_flight_time(self) -> float:
        """Total cumulative flight time [s] for all firefighters."""
        fleet_time_flown = 0
        for agent in self.firefighters.firefighters:
            fleet_time_flown += agent.cumulative_flight_time
        return fleet_time_flown

    @property
    def atm_pos(self):  # noqa D102
        return np.array(
            [agent.pos for agent in self.firefighters.air_traffic_managers]
        )

    @Output(target_key=TargetKey.SIMULATION)
    def min_distance_airport_to_fire(self) -> float:  # noqa D102
        min_distance = math.inf
        for center in self.ignition_centers:
            _, distance = self.agent_methods.nearest_position(
                self.atm_pos, center.pos
            )
            min_distance = min(min_distance, distance)
        return min_distance

    @Output(target_key=TargetKey.SIMULATION)
    def max_distance_airport_to_fire(self) -> float:  # noqa D102
        max_distance = -math.inf
        for center in self.ignition_centers:
            distances = self.agent_methods.distance(center.pos, self.atm_pos)
            max_idx = np.argmax(distances)
            farthest_distance = distances[max_idx]
            max_distance = max(max_distance, farthest_distance)
        return max_distance

    @Output(target_key=TargetKey.SIMULATION)
    def min_distance_water_to_fire(self) -> float:  # noqa D102
        min_distance = math.inf
        for center in self.ignition_centers:
            if len(self.firefighters.water_sources) != 0:
                water_source_pos = np.array(
                    [agent.pos for agent in self.firefighters.water_sources]
                )
                _, distance = self.agent_methods.nearest_position(
                    water_source_pos, center.pos
                )
            else:
                distance = math.inf
            min_distance = min(min_distance, distance)
        return min_distance

    @Output(target_key=TargetKey.SIMULATION)
    def min_distance_resupply_to_fire(self) -> float:  # noqa D102
        min_airport_distance = self.min_distance_airport_to_fire
        min_water_distance = self.min_distance_water_to_fire
        if min_airport_distance < min_water_distance:
            return min_airport_distance
        return min_water_distance

    @Output(target_key=TargetKey.SIMULATION)
    def min_resupply_loc_is_water(self) -> bool:  # noqa D102
        min_water_distance = self.min_distance_water_to_fire
        min_resupply_distance = self.min_distance_resupply_to_fire
        return min_resupply_distance == min_water_distance

    @Output(target_key=TargetKey.SIMULATION)
    def average_distance_airport_to_fire(self):  # noqa D102
        average_distance = 0
        for center in self.ignition_centers:
            distances = self.agent_methods.distance(center.pos, self.atm_pos)
            average_distance += np.average(distances)
        average_distance /= len(self.ignition_centers)
        return average_distance

    @Output
    def total_fire_cost(self) -> float:
        """Return cost of burnt area in Euros."""
        total_cost = 0

        for terrain_type in TerrainTypes:
            total_cost += (
                COSTS_TABLE[terrain_type]
                * self.area_burnt_by_type(terrain_type)
                * M2_TO_HECTARS
            )

        return round(total_cost, 2)

    @Output
    def total_fire_emissions(self) -> float:
        """Return emissions cost in tons of CO2."""
        total_emissions = 0

        for terrain_type in TerrainTypes:
            total_emissions += (
                EMISSIONS_TABLE[terrain_type]
                * self.area_burnt_by_type(terrain_type)
                * M2_TO_HECTARS
            )

        return round(total_emissions, 2)

    @Output
    def total_casualties(self) -> float:
        """Return the number of casualties."""
        total_casualties = 0

        for terrain_type in TerrainTypes:
            total_casualties += (
                CASUALTIES_TABLE[terrain_type]
                * self.area_burnt_by_type(terrain_type)
                * PEOPLE_PER_HOUSEHOLD
                * M2_TO_HECTARS
            )

        return int(total_casualties)

    def export_env_image(
        self, img_path: str | None = None, img_name: str | None = None
    ) -> None:
        """Exports a '.png' image of the simulation environment state.

        Args:
            img_path: An optional py:type:`str` sepcifying the path of
                the image file to be exported. The file path defaults to
                the `figures` directory of the SoSID example.
            img_name: An optional py:type:`str` sepcifying the name of
                the image file to be exported. The file name defaults to
                the time and date of the export in 'hhmmss_DDMMYYYY'
                format.
        """
        fire_states = self.wildfire.fire_states
        fire_states = np.array(
            [list(map(COLOR_TABLE.__getitem__, row)) for row in fire_states]
        )
        color_map = self.environment.terrain.features.colormap
        hillshade = self.environment.terrain.elevation.hillshade_matplotlib()
        hillshade = np.reshape(np.repeat(hillshade, 4), np.shape(color_map))

        fire_img = Image.fromarray(np.uint8(fire_states))
        features_img = Image.fromarray(np.uint8(color_map * hillshade))

        img = Image.alpha_composite(features_img, fire_img)

        default_path = FIGURE_DIR
        default_name = (datetime.now()).strftime("%H%M%S_%m%d%Y") + ".png"

        path = img_path if img_path else default_path
        name = img_name + ".png" if img_name else default_name

        img.save(path / name)

    def step_all_models(self) -> None:
        """Enable adaptive time stepping for Wildfire model.

        If adaptive time step is enabled, it creates the task flow of
        the two models. The method considers the given fixed-time step
        for the ABM and an adaptive time step for the fire model, which
        is computed dynamically (in each iteration based on the fire
        spread rates).

        Otherwise, if adaptive time step is disabled, it advances all
        models by a single :py:attr:`time_step`.
        """
        if self.parameters.enable_adaptive_time_step:
            if self.iterations == 0:
                # If it is the first iteration, run both models
                [model.step() for model in self.models]

            elif self.wildfire.model_time_step > self.time_step:
                # If Wildfire model has greater timestep than
                # ABM run both models with the same timestep
                self.wildfire.model_time_step = self.time_step
                [model.step() for model in self.models]
            else:
                # If Fire model ideal timestep lower than ABM,
                # let fire model run adaptively
                while (
                    self.wildfire.internal_model_time
                    < self.firefighters.time_at_next_iter
                ):
                    # Set upper limit on timestep, actual time step is
                    # computed within fire model
                    self.wildfire.model_time_step = (
                        self.fire_model_max_time_step
                    )
                    self.wildfire.step()
                self.firefighters.step()
        else:
            super().step_all_models()

    @property
    def fire_model_max_time_step(self) -> timedelta:
        """Fire model time step upper threshold for next iteration."""
        fire_model_time_at_next_iter = (
            self.wildfire.internal_model_time + self.wildfire.model_time_step
        )
        if fire_model_time_at_next_iter > self.firefighters.time_at_next_iter:
            # Ensure fire model does not surpass ABM model time
            return (
                self.firefighters.time_at_next_iter
                - self.wildfire.internal_model_time
            )
        return self.wildfire.model_time_step
