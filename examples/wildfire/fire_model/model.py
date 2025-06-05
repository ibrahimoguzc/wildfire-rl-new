# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property, lru_cache

import numpy as np
from PyQt5 import QtGui

from examples.wildfire.fire_model.jit_funcs.cpu import MOORE_RADIUS
from examples.wildfire.fire_model.jit_funcs.cpu import step as step_cpu
from examples.wildfire.fire_model.states import (
    BURNT,
    COLOR_TABLE,
    COMBUSTIBLE,
    EARLY_BURNING,
    EXTINGUISHING,
    FULL_BURNING,
    NONFLAMMABLE,
    SUPPRESSED,
)
from sosid.environment.terrain import COMBUSTIBILITY_TABLE
from sosid.gui.items import IndexedImageItem
from sosid.model.ca import CellularAutomataModel, GridData, RasterizedShape
from sosid.model.ca.neighborhood import MooreNeighborhood
from sosid.model.transform import index_to_pos, pos_to_index
from sosid.typedef import Position

# TODO enable method to update array values, on mask, new values,
# TODO test if there is any speed improvement form using Numba classdef


@dataclass(frozen=True)
class CPUFireData(GridData):
    # User-Assignable Values
    temperatures: np.float64 = 25
    wind_speeds: np.float64 = 5
    wind_aspects: np.float64 = 0
    humidities: np.float64 = 30
    terrain_slopes: np.float64 = 0
    terrain_aspects: np.float64 = 0
    combustibilities: np.float64 = 1

    # Arrays used to cache state/intermediate values. DO NOT MODIFY
    fire_states: np.uint8 = COMBUSTIBLE
    spread_rates: np.float64 = 0
    intermediate_states: np.float64 = 2
    prop_aspect: np.float64 = np.nan
    can_ignite: bool = False
    can_extinguish: bool = False


class CPUFireModel(CellularAutomataModel):
    """Implementation of the Rui, 2018 Cellular Automata Forest Fire
    Spread Simulation which accounts for the influence of combustibles,
    wind, temperature, humidity, and slope.

    Args:
        temperature: Ambient Temperature in degree Celcius (C)
        wind_speed: Wind Speed in meter per second (m/s)
        humidity: Relative Air Humidity as a percentage (%)

    Note:
        Instead of padding the FireData
    """

    def __init__(self, simulation: object | None = None) -> None:
        env = simulation.environment
        self.atmosphere = env.atmosphere
        self.terrain = env.terrain
        self.__data__ = CPUFireData(
            temperatures=self.atmosphere.temperature,
            wind_speeds=self.atmosphere.wind_speed,
            wind_aspects=self.atmosphere.wind_aspect,
            humidities=self.atmosphere.relative_humidity,
            terrain_slopes=self.terrain.elevation.slopes,
            terrain_aspects=self.terrain.elevation.aspects,
            combustibilities=self.terrain.features.combustibilities,
        )
        self.__cache__ = {}  # Cache of per-step constant properties
        super().__init__(simulation)
        self.fire_indices = np.zeros((np.prod(self.shape), 2), dtype=np.int64)
        self.total_suppressed_burn_cells = 0
        self.burnt_area_samples = []
        self.sampled_times = []
        self.set_fire_states()
        self.n_burning = 0
        self.initial_combustibilities = (
            self.terrain.features.combustibilities.copy()
        )
        self._model_time_step = simulation.time_step
        self._model_time = simulation.timer.mission_time

    def reset(self):
        """Resets the model to its initial state."""
        self.__init__(self.simulation)

    @cached_property
    def shape(self):
        """Returns number of rows and columns of the CA grid."""
        return self.terrain.grid_shape

    @property
    def fire_in_bounds(self):
        """Checking whether the fire is within the allowed boundaries.

        Returns `True` if all firefronts are within the CA grid fire
        propagation boundaries

        """
        cells_on_fire = self.burning_indices
        in_bounds = (
            (cells_on_fire[:, 0] > MOORE_RADIUS)
            & (cells_on_fire[:, 0] < self.shape[0] - 1 - MOORE_RADIUS)
            & (cells_on_fire[:, 1] > MOORE_RADIUS)
            & (cells_on_fire[:, 1] < self.shape[1] - 1 - MOORE_RADIUS)
        )
        return (in_bounds).all()

    def ignite(self, ignition_centers: tuple[Position, ...]):
        """Ignites cells on fire at the specified ``positions``."""
        for center in ignition_centers:
            i, j = pos_to_index(
                center.fire_map_pos,
                grid_description=self.terrain.grid_description,
            )
            neighborhood = MooreNeighborhood(
                radius=MOORE_RADIUS, include_center=True
            )
            for n, (i_offset, j_offset) in enumerate(neighborhood.as_tuple()):
                i_fire, j_fire = i + i_offset, j + j_offset
                self.__data__.fire_states[i_fire, j_fire] = FULL_BURNING
                self.fire_indices[n + self.n_burning] = i_fire, j_fire
            self.n_burning += neighborhood.cell_count

    def suppress(self, suppression_area: RasterizedShape):
        """Updates fire model based on suppression area.

        Suppressed indices are turned into NONFLAMMABLE and the stored
        data of the indices is updated to reflect this (spreadability,
        combustibility, etc.). Suppressed burn cells are updated based
        on cells that were extinguished (burning, but not burnt).
        """
        suppression_indices = suppression_area.nonzero(self.shape)
        cell_states = self.fire_states[suppression_indices]
        suppressed_burn_cells = np.count_nonzero(
            (cell_states < BURNT) & (cell_states >= EARLY_BURNING)
        )
        self.total_suppressed_burn_cells += suppressed_burn_cells

        self.__data__.fire_states[suppression_indices] = SUPPRESSED
        if "fire_states" in self.__cache__:
            self.__cache__["fire_states"][suppression_indices] = SUPPRESSED
        self.__data__.combustibilities[suppression_indices] = 0
        self.__data__.intermediate_states[suppression_indices] = SUPPRESSED
        self.__data__.spread_rates[suppression_indices] = 0
        if suppressed_burn_cells > 0:
            self.update_fire_area(suppression_indices)

    def update_fire_area(self, suppression_indices):
        """Updates the fire position array to exclude suppressed areas."""
        pos_suppress = index_to_pos(
            suppression_indices,
            self.terrain.grid_description,
            self.origin,
        )
        mask = np.ones(len(self.fire_positions), dtype=bool)
        for elem in pos_suppress:
            mask &= np.any(self.fire_positions != elem, axis=1)
        self.__cache__["fire_positions"] = self.__cache__["fire_positions"][
            mask, ...
        ]
        self.__cache__["burning_indices"] = self.__cache__["burning_indices"][
            mask, ...
        ]

    def set_fire_states(self):
        nonflammable_indices = np.nonzero(self.__data__.combustibilities == 0)
        self.__data__.fire_states[nonflammable_indices] = NONFLAMMABLE

    @property
    def model_time_step(self) -> timedelta:
        """Time step to be used for the fire model."""
        if self.simulation.parameters.enable_adaptive_time_step:
            return self._model_time_step
        return self.simulation.time_step

    @model_time_step.setter
    def model_time_step(self, time_step: timedelta) -> None:
        if (
            not self.simulation.parameters.enable_adaptive_time_step
            and time_step != self.simulation.time_step
        ):
            raise ValueError(
                "Time step can not be changed when `adaptive_time_step` is disabled"
            )
        self._model_time_step = time_step

    # TODO see if using asdict property and unpacking it is faster!
    def step(self):  # noqa D102
        # Preventing duplicate LOAD_FAST
        data = self.__data__
        sim = self.simulation
        self.update_atmosphere

        try:
            self.n_burning, ideal_time_step_min = step_cpu(
                # TODO Convert the fire-model to use seconds instead of
                # minutes
                time_step=sim.time_step.total_seconds() / 60,
                temperatures=data.temperatures,
                wind_speeds=data.wind_speeds,
                wind_aspects=data.wind_aspects,
                humidities=data.humidities,
                terrain_slopes=self.terrain.slopes,
                terrain_aspects=self.terrain.aspects,
                combustibilities=self.terrain.combustibilities,
                fire_states=data.fire_states,
                spread_rates=data.spread_rates,
                intermediate_states=data.intermediate_states,
                prop_aspect=data.prop_aspect,
                can_ignite=data.can_ignite,
                can_extinguish=data.can_extinguish,
                fire_indices=self.fire_indices,
                n_burning=self.n_burning,
                cell_size=sim.parameters.cell_size,
                correction_coefficient=sim.parameters.correction_coefficient,
                enable_adaptive_time_step=sim.parameters.enable_adaptive_time_step,
            )
            self.internal_model_time += self.model_time_step
            self.model_time_step = timedelta(seconds=ideal_time_step_min * 60)

        except IndexError as e:
            print("Index error:", e.args[0])

        self.check_output_sampling_time()

        # Clear property cache at each simulation step
        self.__cache__.clear()

        # Halt the simulation if fire is fully extinguished
        if not (self.fire_positions).size:
            print("\nMission Completed")
            sim.stop()

        # Halt the simulation if fire reaches the endge of the map
        if not self.fire_in_bounds:
            print("\nMission Failed")
            sim.stop()

    OUTPUT_SAMPLE_TIMER = 0

    def retrieve_fire_states(self, indices):
        """Outputs the fire states of input indices."""
        indices = tuple(np.array(indices).T)
        fire_states = self.fire_states[indices]
        return fire_states

    def burnt_area_for_combustibility(
        self, combustibility: COMBUSTIBILITY_TABLE
    ):
        """Return burnt area in SI meters^2 for terrain type."""
        burnt_mask = (self.fire_states >= FULL_BURNING) | (
            self.fire_states == SUPPRESSED
        )
        # Count the number of cells in the features that satisfy the
        # mask
        combust = self.initial_combustibilities
        count = np.sum(burnt_mask & (combust == combustibility))
        return int(count * self.simulation.parameters.cell_size**2)

    @property
    def update_atmosphere(self) -> None:
        """Call for update of atmospheric properties."""
        # Dont reassign unless values have changed
        self._update_atmosphere(self.atmosphere.cache_access)

    @lru_cache(maxsize=1)
    def _update_atmosphere(self, cache_access: int) -> int:
        """Update data when atmosphere values have been updated."""
        self.__data__.temperatures[:, :] = self.atmosphere.temperature
        self.__data__.humidities[:, :] = self.atmosphere.relative_humidity
        self.__data__.wind_speeds[:, :] = self.atmosphere.wind_speed
        self.__data__.wind_aspects[:, :] = self.atmosphere.wind_aspect
        return cache_access

    def check_output_sampling_time(self) -> None:
        """Triggers output when sampling time is reached."""
        self.OUTPUT_SAMPLE_TIMER -= self.simulation.time_step.total_seconds()
        if self.OUTPUT_SAMPLE_TIMER <= 0:
            self.sample_outputs()
            self.OUTPUT_SAMPLE_TIMER = (
                self.simulation.parameters.output_sampling_time
            )

    def sample_outputs(self) -> None:
        """Sample relevant outputs and store."""
        self.burnt_area_samples.append(self.burnt_area)
        self.sampled_times.append(
            self.simulation.timer.mission_runtime.total_seconds()
        )

    @property
    def origin(self) -> tuple[float, float]:
        """Mercator projection coordinate of local coordinate origin."""
        return self.simulation.environment.terrain.origin

    @property
    def burnt_area(self):
        """Return total burnt area in SI meters^2."""
        return (
            (np.count_nonzero(self.fire_states == BURNT))
            + (np.count_nonzero(self.fire_states == EXTINGUISHING))
            + (np.count_nonzero(self.fire_states == FULL_BURNING))
            + self.total_suppressed_burn_cells
        ) * self.simulation.parameters.cell_size**2

    @property
    def burning_indices(self):
        if "burning_indices" in self.__cache__:
            return self.__cache__["burning_indices"]
        burning_indices = self.fire_indices[: self.n_burning, :]
        self.__cache__["burning_indices"] = burning_indices
        return burning_indices

    @property
    def fire_positions(self) -> list[Position]:
        """Return positions of cells currently on fire."""
        if "fire_positions" in self.__cache__:
            return self.__cache__["fire_positions"]
        fire_indices = self.burning_indices
        i_idx, j_idx = fire_indices.T
        positions = index_to_pos(
            (i_idx, j_idx),
            self.terrain.grid_description,
            origin=self.origin,
        )
        self.__cache__["fire_positions"] = positions
        return positions

    @property
    def fire_states(self):
        if "fire_states" in self.__cache__:
            return self.__cache__["fire_states"]
        states = self.__data__.fire_states
        self.__cache__["fire_states"] = states
        return states

    @property
    def prop_aspect(self):
        """Return prop aspect."""
        return self.__data__.prop_aspect

    def get_spread_rates(self, indices=None):
        """Return spread rates."""
        if indices is None:
            return self.__data__.spread_rates
        indices = tuple(np.array(indices).T)
        return self.__data__.spread_rates[indices]

    @property
    def internal_model_time(self) -> datetime:
        """Internal time of the model.

        This parameter is used to ensure fire model and ABM model run in sync
        when adaptive time step is enabled.
        """
        return self._model_time

    @internal_model_time.setter
    def internal_model_time(self, time: datetime) -> None:
        """Increment internal model time by ``time``."""
        self._model_time = time

    def __gui_repr__(self):
        """Implements the GUI representation protocol."""
        x, y = self.origin
        height = self.terrain.grid_dimensions[1]
        width = self.terrain.grid_dimensions[0]
        return {
            "object": self,
            "painter": IndexedImageItem,
            "on_init": lambda p, obj: p(
                obj.__data__.fire_states,
                colorTable=tuple(COLOR_TABLE.values()),
                compositionMode=(QtGui.QPainter.CompositionMode_SourceOver),
                rect=(x, y, width, height),
            ),
            "on_update": lambda item, obj: item.setImage(
                obj.__data__.fire_states
            ),
            "z_order": 2,
            "scale": 1,
        }
