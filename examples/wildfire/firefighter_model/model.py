# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import math
import pickle
from collections import defaultdict
from datetime import datetime
from functools import cached_property
from itertools import chain

import numpy as np
import shapely.geometry as geom
from mesa.space import ContinuousSpace
from scipy.spatial.distance import cdist
from skimage.draw import ellipse, ellipse_perimeter, line

from examples.wildfire.fire_model.jit_funcs.cpu import MOORE_RADIUS
from examples.wildfire.fire_model.states import (
    BURNT,
    EARLY_BURNING,
    SUPPRESSED,
)
from examples.wildfire.firefighter_model.agents import (
    AirTrafficManager,
    IgnitionCenter,
    ProtectionLocation,
    SuppressionUAV,
    WaterSourceManager,
)
from sosid.environment.terrain import TerrainTypes
from sosid.model.abm.agent import Agent
from sosid.model.abm.model import AgentBasedModel
from sosid.model.abm.schedule import RandomActivationByBreed
from sosid.model.ca.jit_funcs.geom2d import calculate_confidence_area
from sosid.model.transform import (
    gps_to_pos,
    pos_to_index,
)


class FireBlockData:
    def __init__(self):
        self.fire_block_indices = None
        self.current_block_index = 0
        self.fire_distance_offset = 30  # number of cells
        self.update_iterations = 1000
        self.max_spread_rate_angle = None
        self.fire_contained = False
        self.closing_index = None
        self.fire_encircled = False


class FirefighterModel(AgentBasedModel, FireBlockData):
    def __init__(self, simulation: object | None = None) -> None:
        super().__init__(simulation)
        self.__cache__ = {}
        self.schedule = RandomActivationByBreed(model=self)
        width, height = self.simulation.environment.dimensions
        self.space = ContinuousSpace(
            x_max=width,
            y_max=height,
            torus=False,
        )  # type: ignore
        self.agents_by_type = defaultdict(list)  # type: ignore
        self.agents = []  # type: ignore
        self.unique_aircraft_definitions: list[SuppressionUAV] = []
        self.air_traffic_managers
        self.water_sources
        self.firefighters
        self.protection_locations
        # Check feasibility only if the water sources are polygons.
        self.feasible_water_sources = self.all_feasible_water_sources
        FireBlockData.__init__(self)
        self.update_urban_areas()
        self._model_time = simulation.timer.mission_time

    def reset(self):
        """Resets the model to its initial state."""
        FireBlockData.__init__(self)
        for _, agent in enumerate(self.firefighters):
            agent.unique_id = agent.init_states["unique_id"]
            agent.model = agent.init_states["model"]
            agent.pos = agent.init_states["pos"]
            agent.__idle_timer__ = None
            agent.__init__(**agent.init_states)

        for _, air_traffic_manager in enumerate(self.air_traffic_managers):
            air_traffic_manager.last_takeoff = (
                self.simulation.timer.mission_start
            )
        self.internal_model_time = self.simulation.timer.mission_time

    @property
    def time_step(self):
        return self.simulation.time_step

    @cached_property
    def parameters(self):  # noqa D102
        """Returns simulation parameters."""
        return self.simulation.parameters

    @cached_property
    def air_traffic_managers(self):
        """Create and return agents of class `AirTrafficManager`."""
        for airport in self.parameters.airports:
            pos = airport.get_pos(
                self.simulation.environment.terrain.top_left_bounds
            )

            air_traffic_controller = AirTrafficManager(
                unique_id=self.get_unique_id(),
                model=self,
                icon=airport.icon,
                takeoff_interval=self.parameters.takeoff_interval,
                takeoff_landing_types=airport.takeoff_landing_types,
            )
            self.add_agent(air_traffic_controller, pos)
        return self.agents_by_type[AirTrafficManager]

    @cached_property
    def firefighters(self):
        """Create and return agents of class `SuppressionUAV`."""
        for ac_type_id, agent_definition in enumerate(self.parameters.agents):
            for base_index, agent_count in enumerate(
                agent_definition.agents_per_base
            ):
                home_base = self.air_traffic_managers[base_index]
                starting_pos = np.array(home_base.pos, dtype=np.float64)
                for _ in range(agent_count):
                    suppression_uav = SuppressionUAV(
                        unique_id=self.get_unique_id(),
                        pos=starting_pos,
                        model=self,
                        home_base=home_base,
                        output_id_name=agent_definition.output_id_name,
                        takeoff_landing_type=agent_definition.takeoff_landing_type,
                        payload_capacity=agent_definition.payload,
                        suppressant_flow_rate=agent_definition.flow_rate,
                        can_scoop=agent_definition.can_scoop,
                        icon_type=agent_definition.icon,
                        propulsion_input=agent_definition.propulsion_input,
                        profile_parameters=agent_definition.profile_parameters,
                        autonomous=agent_definition.autonomous,
                        empty_mass=agent_definition.empty_mass,
                        mtom=agent_definition.mtom,
                        scooping_distance=agent_definition.scooping_distance,
                        span=agent_definition.span,
                        ac_type_id=ac_type_id,
                        suppression_tactic=agent_definition.suppression_tactic.main,
                        change_condition=(
                            agent_definition.suppression_tactic.alternative.change_condition
                            if agent_definition.suppression_tactic.alternative
                            else None
                        ),
                        alternative_tactic=(
                            agent_definition.suppression_tactic.alternative.alternative_tactic
                            if agent_definition.suppression_tactic.alternative
                            else None
                        ),
                        threshold=(
                            agent_definition.suppression_tactic.alternative.threshold
                            if agent_definition.suppression_tactic.alternative
                            else None
                        ),
                    )
                    self.add_agent(suppression_uav, starting_pos)
            self.unique_aircraft_definitions.append(suppression_uav)
        return self.agents_by_type[SuppressionUAV]

    @cached_property
    def agent_scooping_dimensions(self) -> dict:
        """Returns the scooping dimensions of unique aircraft type."""
        agent_scooping_dimensions = {}
        for ac_type_id, agent in enumerate(self.unique_aircraft_definitions):
            agent_scooping_dimensions[ac_type_id] = (
                agent.scooping_distance,
                agent.scooping_width,
            )
        return agent_scooping_dimensions

    @cached_property
    def largest_suppression_patch(self) -> float:
        """Returns the largest suppression patch dimensions (cells)."""
        largest_dim = 0
        for agent in self.unique_aircraft_definitions:
            patch_size = agent.suppression_patch(
                agent.payload, agent.flow_rate
            )
            largest_dim = max(largest_dim, max(patch_size))
        return largest_dim

    @cached_property
    def smallest_suppression_patch(self) -> float:
        """Returns the largest smaller suppression patch dimensions."""
        smallest_dim = 0
        # Loop over unique aircraft definitions
        for agent in self.unique_aircraft_definitions:
            patch_size = agent.suppression_patch(
                agent.payload, agent.suppressant_flow_rate
            )
            smallest_dim = max(smallest_dim, min(patch_size))
        return smallest_dim

    @cached_property
    def ignition_centers(self):
        """Create, initialize, and return ignition centers as agents."""
        for location in self.parameters.ignition_centers:
            pos = location.get_pos(
                self.simulation.environment.terrain.top_left_bounds
            )
            ignition_center = IgnitionCenter(
                unique_id=self.get_unique_id(), model=self
            )
            self.add_agent(ignition_center, pos)
        return self.agents_by_type[IgnitionCenter]

    @cached_property
    def mean_ignition_center_index(self):
        # TODO consider using separate fire block indices for each
        # center
        sum_index = np.zeros(2)
        for center in self.simulation.ignition_centers:
            index = pos_to_index(
                pos=center.fire_map_pos,
                grid_description=self.simulation.environment.terrain.grid_description,
            )
            sum_index = np.add(sum_index, index)

        return np.round(
            sum_index / len(self.simulation.ignition_centers)
        ).astype(np.int64)

    @cached_property
    def water_sources(self) -> list[WaterSourceManager]:
        """Create and return agents of class `WaterSourceManager`."""
        for source in self.parameters.water_sources:
            pos = source.get_pos(
                self.simulation.environment.terrain.top_left_bounds
            )
            water_source_controller = WaterSourceManager(
                unique_id=self.get_unique_id(),
                model=self,
                water_shell=pos,
                water_holes=pos,
                always_feasible=True,
            )
            self.add_agent(water_source_controller, pos)
        if self.parameters.deploy_osm_waters:
            # Open water_sources file
            with self.parameters.terrain_inputs.water_sources_file.open(
                "rb",
            ) as file:
                water_edges = pickle.load(file, encoding="bytes")  # noqa: S301

            for source in water_edges.values():
                # Minimum 4 edges are required to form the shape.
                if len(source["shell_coords"]) >= 4:  # noqa: PLR2004
                    shell = source["shell_coords"]
                    holes = []

                    # new_water already converted to position, OSM
                    # waters must be converted
                    if not source["new_water"]:
                        shell = gps_to_pos(
                            shell,
                            self.simulation.environment.terrain.top_left_bounds,
                        )
                        if source["holes_coords"]:
                            for hole in source["holes_coords"]:
                                hole_pos = gps_to_pos(
                                    hole,
                                    self.simulation.environment.terrain.top_left_bounds,
                                )
                                holes.append(hole_pos)

                    water_source_controller = WaterSourceManager(
                        unique_id=self.get_unique_id(),
                        model=self,
                        water_shell=shell,
                        water_holes=holes,
                    )
                    pos = np.array(
                        [
                            water_source_controller.representative_point.x,
                            water_source_controller.representative_point.y,
                        ],
                        dtype=np.float64,
                    )
                    self.add_agent(water_source_controller, pos)
        return self.agents_by_type[WaterSourceManager]

    @cached_property
    def protection_locations(self):
        """Create and return agents of class `ProtectionLocation`."""
        for location in self.parameters.protection_locations:
            pos = location.get_pos(
                self.simulation.environment.terrain.top_left_bounds
            )
            protection_location_controller = ProtectionLocation(
                unique_id=self.get_unique_id(), model=self
            )
            self.add_agent(protection_location_controller, pos)
        return self.agents_by_type[ProtectionLocation]

    @cached_property
    def all_feasible_water_sources(self) -> dict:
        """Returns nodes for each water source accessible to the agent.

        Returns:
            dict: A dictionary containing the feasible water source
            nodes for each agent type.
                  The keys are the unique IDs of the agent types, and
                  the values are the corresponding lists of feasible
                  nodes.
        """
        water_sources = self.water_sources
        feasible_nodes = {}
        feasible_nodes_by_ac_type = {}
        for ac_type_id, (
            scooping_distance,
            scooping_width,
        ) in self.agent_scooping_dimensions.items():
            for water_source in water_sources:
                is_feasible, nodes = self.evaluate_water_source_feasibility(
                    water_source, scooping_width, scooping_distance
                )
                if is_feasible:
                    feasible_nodes[water_source.unique_id] = nodes
            locations = list(feasible_nodes.values())
            expanded_locations = list(chain.from_iterable(locations))
            feasible_nodes_by_ac_type[ac_type_id] = expanded_locations
        return feasible_nodes_by_ac_type

    def evaluate_water_source_feasibility(
        self,
        water_source: WaterSourceManager,
        scooping_width: float,
        scooping_distance: float,
        num_points: int = 10,
        num_angles: int = 4,
    ) -> tuple[bool, list[tuple[float, float]]]:
        """Checks water polygons based on agent traits.

        Args:
            water_source (object): Water source agent.
            scooping_width(float): Scooping width of the agent which
            refers to the span.
            scooping_distance (float): Scooping distance of the
            agent.
            num_points (int, optional): Number of points to generate in
            the value vectors. Default is 10.
            num_angles (int, optional): Number of angles to consider in
            the value vectors. Default is 4.

        Returns:
            bool: True if there are feasible nodes, False otherwise.
            list: List of nodes within the water source for which the
            agent's confidence area fits inside.
        """
        if water_source.always_feasible:
            feasible_nodes = [water_source.pos]
            return water_source.always_feasible, feasible_nodes

        # Create points and directions to test in the water source
        x_values, y_values, theta_values = self.generate_value_vectors(
            water_source.edges, num_points, num_angles
        )

        feasible_nodes = []  # A list which holds all the feasible spots

        # Test each point and direction
        for x, y in np.ndindex(num_points, num_points):
            node = (x_values[x], y_values[y])  # The current spot to test
            if not any(
                np.array_equal(node, a) for a in feasible_nodes
            ) and water_source.polygon.contains(geom.Point(node)):
                for theta in theta_values:  # Test each direction at this spot
                    # Draw the imaginary scooping area
                    confidence_edges = geom.Polygon(
                        calculate_confidence_area(
                            node[0],
                            node[1],
                            scooping_width,
                            scooping_distance,
                            theta,
                        )
                    )
                    # Check if this area stays within the water source
                    if np.all(water_source.polygon.contains(confidence_edges)):
                        feasible_nodes.append(node)
                        break

        # Indicates if there are any feasible spots
        feasible = bool(feasible_nodes)

        return feasible, feasible_nodes

    @staticmethod
    def generate_value_vectors(
        input_vector: np.ndarray, num_points: int, num_angles: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates vectors x, y and theta based on a 2D input vector.

        Args:
            input_vector (ndarray): The input vector.
            num_points (int): The number of points.
            num_angles (int): The number of angles.

        Returns:
            Tuple: (x_values, y_values, theta_values)
                x_values (ndarray): The vector of x-values.
                y_values(ndarray): The vector of y-values.
                theta_values (ndarray): The vector of angle-values.
        """
        x_min = np.min(input_vector[:, 0])
        x_max = np.max(input_vector[:, 0])
        y_min = np.min(input_vector[:, 1])
        y_max = np.max(input_vector[:, 1])

        x_values = np.linspace(start=x_min, stop=x_max, num=num_points)
        y_values = np.linspace(start=y_min, stop=y_max, num=num_points)
        theta_values = np.linspace(start=0, stop=180, num=num_angles)
        return x_values, y_values, theta_values

    def step(self):  # noqa: ANN201, D102
        # Create and update fire blocking indices
        if self.simulation.iterations > 0:
            self.__cache__.clear()
            self.fire_encircled = (self.closing_index is not None) & (
                self.current_block_index == self.closing_index
            ) | self.fire_encircled

            if self.fire_block_indices is None:
                # Create first fire blocking indices when the response
                # time has elapsed
                if self.simulation.timer.mission_runtime.seconds >= int(
                    self.simulation.parameters.response_time
                ):
                    self.create_fire_block_ellipse()
            elif self.simulation.iterations % self.update_iterations == 0:
                # Check if fire can be contained
                self.fire_contained = self.is_fire_containable()
                if not self.fire_contained:
                    # Otherwise keep expanding and updating the fire
                    # block ellipse
                    self.update_fire_block_ellipse()

        self.schedule.step()
        self.internal_model_time += self.simulation.time_step

    def update_urban_areas(self):
        """Creates urban areas as ellipses based on input parameters."""
        if self.parameters.urban_locations:
            indices = []
            for location in self.parameters.urban_locations:
                idx = pos_to_index(
                    location.get_pos(
                        self.simulation.environment.terrain.fire_map_top_left_bounds
                    ),
                    grid_description=self.simulation.environment.terrain.grid_description,
                )
                cell_x = round(location.radius[1] / self.parameters.cell_size)
                cell_y = round(location.radius[0] / self.parameters.cell_size)
                rr, cc = ellipse(
                    idx[0],
                    idx[1],
                    cell_x,
                    cell_y,
                    self.simulation.environment.terrain.grid_shape,
                    rotation=location.angle,
                )
                urban_indices = np.array([rr, cc]).T
                indices.append(urban_indices)
            indices = np.vstack(indices)
            self.simulation.environment.terrain.override_map_features(
                indices, TerrainTypes.RESIDENTIAL
            )

    def create_fire_block_ellipse(self):
        """Create fire block ellipse for the agents to follow and block
        the fire from extending pass it.
        """
        # Get burning indices
        fire_indices = self.wildfire.fire_indices[: self.wildfire.n_burning]

        # Get fire boundaries
        top = np.min(fire_indices.T[0])
        bottom = np.max(fire_indices.T[0])
        left = np.min(fire_indices.T[1])
        right = np.max(fire_indices.T[1])

        # Find ellipse radius for minor and major semi-axes
        row_radius = int((bottom - top) / 2)
        column_radius = int((right - left) / 2)

        # Find center coordinates for ellipse
        row_center = top + row_radius
        column_center = left + column_radius

        # Create ellipse
        rr, cc = ellipse_perimeter(
            row_center,
            column_center,
            row_radius + self.fire_distance_offset,
            column_radius + self.fire_distance_offset,
        )
        self.fire_block_indices = np.array([rr, cc]).T

        # Sort indices in clockwise order
        center = np.array([row_center, column_center])
        self.fire_block_indices = self.sort_indices_clockwise(
            self.fire_block_indices, center
        )

        # Add lower and upper bounds to the array so that the ellipse
        # indices don't exceed the limits of the map
        # 2 spots are required since the computation of the Moore
        # Neighborhood is done on each neighbor cell of the fire model
        # (1 neighbor cell looks at the neighbor cell of itself
        # (--> = n+2 cells observed))
        lower_bounds = np.array([0, 0])
        upper_bounds = np.array(
            [
                self.wildfire.shape[0] - MOORE_RADIUS - 1,
                self.wildfire.shape[1] - MOORE_RADIUS - 1,
            ]
        )

        self.fire_block_indices = np.clip(
            self.fire_block_indices, lower_bounds, upper_bounds
        )
        self.verify_block_indices()

    def verify_block_indices(self, starting_idx=0):
        """Ensures the fire block is well represented in grid.

        The fire block may have areas where it traverses across
        diagonals in the grid. Whilst this is realistic for a positional
        argument, due to the fire model's behavior, fire can leap across
        diagonal grid spaces and therefore it is necessary to patch
        together diagonals.
        """
        prev = self.fire_block_indices[starting_idx, :]
        new_indices = self.fire_block_indices
        n_new = 0
        for idx in range(starting_idx, len(self.fire_block_indices)):
            current = self.fire_block_indices[idx, :]
            if not (prev == current).any():
                new = np.array([prev[0], current[1]])
                new_idx = idx + n_new
                new_indices = np.insert(new_indices, new_idx, new, axis=0)
                n_new += 1
            prev = current
        self.fire_block_indices = new_indices

    def update_fire_block_ellipse(self):
        """Update fire block ellipse and connect it with the last fire
        block ellipse.
        """
        # Save the current fire block indices
        current_fire_block_indices = self.fire_block_indices[
            self.current_block_index, :
        ]

        # Create the new ellipse
        self.create_fire_block_ellipse()

        # Find closest point in the new ellipse
        distance_array = np.array(
            [
                Agent.distance(current_fire_block_indices, point)
                for point in self.fire_block_indices
            ]
        )
        self.current_block_index = np.argmin(distance_array)

        # Draw line between current fire block indices and the closest
        # point in the new ellipse
        rr, cc = line(
            current_fire_block_indices[0],
            current_fire_block_indices[1],
            self.fire_block_indices[self.current_block_index, 0],
            self.fire_block_indices[self.current_block_index, 1],
        )
        fire_block_line = np.array([rr, cc]).T

        # Insert line into fire block indices
        self.fire_block_indices = np.insert(
            self.fire_block_indices,
            self.current_block_index,
            fire_block_line,
            axis=0,
        )

    def is_fire_containable(self) -> bool:
        """Check if fire is containable and close fire block ellipse."""
        # Avoid multiple method calls after fire is contained
        if self.fire_contained or self.wildfire.n_burning == 0:
            return True

        # Analyze future fire block cells to see if fire block would be
        # contained. Offsetting done to reduce likelihood of corners
        # where conditions are incorrectly identified as passed.
        index_offset_low = self.current_block_index + int(
            len(self.fire_block_indices) * 10 / 360
        )
        index_offset_high = self.current_block_index + int(
            len(self.fire_block_indices) * 30 / 360
        )
        for future_idx in range(index_offset_low, index_offset_high):
            future_idx %= len(self.fire_block_indices)
            future_fire_block_idx = self.fire_block_indices[future_idx, :]
            if self.wildfire.retrieve_fire_states(
                future_fire_block_idx
            ) not in [SUPPRESSED, BURNT]:
                break

        # Create line trace between future fire block indices and
        # ignition center.
        rr, cc = line(
            future_fire_block_idx[0],
            future_fire_block_idx[1],
            self.mean_ignition_center_index[0],
            self.mean_ignition_center_index[1],
        )
        line_trace = np.array([rr, cc]).T

        closeable, short_line_trace = self.create_closing_line(line_trace)

        # Insert short line trace into fire block indices.
        if not closeable:
            return False
        if len(short_line_trace):
            self.fire_block_indices = np.insert(
                self.fire_block_indices,
                future_idx,
                short_line_trace,
                axis=0,
            )
        else:
            return False

        # Save the index where the line trace ends so the agents can
        # compare and end the mission.
        self.closing_index = (future_idx + len(short_line_trace)) % len(
            self.fire_block_indices
        )

        return True

    def create_closing_line(self, line_trace):
        """Checks if future fire block will allow for fire line closing
        Returns:
            bool: if fire block is closable.
            ndarray: set of indices from future fire block index and
            previous suppressed area (previous fire block index) which
            represents the closing line of the fire line.
        """
        closable = True
        short_line_trace = []
        for i in range(len(line_trace)):
            idx = line_trace[i]
            state = self.wildfire.retrieve_fire_states(tuple(idx))
            if state == SUPPRESSED:
                if i >= self.smallest_suppression_patch:
                    break
                continue
            if state >= EARLY_BURNING:
                closable = False
                break
            short_line_trace.append(idx)
        short_line_trace = np.array(short_line_trace)

        return closable, short_line_trace

    def sort_indices_clockwise(self, indices, center):
        """Sort unordered array of indices clockwise."""
        # Calculate angles with respect to the center
        angles = np.arctan2(
            indices[:, 0] - center[0], indices[:, 1] - center[1]
        )

        if self.max_spread_rate_angle is None:
            self.max_spread_rate_angle = self.find_max_spread_rate_angle(
                center
            )

        # Offset the angles with initial starting position of the
        # fastest spread rate angle
        angles = [
            (a + math.pi - self.max_spread_rate_angle) % (math.pi * 2)
            for a in angles
        ]

        # Sort indices based on angles in a clockwise order
        sorted_indices = np.argsort(angles)

        # Extract sorted indices
        clockwise_indices = indices[sorted_indices]

        return clockwise_indices

    def find_max_spread_rate_angle(self, center: tuple):
        """Find the direction at which the fire spreads the fastest.

        center : Reference point from which the angle is calculated.
        """
        # Get the index for the maximum fire spread rate
        burning_indices = self.wildfire.burning_indices
        fire_spread_rates = self.wildfire.get_spread_rates(burning_indices)
        max_spread_rate_index = np.argmax(fire_spread_rates)

        # Find the angle between max spread rate point and center
        max_spread_rate_angle = np.arctan2(
            burning_indices[max_spread_rate_index, 0] - center[0],
            burning_indices[max_spread_rate_index, 1] - center[1],
        )

        # Transform from (-180, 180) to (0, 360)
        max_spread_rate_angle += math.pi

        return max_spread_rate_angle

    @property
    def min_distance_fire_urban(self):
        """Return shortest distance from fire to residential areas."""
        if "min_distance_fire_urban" in self.__cache__:
            return self.__cache__["min_distance_fire_urban"]
        fire_indices = self.wildfire.burning_indices
        # in instances where fire model stops and agents run last step
        if not fire_indices.size:
            return 0
        urban_indices = self.simulation.environment.terrain.urban_indices

        min_dist = self.parameters.cell_size * np.min(
            cdist(fire_indices, urban_indices)
        )

        self.__cache__["min_distance_fire_urban"] = min_dist
        return min_dist

    @property
    def min_distance_fire_fireblock(self):
        """Return shortest distance from fire to fire block."""
        if "min_distance_fire_fireblock" in self.__cache__:
            return self.__cache__["min_distance_fire_fireblock"]
        block_indices = self.fire_block_indices[: self.current_block_index]
        fire_indices = self.wildfire.burning_indices
        # in instances where fire model stops and agents run last step
        if not fire_indices.size:
            return 0

        min_dist = self.parameters.cell_size * np.min(
            cdist(fire_indices, block_indices)
        )
        self.__cache__["min_distance_fire_fireblock"] = min_dist
        return min_dist

    @property
    def positions(self):
        """Array of agent positions.

        Has the shape (len(self.agents)), 2).

        One can access the position of an agent using its `unique_id`
        as follows:

            agent_pos = self.positions[..., unique_id]

        Returns:
            A writable view of the Numpy array.

        """
        return self.space._agent_points.view()

    @property
    def wildfire(self):
        """Exposing fire-model to agents."""
        return self.simulation.wildfire

    def calculate_utility(self):
        pass

    @property
    def time_at_next_iter(self) -> datetime:
        """Time of the ABM model after an additional iteration."""
        return self.internal_model_time + self.time_step

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
