# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import json
from pathlib import Path

from examples.wildfire.paths import DATA_DIR, TERRAIN_DIR
from sosid.model.transform import bounding_box_coordinates
from src.sosid.environment.terrain_gen import (
    HeightMappedTerrainGeneration,
    TerrainFeaturesGeneration,
    TerrainGenResponses,
    initialize_parameters,
)


def TerrainGeneration(
    overwrite: bool = False,
    input_file: Path | None = None,
):
    """Generates height and features raster data.

    Args:
        overwrite (bool, optional): Automatically downloads new
        terrain data if inputted as 'True'. Defaults to False.
    """
    parameters = initialize_parameters(
        data_path=DATA_DIR, input_file=input_file
    )
    parameters.coordinates = bounding_box_coordinates(
        parameters.center_coords,
        parameters.height_cells,
        parameters.width_cells,
        parameters.resolution,
    )
    elevation = HeightMappedTerrainGeneration(parameters, overwrite)
    overwrite = elevation.create_new
    features = TerrainFeaturesGeneration(parameters, overwrite)
    in_storage = TerrainGenResponses.IN_STORAGE
    # If meta or terrain file was missing from storage, write new metafile
    if (
        overwrite is True
        or elevation.response > in_storage
        or features.response > in_storage
    ):
        # Create meta file
        meta_elevation, meta_features = elevation.meta, features.meta
        meta = {**parameters.model_dump(), **meta_elevation, **meta_features}
        metafile_path = parameters.path_for_terrain / (
            parameters.label + ".meta"
        )
        with open(metafile_path, "w+") as metafile:
            metafile.write(json.dumps(meta, default=str))

        # Postprocess features file based on user settings
        if parameters.override_water_0_elevation:
            print("Overriding the 0 elevation areas to water")
            features.water_at_0_elev(elevation)
        if parameters.convert_noncombustible:
            print("Overriding non-combustible terrain to field type")
            features.change_noncombust()


INPUT_FILE = TERRAIN_DIR / "terrain_gen" / "terrain_gen_input.json"
if __name__ == "__main__":
    TerrainGeneration(True, INPUT_FILE)
