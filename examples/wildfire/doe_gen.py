# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import json

from examples.wildfire.paths import DATA_DIR
from examples.wildfire.simulation import WildfireParameters
from sosid.doe.doe_generator import FullFactDoEGenerator

doe_gen_file = "doe_gen_test.json"


def wildfire_doe_generator(doe_file: str) -> None:
    """Generate DoE for wildfire."""
    with open(DATA_DIR / "doe" / "gen_input" / doe_file) as f:
        doe_specs = json.load(f)
    default_params_file = doe_specs["default_params_file"]
    with open(DATA_DIR / "scenarios" / "inputs" / default_params_file) as f:
        default_params = f.read()
    # validate default parameters
    default_params = WildfireParameters.model_validate_json(default_params)
    doe_params = doe_specs
    del doe_params["default_params_file"]
    FullFactDoEGenerator(
        default_parameters=default_params,
        doe_params=doe_params,
        project_name="project_name",
        data_directory=DATA_DIR / "doe" / "gen_input",
    )


if __name__ == "__main__":
    wildfire_doe_generator(doe_gen_file)
