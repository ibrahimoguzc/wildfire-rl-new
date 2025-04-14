# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains common paths (directories) for the Wildfire example."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"
FIGURE_DIR = ROOT_DIR / "figures"
SCENARIOS_DIR = DATA_DIR / "scenarios"
TERRAIN_DIR = DATA_DIR / "terrain"
STATIC_DIR = ROOT_DIR / "static"
ATMOSPHERE_DIR = DATA_DIR / "atmosphere"
