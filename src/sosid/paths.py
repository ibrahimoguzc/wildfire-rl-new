# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains commonly used paths within the sosid package."""

from pathlib import Path

import pkg_resources

# https://stackoverflow.com/questions/1270951/how-to-refer-to-relative-paths-of-resources-when-working-with-a-code-repository

INSTALL_PATH = Path(pkg_resources.resource_filename("sosid", ""))
ICONS_PATH = INSTALL_PATH / "static" / "icons"
TEMPLATES_PATH = INSTALL_PATH / "gui" / "templates"
