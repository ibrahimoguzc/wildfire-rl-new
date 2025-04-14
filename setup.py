# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Entry point for setuptools, configuration is in setup.cfg."""

from setuptools import setup

# TODO move metadata from setup.cfg into here so that copyright works

# Copyright example below
# __title__ = 'mesa'
# __version__ = '0.8.6'
# __license__ = 'Apache 2.0'
# __copyright__ = 'Copyright %s Project Mesa Team' % datetime.date.today().year
if __name__ == "__main__":
    setup(use_scm_version=True)
