# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from typing import Any


class PostponedImportError:
    def __init__(self, package: str) -> None:
        self.package = package
        self.error = ImportError(f"{package} not available")

    def __getattribute__(self, _) -> Any:
        raise self.error

    def __call__(self, *args, **kwargs):
        raise self.error
