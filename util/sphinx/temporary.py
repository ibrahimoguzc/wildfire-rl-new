# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains utilities for temporary Sphinx builds for use in tests."""

import os
import shutil
import sys
from typing import IO

from docutils import nodes
from docutils.parsers.rst import directives, roles
from sphinx.application import Sphinx
from sphinx.testing.path import path
from sphinx.testing.util import SphinxTestApp


class TempSphinxApp(Sphinx):
    """Temporary :py:obj:`Sphinx` object usable as a context manager.

    This class is useful for running a Sphinx build process within
    a pytest temporary directory, while providing context manager
    functionality for automated cleanup. Simply pass in the pytest
    ``tmpdir`` fixture to ``builddir``. Usage is as follows::

        import pytest
        def test_build(tmpdir):
        with TempSphinxApp(SRCDIR, tmpdir) as app:
            app.build()

    Note:
        If more flexibility is required such as running the sphinx-build
        process within a new environment use the
        :py:class:`SphinxTestApp` provided within
        :py:module:`sphinx.testing.util`

    Args:
        srcdir: Absolute path to directory containing conf.py
        builddir: Absolute path to the directory Sphinx builds to
        buildername: Sets the builder used by Sphinx (i.e. html, dummy)
        confoverrides: Overrides options set in conf.py
        status: String buffer for Sphinx status messages
        warning: String buffer for Sphinx warning messages
        warningiserror: Stops Sphinx when it encounters warnings
        delete_output: Sets if the built files should be deleted on exit
        assert_empty: Sets if ``builddir`` should be empty on entry

    """

    def __init__(
        self,
        srcdir: str | path,
        builddir: str | path = None,
        buildername: str = "html",
        confoverrides: dict = None,
        status: IO = None,
        warning: IO = None,
        warningiserror: bool = True,
        delete_output: bool = True,
        assert_empty: bool = True,
    ):
        # Ensuring srcdir is a path object
        self.srcdir = path(srcdir)

        # Assigning new builddir attribute for convenience in clean-up
        self.builddir = (
            srcdir / "_build" if builddir is None else path(builddir)
        )
        if assert_empty:
            assert os.listdir(self.builddir) == []

        # Storing present state in private variables
        self._save_state()

        # Assigning
        self.delete_output = delete_output

        try:
            # Instantiating the Sphinx object with modified inputs
            super().__init__(
                srcdir=self.srcdir,
                confdir=self.srcdir,
                outdir=self._make_buildsubdir(buildername),
                doctreedir=self._make_buildsubdir("doctrees"),
                buildername=buildername,
                confoverrides={} if confoverrides is None else confoverrides,
                status=status,
                warning=warning,
                warningiserror=warningiserror,
            )
        except Exception:  # Handling failed initializaiton of Sphinx
            self.cleanup()
            raise

    def _save_state(self):
        """Stores starting state in private variables for safe exit."""
        self._saved_path = sys.path[:]
        self._saved_directives = directives._directives.copy()
        self._saved_roles = roles._roles.copy()

        self._saved_nodeclasses = {
            v for v in dir(nodes.GenericNodeVisitor) if v.startswith("visit_")
        }

    def _make_buildsubdir(self, subdir: str) -> path:
        """Creates a sub-directory within :py:attr:`builddir`."""
        subdir = self.builddir.joinpath(subdir)
        subdir.makedirs(exist_ok=True)
        return subdir

    def __enter__(self) -> object:
        """Executed when entering the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Executed when leaving the context manager."""
        self.cleanup()

    def cleanup(self):
        """Restores previous Sphinx state and deletes build files.

        This is simply the implementation taken from the
        :py:class:`SphinxTextApp` which requires saved-states be stored
        in private variables. Within this class, state-saving is
        currently implemented by :py:meth:`save_state`.

        Once the states are restored (or even if they fail) the
        directory created for the built files is torn down if
        :py:attr:`delete_output` is :py:obj:`True`. Thus all files
        and subfolders contained within :py:attr:`builddir` are
        recursively deleted.
        """
        try:
            SphinxTestApp.cleanup(self)
        except Exception:
            raise
        finally:
            if os.path.isdir(self.builddir) and self.delete_output:
                shutil.rmtree(self.builddir, ignore_errors=True)

    def __repr__(self) -> str:
        """Uses string representation of :py:class:`SphinxTestApp`."""
        return SphinxTestApp.__repr__(self)
