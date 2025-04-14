# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

# TODO add fixture in cfg to configure if building tests are run or not

import os

import pytest

from docs.ext import numba_patch
from tests.test_docs.source.numba_patch.funcs import (
    cuda_func,
    device_func,
    jitted_func,
    normal_func,
)
from util.sphinx.parsers import SphinxFunctionNameExtractor
from util.sphinx.temporary import TempSphinxApp

#: List of compatible and incompatible types in (func, patch_required)
# syntax
FUNCTION_TEST_CASES = [
    (device_func, True),
    (normal_func, False),
    (jitted_func, True),
    (cuda_func, True),
]

#: List of patched documenters, EXPAND LIST WHEN NEW DOCUMENTERS ARE
# ADDED!
PATCHED_DOCUMENTERS = [
    numba_patch.PatchedModuleDocumenter,
    numba_patch.PatchedFunctionDocumenter,
]

#: Root directory for the Sphinx test-builds
TEST_ROOT = os.path.abspath("tests/test_docs/source/numba_patch")

_func_parser = SphinxFunctionNameExtractor()
#: (subdir, parser) format where subdir contains .rst and conf.py files
DIRECTIVE_TEST_CASES = {
    "argnames": "subdir, parser",
    "argvalues": [
        ("autofunction", _func_parser),
        ("automodule", _func_parser),
    ],
}


@pytest.mark.parametrize("test_function, expected_result", FUNCTION_TEST_CASES)
def test_patch_required(test_function, expected_result):
    """Tests if incompatible Numba-types are correctly identified."""
    assert numba_patch.patch_required(test_function) == expected_result


def test_isdocumenter():
    """Tests if a derived :py:class:`Documenter` can be identified."""
    from sphinx.ext.autodoc import FunctionDocumenter

    assert numba_patch.isdocumenter(FunctionDocumenter)


def test_get_documenters():
    """Tests if the required documenters are retrieved from the module.

    The :py:meth:`get_documenters` should return all
    :py:class:`Documenter` classes defined within the numba_patch
    module, while neglecting those that were imported
    """
    result = {documenter for documenter in numba_patch.get_documenters()}
    assert set(PATCHED_DOCUMENTERS).difference(result) == set()


def test_monkey_patch(tmpdir):
    """Tests if the patched documenters have been registered."""
    with TempSphinxApp(TEST_ROOT, tmpdir, buildername="dummy") as app:
        registered_documenters = app.registry.documenters.values()
        for documenter in PATCHED_DOCUMENTERS:
            assert documenter in registered_documenters


@pytest.mark.parametrize(**DIRECTIVE_TEST_CASES)
def test_autodirectives(subdir, parser, tmpdir):
    """Tests if the monkey-patched autodirectives work.

    This is accomplished by building the source-files contained within
    sub-directories within :py:attr:`TEST_ROOT` to HTML and then
    parsing the output HTML to make sure that functions specified in
    :py:module:`tests.test_docs.source.funcs` are present.

    Note:
        The sources for each test-case defined by
        :py:attr:`DIRECTIVE_TEST_CASES` must be segregated into
        sub-directories within the :py:attr:`TEST_ROOT` directory. This
        is necessary to isolate the Sphinx source .rst files, and
        prevent Sphinx from collecting additional .rst files from the
        other test-cases.

        When defining a new autodocumenter such as
        :py:class:`PatchedClassDocumenter` make sure to create a new

    """
    srcdir = os.path.join(TEST_ROOT, subdir)
    with TempSphinxApp(srcdir, tmpdir, delete_output=False) as app:
        app.build()
        html_path = app.outdir / "index.html"
        retrieved_funcs = parser.find_functions(html_str=html_path.read_text())
        assert retrieved_funcs == {
            "device_func",
            "jitted_func",
            "normal_func",
            "cuda_func",
        }


def test_setup():
    """Tests if :py:function:`setup` returns the right configuration.

    Note:
        The build process already tests :py:function:`setup` with an
        active :py:obj:`Sphinx` object. Therefore, this test only

    """
    from sphinx import __display_version__ as __version__

    assert numba_patch.setup(None) == {
        "version": __version__,
        "parallel_read_safe": True,
    }
