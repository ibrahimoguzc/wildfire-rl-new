# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""**Patches Sphinx's retrieval of Python objects from Numba types**.

This module contains the Sphinx extension that monkey-patches the
retreival of Python objects from source files containing incompatbile
Numba types. In essence, this modification makes it so that if Sphinx
encounters a type within :py:attr:`NUMBA_TYPES`, then it will replace it
with the underlying Python object stored in :py:attr:`py_func`. The
motivation for this extension was to keep the documentation and source
code independent of each other, and to ensure that a new developer can
simply program in Numba without having to patch imports to check if
Sphinx is running or not.

Note:
    Sphinx mainly has an issue with CUDA Device Functions as currently
    they have no :py:attr:`__doc__`. This means that Sphinx skips these
    unless the directive `:undoc-members:` is used. However, even if
    the developer monkey-patches the :py:attr:`__doc__` at build-time,
    the problem still remains that Sphinx might will not know how to
    document a :py:obj:`cuda.compiler.DeviceFunctionTemplate`. As such,
    the simplest implementation is to just patch Sphinx's object
    retrieval as is done currently, so that Sphinx is not aware of the
    incompatible Numba type.

Tip:
    If additional patched documenters are desired, then they **MUST**
    inherit from the respective :py:class:`Documenter` specialization
    they intent to replace. The name of the additional documenter is not
    important. However, for clarity it is advised to stick to the
    `Patched{DocumenterName}` naming convention used by
    :py:class:`PatchedModuleDocumenter` and
    :py:class:`PatchedFunctionDocumenter`


See Also:
    https://github.com/sphinx-doc/sphinx/issues/3783

"""

import inspect
import sys
from functools import lru_cache
from typing import Any, NewType

import numba
import sphinx
from numba import cuda
from sphinx import __display_version__ as __version__
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Documenter, FunctionDocumenter, ModuleDocumenter


@lru_cache(maxsize=1)
def numba_types() -> tuple[type]:
    """Numba types that are incompatible w/ Sphinx."""
    return (
        cuda.dispatcher.CUDADispatcher,
        numba.core.registry.CPUDispatcher,
    )


LOGGER = sphinx.util.logging.getLogger(__name__)
"""SphinxLoggerAdapter: Active Sphinx debug-logger of module. Methods of
this object are used to print messages to the terminal"""

MemberList = NewType("MemberList", list[tuple[str, object]])


class PatchedModuleDocumenter(ModuleDocumenter):
    """Patches the Sphinx `automodule` directive.

    This overwrites the :py:meth:`get_object_members` of the
    :py:class:`ModuleDocumenter` so as to allow Sphinx to build
    documentation from the Numba objects within a module.
    """

    def get_object_members(self, want_all: bool) -> tuple[bool, MemberList]:
        r"""Gets the module-members utilizing the super-class method.

        Once the members are retrieved, the underlying Python objects
        are extracted from any incompatible Numba type using
        :py:meth:`patch_members`.

        Args:
            want_all (bool): Sets if all members should be retrieved

        Returns:
            Updated members with incompatible Numba types replaced by \
                their underlying Python functions

        """
        want_all, members = super().get_object_members(want_all=want_all)
        return want_all, self.patch_members(members)

    # TODO rename for clarity
    @staticmethod
    def patch_members(members: MemberList) -> MemberList:
        r"""Patches Numba objects with the underlying py:attr:`py_func`.

        This static method iterates a list of members to find Numba
        objects that are incompatible with Sphinx. These objects are
        then replaced by their underlying Python object contained
        within py:attr:`py_func`. All other members are untouched.

        Args:
            members: Contains member_name, member (obj) pairs

        Returns:
            Updated members with incompatible Numba types replaced by \
                their underlying Python functions

        """
        return [
            (name, obj.py_func) if patch_required(obj) else (name, obj)
            for name, obj in members
        ]


class PatchedFunctionDocumenter(FunctionDocumenter):
    """Patches the Sphinx `autofunction` directive."""

    def import_object(self) -> Any:
        import_successful = super().import_object()
        if import_successful and patch_required(self.object):
            self.object = self.object.py_func
        return import_successful


def patch_required(obj: Any) -> bool:
    """Returns if running a monkey-patch is necessary for ``obj``."""
    return True if type(obj) in numba_types() else False


def isdocumenter(obj: Any) -> bool:
    """Return if ``obj`` inherits from :py:class:`Documenter`."""
    return issubclass(obj, Documenter) if inspect.isclass(obj) else False


def get_documenters(module: str = __name__) -> tuple[type[Documenter]]:
    """Gets all :py:class:`Documenter` definitions within ``module``.

    Args:
        module: Name of the current Python module

    Returns:
        All sub-classes of :py:class:`Documenter` in the current
        ``module``

    """
    documenters = inspect.getmembers(sys.modules[module], isdocumenter)
    return [
        documenter
        for name, documenter in documenters
        # Filtering Documenters defined outside of current ``module``
        if documenter.__module__ == module
    ]


def setup(app: Sphinx) -> dict[str, Any]:
    """Sphinx extension setup function.

    When the extension is loaded, Sphinx imports this module and
    executes the ``setup()`` function, which in turn notifies Sphinx of
    everything the extension offers.

    Args:
        app: Application object representing the Sphinx process

    See Also:
        `The Sphinx documentation on Extensions
        <http://sphinx-doc.org/extensions.html>`_
        `The Extension Tutorial <http://sphinx-doc.org/extdev/tutorial.html>`_
        `The Extension API <http://sphinx-doc.org/extdev/appapi.html>`_

    Returns:
        Default extension configuration dictionary

    """
    ext_config = {"version": __version__, "parallel_read_safe": True}
    if isinstance(app, Sphinx):
        app.setup_extension("sphinx.ext.autodoc")  # Ensure auto-doc is set-up
        for documenter in get_documenters(__name__):
            LOGGER.info(
                f"Exposing auto{documenter.objtype} directive to Numba types",
                color="green",
            )
            app.registry.add_documenter(documenter.objtype, documenter)
    else:  # probably called by tests
        pass

    return ext_config
