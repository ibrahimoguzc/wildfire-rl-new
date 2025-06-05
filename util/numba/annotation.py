# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""This module aims to convert Python type-hints into Numba signatures.

One of the issues with testing Numba is that `coverage.py`_ will not
work as it cannot trace the path within the JIT-compiled Python
function. Therefore, a workaround is to first test the underlying Python
function, given by :py:attr:`py_func`, and later check to make sure that
it can be compiled during additional tests. Furthermore, due to how
the types of objects passed to Numba determine if the function can
be JIT compiled, it is important to declare what types the function can
be used with.

However, it is redundant to provide this type information in both the
Numba signature as well as in the Python type-hints of the
:py:attr:`py_func`. Therefore, it makes sense to use this typing
information as a Numba signature since then it is possible to retrieve
typing information for Numba, Sphinx, and Mypy from a single source of
truth. This also enforces correct type-hints as Numba will not be able
to compile if the user provides an incorrect (or missing) type-hint.

The downside to this is that Numba, as of version 0.46, offers only
type-inference for actual data being passed to a function at
compile-time and has no functionality to translate :py:obj:`type`.
Therefore, this module is required to convert these type-hints, into
Numba compatible signatures in order for eager JIT compilation to work.

Note:
    AOT Compilation for JIT funcitons is not recommended as shown by
    `chrisb`_ and `MSeifert`_. Therefore, AOT compilation is not tested
    and also not supported.

See Also:
    Numba issue `#2234`_

.. _coverage.py: https://coverage.readthedocs.io/en/latest/
.. _chrisb: https://stackoverflow.com/a/51089874
.. _MSeifert: https://stackoverflow.com/q/35121091
.. _#2234: https://github.com/numba/numba/issues/2334

"""

import inspect
import sys
import typing
from collections.abc import Callable
from typing import Any

import numba

# TODO add support for inspecting the output as string
# TODO add support for converting Numpy types

# Sets the default precision in bits based on system type
type_suffix = "64" if sys.maxsize > 2**32 else "32"
BUILTIN_MAPPING = {
    int: getattr(numba, f"int{type_suffix}"),
    float: getattr(numba, f"float{type_suffix}"),
    bool: numba.boolean,
    complex: getattr(numba, f"float{type_suffix}"),
    tuple: numba.types.Tuple,
    dict: numba.types.DictType,
    list: numba.types.List,
    set: numba.types.Set,
    # None is a singleton instance, therefore we must use type(None)
    type(None): numba.types.void,
}
"""Dict[Type, Type]: Mapping from Python builtins to Numba types.

This dictionary should be expanded first before creating a new
:py:class:`TypeHintConverter`, unless dealing with edge-cases such as
:py:class:`typing._GenericAlias` or :py:class:`typing.Optional`.
"""


def get_signature(func: Callable):
    """Gets a Numba signature from the annotations of ``func``.

    Args:
        func: A Python function (i.e. the :py:attr:`py_func` of a
            :py:func:`numba.jit` decorated function)

    Returns:
        A Numba signature that can be used to eager JIT-compile

    """
    typehints, return_index = list(func.__annotations__.values()), -1
    assert list(func.__annotations__.keys())[return_index] == "return"

    numba_types = []
    for type_hint in typehints:
        converter = TypeHintConverter.get_converter(type_hint)
        numba_types.append(converter(type_hint).get_signature())
    return_type = numba_types.pop(return_index)
    return return_type(*numba_types)


class RegisterConverter(type):
    """Registers any :py:class:`TypeHintConverter` on construction.

    As this is a metaclass definition, anytime a subclass of
    :py:class:`TypeHintConverter is defined in any file, it will be
    added to the :py:attr:`__converters__`. This attribute is then also
    accessible by subclasses.
    """

    def __init__(cls, name, bases, cls_dict):
        super().__init__(name, bases, cls_dict)

        # bases ensures this doesn't run for the TypeHintConverter ABC
        if cls not in cls.__converters__.values() and bases:
            cls.__converters__.update((c, cls) for c in cls.can_convert)


class TypeHintConverter(metaclass=RegisterConverter):
    """Base-Class for all type-hint converters."""

    #: Registry of converters populated by the metaclass
    __converters__ = {}

    #: Optimization step to reduce size of the object dictionary
    __slots__ = ["type_hint"]

    #: The type-class for which the converter can get a Numba signature
    can_convert = NotImplementedError

    def __init__(self, type_hint: Any):
        self.type_hint = type_hint

    def get_signature(self):
        """Gets the correct Numba signature for a single type-hint."""
        return self.type_hint

    @classmethod
    def get_converter(cls, type_hint: Any) -> type:
        """Gets the converter required for ``type_hint``.

        Converters register which class they can convert, therefore use
        of :py:meth:`ensure_class` on ``type_hint`` is warranted.
        However, :py:meth:`get_signature` shouldn't invoke
        :py:meth:`ensure_class` on :py:attr:`type_hint` since we want to
        convert the type hint object and not its class (type).
        """
        try:
            return cls.__converters__[cls.ensure_class(type_hint)]
        except KeyError:
            raise KeyError(f"No registered converter found for {type_hint}")

    @staticmethod
    def ensure_class(type_hint: Any) -> type:
        """Ensures that the ``type_hint`` is a type object (class)."""
        return type_hint if inspect.isclass(type_hint) else type_hint.__class__


class BuiltinConverter(TypeHintConverter):
    """Converts Python Built-in Data Types."""

    can_convert = list(BUILTIN_MAPPING.keys())

    # Adding None object into mapping (this returns a new dict)
    mapping = {**BUILTIN_MAPPING, None: BUILTIN_MAPPING[type(None)]}

    # TODO Remove try except to simplify the code
    def get_signature(self) -> numba.types.abstract.Type:
        """Gets the corresponding Numba signature for a builtin type."""
        try:
            return self.mapping[self.type_hint]
        except KeyError:
            raise KeyError(f"No registered Numba type for {self.type_hint}")


class NumbaConverter(TypeHintConverter):
    """Registers valid Numba signature types to the converter.

    No operation is done on these types, as they are already valid
    Numba signatures!
    """

    can_convert = {
        numba.types.Boolean,
        *numba.types.abstract.Type.__subclasses__(),
        *numba.types.scalars.Number.__subclasses__(),
        *numba.types.Sequence.__subclasses__(),
        *numba.types.containers.Buffer.__subclasses__(),
    }


class GenericAliasConverter(TypeHintConverter):
    """Converts type-hints such as Tuple[Tuple[int, int]].

    Note:
        This converter is complicated by the fact that Optional[int]
        which is a :py:class:`typing._GenericAlias` shares its
        :py:attr:`__origin__` with :py:class:`typing.Union`. Currently,
        Union is not a supported Numba type. Therefore, the selected
        option is to simply convert all Union types to Optional types.
        However, this may require tweaking in the future.

    """

    can_convert = [
        typing._GenericAlias,
        typing._UnionGenericAlias if sys.version_info > (3, 9) else (),
    ]

    #: Handling problematic origins such as typing.Optional
    origin_mapping = {typing.Union: numba.types.Optional}

    @property
    def origin_signature(self) -> numba.types.abstract.Type:
        """The converted Numba class of the :py:class:`_GenericAlias`.

        Examples:
            >>> conv = GenericAliasConverter
            >>> conv(List[int]).origin_signature
            numba.types.List
            >>> conv(Tuple[int]).origin_signature
            numba.types.Tuple

        """
        origin = self.type_hint.__origin__
        if origin in self.origin_mapping:
            return self.origin_mapping[origin]
        converter = self.get_converter(origin)
        return converter(origin).get_signature()

    @property
    def args_signature(self) -> tuple[numba.types.abstract.Type]:
        """The arguments provided to :py:class:`_GenericAlias`.

        Caution:
            Return type must be a :py:obj:`tuple` if there is more than
            one argument. If the return type is a :py:obj:`list` Numba
            will raise an error about a list being unhashable.

        """
        args_signature = []
        for arg in self.type_hint.__args__:
            converter = self.get_converter(arg)
            args_signature.append(converter(arg).get_signature())
        return (
            tuple(args_signature)
            if len(args_signature) > 1
            else args_signature[0]
        )

    def get_signature(self):
        """Gets the fully-formed Numba signature of the GenericAlias."""
        if self.origin_signature is numba.types.Optional:
            return self.origin_signature(self.args_signature[0])
        return self.origin_signature(self.args_signature)
