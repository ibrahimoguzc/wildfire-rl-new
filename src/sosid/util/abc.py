# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains a modified ABCMeta class from the Python Standard Library.

This modification allows one to enforce the creation of object
attributes through the use of the abstractattribute decorator. The
original code for this modification was created by `krassowski`_
and released under Creative Commons.

.. _krassowski: https://stackoverflow.com/a/50381071
"""

# Renaming to avoid naming conflict
from abc import ABCMeta as ABCMetaStdLib
from abc import abstractclassmethod, abstractmethod, abstractstaticmethod
from collections.abc import Callable
from functools import update_wrapper
from typing import TYPE_CHECKING, TypeVar

__all__ = [
    "ABC",
    "ABCMeta",
    "abstractattribute",
    "abstractclassmethod",
    "abstractinterface",
    "abstractmethod",
    "abstractstaticmethod",
]

CallableT = TypeVar("CallableT", bound=Callable)

if TYPE_CHECKING:  # Add support for static type checkers
    from builtins import property as abstractattribute
else:

    class abstractattribute:
        """A decorator indicating abstract class or instance attributes.

        This can either be used as a decorator or assigned directly to a
        class attribute. Both usages are as follows::

            class C(metaclass=ABCMeta):
                foo = abstractattribute()

                @abstractattribute
                def bar(self): ...

        The second usage style is beneficial to add documentation and
        type annotation to the abstract attribute.

        Caution: The :py:class:`abstractattribute` decorator cannot be
            used along with the :py:class:`property` decorator.
        """

        __isabstractattribute__ = True  # Not required but adheres to ABC style

        def __init__(self, funcobj: callable = lambda obj: NotImplemented):
            update_wrapper(self, funcobj)  # Updating docs/annotations of attr


def abstractinterface(funcobj: CallableT) -> CallableT:
    """A decorator indicating a required interface for composition.

    An abstractinterface can be defined within the component class,
    and enforces that interface be defined in any composite class
    using it in a composition. The component class must inherit
    metaclass `ComponentMeta`.

    This function can both be used to refer to methods on the composites
    or to properties. ::

        class C(metaclass=ComponentMeta):
            @abstractinterface
            def foo(self, bar): ...

            @property
            @abstractinterface
            def bar(self): ...

    Note: Component class must inherit from `ComponentMeta` for the
    interfaces to be enforced.
    """
    funcobj.__isabstractinterface__ = True
    return funcobj


class ABCMeta(ABCMetaStdLib):
    """Metaclass for defining Abstract Base Classes (ABCs).

    Use this metaclass to create an ABC.  An ABC can be subclassed
    directly, and then acts as a mix-in class.  You can also register
    unrelated concrete classes (even built-in classes) and unrelated
    ABCs as 'virtual subclasses' -- these and their descendants will be
    considered subclasses of the registering ABC by the built-in
    issubclass() function, but the registering ABC won't show up in
    their MRO (Method Resolution Order) nor will method implementations
    defined by the registering ABC be callable (not even via super()).

    Note:
        This extends the `ABCMeta` class from the Python Standard
        Library in order to run a post-initialization check that allows
        for usage of abstract instance attributes.
    """

    def __init__(cls, name, bases, cls_dict):
        """Adds a :py:attr:`__abstractattributes__` set inside ``cls``.

        This method is called multiple times if inheriting from a base
        class which specifies :py:class:`ABCMeta` as its metaclass.
        As a result, if a set already exists from the base-class, it is
        copied to prevent modifying it in-place and then updated.
        """
        super().__init__(name, bases, cls_dict)

        # Get abstract attributes from base class.
        abstracts = set(getattr(cls, "__abstractattributes__", set()))

        # Updating set with all abstract attributes within ``cls``
        abstracts.update(
            attr
            for attr, obj in cls_dict.items()
            if isinstance(obj, abstractattribute)
        )
        cls.__abstractattributes__ = frozenset(abstracts)

    def __call__(cls, *args, **kwargs):
        """Checks if all :py:class:`abstractattribute` are overridden.

        This method is called on class instantiation and is responsible
        for returning the instantiated object. After the object is
        instantiated, all :py:class:`abstractattribute` definitions are
        checked.

        Raises:
            TypeError: If any :py:class:`abstractattribute` are not
                overridden by class or instance attributes.

        """
        # Instantiating the class using method from type
        # This calls :py:meth:`__init__()` and returns an object
        obj = super().__call__(*args, **kwargs)

        # Checking if abstract attributes exist post __init__() on obj
        abs_attrs = cls.__abstractattributes__  # Preventing multiple LOAD_FAST
        if abs_attrs:
            obj_abstractattributes = {
                attr
                for attr in abs_attrs
                if isinstance(getattr(obj, attr), abstractattribute)
            }

            # Raising error if any abstractattributes exist
            if obj_abstractattributes:
                raise TypeError(
                    "Can't instantiate abstract class {} without"
                    " abstract attributes: {}".format(
                        cls.__name__, ", ".join(sorted(obj_abstractattributes))
                    )
                )
        return obj


class ABC(metaclass=ABCMeta):
    """Provides a standard way to create an ABC using inheritance."""

    __slots__ = ()


class ComponentMeta(ABCMeta):
    """Metaclass for all component classes requiring an interface.

    This metaclass can be used to create components that require access
    to certain attributes or methods of the composite class.
    """

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        def create_property(attr: str, property_ns: property) -> property:
            def getter(self):
                return getattr(self.__composite_instance__, attr)

            def setter(self, value):
                setattr(self.__composite_instance__, attr, value)

            def deleter(self):
                delattr(self.__composite_instance__, attr)

            if property_ns.fget is None:
                getter = None
            else:
                update_wrapper(getter, property_ns.fget)
            if property_ns.fset is None:
                setter = None
            else:
                update_wrapper(setter, property_ns.fset)
            if property_ns.fdel is None:
                deleter = None
            else:
                update_wrapper(deleter, property_ns.fdel)
            return property(getter, setter, deleter)

        abstracts = {
            attr
            for attr, value in namespace.items()
            if getattr(value, "__isabstractinterface__", False)
        }
        for base in bases:
            for attr in getattr(base, "__abstractinterfaces__", set()):
                if getattr(
                    namespace.get(attr), "__isabstractinterface__", False
                ):
                    abstracts.add(attr)
        abstract_properties = {
            attr
            for attr, obj in namespace.items()
            if (
                isinstance(obj, property)
                and getattr(obj.fget, "__isabstractinterface__", False)
            )
        }
        abstracts.update(abstract_properties)

        for attr in abstract_properties:
            namespace[attr] = create_property(attr, namespace[attr])

        namespace["__abstractinterfaces__"] = frozenset(abstracts)
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __call__(cls, *args, **kwargs):
        """Checks if all :py:class:`abstractinterfaces` are overridden.

        This method is called on class instantiation and is responsible
        for returning the instantiated object. After the object is
        instantiated, all :py:class:`abstractinterfaces` definitions in
        the composite instance are checked. In addition, the
        abstractinterface definition in component class instance is
        linked to that of the composite class instance.

        Raises:
            TypeError: If any :py:class:`abstractinstances` are not
            defined properly or are missing in the composite instance.

        """
        base_msg = (
            f"Can't instantiate component class {cls.__name__} without"
            f" defining"
        )
        # Instantiating the class using method from type
        # This calls :py:meth:`__init__()` and returns an object
        obj = super().__call__(*args, **kwargs)
        composite = getattr(obj, "__composite_instance__", None)
        if composite is None:
            raise TypeError(f"{base_msg} the composite instance.")

        # Checking if abstractinterfaces exist post __init__() on obj
        abs_ints = cls.__abstractinterfaces__  # Preventing multiple LOAD_FAST
        composite_cls = composite.__class__
        undefined_interfaces = {
            attr for attr in abs_ints if not hasattr(composite_cls, attr)
        }
        if undefined_interfaces:
            msg = (
                f"{base_msg} interface methods or properties in"
                f" {composite_cls.__name__}: {undefined_interfaces}."
            )
            raise TypeError(msg)
        for attr in abs_ints:
            if isinstance(getattr(cls, attr), property):
                component_prop = getattr(cls, attr)
                composite_prop = getattr(composite_cls, attr)
                if not isinstance(composite_prop, property):
                    msg = (
                        f"{base_msg} {attr} in {composite_cls.__name__} as a"
                        f" property."
                    )
                    raise TypeError(msg)
                required_methods = {
                    method
                    for method in ("fget", "fset", "fdel")
                    if getattr(component_prop, method) is not None
                }
                available_methods = {
                    method
                    for method in required_methods
                    if getattr(composite_prop, method) is not None
                }
                if missing := required_methods - available_methods:
                    msg = (
                        f"{base_msg} {attr} in {composite_cls.__name__} with"
                        f" methods: {missing}."
                    )
                    raise TypeError(msg)
            else:
                func = getattr(composite, attr)
                if not callable(func):
                    msg = (
                        f"{base_msg} {attr} in {composite_cls.__name__} as a"
                        f" callable."
                    )
                    raise TypeError(msg)
                setattr(obj, attr, func)
        return obj


class Component(metaclass=ComponentMeta):
    """Base class for all component class requiring an interface."""

    __slots__ = ("__composite_instance__",)

    def __init__(self, composite: object) -> None:
        self.__composite_instance__ = composite
        super().__init__()
