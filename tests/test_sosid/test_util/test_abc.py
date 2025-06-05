# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import re

import pytest

from sosid.util.abc import (
    ABC,
    ABCMeta,
    Component,
    abstractattribute,
    abstractclassmethod,
    abstractinterface,
    abstractmethod,
    abstractstaticmethod,
)


def assert_type_error(abstract_cls, abstract_attrs: str, *args) -> None:
    """Validates the raised :py:exception:`TypeError`.

    Args:
        abstract_cls: The class using :py:class:`ABCMeta` to instantiate
        abstract_attrs: The names of the abstract attributes which
            should appear in the :py:exception:`TypeError`

    """
    with pytest.raises(
        TypeError,
        match=re.compile(f".*({abstract_attrs})"),
    ):
        abstract_cls(*args)


def test_std_library():
    """Tests ABC decorators from the Standard Library."""

    class StdLibraryTester(metaclass=ABCMeta):
        @abstractclassmethod
        def method(cls): ...

        @abstractmethod
        def class_method(self): ...

        @abstractstaticmethod
        def static_method(self): ...

    assert_type_error(StdLibraryTester, "class_method, method, static_method")


def test_ABC():
    """Tests if the ABC helper class functions properly."""

    class ABCTester(ABC):
        @abstractmethod
        def method(self): ...

    with pytest.raises(
        TypeError,
        match=re.compile(".*(method)"),
    ):
        ABCTester()


SINGLETON_OBJ = object()


class SimpleUsage(metaclass=ABCMeta):
    """Example of one-line usage of abstractattribute."""

    abstract_attribute = abstractattribute()

    def __init__(self):
        self.abstract_attribute = SINGLETON_OBJ


class DecoratorUsage(metaclass=ABCMeta):
    """Example of decorator usage of abstractattribute."""

    def __init__(self):
        self.abstract_attribute = SINGLETON_OBJ

    @abstractattribute
    def abstract_attribute(self) -> object:
        """Sample documentation."""


@pytest.mark.parametrize("usage", [SimpleUsage, DecoratorUsage])
def test_classattribute(usage, monkeypatch):
    """Checks if class attributes function properly."""
    # Removing instance attribute by using __init__ of object
    monkeypatch.setattr(usage, "__init__", object.__init__)

    # Tests if the correct error is raised by the abstract attribute
    assert_type_error(usage, "abstract_attribute")

    # Checking if replacing the abstract attribute raises no error
    monkeypatch.setattr(usage, "abstract_attribute", SINGLETON_OBJ)
    assert usage()


@pytest.mark.parametrize("usage", [SimpleUsage, DecoratorUsage])
def test_instanceattribute(usage, monkeypatch):
    """Checks if instance attributes function properly."""
    # Checking if abstract attribute replaced in __init__ raises
    # no errors
    obj = usage()
    assert obj.abstract_attribute is SINGLETON_OBJ

    # Removing instance attribute by using __init__ of object
    monkeypatch.setattr(usage, "__init__", object.__init__)

    # Tests if the correct error is raised by the abstract attribute
    assert_type_error(usage, "abstract_attribute")


def test_updatewrapper():
    """Tests if the docstring/annotations are correctly updated."""
    assert DecoratorUsage.abstract_attribute.__doc__ == "Sample documentation."
    assert DecoratorUsage.abstract_attribute.__annotations__ == {
        "return": object
    }


def test_call():
    """Tests if overridding :py:meth:`__call__` functions ."""

    class CustomCall(SimpleUsage):
        foo = None

        def __call__(self, foo_value):
            self.foo = foo_value

    obj = CustomCall()
    obj(True)
    assert obj.foo


def test_property():
    """Tests if :py:class:`property` decorator methods are lazy.

    The asserts check the following:

    #. Checking if abstract attribute exists within the test class

    #. Ensuring property method not called on class initialization

    #. Initializing object from test-class class

    #. Checking if property method is once again not called

    #. Making sure poperty method can return a value

    #. Ensuring that calling foo updates property_called
    """

    class LazyProperty(SimpleUsage):
        property_called = False

        @property
        def foo(self):
            self.property_called = True
            return True

    assert "abstract_attribute" in LazyProperty.__abstractattributes__
    assert not LazyProperty.property_called
    obj = LazyProperty()
    assert not obj.property_called
    assert obj.foo
    assert obj.property_called


def test_slots():
    """Tests proper function on a class that defines slots."""

    class SlotsDefinition(metaclass=ABCMeta):
        __slots__ = ["spam"]

        def __init__(self):
            self.spam = None

        foo = abstractattribute()

        @abstractattribute
        def bar(self): ...

    # Checking that abstract attribute retrieval works with __slots__
    assert all(
        attr in SlotsDefinition.__abstractattributes__
        for attr in ("foo", "bar")
    )
    assert_type_error(SlotsDefinition, "bar, foo")


def test_inheritance():
    """Tests if abstract attributes of are inherited."""

    class Base(metaclass=ABCMeta):
        foo = abstractattribute()

    class Specialization(Base):
        bar = abstractattribute()

    # Testing that derived class inherits abstract attrs from base-class
    assert all(
        attr in Specialization.__abstractattributes__
        for attr in ("foo", "bar")
    )
    # Testing that __abstractattributes__ in Base are unchanged
    assert "bar" not in Base.__abstractattributes__


class DecoratorComponent(Component):
    """Class used in composition and defines an abstract interface."""

    @abstractinterface
    def abstract_interface(self):
        pass


class PropertyComponent(Component):
    """Class used in composition and defines an abstract interface."""

    @property
    @abstractinterface
    def abstract_interface(self):
        pass


class PropertyComponentWithSetDel(Component):
    @property
    @abstractinterface
    def abstract_interface(self):
        pass

    @abstract_interface.setter
    def abstract_interface(self, value):
        pass

    @abstract_interface.deleter
    def abstract_interface(self):
        pass


class CompatibleComposite:
    """Composite class which implements an interface."""

    def abstract_interface(self):
        return SINGLETON_OBJ


class IncompatibleComposite:
    """Composite class which does not implement an interface."""


class MutableCompatibleComposite:
    """Composite class which implements a mutable interface."""

    def __init__(self, value):
        self.value = value

    def step(self):
        self.value += 1

    def abstract_interface(self):
        return self.value


class CompositeWithMinorPropertyInterface:
    """Composite class which implements an property as interface."""

    def __init__(self, value):
        self.value = value

    @property
    def abstract_interface(self):
        return self.value


class CompositeWithFullPropertyInterface:
    """Composite class which implements an property as interface."""

    def __init__(self, value):
        self.value = value

    def step(self):
        self.value += 1

    @property
    def abstract_interface(self):
        return self.value

    @abstract_interface.setter
    def abstract_interface(self, value):
        self.value = value

    @abstract_interface.deleter
    def abstract_interface(self):
        self.value = None


def test_interface_enforcement():
    """Test whether interface is enforced."""
    composite = CompatibleComposite()
    obj = DecoratorComponent(composite)
    assert obj.abstract_interface() is SINGLETON_OBJ
    composite = IncompatibleComposite()
    assert_type_error(DecoratorComponent, "abstract_interface", composite)


def test_interface_mutability():
    """Test whether interface link is mutable."""
    composite = MutableCompatibleComposite(value=1)
    obj = DecoratorComponent(composite)
    assert obj.abstract_interface() == 1
    composite.step()
    assert obj.abstract_interface() == 2


@pytest.mark.parametrize(
    ("component_cls", "has_setter", "has_deleter"),
    [
        (PropertyComponent, False, False),
        (PropertyComponentWithSetDel, True, True),
    ],
)
def test_interface_property(component_cls, has_setter, has_deleter):
    """Test whether incorrect usage of interface is caught."""
    composite = CompositeWithFullPropertyInterface(value=1)
    obj = component_cls(composite)
    assert obj.abstract_interface == 1
    composite.step()
    assert obj.abstract_interface == 2
    if has_setter:
        obj.abstract_interface = 4
        assert composite.abstract_interface == 4
    else:
        with pytest.raises(AttributeError):
            obj.abstract_interface = 4
    if has_setter:
        del obj.abstract_interface
        assert composite.abstract_interface is None
    else:
        with pytest.raises(AttributeError):
            del obj.abstract_interface


@pytest.mark.parametrize(
    ("composite", "component_cls"),
    [
        (CompatibleComposite(), PropertyComponent),
        (CompositeWithFullPropertyInterface(1), DecoratorComponent),
        (CompositeWithMinorPropertyInterface(1), PropertyComponentWithSetDel),
    ],
)
def test_interface_property_invalid(composite, component_cls):
    """Test whether incorrect usage of interface is caught."""
    assert_type_error(component_cls, "abstract_interface", composite)
