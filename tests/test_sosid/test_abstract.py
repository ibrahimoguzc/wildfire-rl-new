# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from sosid.abstract import Writable
from sosid.output import Output


class Foo(Writable):
    @Output()
    def spam(self):  # noqa: D102
        return "I am spam"


class Bar(Foo):
    @Output()
    def eggs(self):  # noqa: D102
        return "I am eggs"


def test_writable():
    """Test that :py:class:`Writable` gathers outputs of superclases."""
    bar = Bar()
    assert bar.output_collector() == {  # noqa: S101
        "spam": "I am spam",
        "eggs": "I am eggs",
    }


def test_writable_inheritance():
    """Test that :py:class:`Output` can be inherited."""

    class Ham(Foo):
        @Output()
        def bacon(self):
            return "I am bacon"

    class MultiInheritance(Bar, Ham):
        pass

    multi = MultiInheritance()
    assert multi.output_collector() == {  # noqa: S101
        "spam": "I am spam",
        "bacon": "I am bacon",
        "eggs": "I am eggs",
    }


def test_writable_overwriting():
    """Tests overwritting a :py:class:`Output` on subclass.

    The intended behavior is that the output of the superclass
    will not be included.
    """

    class Quz(Bar):
        def eggs(self):
            return "I am eggs"

    quz = Quz()
    assert quz.output_collector() == {"spam": "I am spam"}  # noqa: S101


def test_writable_multiple_definitions():
    """Test that a :py:class:`Output` is only collected once."""

    class Quz(Bar):
        @Output()
        def eggs(self):
            return "I am new eggs"

        @Output()
        def spam(self):
            return "I am spam"

    quz = Quz()
    assert quz.output_collector() == {  # noqa: S101
        "spam": "I am spam",
        "eggs": "I am new eggs",
    }
