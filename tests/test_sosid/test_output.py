# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from sosid.output import Output


class Foo:
    @Output
    def spam(self):
        return "I am spam"


def test_output_class_level():
    assert isinstance(Foo.spam, Output)
    assert Foo.spam.name == "spam"


def test_output_instance_level():
    foo = Foo()
    assert foo.spam == "I am spam"


def test_output_is_settable():
    foo = Foo()
    foo.spam = "eggs"
    assert foo.spam == "eggs"

    # Ensure mutation is bound to instance
    foo = Foo()
    assert foo.spam == "I am spam"
