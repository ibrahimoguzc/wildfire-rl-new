# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Collection of re-usable tests (snippets) that can be imported.

This solves the problem of typically violating the Don't Repeat Yourself
(DRY) principal during testing.

For snippets that begin with "test" you can simply import it into a test
module and let pytest discover that snippet automatically::

    from tests.snippets import test_jit_compile  # noqa: F401

"""

from typing import Any

import pytest


@pytest.mark.retrieve_jit_funcs
def test_jit_compile(jit_func, compile_jit):
    """Tests if imported JIT funcs compile eagerly Just-in-Time (JIT).

    Note:
        ``jit_func`` and ``compile_jit`` are Pytest fixtures that are
        examples of Dependency Injection. To understand how this works
        refer to `pytest.fixtures`_ and the implementation in the
        `conftest.py` file in the `tests/` directory.

    .. _pytest.fixtures: https://docs.pytest.org/en/2.8.7/fixture.html

    """
    compile_jit(jit_func)


class Scenario:
    """Lazy container for a test class instantiation.

    Due to the combined use of the :py:class:`functools.property`
    decorator with a cached value stored in :py:attr:`__object__`,
    :py:attr:`obj` is evaluated lazily when required by a test and
    not instantiated during test-discovery. This ensures that the
    test-discovery process is not bogged down by slow object
    instantiations. This is especially useful when the user wants to
    test only a small portion of the code-base.

    Args:
        test_class: Test class to be lazily instantiated after discovery
        label: Label of the current scenario
        args: Positional arguments passed to ``test_class``
        kwargs: Keyword arguments passed to ``test_class``

    Note:
        It is important to create this additional Scenario object per
        scenario definition, since we cannot guarantee that the
        ``args`` and ``kwargs`` supplied by the user will be hashable!

    """

    # Reduces memory-usage of the Scenario object by avoiding dict
    __slots__ = ["__object__", "args", "kwargs", "label", "test_class"]

    def __init__(self, test_class: type, label: str, args, kwargs):
        self.test_class = test_class
        self.label = label
        self.args = args
        self.kwargs = kwargs
        self.__object__ = None

    @property
    def obj(self) -> Any:
        """Returns the instantiated :py:attr:`test_class`."""
        if self.__object__:
            return self.__object__
        return self.test_class(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        """Override string representation method to return label.

        This provides a more informative message in the event of a
        failure with the current :py:class:`Scenario`.
        """
        return self.label


# TODO add set_attributes method
class ScenarioTestSuite:
    """Paramterizes a `test_class` as definited by :py:attr:`scenarios`.

    This implements the functionality provided by `testscenarios`_ for
    pytest. The advantage is to have a low verbosity way to define many
    potential objects, and have them instantiated only once. The basic
    usage is simple, but requires that the testclass inherits from
    ScenarioTestSuite as follows::

        class Foo:  # A class we want to parametrically instantiate
            def __init__(self, spam: int = 0, *, ham: int = 1.0):
                self.spam = spam
                self.ham = ham

            def bar(self):
                return self.spam + self.ham


        class TestClass(ScenarioTestSuite):
            test_class = Foo
            scenarios = {
                "argnames": "label, args, kwargs",
                "argvalues": [
                    ("Foo1", (1,), {"ham": 1}),
                    ("Foo2", (2,), {"ham": 2}),
                ],
            }

            def test_bar(self, scenario):
                expected_results = {"Foo1": 2, "Foo2": 4}
                result = scenario.obj.bar()
                assert result == expected_results[scenario.label]

    Attributes:
        test_class: Test class to be parametrically instantiated
        scenarios: Dictionary containing scenario labels and arguments
        scenarios_typespec: Defines keys and types necessary for
            a valid definition of a :py:attr:`scenarios` dictionary
        argtypes_typespec: Defines keys and types necessary for a
            a valid argtypes definition within :py:attr:`scenarios`

    .. _testscenarios: https://pypi.org/project/testscenarios/

    """

    test_class: type
    scenarios: dict

    scenarios_typespec = {"argnames": str, "argvalues": list}
    argtypes_typespec = {"label": str, "args": tuple, "kwargs": dict}

    def pytest_generate_tests(self, metafunc):
        """Overrides generation of tests for ``metafunc`` defs in class.

        Currently, the implementation below checks if a test asks for
        the `scenario` fixture. If it does, then it paramterizes the
        test provided by ``metafunc`` by injecting a
        :py:class:`Scenario` instance with `label`, `args`, and `kwargs`
        specified by the :py:attr:`scenarios` dictionary into the
        scenario fixture.

        Args:
            metafunc: Test inspection helper that is used for generating
                tests based on test configuration or values specified in
                the class or module where a test function is defined

        Note:
            Although it may seem like extra effort to specify a
            scenarios :py:obj:`dict` with ``label``, ``args``, and
            ``kwargs`` and then instantiate it later through the
            ``scenario`` fixture, the benefit of this approach is that
            it adheres to Pytest parametrization syntax and is truly the
            shortest way to write test-scenarios. For instance, if
            :py:class:`Scenario` objects were instantiated directly
            outside of the test-class, then one would be required to
            repeat the words `Scenario` and the test class `Foo`
            needlessly as follows::

                FOO_SCENARIOS = {
                    "argnames": "scenario_object",
                    "argvalues": [
                        (Scenarios("Foo1", Foo, (2,), {"ham": 1})
                        (Scenarios("Foo1", Foo, (2,), {"ham": 2})
                    ]
                }

        """
        if self.is_valid_scenarios():
            if "scenario" in metafunc.fixturenames:
                argvalues = self.get_completed_argvalues()
                metafunc.parametrize(
                    # Full list of argnames required for fixture
                    argnames="scenario_object",
                    argvalues=self.get_scenarios(argvalues),
                    scope="class",
                    ids=self.get_labels(argvalues),
                )

    @pytest.fixture(scope="class")
    def scenario(self, scenario_object: Scenario) -> Scenario:
        """Fixture responsible for providing Scenario instances."""
        return scenario_object

    @property
    def specified_argnames(self) -> list[str]:
        """Lists argnames specified by :py:attr:`scenarios`."""
        return self.scenarios["argnames"].replace(" ", "").split(",")

    def get_completed_argvalues(self) -> list[tuple[str, tuple, dict]]:
        """Ensures Scenario specifications are complete.

        If a scenario specification has been provided that lacks
        one of "label", "args", or "kwargs", these values will be
        filled in by empty instantiations of their respective type
        specified by :py:attr:`argtypes_typespec`. An example of this is
        how if the user only wants to specify positional arguments, the
        keyword arguments will be filled in with an empty dict::

            TESTCLASS_SCENARIOS = {
                "argnames": "label, args",
                "argvalues": [("s1", (170,)), ("s2", (2,))],
            }

            completed_argvalues = [("s1", (170,), {}), ("s2", (2,), {})]

        Furthermore, this ensures that the user can specify the
        arguments in any order they like, as long as it matches the
        argnames they have specified in the scenarios dictionary of the
        test class.
        """
        # Placeholder dictionary with valid types instantiated
        complete_argdict = {k: v() for k, v in self.argtypes_typespec.items()}
        completed_argvalues = []
        argnames = self.specified_argnames
        for argvalue in self.scenarios["argvalues"]:
            # Ensuring scenario has the correct number of arguments
            assert len(argvalue) == len(argnames), (
                f"Scenario {argvalue} is missing an argvalue"
            )

            specified_argdict = {}  # Dictionary of specified arguments
            for i, arg in enumerate(argvalue):
                # Getting name of current argument
                argname = argnames[i]

                # Ensuring current argument is of the correct type
                assert isinstance(arg, self.argtypes_typespec[argname])

                # Adding current argument by name to specified_argdict
                specified_argdict[argname] = arg

            # Overwriting complete_argdict with specified arguments
            complete_argdict.update(specified_argdict)
            completed_argvalues.append(tuple(complete_argdict.values()))

        return completed_argvalues

    def get_labels(self, argvalues) -> list[str] | None:
        """Retrieves label from validated_argvalues.

        Note:
            Due to the order of :py:attr:`argtypes_typespec` the label
            will always be at index 0.

        """
        if "label" in self.specified_argnames:
            return [argvalue[0] for argvalue in argvalues]

    def get_scenarios(
        self, argvalues: list[tuple[str, tuple, dict]]
    ) -> list[tuple[Scenario]]:
        """Instantiates :py:class:`Scenario` list from ``argvalues``."""
        scenarios = []
        for label, args, kwargs in argvalues:
            scenario = Scenario(
                test_class=self.test_class,
                label=label,
                args=args,
                kwargs=kwargs,
            )
            scenarios.append(scenario)
        return scenarios

    def is_valid_scenarios(self) -> bool | None:
        """Checks if :py:attr:`scenarios` is valid."""
        assert hasattr(self, "scenarios"), "No scenarios definition provided"
        assert isinstance(self.scenarios, dict), "Scenarios must be a dict"
        assert all(k in self.scenarios for k in self.scenarios_typespec.keys())
        assert all(
            isinstance(self.scenarios[k], v)
            for k, v in self.scenarios_typespec.items()
        ), f"Scenarios must be specified in {self.scenarios_typespec} format"
        return True
