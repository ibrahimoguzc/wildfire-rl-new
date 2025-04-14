# Testing Numba JIT Functions

* Status: Accepted
* Deciders: San Kilkis
* Date: 2019-09-26

## Context and Problem Statement

The main problem encountered while testing Numba is that sometimes there exists
a discrepancy between what the programmer wanted to write, and what Numba can
compile or what the compiled JIT-function outputs. In order to address this
uncertainty, it is very important to strictly test Numba JIT functions as
recommended by a [SciPy 2017 Tutorial]. Making matters worse, [coverage.py]
cannot run on JIT-compiled functions and therefore, will output 0% coverage for
these lines. As a result, it becomes more difficult to understand which paths
of a function have been tested. Therefore, there is a need to come up with a
proper testing methodology that can address these concerns.

## Decision Drivers

* Verbosity/Boilerplate
* Coverage
* Performance

## Considered Options

* Test the JIT-function as any other
* Test the Underlying Python function
* Test both the JIT-function and Python Function and Compare Results

## Decision Outcome

Chosen option: Test both the JIT-function and Python Function and Compare
Results. The proposed workaround to this problem is to first test the
underlying Python function, given by `py_func`, compare it with the
JIT-compiled result, and later check function annotations with additional
tests. The first two steps are the most important ones as it is important to
make sure that the Python code and JIT-compiled code produce the same results,
especially when using `fast_math` flags. The following testing strategy is
adopted

1. Run both the underlying Python function and JIT-compiled function
2. Compare results of underlying Python function and JIT-compiled function
3. **Optional:** Test JIT compilation with function annotations (Highly
   Recommended)

``` python
from foo.jit_funcs import bar  # bar is a numba.jit decorated function
from tests.snippets import test_jit_compile  # noqa: F401

# The above snippet is recognized by pytest, which then JIT compiles bar from
# its annotations when test_jit_compile is run.

test_function(func_args, expected_result):
    result, jit_result = bar.py_func(*func_args), bar(func_args)
    assert result == jit_result
    assert result == expected_result
```

Refer to [Option 3] for an example of a test implementing only Step 1 and 2.

### Positive Consequences

* Confidence in JIT compiled functions increase
* [coverage.py] works, increasing insight into which code paths are tested

### Negative Consequences

* Tests take more time to execute, especially on slower machines
* Tests become more verbose

### Motivation for Testing JIT Compilation from Annotations (Step 3)

Although Steps 1 and 2 address issue of ensuring that the JIT function
outputs the expected results, it is not guaranteed that the end-user will be
able to run the function on his data. This is due to how the types of objects
passed to Numba determine if the function can be JIT compiled. As such, it is
important to declare what types the function can be used with to increase
transparency for the end-user.

However, it is redundant to provide this type information in both the Numba
signature as well as in the Python type-hints of the `py_func`. Therefore, it
makes sense to use this typing information as a Numba signature since then it
is possible to retrieve typing information for Numba, Sphinx, and Mypy from a
single source of truth. This also enforces correct type-hints as Numba will not
be able to compile if the user provides an incorrect (or missing) type-hint.

The downside to this is that Numba, as of version 0.46, offers only
type-inference for actual data being passed to a function at compile-time and
has no functionality to translate `type`. Therefore, this module is required to
convert these type-hints, into Numba compatible signatures in order for eager
JIT compilation to work.

Therefore, an annotations plug-in was specifically developed for Numba and can
be used as follows:

``` python
import numba
from util.numba.annotation import AnnotationInterpreter

@numba.jit(nopython=True)
def simple_sum(a: int, b: int) -> int:
    return a + b

AnnotationInterpreter(test.py_func).convert_typehints()
# Output = (int64, int64) -> int64
```

Since each Numba function should be tested once and only once for correct
compilation, a Pytest fixture and marker were created. These can be used as
follows to JIT-compile all imported JIT functions in a test module using their
annotations:

``` python
@pytest.mark.retrieve_jit_funcs
def test_jit_compile(jit_func, jit_compile):
    jit_compile(jit_func)
```

This addresses the problem of having to re-compile a JIT-function for each
parametrized test or having to gather the JIT functions manually. Taking the
implementation one step further, a test snippet was created that contains the
`test_jit_compile` function from above. Therefore, it can simply be imported
into a test module and it will automatically test all JIt functions used within
that test-module for JIT compilation.

``` python
from tests.snippets import test_jit_compile  # noqa: F401
```

## Pros and Cons of the Options

### Test the JIT-function as any other

``` python
from foo.jit_funcs import bar  # bar is a numba.jit decorated function

test_function(func_args, expected_result):
    assert bar(func_args) == expected_result
```

* Good, because actual behavior of the JIT function is tested
* Bad, because programmed code (underlying `pyfunc`) is not tested
* Bad, because [coverage.py] does not work

### Test the Underlying Python function

``` python
from foo.jit_funcs import bar  # bar is a numba.jit decorated function

test_function(func_args, expected_result):
    assert bar.py_func(func_args) == expected_result
```

* Good, because code [coverage.py] works as expected
* Good, because performance is good from not having to compile JIT-funcs
* Bad, because it is not certain that the JIT function can be compiled
* Bad, because actual behavior of the JIT function is not tested

### Test both the JIT-function and Python Function and Compare Results

``` python
from foo.jit_funcs import bar  # bar is a numba.jit decorated function

test_function(func_args, expected_result):
    result, jit_result = bar.py_func(*func_args), bar(func_args)
    assert result == jit_result
    assert result == expected_result
```

* Good, because you are confident that what you program is what you get
* Good, because [coverage.py] works
* Bad, because tests take more time to run
* Bad, because tests become more verbose

<!-- Un-wrapped URLs below -->
[coverage.py]: https://coverage.readthedocs.io/en/latest/
[SciPy 2017 Tutorial]: https://www.youtube.com/watch?v=1AwG0T4gaO0
[Option 3]: #test-both-the-jit-function-and-python-function-and-compare-results
