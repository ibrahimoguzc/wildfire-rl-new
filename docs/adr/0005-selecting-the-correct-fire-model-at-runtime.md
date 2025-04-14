# Selecting the Correct Fire Model at Runtime

* Status: Accepted
* Deciders: San Kilkis
* Date: 2019-09-17

## Context and Problem Statement

Although the fire-model has now been successfully ported onto the GPU utilizing
CUDA, an emergent issue is now dealing with the various configurations that the
end-user system can have. For example, on a machine with no CUDA device, the
fire model should run on the CPU. On the other hand, for an end-user who has a
CUDA enabled device, they should have the option to chose if the model runs on
the CPU or GPU. The complication to this problem arises from the fact that
the fire model common functions contained within
`examples\wildfire\fire_model\com` need to be JIT compiled separately
in order to maximize performance through use of function inlining which removes
the overhead needed to call functions.

To illustrate this complication, imagine a summing function that takes
two float values and returns their summed value:

``` python
def summed(a: float, b: float): -> float
    return a + b
```

The above function can then be optimally compiled for the CPU & GPU targets
respectively as follows:

``` python
summed_cpu = numba.jit(nopython=True, fastmath=True)
summed_gpu = cuda.jit(device=True, inline=True)
```

## Decision Drivers

* Simplicity
* Maintainability
* Preserving Commonality between Fire Models (CPU & GPU)

## Considered Options

* Compile with the [@numba.jit] decorator and don't use the `inline` option
* Monkey patch the imported functions with their JIT compiled counterparts
* Create a custom decorator that returns the required JIT compiler

## Decision Outcome

Chosen option: Compile with the [@numba.jit] decorator and don't use the
`inline` option. This option is simple, easy to maintain, and preserves
commonality between the fire model for the CPU and GPU.

### Positive Consequences

* Simple, elegant solution that is easy to maintain
* Preserves commonality between the CPU and GPU fire model functions

### Negative Consequences

* Performance penalty may be payed in the future if Numba implicit behavior
  changes

## Pros and Cons of the Options

### Compile with the [@numba.jit] decorator and don't use the `inline` option

One of the nicest features of the [@cuda.jit] decorator is that it
automatically translates [@numba.jit] compiled functions to device funtions
that can execute on the GPU (device). However, the aforementioned `inline`
feature cannot be specified when using the [@numba.jit] decorator. Through
testing the difference between [@cuda.jit] compiled functions with the `inline`
option and the plain [@numba.jit] was unmeasurable.

* Good, because this implementation is very simple
* Good, because it doesn't seem to negatively effect performance
* Bad, because if the implicit Numba behavior of inlining [@numba.jit]
  decorated within a CUDA context changes in the future, a performance penalty
  may be incurred.

### Monkey patch the imported functions with their JIT compiled counterparts

A potential solution is to not decorate the common functions at all, and only
compile them when required by the respective fire-model. A sample
implementation would be as follows:

``` python
    from .postprocess import postprocess

    if TARGET == "CPU":
        jit_compiler_= numba.jit(nopython=True, fastmath=True)
    elif TARGET == "GPU":
        jit_compiler = cuda.jit(device=True, inline=True)

    postprocess = jit_compiler(postprocess)
```

* Bad, because it is difficult to maintain unless the dependencies are
automatically filtered and recursively compiled. However, this needlessly adds
complexity.
* Bad, because programmers and linters alike will go crazy
* Bad, because it is difficult to keep track of the order in which the common
functions need to be compiled. (Some functions call other functions, which
then need to be compiled first)

### Create a custom decorator that returns the required JIT compiler

Another option is to create a decorator that returns the correctly decorated
(JIT-compiled) python function. This option could be implemented as follows:

``` python
def auto_jit(f: Callable):
    """Compiles ``f`` on the GPU or CPU based on user configuration."""
    TARGET = os.environ.get("TARGET", default=None)
    if TARGET == "GPU":
        from numba import cuda

        compiler = cuda.jit(device=True, inline=True)
    elif TARGET == "CPU":
        import numba

        compiler = numba.jit(nopython=True, fastmath=True)
    else:
        pass
    return compiler(f)
```

* Good, because it is maintainable (doesn't require manual compiling through
  monkey-patching)
* Bad, because it requires use of environment variables or configuration files.
  As such, when the use wants to switch between using the CPU and GPU, the
  common functions need to be reloaded as they are compiled at run-time (during
  import).

<!-- Un-wrapped URLs -->
[@cuda.jit]: https://numba.pydata.org/numba-doc/dev/cuda-reference/kernel.html?highlight=cuda%20jit#numba.cuda.jit
[@numba.jit]: https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html?highlight=numba%20jit#numba.jit
