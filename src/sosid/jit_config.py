"""Contains config options for Numba JIT decorated functions."""

FAST_MATH_FLAGS = {
    # Refer to https://llvm.org/docs/LangRef.html#fast-math-flags
    "nnan": False,  # Propagation dir. requires nan be returned!
    "ninf": True,
    "nsz": True,
    "arcp": True,
    "contract": True,
    "afn": True,
    "reassoc": True,
}

BASE_JIT_KWARGS = {
    "nopython": True,
    "cache": True,
}
