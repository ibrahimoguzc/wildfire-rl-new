master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "docs.ext.numba_patch",  # Exposes CUDA device funcs to Sphinx
]
source_suffix = ".rst"
