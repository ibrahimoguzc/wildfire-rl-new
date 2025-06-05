"""Contains patched `pyqtgraph` exporters."""

import types
from functools import lru_cache
from typing import Any

from PyQt5 import QtGui
from pyqtgraph import exporters
from pyqtgraph.exporters.SVGExporter import _generateItemSvg as _genItemSvg
from pyqtgraph.exporters.SVGExporter import generateSvg as genSvg

__all__ = [
    "CSVExporter",
    "HDF5Exporter",
    "ImageExporter",
    "MatplotlibExporter",
    "PrintExporter",
    "SVGExporter",
]

CM = QtGui.QPainter.CompositionMode
COMPOSITION_MODE_TO_SVG_MODE = {
    CM.CompositionMode_Multiply: "multiply",
    CM.CompositionMode_Screen: "screen",
    CM.CompositionMode_Overlay: "overlay",
    CM.CompositionMode_Darken: "darken",
    CM.CompositionMode_Lighten: "lighten",
    CM.CompositionMode_ColorDodge: "color-dodge",
    CM.CompositionMode_ColorBurn: "color-burn",
    CM.CompositionMode_HardLight: "hard-light",
    CM.CompositionMode_SoftLight: "soft-light",
}
"""Map :py:attr:`QPainter.CompositionMode` to SVG blend mode

Refer to: https://www.w3.org/TR/compositing-1/#mix-blend-mode
."""


@lru_cache(maxsize=2)
def patch_function_globals(
    func: types.FunctionType, patched_globals: tuple[tuple[str, Any]]
) -> types.FunctionType:
    """Patch `__globals__` of ``func`` with ``patched_globals``."""
    globals = func.__globals__.copy()
    globals.update(patched_globals)
    return types.FunctionType(
        code=func.__code__,
        globals=globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )


def _generateItemSvg(item, nodes=None, root=None, options={}):
    """Traverse child tree of ``item`` to convert them to SVG.

    This patches the original implementation to add support for
    SVG blend-modes.
    """
    # Patch _genItemSVG to call new _generateItemSVG recursively
    patched_genItemSvg = patch_function_globals(_genItemSvg, PATCHED_GLOBAL)
    output = patched_genItemSvg(item, nodes, root, options)
    if isinstance(output, tuple):
        g1, defs = output
        paintMode = getattr(item, "paintMode", None)
        if paintMode:
            svg_mode = COMPOSITION_MODE_TO_SVG_MODE.get(paintMode, "normal")
            g1.setAttribute("style", f"mix-blend-mode:{svg_mode}")
        return g1, defs
    return output


PATCHED_GLOBAL = (("_generateItemSvg", _generateItemSvg),)

# Create monkeypatched generateSvg function
patched_genSvg = patch_function_globals(genSvg, PATCHED_GLOBAL)


class SVGExporter(exporters.SVGExporter):
    """Patched SVGExporter that allows Composition Mode."""

    # Patch generateSvg function from within export method
    export = patch_function_globals(
        exporters.SVGExporter.export, (("generateSvg", patched_genSvg),)
    )


# Remove old SVGExporter and register patched one
(exporter_list := exporters.Exporter.Exporters).pop(
    exporter_list.index(exporters.SVGExporter)
)
SVGExporter.register()


# Make un-patched exporters importable from this module
class ImageExporter(exporters.ImageExporter):  # noqa: D101
    pass


class MatplotlibExporter(exporters.MatplotlibExporter):  # noqa: D101
    pass


class CSVExporter(exporters.CSVExporter):  # noqa: D101
    pass


class PrintExporter(exporters.PrintExporter):  # noqa: D101
    pass


class HDF5Exporter(exporters.HDF5Exporter):  # noqa: D101
    pass
