# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains drawable items for display in a :py:class:`ViewBox`."""

from collections.abc import Sequence
from pathlib import WindowsPath
from typing import Union

import numpy as np
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets
from pyqtgraph import GraphicsObject, ImageItem
from pyqtgraph.functions import mkBrush, mkPen

from sosid.paths import ICONS_PATH

MARKER_ICON = ICONS_PATH / "marker.svg"
"""str: Sets the default marker icon."""

# TODO Consider moving into typedef
Color = Union[str, Sequence[int], QtGui.QColor]  # Color Type-Hint


# TODO Make this class much simpler with a base Qt.QImage
# TODO Finish docstring by adding Args and Attributes sections
class IndexedImageItem(ImageItem):
    """Enables fast rendering of 2D :py:obj:`numpy.uint8` images.

    See Also:
        # TODO talk about IndexedArrays from the following website:
        https://doc.qt.io/qt-5/qimage.html#pixel-manipulation

    """

    def __init__(
        self,
        image: np.ndarray,  # TODO consider replacing with numba.uint8[:, :]
        colorTable: Sequence[tuple[int, int, int]] | Sequence[int],
        axisOrder: str = "row-major",
        **kwargs,
    ):
        GraphicsObject.__init__(self)
        self.image = image
        self.axisOrder = axisOrder

        # Disabling unused attributes
        self.menu = None
        self.qimage = None
        self.paintMode = None
        self.autoDownsample = False
        self.drawKernel = None
        self.border = None
        self.removable = False

        # Setting Attributes
        self.setColorTable(colorTable)
        if image is not None:  # Preserving compatibility w/ ImageItem
            self.setImage(image, **kwargs)
            self._renderRequired = True
            self._unrenderable = False
        else:
            self.setOpts(**kwargs)
            self._renderRequired = False
            self._unrenderable = True

    def setColorTable(
        self, colorTable: Sequence[tuple[int, int, int, int]] | list[int]
    ) -> None:
        """Sets :py:attr:`colorTable` to ``colorTable`` in Qt format.

        The Qt color format internally represents the RGB triplet as a
        single integer. To recover separate RGB(A) channels in the
        form of a tuple use :py:meth:`QtGui.Qcolor.fromRgba`. as
        follows::

            qColorInt = QtGui.qRgb(255, 255, 255)
            r, g, b, a = QtGui.QColor.fromRgba(qColorInt).getRgb()


        Note:
            If the :py:attr:`qimage` already exists, then its colorTable
            is updated by this method.

        Args:
            colorTable: Contains a sequence of either integer values
                0-255 for each color channel (Red (R), Green (G), and
                Blue (B)), or the single integer RGBA spec. of Qt.

        """
        if any(isinstance(e, int) for e in colorTable):  # Already Qt RGBA
            self.colorTable = colorTable
        else:
            self.colorTable = [QtGui.qRgba(*entry) for entry in colorTable]

        if self.qimage:  # Updating color table of qimage if it exists
            self.qimage.setColorTable(colorTable)

    def render(self) -> None:
        """Converts to :py:class:`QtGui.QImage` for display."""
        image = self.image
        if self.axisOrder == "col-major":
            image = image.transpose((1, 0, 2)[: image.ndim])
        self.qimage = QtGui.QImage(
            image.ctypes.data,
            image.shape[1],
            image.shape[0],
            QtGui.QImage.Format_Indexed8,
        )
        self.qimage.setColorTable(self.colorTable)

    # TODO finish docstring
    def setImage(self, image, **kwargs) -> None:
        if image is None and self.image is None:
            return
        gotNewData = True
        shapeChanged = self.image is None or image.shape != self.image.shape
        # Important to not modify original data!
        self.image = image.view(np.ndarray)
        # if self.image.shape[0] > 2**15-1 or self.image.shape[1] > 2**15-1:
        #     if 'autoDownsample' not in kargs:
        #         kargs['autoDownsample'] = True
        if shapeChanged:
            self.prepareGeometryChange()
            self.informViewBoundsChanged()

        self.setOpts(update=False, **kwargs)
        self.update()

        if gotNewData:
            self.sigImageChanged.emit()

    def setOpts(self, update=True, **kwargs) -> None:
        if "axisOrder" in kwargs:
            val = kwargs["axisOrder"]
            if val not in ("row-major", "col-major"):
                raise ValueError(
                    'axisOrder must be either "row-major" or "col-major"'
                )
            self.axisOrder = val
        if "opacity" in kwargs:
            self.setOpacity(kwargs["opacity"])
        if "compositionMode" in kwargs:
            self.setCompositionMode(kwargs["compositionMode"])
        if "border" in kwargs:
            self.setBorder(kwargs["border"])
        if "removable" in kwargs:
            self.removable = kwargs["removable"]
            self.menu = None
        if "rect" in kwargs:
            self.setRect(kwargs["rect"])
        if update:
            self.update()


class MarkerItem(QtSvg.QGraphicsSvgItem):
    """Creates a Marker at specified position from a SVG file.

    Args:
        svg_file: Absolute path to a SVG file.
        size: Width, height dimensions in logical pixel units.
        pos: Position along the height and width axes in logical pixel
            units. By default the origin is at the top-left corner
            of the ViewBox, positive x points to the right, positive
            y points down.
        aspect: Angle in degrees that governs the orientation of the
            `MarkerItem` icon. Defaults to 0 deg (North) with an aspect
            of 90 deg (East) rotating the icon Clockwise (CW+).
        lockAspect: Sets if the original aspect ratio of the SVG
            file is preserved during scaling.
        parent: :py:class:`QWidget` parent instance.

    """

    def __init__(
        self,
        svg_file: WindowsPath = MARKER_ICON,
        size: tuple[float, float] = (10, 10),
        pos: tuple[float, float] = (0, 0),
        aspect: float = 0.0,
        lockAspect: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(str(svg_file), parent)
        self.renderer = self.renderer()
        self.lockAspect = lockAspect
        self.setSize(*size)
        self.setPos(*pos)
        self.setRotation(aspect)
        self._color_effect = None

    def setSize(self, width: float, height: float) -> None:
        """Sets the ``size`` of the current :py:class:`Marker` object.

        If :py:attr:`lockAspect` is ``True`` then this method will check
        that the provided dimensions have the same aspect ratio as the
        base SVG file. In order to keep the resizing algorithm
        predictable, if there is an aspect ratio mismatch, then the
        largest dimension provided will be kept constant while the other
        dimension is scaled. However, if a square size is provided
        (``width == height``) then the algorithm will default to scaling
        the ``width`` to maintain the correct aspect ratio.

        Args:
            width: Width of the icon in logical pixel units.
            height: Height of the icon in logical pixel units.

        """
        if self.lockAspect:
            # Computing Aspect Ratios (AR)
            default_ar = self.calcAspectRatio(self.renderer.defaultSize())
            current_ar = self.calcAspectRatio((width, height))

            if default_ar == current_ar:  # No need to correct AR
                pass
            elif width <= height:  # Scale width & keep height constant
                width = default_ar * height
            elif width > height:  # Scale height & keep width constant
                height = width / default_ar

            self._size = (width, height)

    def setColor(self, color: tuple[int, int, int, int]) -> None:
        """Sets the color of the marker."""
        if self._color_effect is None:
            self._color_effect = QtWidgets.QGraphicsColorizeEffect()
            self.setGraphicsEffect(self._color_effect)
        self._color_effect.setColor(QtGui.QColor(*color))

    def paint(self, painter: QtGui.QPainter, *args) -> None:
        """Ensures marker is rendered with correct size and position.

        This works by painting the SVG maker within the specified
        bounding rectangle provided by :py:meth:`boundingRect`.
        Therefore, one must ensure that :py:meth:`boundingRect` produces
        the a correctly positioned and sized :py:obj:`QRectF`.
        """
        self.renderer.render(painter, self.boundingRect())

    def boundingRect(self) -> QtCore.QRectF:
        """Provides a centered bounding rectangle.

        This effectively translates the SVG image such that its center
        point is the anchor. By default the anchor is the top-left
        corner, and as a result translating the bounding rectangle
        up and to the left centers the image.
        """
        (width, height) = self._size
        return QtCore.QRectF(-width / 2, -height / 2, *self._size)

    @staticmethod
    def calcAspectRatio(
        size: tuple[float, float] | QtCore.QSize,
    ) -> float:
        """Calculates the aspect ratio (width / height) for a ``size``.

        Args:
            size: Width, height dimensions in logical pixel units.

        Returns:
            Aspect ratio (width / height) calculated from ``size``

        """
        if isinstance(size, QtCore.QSize):
            return size.width() / size.height()
        return size[0] / size[1]


# TODO Add methods to allow setting properties within Qt main-loop
# TODO Enlarge default size to be more legible
class AxisItem(QtWidgets.QGraphicsPolygonItem):
    """Paints a simple arrow and text-label to represent an axis.

    The Image Coordiante System is defined such that the x-axis points
    to the right and the y-axis points down. The length refers
    to the dimension measured along the direction the arrow points
    towards. Therefore, the width measures the transverse dimension.

    Args:
        pos: Position (width, height) in the Image Coordinate System.
        label: Axis label (i.e. x, y, or z). Defaults to "x".
        angle: Clockwise angle measured from the x-axis of the Image
            Coordinate System that sets the direction that the arrow is
            pointing. Defaults to 0.
        color: Color assigned to the axis. Defaults to Red.
        tail_length: Length of the tail (line) of the arrow. Defaults to
            45 pixels.
        tail_width: Width of the tail (line) of the arrow. Defaults to
            5 pixels.
        head_length: Length of the head (triangle) of the arrow.
            Defaults to 5 pixels.
        head_width: Width of the head (triangle) of the arrow.
            Defaults to 5 pixels.
        parent: Qt parent object. Defaults to None.

    """

    def __init__(
        self,
        pos: tuple[int, int] = (0, 0),
        label: str = "x",
        angle: float = 0,
        color: Color = "r",
        tail_length: float = 45,
        tail_width: float = 1,
        head_length: float = 5,
        head_width: float = 5,
        parent: object = None,
    ):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent)
        # Setting Attributes
        self.label = label
        self.angle = angle
        self.color = color
        self.tail_length = tail_length
        self.head_length = head_length
        self.tail_width = tail_width
        self.head_width = head_width

        # Creating Arrow and Setting Position and Orientation
        self.setPos(*pos)  # Must be set before creating text!
        self.arrow = self.makeArrow()
        self.text_pos = self.getTextPos()
        self.brush = mkBrush(color)
        self.pen = mkPen(color)
        self.setPolygon(self.arrow)

        # Ensuring that the axis does not move when panning/zooming
        self.setFlags(self.flags() | self.ItemIgnoresTransformations)

    def paint(self, p, *args):
        """Responsible for rendering the :py:class:`AxisItem`."""
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        # TODO Use QRectF such that the text is centered!
        p.drawText(self.text_pos, self.label)
        p.drawPolygon(self.arrow)

    def makeArrow(self) -> QtGui.QPolygonF:
        """Constructs a simple arrow polygon at position (0, 0).

        The arrow is oriented in the correct direction as specified by
        :py:attr:`angle`. Where the angle is measured Clockwise Positiv
        (CW+) from the global x-axis in the Image Coordinate System.

        Note:
            The rotation is applied to the arrow polygon and not to the
            :py:obj:`QGraphicsPolygonItem` so that the text is not
            rotated. # TODO add reference to Sphinx section

        """
        tl, hl = self.tail_length, self.head_length
        tw, hw = self.tail_width, self.head_width
        arrow = QtGui.QPolygonF()
        arrow_pts = [
            (0, 0),
            (0, -tw / 2),
            (tl, -tw / 2),
            (tl, -hw / 2),
            (tl, -hw / 2),
            (tl + hl, 0),
        ]
        arrow_pts.extend([(x, -y) for x, y in arrow_pts[-2:0:-1]])
        for x, y in arrow_pts:
            arrow.append(QtCore.QPointF(x, y))

        # Rotation transformation
        rotation = QtGui.QTransform().rotate(self.angle)
        return rotation.map(arrow)

    # TODO Chage to makeTextBox after painting text with QRectF
    def getTextPos(self) -> tuple[int, int]:
        """Returns the position where the axis-label should be."""
        text_x = (self.tail_length + self.head_length) + 10  # pad value
        transform = QtGui.QTransform().rotate(self.angle).translate(text_x, 0)
        return transform.map(self.pos())

    def dataBounds(self, ax: int, frac, orthoRange=None) -> list[float]:
        """Required for auto-ranging :py:class:`ViewBox`."""
        br = self.boundingRect()
        return [br.left(), br.right()] if ax == 0 else [br.top(), br.bottom()]

    def pixelPadding(self) -> int:
        """Required for auto-ranging :py:class:`ViewBox`."""
        return 0


class BorderItem(QtWidgets.QGraphicsRectItem):
    """A custom QGraphicsRectItem for creating and displaying border."""

    def __init__(
        self,
        color: Color = "g",
        parent: object = None,
        width: float = 5120,
        height: float = 5120,
        ax: float = 0,
        ay: float = 0,
        thickness: int | None = 3,
    ):
        QtWidgets.QGraphicsRectItem.__init__(self, parent)
        self.color = color
        self.width = width
        self.height = height
        self.ax = ax
        self.ay = ay

        # Creating Rectangle and Setting Position and Orientation
        self.rectangle = self.makeRect()
        self.pen = mkPen(color, width=thickness)
        self.setRect(self.rectangle)

    def paint(self, p, *args):
        """Responsible for rendering the :py:class:`BorderItem`."""
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.pen)
        p.drawRect(self.rectangle)

    def makeRect(self):
        """Constructs a simple rectangle at position (ax, ay)."""
        rectangle = QtCore.QRectF()
        rectangle.setBottomRight(
            QtCore.QPointF(self.ax + self.width, self.ay + self.height)
        )
        rectangle.setTopLeft(QtCore.QPointF(self.ax, self.ay))
        return rectangle
