# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains primitive shapes for affecting CA through rasterization.

These shapes provide a way to transfer the actions performed by agents
in the continous space into the cellular space of a Cellular Automata
(CA) model. All shapes are centered at a provided ``idx`` in order to
make it easy to define an area of effect that a certain action has on
the CA model.
"""

import numpy as np
from PIL import Image, ImageDraw

from sosid.util.abc import ABCMeta, abstractmethod

__all__ = ["Ellipse", "LineSegment", "Rectangle"]


class RasterizedShape(metaclass=ABCMeta):
    """Abstract Base Class (ABC) definition of a rasterized shape.

    This ABC, simplifies rasterization of *centered* shapes onto a
    supplied array with :py:meth:`boolean_mask`. This is necessary
    because the :std:doc:`PIL` package draws shapes such that the
    upper-left corner of their bounding-box is coincident with the
    suppled ``idx``.

    Args:
        # TODO decide on what to do with these coordinate systems
        idx: Index (i, j) in the cellular space coordinate system
        size: Sets the dimensions in x and y direction
            of the base :std:doc:`PIL` image (canvas). Depending on the
            shape this canvas can expand during rotation. Therefore,
            it is important to access the `size` attribute of the
            resultant :py:attr:`shape`.
        aspect: Compass direction that the x-axis of the primitive shape
            points toward.

    Attributes:
        canvas (Image.Image): The image onto which the
            :py:class:`RasterizedShape` is drawn.

    """

    __slots__ = ["aspect", "canvas", "idx", "size"]

    def __init__(
        self, idx: tuple[int, int], size: tuple[int, int], aspect: float
    ):
        # Ensure both values in size argument are larger than zero
        if 0 in size:
            _error_msg = (
                "Both values of RasterizedShape size have to be larger than 0."
                f" Size: {size}"
            )
            raise ValueError(_error_msg)

        self.idx = idx
        self.size = size
        self.aspect = aspect
        self.canvas = Image.new(mode="1", size=self.size, color=0)

        # Drawing image onto canvas on entry to prevent redraws
        self.painter((0, 0, *size), fill=1)

    @property
    @abstractmethod
    def painter(self) -> ImageDraw.ImageDraw:
        """Returns the :std:doc:`PIL` method used for drawing.

        This must be overridden in sub-classes. For an ellipse the
        required syntax would be as follows::

            ImageDraw.Draw(self.canvas).ellipse

        """

    @property
    def rotated(self) -> Image.Image:
        """Rotates :py:attr:`canvas` to orient the primitive shape.

        The :std:doc:`PIL` module defines a CCW+ rotation angle. In
        order to use the use compass direction defined by
        :py:attr:`aspect` a conversion is performed.

        Note:
            Using an Affine transform in the :std:doc:`PIL` module takes
            ~50x more time than a simple rotation. Since the primities
            only need to be oriented correctly within the 2D cellular
            space, it is sufficient to accomplish this with only a
            translation and rotation. However, if skewing or reflecting
            these shapes is required in the future, then use of an
            Affine transform is required.

        """
        return self.canvas.rotate(360 - (self.aspect - 90) % 360, expand=True)

    @property
    def shape(self) -> Image.Image:
        """Returns the correctly oriented rasterized shape."""
        return self.rotated if self.aspect != 90 else self.canvas

    @property
    def positioned_bbox(self) -> tuple[int, int, int, int]:
        """Returns a centered bounding-box at :py:attr:`idx`.

        Note:
            Here x_p and y_p refer are pixel coordinates within
            the Image Coordinate System.

        """
        # Localizing object vars to prevent repeating LOAD_FAST calls
        (y_p, x_p), shape = self.idx, self.shape
        width, height = shape.size  # Important to use resultant size!

        # Translating position up and to the left to center bbox
        x_p -= int(width // 2)
        y_p -= int(height // 2)
        return (x_p, y_p, x_p + width, y_p + height)

    _bounds_msg = "Provided position {} lies outside of the desired mask."

    def nonzero(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Returns corrected indices of the :py:class:`RasterizedShape`.

        The returned indices can be used to directly modify an array
        as follows::

            >>> array = np.zeros((5, 5))
            >>> ellipse = Ellipse(idx=(2, 2), major=3, minor=1)
            >>> array[ellipse.nonzero(array.shape)] = 1
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )

        For a rotated ellipse, this would be as follows::

            >>> array = np.zeros((5, 5))
            >>> ellipse = Ellipse(
            ...     idx=(2, 2), major=3, minor=1, aspect=135
            ... )
            >>> array[ellipse.nonzero(array.shape)] = 1
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            )

        Args:
            shape: The shape of the array (n_rows, n_cols) for which the
                nonzero indices are desired. This helps to clip the
                values of the output indices to within bounds of the
                target array, thereby avoiding out of bounds errors.

        Note:
            This is an O(1) method w.r.t the size of an array which
            needs to be masked with a shape, as compared to
            :py:meth:`boolean_mask` which is O(N). This is because this
            method does not need to first create an image of the correct
            proportions to return the required indices. Therefore, it is
            for performance reasons it is advised to to use this method.

            However, :py:meth:`boolean_mask` is a left here to flaunt my
            insane Python optimization skills. Just kidding it is here
            to provide an additional method to compare to for testing
            purposes.

        """
        if all(0 <= i < i_max for i, i_max in zip(self.idx, shape)):
            i_idx, j_idx = np.nonzero(self.shape)

            # Adding offsets to the indices based on bounding-box origin
            # Note that bbox pixels are converted to CA coordinates
            j_origin, i_origin, _, _ = self.positioned_bbox
            i_idx += i_origin
            j_idx += j_origin

            # Clipping output to ensure indices are within bounds
            i_max, j_max = shape
            np.clip(i_idx, a_min=0, a_max=i_max - 1, out=i_idx)
            np.clip(j_idx, a_min=0, a_max=j_max - 1, out=j_idx)
            return (i_idx, j_idx)
        raise ValueError(self._bounds_msg.format(self.idx))

    def boolean_mask(self, mask_shape: tuple[int, int]) -> np.ndarray:
        """Creates a boolean mask to apply :py:attr:`shape` to an array.

        The returned boolean mask can then be used to modify the cells
        of an array which lie within the :py:attr:`shape` as follows::

            >>> array = np.zeros((5, 5))
            >>> ellipse = Ellipse(idx=(2, 2), major=3, minor=1)
            >>> mask = ellipse.boolean_mask(mask_shape=array.shape)
            >>> array[np.nonzero(mask)] = 1
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )

        For a rotated ellipse, this would be as follows::

            >>> array = np.zeros((5, 5))
            >>> ellipse = Ellipse(
            ...     idx=(2, 2), major=3, minor=1, aspect=135
            ... )
            >>> mask = ellipse.boolean_mask(mask_shape=array.shape)
            >>> array[np.nonzero(mask)] = 1
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            )

        Args: mask_shape: Number of rows and columns of the required
            boolean mask. Shape here refers to the Numpy definition of
            the shape of an array.

        Raises:
            ValueError: If a ``mask_shape`` is provided that is too
                small place a :py:class:`RasterizedShape` at
                :py:attr:`idx`.

        """
        # Localizing object vars to prevent repeating LOAD_FAST calls
        idx, shape = self.idx, self.shape
        if all(0 < i < i_max for i, i_max in zip(idx, mask_shape)):
            # Creating and pasting shape into mask
            mask = Image.new(mode="1", size=mask_shape)
            mask.paste(shape, box=self.positioned_bbox, mask=shape)

            # Converting into a bool array ensuring the center is masked
            (mask_array := np.array(mask, dtype=bool))[self.idx] = True

            return mask_array
        raise ValueError(self._bounds_msg.format(idx))


class Ellipse(RasterizedShape):
    """Rasterized ellipse centered at the supplied ``idx``.

    Args:
        idx: Index (i, j) in the cellular space coordinate system
        major: Length of the major axis
        minor: Length of the minor axis
        aspect: Compass direction that the major axis of the ellipse
            points towards. For an ellipse supplied with an aspect of
            180 SI degree, this would mean that the major axis is
            oriented vertically (North-South). Defaults to 90.

    """

    def __init__(
        self,
        idx: tuple[int, int],
        major: int,
        minor: int,
        aspect: float | None = 90,
    ):
        super().__init__(idx, (major, minor), aspect)

    @property
    def painter(self):
        """:std:doc:`PIL` method used to paint an ellipse."""
        return ImageDraw.Draw(self.canvas).ellipse


class Rectangle(RasterizedShape):
    """Rasterized rectangle centered at the supplied ``idx``.

    Args:
        idx: Index (i, j) in the cellular space coordinate system
        width: Length of rectangle on the x-axis
        height: Length of rectangle on the y-axis
        aspect: Compass direction that the width axis of the rectangle
            points towards. For a rectangle with a width larger than
            its length, an aspect of 180 SI degree would result in
            the longer axis of the rectangle oriented vertically
            (North-South). Defaults to 90.

    """

    def __init__(
        self,
        idx: tuple[int, int],
        width: int,
        height: int,
        aspect: float | None = 90,
    ):
        super().__init__(idx, (width, height), aspect)

    @property
    def painter(self):
        """:std:doc:`PIL` method used to paint a rectangle."""
        return ImageDraw.Draw(self.canvas).rectangle


class LineSegment(Rectangle):
    """Rasterized line-segment centered at the supplied ``idx``.

    Args:
        idx: Index (i, j) in the cellular space coordinate system.
        width: Length of the line-segment.
        stoke: Thickness of the line segment.
        aspect: Compass direction that the line-segment is oriented.
            Defaults to 90 (a horizontal line).

    """

    def __init__(
        self,
        idx: tuple[int, int],
        length: int,
        stroke: int | None = 1,
        aspect: float | None = 90,
    ):
        super().__init__(idx, length, stroke, aspect)
