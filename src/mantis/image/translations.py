"""Functions for translating (moving all pixels x/y steps) images."""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from mantis.image.checks import ensure_img_is_array, ensure_array_dimension

_TRANS_FLOAT_COMP_TOL: float = 5e-1
"""Tolerance for comparing floats which specify image translations."""


def translate_image(
    img: ndarray,
    translation_x: int | float = 0,
    translation_y: int | float = 0,
) -> ndarray:
    """
    Translates (rolls) image accross the given x and y translation values.

    :param img: 3D numpy array (e.g. representing an image with heigh, width, and color/intensity).
    :param translation_x: (optional)
        Value specifying the desired translation in the x-direction.
        Can be an integer to represent the number of pixels to shift by, or a float
        on domain [-1.0, 1.0] to shift by a fraction of the image width.
        The result is rounded to the nearest integer pixel value in case a fraction is passed.
        Defaults to 0 (no translation in x-direction).
    :param translation_y: (optional)
        Value specifying the desired translation in the y-direction.
        Can be an integer to represent the number of pixels to shift by, or a float
        on domain [-1.0, 1.0] to shift by a fraction of the image height.
        The result is rounded to the nearest integer pixel value in case a fraction is passed.
        Defaults to 0 (no translation in y-direction).

    :raises ValueError: if translations contain fractional values < -1.0 or > 1.0.

    .. note ::
         The positive x direction == left -> right, and the positive y direction == top-> bottom.
    """
    ensure_img_is_array(img)
    ensure_array_dimension(
        img,
        dim=3,
        errmsg=(
            "Images expected to be 3-dimensional; " "having a width, height, and intensity/color dimension."
        ),
    )
    no_change_vals = (0, -1.0, 1.0)
    if translation_x in no_change_vals and translation_y in no_change_vals:
        return img

    if translation_x:
        translation_x = interpret_img_translation_value(img=img, translation=translation_x, axis="x")
        img = np.roll(img, shift=translation_x, axis=0)
    if translation_y:
        translation_y = interpret_img_translation_value(img=img, translation=translation_y, axis="y")
        img = np.roll(img, shift=translation_y, axis=1)
    return img


def interpret_img_translation_value(img: ndarray, translation: float | int, axis: str) -> int:
    """
    Checks bounds for an image translation value and converts it to be interpreted as pixels.

    :param img: numpy array (representing an image).
    :param translation: Value specifying the desired translation in the x-, or y-direction.
        Can be an integer to represent the number of pixels to shift by, or a float
        on domain [-1.0, 1.0] to shift by a fraction of the image width/height.
        The result is rounded to the nearest integer pixel value in case a fraction is passed.
        Defaults to 0 (no translation).
    :param axis: string indicating translation axis; must be 'x' or 'y'.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"Argument to 'axis' parameter must be 'x' or 'y', got {axis!r}.")

    if isinstance(translation, float):
        valid_fractional_translation: bool = (
            -1.0 - _TRANS_FLOAT_COMP_TOL <= float(translation) <= 1.0 + _TRANS_FLOAT_COMP_TOL
        )

        if not valid_fractional_translation:
            raise ValueError("Passed a float translation which is not within domain [-1.0, 1.0].")
        elif axis == "x":
            translation = round(img.shape[1] * translation) % img.shape[1]
        elif axis == "y":
            translation = round(img.shape[0] * translation) % img.shape[0]
    elif isinstance(translation, int):
        if axis == "x":
            translation %= img.shape[1]
        elif axis == "y":
            translation %= img.shape[1]
    else:
        raise TypeError(f"Expected int or float for image translation value, got {type(translation)}.")

    return int(translation)
