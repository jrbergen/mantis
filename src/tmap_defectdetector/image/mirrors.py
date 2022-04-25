"""Functions for mirroring images."""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

from src.tmap_defectdetector.image.checks import ensure_img_is_array, ensure_square_img


def mirror_horizontal(img: ndarray) -> ndarray:
    """Mirrors/flips an image represented as numpy array horizontally."""
    ensure_img_is_array(img)
    return cv.flip(img, 0)


def mirror_vertical(img: ndarray) -> ndarray:
    """Mirrors/flips an image represented as numpy array vertically."""
    ensure_img_is_array(img)
    return cv.flip(img, 1)


def mirror_diag_topleft_bottomright(img: ndarray) -> ndarray:
    """
    Mirrors/flips a square image (as numpy array) diagonally along the axis
    spanning from the top-left corner to the bottom-right corner.
    """
    ensure_img_is_array(img)
    ensure_square_img(img)
    return np.rot90(np.fliplr(img))


def mirror_diag_bottomleft_topright(img: ndarray) -> ndarray:
    """
    Mirrors/flips a square image (as numpy array) diagonally along the axis
    spanning from the bottom-left corner to the top-right corner.
    """
    ensure_img_is_array(img)
    ensure_square_img(img)
    return np.rot90(np.flipud(img))
