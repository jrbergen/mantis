from __future__ import annotations

import cv2
import numpy as np


def rgb_to_grayscale(img: np.ndarray):
    imshape = img.shape
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(*imshape[:2], 1)


def grayscale_to_binary(img: np.ndarray, thresh_low: int = 128, maxval: int = 255):
    return cv2.threshold(img, thresh_low, maxval, cv2.THRESH_BINARY)[1]
