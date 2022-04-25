"""Functions for rotating images."""
import inspect

import numpy as np
from numpy import ndarray


def rotate_square(img: ndarray, degrees: int) -> ndarray:
    """
    Rotates image (as numpy array) with multiples of 90 degrees / .5pi rad.

    :raises ValueError: if degrees is not a multiple of 90 or 0.
    """
    degrees %= 360
    match degrees:
        case 0:
            return img
        case 90:
            return np.rot90(img)
        case 180:
            return np.rot90(img, k=2)
        case 270:
            return np.rot90(img, k=3)
        case _:
            raise ValueError(
                f"Function {inspect.currentframe().f_code.co_name} only "
                f"allows rotation in multiples of 90 degrees (.5pi rad)."
            )
