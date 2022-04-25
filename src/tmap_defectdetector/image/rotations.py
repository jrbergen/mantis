"""Functions for rotating images."""
import inspect
import warnings

import numpy as np
from numpy import ndarray


def rotate_img(img: ndarray, degrees: int) -> ndarray:
    """
    Rotates image (as numpy array) with multiples of 90 degrees / .5pi rad.

    :raises ValueError: if degrees is not a multiple of 90 or 0.
    :raises ValueError: if image is not square.
    """

    if img.shape == (1,):
        warnings.warn(
            "Tried to rotate singular image (image with 1 pixel). Returned image unaltered.", UserWarning
        )
        return img

    if img.size == 0:
        warnings.warn("Tried to rotate empty image. Returned image unaltered.", UserWarning)
        return img

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


def rotate_img_square(img: ndarray, degrees: int) -> ndarray:
    """
    Rotates image (as numpy array) with multiples of 90 degrees / .5pi rad.

    :raises ValueError: if degrees is not a multiple of 90 or 0.
    :raises ValueError: if image is not square.
    """

    if img.size > 0 and (len(img.shape) <= 1 or img.shape[0] != img.shape[1]):
        raise ValueError(f"Expected square image/matrix; got shape {img.shape!r}.")

    return rotate_img(img=img, degrees=degrees)


if __name__ == "__main__":
    a = rotate_img_square(np.array([1, 2]), 90)

    print(f"Temporary breakpoint in {__name__}")
