"""Contains (validation) checks for images."""

from __future__ import annotations
from numpy import ndarray


def ensure_img_is_array(img: ndarray) -> None:
    """
    Checks if images is an instance of numpy.ndarray.

    :raises TypeError: If img is not an instance of np.ndarray
    """
    if not isinstance(img, ndarray):
        raise TypeError(f"Image must be an instance of {ndarray.__qualname__}, got type {type(img)} instead.")


class ImageDimensionError(ValueError):
    pass


def ensure_array_dimension(arr: ndarray, dim: int, errmsg: str = "") -> None:
    """Ensures an array is of a particular dimension.

    :param arr: numpy array to check.
    :param dim: number of dimensions to enforce.
    :param errmsg: (optional) information to add to error if dimension is invalid (default = "").

    :raises ImageDimensionError: if array is not of the desired dimension.
    """
    if arr.ndim != dim:
        full_err = f"Array must be of dimension {dim}; got array of dimension {arr.ndim}."
        if errmsg:
            full_err += f", {errmsg}"
        raise ImageDimensionError(errmsg)


def ensure_square_img(img: ndarray) -> None:
    """
    Checks if image dimensions are square.

    :raises ValueError: if image dimensions are not equal (i.e. not a square image).
    :raises ValueError: if empty 'image' is passed.
    """
    if img.size == 0:
        raise ValueError("Empty image of shape 0x0 is not considered square.")

    if img.size > 1:
        if len(img.shape) == 1:
            raise ValueError(
                "Operation is supported only for square images.\n" f"Got image of dimensions {img.shape[0]}x1."
            )

        if img.shape[0] != img.shape[1]:
            raise ValueError(
                "Operation is supported only for square images.\n"
                f"Got image of dimensions {img.shape[0]}x{img.shape[1]}."
            )
