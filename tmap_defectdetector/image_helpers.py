"""Contains image-related functionality that wasn't already in a library AFAIK"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def file_is_image(potential_imgfile: os.PathLike) -> bool:
    """
    Checks whether a file is an image file based on whether PIL.Image can load it.

    :param potential_imgfile: Path or Path-like argument pointing to the path of a file
        for which we want to know whether it is an image.
    :raises FileNotFoundError: If non-existent Path is passed.
    """
    if not (_potential_imgpath := Path(potential_imgfile).resolve()).exists():
        raise FileNotFoundError(
            f"Cannot check whether file is image: {str(potential_imgfile)!r} doesn't exist."
        )

    try:
        Image.open(_potential_imgpath)
        return True
    except UnidentifiedImageError:
        return False
    except Exception as err:
        raise IOError(
            f"Unexpected exception occured whilst testing whether this file is an image: "
            f"{str(_potential_imgpath)!r}."
        ) from err
