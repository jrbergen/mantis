from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import TypeAlias, Iterable

import pandas as pd
from PIL.Image import Image

from tmap_defectdetector.logger import log

ImageCollection: TypeAlias = list[Image] | tuple[Image, ...]
LabelCollection: TypeAlias = list[pd.DataFrame] | tuple[pd.DataFrame, ...]
Translation: TypeAlias = tuple[float, float] | tuple[int, int]


class AbstractDataSet(ABC):

    label_data: list[pd.DataFrame]

    @abstractmethod
    def amplify_data(self):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case the training data comprises images.
        """
        pass

    @abstractmethod
    def add_label_data(self, label_path_or_data: Path | pd.DataFrame):
        ...

    def amplify_data(self, *args, **kwargs):
        ...


class ImageDataSet(AbstractDataSet):

    _VALID_MIRROR_AXES: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 360)
    DEFAULT_TRANSLATIONS: Iterable[Translation] = (
        (0.5, 0.5),
        (0.5, -0.5),
        (-0.5, -0.5),
        (-0.5, 0.5),
    )
    DEFAULT_ROTATIONS: Iterable[int] = (90, 180, 270)
    DEFAULT_MIRROR_AXES: Iterable[float | int] = (0, 45, 90, 135)

    def __init__(self, data: ImageCollection = tuple(), label_data: list[pd.DataFrame] = tuple()):
        self.data: list[Image] = list(data)
        self.label_data = list(label_data)

    def amplify_data(
        self,
        translations: Iterable[float | int] = DEFAULT_TRANSLATIONS,
        rotations: Iterable[float | int] = DEFAULT_ROTATIONS,
        mirror_axes: Iterable[float | int] = DEFAULT_MIRROR_AXES,
        *args,
        **kwargs,
    ):
        for translation in translations:
            self.data += self.generate_translation(translation[0], translation[1])
        for rotation in rotations:
            self.data += self.generate_rotation(rotation)
        for cur_axis in mirror_axes:
            self.data += self.generate_mirrors(axis=cur_axis)

    def generate_mirrors(self, axis: float | int) -> list[Image]:
        """
        Generates mirrored image to increase number of samples.

        :param axis: axis about which to mirror images. Must be a diagonal, the horizontal,
            or the vertical axis, as we currently expect square images which have
            four reflection symmetry axes.
            Angles and rotation direction are defined in degrees, using the unit circle
            as reference (starting at horziontal and increasing counter-clockwise) i.e.:
            0 = 180 = 360 = mirror accross horizontal axis.
            45 = 225 = mirror accross diagonal from upper left to lower right.
            90 = 270 = mirror accross vertical axis.
            135 = 315 = mirror accross diagonal from upper left to lower right.
        """
        axis = round(axis)
        if axis not in ImageDataSet._VALID_MIRROR_AXES:
            raise ValueError(
                f"Mirror axes must be one of: "
                f"{', '.join(str(n) for n in ImageDataSet._VALID_MIRROR_AXES)}."
            )

        raise NotImplementedError(
            f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}' implementation is not finished yet."
        )

    def generate_translation(self, x: float, y: float) -> list[Image]:
        """Generates translated image to increase number of samples."""
        raise NotImplementedError(
            f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}' implementation is not finished yet."
        )

    def generate_rotation(self, rot_degrees: float) -> list[Image]:
        """Generates rotated image to increase number of samples."""
        raise NotImplementedError(
            f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}' implementation is not finished yet."
        )

    def add_label_data(self, label_path_or_data: Path | pd.DataFrame):
        match label_path_or_data:
            case Path():
                if label_path_or_data.suffix.lower() in (accepted_extensions := (".csv", ".tsv")):

                    clean_whitespace(label_path_or_data)

                    data_lines = label_path_or_data.read_text().split(" ")
                    dialect = csv.Sniffer().sniff(data_lines[0])
                    self.label_data.append(pd.read_csv(label_path_or_data, sep=dialect.delimiter))
                else:
                    raise NotImplementedError(
                        f"Label data handling for types other than {', '.join(accepted_extensions)} not (yet) implemented."
                    )
            case pd.DataFrame():
                self.label_data.append(label_path_or_data)
            case _:
                raise TypeError(
                    f"Expected type {pd.DataFrame.__name__} or {Path.__name__}, "
                    f"got {type(label_path_or_data)}."
                )

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: ImageDataSet | ImageCollection | Image) -> ImageDataSet:
        """Currently doesn't handle data labels, which may lead to image/label sets of different sizes..."""
        match other:
            case list() | tuple():
                if all(isinstance(item, Image) for item in other):
                    self.data += list(other)
                else:
                    raise TypeError(
                        f"All items in data to be added must be images ({Image.__qualname__} objects),"
                        " which seems not to be the case."
                    )
            case ImageDataSet():
                self.data += other.data
            case Image():
                self.data.append(other)
            case _:
                raise TypeError(
                    f"Can only add together (lists of) images or other {type(self).__name__!r} "
                    f"datasets to this {type(self).__name__} dataset."
                )
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data.__repr__()})"


def clean_whitespace(
    filepath: Path,
    whitespace_search_regex: re.Pattern = re.compile(r"\s+"),
    replacement: str = " ",
) -> None:
    """
    Removes inconsistent whitespaces in CSV files (e.g. single space for 1st column, multiple for next like in the ELPV dataset)
    and replaces them with singular ones.

    :param filepath: Path to clean whitespaces for.
    :param whitespace_search_regex: regular expression used to match whitespace (as re.Pattern object).
    :param replacement: string to replace whitespaces with.
    """

    newlines = []
    for sep in ("\n", "\r\n", "\r"):
        if len(filepath.read_text(encoding="utf-8").split("\r\n")) == 1:
            log.debug(
                "Huh!?!!?!?! labels.csv from ELPV dataset has no newlines but they are in my text editor? Weird..."
            )

    # with open(filepath, "r") as textfile:
    #    for line in textfile.readlines():
    #        newlines.append(whitespace_search_regex.sub(replacement, line).strip())
    # newlines = [x for y in [line.split(" ") for line in newlines] for x in y]
    # with open(Path(filepath.parent, filepath.stem + "_clean" + filepath.suffix), "w") as newfile:

    #    newfile.writelines(newlines)
    # print(f"Temporary breakpoint in {__name__}")
    # raise NotImplementedError("Somehow the newline characters aren't recognized in the ELPV dataset...")
