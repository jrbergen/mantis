from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import TypeAlias, Iterable

from PIL.Image import Image

ImageCollection: TypeAlias = list[Image] | tuple[Image, ...]
LabelCollection: TypeAlias = list[int | float | str] | tuple[int | float | str]
Translation: TypeAlias = tuple[float, float] | tuple[int, int]

class AbstractDataSet(ABC):
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


class DataSet(ABC):
    def amplify_data(self, *args, **kwargs):
        ...


class ImageDataSet(DataSet):

    _VALID_MIRROR_AXES: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 360)
    DEFAULT_TRANSLATIONS: Iterable[Translation] = ((.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5))
    DEFAULT_ROTATIONS: Iterable[int] = (90, 180, 270)
    DEFAULT_MIRROR_AXES: Iterable[float | int] = (0, 45, 90, 135)

    def __init__(self,
                 data: ImageCollection = (),
                 labels: LabelCollection = ()):
        self.data: list[Image] = list(data)
        self.labels = labels

    def amplify_data(self,
                     translations: Iterable[float | int] = DEFAULT_TRANSLATIONS,
                     rotations: Iterable[float | int] = DEFAULT_ROTATIONS,
                     mirror_axes: Iterable[float | int] = DEFAULT_MIRROR_AXES,
                     *args, **kwargs
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

    def __add__(self, other: ImageDataSet | ImageCollection | Image) -> ImageDataSet:
        """Currently doesn't handle data labels, which may lead to image/label sets of different sizes..."""
        match other:
            case list() | tuple():
                if all(isinstance(item, Image) for item in other):
                    self.data = list(set(self.data) | set(other))
                else:
                    raise TypeError(f"All items in data to be added must be images ({Image.__qualname__} objects),"
                                    " which seems not to be the case.")
            case ImageDataSet():
                self.data += other.data
            case Image():
                self.data.append(other)
            case _:
                raise TypeError(f"Can only add together (lists of) images or other {type(self).__name__!r} "
                                f"datasets to this {type(self).__name__} dataset.")
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data.__repr__()})"
