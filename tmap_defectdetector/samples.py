"""Contains (generic) classes/functions regarding training samples"""

from __future__ import annotations

import inspect

from pathlib import Path
from typing import Optional, Iterable, TypeAlias, Collection

import numpy as np
import pandas as pd
from PIL import Image
from numpy import ndarray, asarray


from tmap_defectdetector.logger import log

DataLabels: TypeAlias = dict[str, Collection[float | int | str]]

# This class is currenlty not used. We might delete it later.
class TrainingSample:
    ...


# This class is currenlty not used. We might delete it later.
class TrainingSampleImage(TrainingSample):
    image: Optional[ndarray | Image]

    __slots__ = ("image", "path", "labels")

    def __init__(
        self,
        image: ndarray | Image = None,
        path: Path = Path(),
        labels: DataLabels = tuple(),
    ):
        """
        Class representing a training image, also storing additonal information
        such as its path or any accompanying labels.
        """
        self.image = self._init_image(image)
        self.path = self._init_path(path)
        self.labels: DataLabels = labels

    @classmethod
    def _init_path(cls, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Image path {str(path)!r} doesn't exist.")
        return Path(path)

    @classmethod
    def _init_image(cls, image: ndarray | Image) -> Image:
        if isinstance(image, ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError(f"Expected ndarray or PIL.Image.Image, got {type(image).__name__!r}")

    @classmethod
    def from_path(cls, path: Path, labels: DataLabels = tuple()):
        return TrainingSampleImage(image=Image.open(path), path=path, labels=labels)

    def __array__(self):
        return asarray(self.image)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(image={self.image}, path={self.path}, labels={self.labels})"


class SampleLabelsBase:

    __slots__ = ("label_or_labels",)
    # Defining attributes in the __slots__ dunder attribute makes attribute access faster.
    # However, if you do this you can no longer dynamically add new attributes to an instantiated class.

    def __init__(
        self,
        label_or_labels: pd.DataFrame,
    ):
        self.label_or_labels: pd.DataFrame = label_or_labels

    @classmethod
    def from_dict(
        cls, label_or_labels: pd.DataFrame | dict[str | int, Iterable[int | float | str]]
    ) -> SampleLabelsBase:
        """
        A SampleLabelsBase instance can be constructed from a dictionary with its keys being strings or ints
        indicating the label category (1 image can have multiple labels), and for each key an Iterable
        (i.e. an instance which defines the __iter__ method such as list, tuple, etc.) containing the actual
        label values (which can be ints, floats, or strings in this case).
        """
        if isinstance(label_or_labels, dict):  # If a dictionary is passed
            # If all dictionary keys are ints or strings,
            # and all values in the dictionary are ints, floats, or strings,
            # the type is as expected and we can instantiate the SampleLabelsBase class.
            if all(
                isinstance(k, (str, int)) and isinstance(v, (int, float, str))
                for k, v in label_or_labels.items()
            ):
                label_or_labels = cls(
                    label_or_labels=pd.DataFrame.from_dict(
                        {k: list(v) for k, v in label_or_labels.items()}
                    )
                )
            else:
                # Otherwise we ant to raise a TypeError
                raise TypeError(
                    "Expected dictionary with strings/ints as keys and an iterable of int/floats/strings as values"
                )

        # If the label_or_labels parameter is not a dict, and also isn't already
        # a DataFrame, we also want to raise a TypeError.
        # Note that this code doesn't check whether the DataFrame is of the expected shape.
        elif not isinstance(label_or_labels, pd.DataFrame):
            raise TypeError(
                f"Expected dict[str, Iterable[float | int | str]], got {type(label_or_labels)!r}."
            )

        return label_or_labels

    def __getitem__(self, item):
        """
        This special/dunder/magic method defines what to do when someone tries to use a square bracket index for an
        instance of this class. E.g. labels = SampleLabelsBase -> something = labels['sulfur'].
        In this case, it passes this through to the self.label_or_labels attribute, which is a DataFrame,
        so when you try labels['sulfur'] it tries to access the DataFrame's 'sulfur' column.
        """
        return self.label_or_labels.__getitem__(item)

    def __getattr__(self, item):
        """
        This special/dunder/magic method defines what to do when someone tries to use an attribute of this isinstance
        which could not be found.
        E.g. labels = SampleLabelsBase -> something = labels.shape (
        In this case, SampleLabelsBase has no 'shape' attribute, so it passes this request to
        the self.label_or_labels attribute, which is a DataFrame,
        so when you try labels.shape it accesses the (existing) attribute of self.labels_or_labels,
        which is a DataFrame, and thus returns the shape of the DataFrame.
        """
        return getattr(self.label_or_labels, item)

    def __add__(self, other: SampleLabelsBase) -> SampleLabelsBase:
        """
        This method defines what using the '+' operator between SampleLabelsBase instances does.
        """
        if not isinstance(other, SampleLabelsBase):
            return NotImplemented
        self.label_or_labels = pd.concatenate([self.label_or_labels, other.label_or_labels])
        return self

    @classmethod
    def from_csv(cls, csv_path: Path):
        """
        This method should instantiate a SampleLabelsBase class from a CSV file.
        It is not applicable for all datasets as samples of course aren't always in CSV format.
        Subclasses must implement this method if this the labels _are_ in CSV format for a particular dataset.
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} not implemented for {cls.__qualname__}."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label_or_labels={self.label_or_labels})"


class SampleLabelsELPV(SampleLabelsBase):
    def __init__(self, label_or_labels: pd.DataFrame | dict[str, Collection[float | int | str]]):
        super().__init__(label_or_labels=label_or_labels)

    @classmethod
    def from_csv(cls, csv_path: Path):
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Couldn't find CSV file to load labels from: {str(csv_path)!r}."
            )

        # np.genfromtxt reads the dataset from the ELPV dataset's labels.csv according to its format.
        labels = pd.DataFrame(
            np.genfromtxt(
                csv_path,
                dtype=["|U19", "<f8", "|U4"],
                names=["path", "probability", "type"],
                encoding="utf-8",
            )
        )
        log.info(f"Read ELPV sample labels from csv file: {str(csv_path)}.")
        return cls(label_or_labels=labels)
