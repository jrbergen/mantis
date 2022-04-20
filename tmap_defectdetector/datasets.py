"""Contains baseclass & concrete implementations for DataSets and more specialized types thereof."""
from __future__ import annotations


import functools
import operator
from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import TypeAlias, Iterable


import pandas as pd
from PIL import Image
from tqdm import tqdm

from tmap_defectdetector.logger import log
from tmap_defectdetector.samples import (
    DataLabels,
    SampleLabelsBase,
)
from tmap_defectdetector.samples import SampleLabelsELPV

ImageCollection: TypeAlias = list[Image] | tuple[Image, ...]

LabelCollection: TypeAlias = Iterable[DataLabels]
Translation: TypeAlias = tuple[float, float] | tuple[int, int]


class DataSet:

    data: pd.DataFrame
    labels: SampleLabelsBase

    # the abstractmethod decorator forces classes inheriting from this class to implement this method before they can be initialized.
    # (requires inheriting from abc.ABC)
    @abstractmethod
    def amplify_data(self):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case the training data comprises images.
        Subclasses should implement this method.
        """
        raise NotImplementedError(
            f"method {inspect.currentframe().f_code.co_name} not implemented for baseclass."
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(data={self.data.__repr__()}, labels={self.labels.__repr__()})"
        )


class ImageDataSet(DataSet, ABC):
    def __init__(self, data: pd.DataFrame, labels: SampleLabelsELPV):
        self.labels: SampleLabelsELPV = labels
        self.data: pd.DataFrame = data

    @abstractmethod
    def amplify_data(self):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case the training data comprises images.
        """
        log.info("Amplyfing dataset (not really, not implemented yet...)")
        log.info("Adding mirrored images to dataset (not really, not implemented yet...)")
        log.info("Adding translated images to dataset (not really, not implemented yet...)")
        log.info("Adding rotated images to dataset (not really, not implemented yet...)")
        log.info("Adding superimposed images to dataset (not really, not implemented yet...)")

    @property
    def images(self):
        """
        'images' is an alias for data, by making it a property the code is executed
        every time the attribute is accessed, making sure it always returns an up-to-date self.data.
        """
        return self.data

    @images.setter
    def images(self, data: pd.DataFrame):
        """
        'images' is an alias for data, and by making it a property we can run code when
         someone tries to set/write this attribute.
        E.g. check that the value passed is indeed a DataFrame, and also make sure that the 'data' attribute gets updated.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for images, got {type(data).__name__}.")
        self.data = data

    @classmethod
    def from_paths(
        cls, data_paths: list[Path] | Path, label_paths: list[Path] | Path
    ) -> ImageDataSet:
        """
        Method to construct an image dataset from a list of paths and a list of paths to label files, which
        subclasses should implement.
        :raises: NotImplemented
        """
        raise NotImplementedError(
            f"method {inspect.currentframe().f_code.co_name} not implemented for baseclass."
        )


class ImageDataSetELPV(ImageDataSet):
    def __init__(self, data: pd.DataFrame, labels: SampleLabelsELPV):
        super().__init__(data=data, labels=labels)  # Pass args to the superclass's constructor.

    @classmethod
    def from_paths(
        cls, data_paths: list[Path] | Path, label_paths: list[Path] | Path
    ) -> ImageDataSet:
        """
        Create an ImageDataSet from one or more data path(s) and one or more label data path(s).
        Assumes label data path is a CSV file in the format as found for the ELPV dataset.
        """

        # Make sure label paths is a list
        if isinstance(label_paths, Path):
            label_paths = [label_paths]

        # Concatenate label sets
        labels = functools.reduce(
            operator.add, [SampleLabelsELPV.from_csv(lp) for lp in label_paths]
        )

        # Make sure data paths is a list
        if isinstance(data_paths, Path):
            data_paths = [data_paths]

        # Load each image with Pillow and put it in a list (tqdm adds progress bar)
        data_objs = []
        for file in tqdm(data_paths, desc="Loading samples (images)...", total=len(data_paths)):
            with Image.open(file) as imgobj:
                data_objs.append(imgobj)

        # Create dataframe with images
        data_df = pd.DataFrame(columns={"images": data_objs})

        # Instantiate ImageDataSet
        return cls(data=data_df, labels=labels)

    def __repr__(self) -> str:
        """String representation of this class, which should translate to a machine-readable string (valid Python)."""
        return (
            f"{type(self).__name__}(data={self.data.__repr__()}, labels={self.labels.__repr__()})"
        )

    def __str__(self) -> str:
        """String representation of this class more suited for humans."""
        return f"{type(self).__name__}()"
