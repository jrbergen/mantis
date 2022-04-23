"""
Add your dataset configurations here by extending DataSetConfig
and defining Schemas (see ELPV dataset below for example).
"""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Iterator, Callable, Collection, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import cv2 as cv

from tmap_defectdetector.dataset.dataset_config_base import (
    DataSetConfig,
    ColSchemaDefectData,
    SchemaEntry,
    ColName,
)

from tmap_defectdetector.logger import log
from tmap_defectdetector.pathconfig.paths import DIR_DATASETS


@dataclass(repr=False)
class ColSchemaSamplesImageData(ColSchemaDefectData):
    """
    Used to specify the schema (column names and data types) for a
    defect dataset/labelset/sampleset pertaining image samples specifically.
    """

    MIRROR_AXIS: SchemaEntry = SchemaEntry(
        "MIRROR_AXIS",
        np.uint8,
        docstring="Number representing mirrored state w.r.t original image.",
    )
    ROT_DEG: SchemaEntry = SchemaEntry(
        "ROT_DEG", np.int16, docstring="Rotation of image in degrees w.r.t. original image."
    )
    TRANSL_X: SchemaEntry = SchemaEntry(
        "TRANSL_X", np.int16, docstring="Translation of Y-pixel positions w.r.t. original image."
    )
    TRANSL_Y: SchemaEntry = SchemaEntry(
        "TRANSL_Y", np.int16, docstring="Translation of X-pixel positions w.r.t. original image."
    )


@dataclass(repr=False)
class ColSchemaLabels(ColSchemaDefectData):
    """Specifies schema for label data."""


@dataclass(repr=False)
class ColSchemaFullImageData(ColSchemaLabels, ColSchemaSamplesImageData):
    """Specifies schema for an image dataset's label _and_ sample data."""


@dataclass(repr=False)
class ColSchemaLabelsELPV(ColSchemaLabels):
    """Specifies schema for ELPV label data."""

    LABEL_FILENAME: SchemaEntry = SchemaEntry(
        "LABEL_FILENAME",
        str,
        docstring="Name of the file which stores a row's data labels, as string.",
    )
    LABEL_PATH: SchemaEntry = SchemaEntry(
        "LABEL_PATH",
        str,
        docstring="Full path of the file which stores a row's data label, as string.",
    )
    TYPE: SchemaEntry = SchemaEntry(
        "TYPE",
        "category",
        docstring="Represents the type of photovoltaic panel (monocrystalline/polycrystalline).",
    )
    PROBABILITY: SchemaEntry = SchemaEntry(
        "PROBABILITY", np.float64, docstring="Expresses the degree of defectiveness."
    )


@dataclass(repr=False)
class ColSchemaSamplesELPV(ColSchemaSamplesImageData):
    """Specifies schema for ELPV sample data."""

    SAMPLE_PATH: SchemaEntry = SchemaEntry(
        "SAMPLE_PATH",
        str,
        docstring="Full path of the file which stores the data labels, as string.",
    )
    SAMPLE: SchemaEntry = SchemaEntry(
        "SAMPLE",
        object,  # np.ndarray (but cvann
        docstring="A numpy array representing the image of the actual photovoltaic panel.",
    )


@dataclass(repr=False)
class ColSchemaFullELPV(ColSchemaFullImageData, ColSchemaLabelsELPV, ColSchemaSamplesELPV):
    """Specifies schema for ELPV label _and_ sample data."""

    pass


class ImageDatasetConfig(DataSetConfig):
    LABEL_SCHEMA: ClassVar[ColSchemaLabels] = ColSchemaLabels()
    SAMPLE_SCHEMA: ClassVar[ColSchemaSamplesImageData] = ColSchemaSamplesImageData()
    FULL_SCHEMA: ClassVar[ColSchemaFullImageData] = ColSchemaFullImageData()

    _RASTER_IMG_EXTENSIONS: set[str] = {
        ".tif",
        ".tiff",
        ".png",
        ".bmp",
        ".raw",
        ".jfif",
        ".jif",
        ".jfi",
        ".jpe",
        ".jpeg",
        ".jpg",
        ".dib",
        ".gif",
        ".webp",
        ".arw",
        ".cr2",
        ".nrw",
        ".k25",
        ".heif",
        ".heic",
    }
    """Valid/expected possible extensions for raster images."""

    def __init__(
        self,
        sample_dirs: os.PathLike | Collection[os.PathLike],
        label_path: os.PathLike,
        sample_col_schema: ColSchemaDefectData = LABEL_SCHEMA,
        label_col_schema: ColSchemaDefectData = SAMPLE_SCHEMA,
        sample_type_desc: str = "solar panel sample image",
    ):
        """
        Provides configuration to load an image dataset for training a defect detection model.

        :param sample_dirs: One ore more path-like object(s) pointing to a directory with sample files.
        :param label_path: A path-like object pointing to corresponding label file.
        :param sample_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param label_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        """
        super().__init__(
            sample_dirs=sample_dirs,
            sample_col_schema=sample_col_schema,
            label_path=label_path,
            label_col_schema=label_col_schema,
            sample_type_desc=sample_type_desc,
        )

    @classmethod
    def file_is_sample(cls, file: Path) -> bool:
        """Checks if potential sample file is an image assuming it has a (correct) extension."""
        return file.is_file() and file.suffix.lower() in ImageDatasetConfig._RASTER_IMG_EXTENSIONS

    def get_sample_paths(
        self,
        filechecker_function: Optional[Callable[[Path], bool]] = None,
        glob_pat: str = "*.*",
        recursive: bool = True,
    ) -> Iterator[Path]:

        _filechecker_function: Callable[[Path], bool] = (
            type(self).file_is_sample if filechecker_function is None else filechecker_function
        )

        for sample_dir in self.sample_dirs:
            sample_dir = Path(sample_dir)
            sample_path_iterator = (
                sample_dir.rglob(glob_pat) if recursive else sample_dir.glob(glob_pat)
            )

            for potential_sample_file in sample_path_iterator:
                if _filechecker_function(potential_sample_file):
                    yield potential_sample_file

    @abstractmethod
    @cached_property
    def full_dataset(self) -> DataFrame:
        """
        Merges sample and label DataFrames into one coherent whole.
        _Must_ be implemented for extending classes.
        """
        raise NotImplementedError("Not implemented for baseclass")

    @abstractmethod
    @cached_property
    def label_data(self) -> DataFrame:
        """
        Provides way to load labels for a specific dataset into DataFrame format.
        This step is seperate from loading the samples as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        _Must_ be implemented for extending classes.
        """
        raise NotImplementedError("Not implemented for baseclass")

    @abstractmethod
    @cached_property
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into DataFrame format.
        This step is seperate from loading the label data as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        _Must_ be implemented for extending classes.
        """
        raise NotImplementedError("Not implemented for baseclass")


class DataSetConfigELPV(ImageDatasetConfig):

    LABEL_SCHEMA: ClassVar[ColSchemaLabelsELPV] = ColSchemaLabelsELPV()
    SAMPLE_SCHEMA: ClassVar[ColSchemaSamplesELPV] = ColSchemaSamplesELPV()
    FULL_SCHEMA: ClassVar[ColSchemaFullELPV] = ColSchemaFullELPV()

    def __init__(
        self,
        sample_dirs: os.PathLike
        | Collection[os.PathLike] = (DIR_DATASETS / "dataset-elpv" / "images",),
        label_path: os.PathLike = Path(DIR_DATASETS / "dataset-elpv" / "labels.csv"),
        sample_col_schema: ColSchemaDefectData = LABEL_SCHEMA,
        label_col_schema: ColSchemaDefectData = SAMPLE_SCHEMA,
        sample_type_desc: str = "solar panel sample image",
    ):
        """
        Provides configuration to load the ELPV dataset for training a defect detection model.

        :param sample_dirs: One ore more path-like object(s) pointing to a directory with sample files.
        :param label_path: A path-like object pointing to corresponding label file.
        :param sample_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param label_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        """
        super().__init__(
            sample_dirs=sample_dirs,
            sample_col_schema=sample_col_schema,
            label_path=label_path,
            label_col_schema=label_col_schema,
            sample_type_desc=sample_type_desc,
        )

    @cached_property
    def full_dataset(self) -> DataFrame:
        """Merges sample and label DataFrames into one coherent whole."""

        # Check that column name and type used for label & sample ID are equal.
        if self.LABEL_SCHEMA.LABEL_SAMPLE_ID != self.SAMPLE_SCHEMA.LABEL_SAMPLE_ID:
            raise ValueError(
                f"Cannot merge label data with sample data: label schema entry "
                f"{self.LABEL_SCHEMA.LABEL_SAMPLE_ID.name!r} is not congruent with "
                f"{self.SAMPLE_SCHEMA.LABEL_SAMPLE_ID.name!r}."
            )
        log.info("Merging label and sample datasets...")
        return self.label_data.merge(
            self.sample_data, how="outer", on=self.LABEL_SCHEMA.LABEL_SAMPLE_ID.name, sort=True
        )

    @cached_property
    def label_data(self) -> DataFrame:
        """
        Provides way to load labels for a specific dataset into DataFrame format.
        This step is seperate from loading the samples as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        """

        lpath = self.label_path
        if not lpath.exists():
            raise FileNotFoundError(f"Couldn't find CSV file to load labels from: {str(lpath)!r}.")

        # np.genfromtxt reads the dataset from the ELPV dataset's labels.csv according to its format.
        log.info("Reading label file data for ELPV dataset...")
        label_df = np.genfromtxt(
            lpath,
            dtype=["|U19", "<f8", "|U4"],
            names=[
                type(self).LABEL_SCHEMA.LABEL_PATH.name,
                type(self).LABEL_SCHEMA.PROBABILITY.name,
                type(self).LABEL_SCHEMA.TYPE.name,
            ],
            encoding="utf-8",
        )
        label_df = pd.DataFrame(label_df)

        log.info("Wrapping up label DataFrame construction...")
        # Make column with path to label file
        label_df[DataSetConfigELPV.LABEL_SCHEMA.LABEL_FILENAME.name] = [lpath] * len(label_df)

        # Add column with ids identifying which labels belong to which samples (in this case the path is used)
        label_df[DataSetConfigELPV.LABEL_SCHEMA.LABEL_SAMPLE_ID.name] = [
            str(Path(p).name) for p in label_df[type(self).LABEL_SCHEMA.LABEL_PATH.name]
        ]

        # Update types
        for col in type(self).LABEL_SCHEMA:
            label_df[col.name] = label_df[col.name].astype(col.type)

        log.info(f"Read ELPV sample labels from file: {str(lpath)}.")
        return label_df

    @cached_property
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into DataFrame format.
        This step is seperate from loading the label data as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        """

        sample_dict: dict[ColName, list] = {colname: [] for colname in self.SAMPLE_SCHEMA.to_dict()}
        log.info("Started reading sample files.")
        # Build dictionary with sample files and initial entries for the relevant columns.
        for sample_id, file in enumerate(
            tqdm(
                self.get_sample_paths(),
                desc="Loading samples (images)...",
                total=len(self.sample_dirs),
                unit=f" {self.sample_type_desc}",
            )
        ):
            sample_dict[self.SAMPLE_SCHEMA.SAMPLE.name].append(cv.imread(str(file.resolve())))
            sample_dict[self.SAMPLE_SCHEMA.SAMPLE_PATH.name].append(str(file.resolve()))
            sample_dict[self.SAMPLE_SCHEMA.LABEL_SAMPLE_ID.name].append(file.name)

            # Initialize data amplification metadata columns
            for colname in (
                self.SAMPLE_SCHEMA.MIRROR_AXIS.name,
                self.SAMPLE_SCHEMA.ROT_DEG.name,
                self.SAMPLE_SCHEMA.TRANSL_X.name,
                self.SAMPLE_SCHEMA.TRANSL_Y.name,
            ):
                sample_dict[colname].append(0)

        log.info("Started DataFrame construction from founr sample files.")
        df_samples: pd.DataFrame = pd.DataFrame.from_dict(sample_dict)

        # Update types
        for entry in self.SAMPLE_SCHEMA.schema_entries:
            df_samples[entry.name] = df_samples[entry.name].astype(entry.type)

        return df_samples
