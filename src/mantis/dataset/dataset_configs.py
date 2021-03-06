"""
Add your dataset configurations here by extending DataSetConfig
and defining Schemas (see ELPV dataset below for example).
"""

from __future__ import annotations

import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Collection, Optional, Type, TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import cv2 as cv

from mantis.dataset.base.dataset_configs_base import (
    ImageDataSetConfig,
    FALLBACK_LABEL_CATEGORY,
)
from mantis.dataset.downloaders import DataSetDownloaderELPV
from mantis.dataset.base.schemas_base import (
    ColName,
)
from mantis.dataset.schemas import (
    SchemaLabelsELPV,
    SchemaSamplesELPV,
    SchemaFullELPV,
    SchemaLabelsWineDetector,
    SchemaSamplesWineDetector,
    SchemaFullWineDetector,
)

from mantis.logger import log
from mantis import DIR_DATASETS
from mantis.dataset.base.downloaders_base import DataSetDownloader

if TYPE_CHECKING:
    from mantis.dataset.base.datasets_base import (
        DefectDetectionDataSetImages,
        DefectDetectionDataSet,
    )
    from mantis.controllers.base import TUIControllerDataSet


class DataSetConfigELPV(ImageDataSetConfig):

    SCHEMA_LABELS: ClassVar[SchemaLabelsELPV] = SchemaLabelsELPV()
    SCHEMA_SAMPLES: ClassVar[SchemaSamplesELPV] = SchemaSamplesELPV()
    SCHEMA_FULL: ClassVar[SchemaFullELPV] = SchemaFullELPV()
    DEFAULT_LABEL_NAMES: tuple[str, ...] = ("pass", "minor_damage", "major_damage", "fail")

    def __init__(
        self,
        name: str = "ELPV Dataset",
        dataset_cls: Type[DefectDetectionDataSetImages] = None,
        sample_dirs: os.PathLike | Collection[os.PathLike] = (DIR_DATASETS / "dataset-elpv" / "images",),
        label_path: os.PathLike = Path(DIR_DATASETS / "dataset-elpv" / "labels.csv"),
        schema_samples: SchemaSamplesELPV = SCHEMA_SAMPLES,
        schema_labels: SchemaLabelsELPV = SCHEMA_LABELS,
        sample_type_desc: str = "solar panel sample image",
        controller_cls: Optional[Type[TUIControllerDataSet]] = None,
        downloader: Optional[Type[DataSetDownloader]] = DataSetDownloaderELPV,
        default_filter_query_str: Optional[str] = f"{SchemaLabelsELPV().TYPE.name}=='poly'",
    ):
        """
        Provides configuration to load the ELPV dataset for training a defect detection model.

        :param name: name/id used to identify this dataset (type).
        :param dataset_cls: (optional) Dataset class to use (defaults to None -> ImageDataSetELPV).
        :param sample_dirs: (optional) One ore more path-like object(s) pointing to a directory with sample files.
            Defaults to DIR_DATASETS / "dataset-elpv" / "images".
        :param label_path: (optional) A path-like object pointing to corresponding label file.
            Defaults to DIR_DATASETS / "dataset-elpv" / "labels.csv"
        :param schema_samples: (optional) SchemaSamples (column specification) object declaring column names and types
            for the samples in this dataset. Defaults to cls.SCHEMA_SAMPLES.
        :param schema_labels: (optional) SchemaLabels (column specification) object declaring column names and types
            for the labels in this dataset. Defaults to cls.SCHEMA_LABELS.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        :param controller_cls: (optional) class which handles interactions with the TUI (Default = None -> TUIControllerELPV).
        :param downloader: (optional) class with a 'download' method to download
            the required data (Default = DataSetConfigELPV)
        :param default_filter_query_str: (optional) Default query string to filter data with.
        """
        if dataset_cls is None:
            from mantis.dataset.datasets import ImageDataSetELPV

            dataset_cls = ImageDataSetELPV
        if controller_cls is None:
            from mantis.controllers.controller_elpv import TUIControllerELPV

            controller_cls = TUIControllerELPV

        super().__init__(
            name=name,
            sample_dirs=sample_dirs,
            schema_samples=schema_samples,
            label_path=label_path,
            schema_labels=schema_labels,
            sample_type_desc=sample_type_desc,
            downloader=downloader,
            dataset_cls=dataset_cls,
            controller_cls=controller_cls,
            default_filter_query_str=default_filter_query_str,
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
                type(self).SCHEMA_LABELS.LABEL_PATH.name,
                type(self).SCHEMA_LABELS.PROBABILITY.name,
                type(self).SCHEMA_LABELS.TYPE.name,
            ],
            encoding="utf-8",
        )
        label_df = pd.DataFrame(label_df)

        log.info("Wrapping up label DataFrame construction...")
        # Make column with path to label file
        label_df[DataSetConfigELPV.SCHEMA_LABELS.LABEL_FILENAME.name] = [lpath] * len(label_df)
        label_df[DataSetConfigELPV.SCHEMA_LABELS.LABEL_CATEGORY.name] = [FALLBACK_LABEL_CATEGORY] * len(
            label_df
        )

        # Add column with ids identifying which labels belong to which samples (in this case the path is used)
        label_df[DataSetConfigELPV.SCHEMA_LABELS.LABEL_SAMPLE_ID.name] = [
            str(Path(p).name) for p in label_df[type(self).SCHEMA_LABELS.LABEL_PATH.name]
        ]

        # Update types
        for col in type(self).SCHEMA_LABELS:
            label_df[col.name] = label_df[col.name].astype(col.type)

        label_df = self.add_categorical_column_to_label_data(label_data=label_df)
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

        sample_dict: dict[ColName, list] = {colname: [] for colname in self.SCHEMA_SAMPLES.to_dict()}
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
            sample_dict[self.SCHEMA_SAMPLES.SAMPLE.name].append(cv.imread(str(file.resolve())))
            sample_dict[self.SCHEMA_SAMPLES.SAMPLE_PATH.name].append(str(file.resolve()))
            sample_dict[self.SCHEMA_SAMPLES.SAMPLE_PATH_CATEGORY.name].append(str(file.resolve()))
            sample_dict[self.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name].append(file.name)

            # Initialize data amplification metadata columns
            for colname in (
                self.SCHEMA_SAMPLES.MIRROR_AXIS.name,
                self.SCHEMA_SAMPLES.ROT_DEG.name,
                self.SCHEMA_SAMPLES.TRANSL_X.name,
                self.SCHEMA_SAMPLES.TRANSL_Y.name,
            ):
                sample_dict[colname].append(0)

        log.info("Started DataFrame construction from founr sample files.")
        df_samples: pd.DataFrame = pd.DataFrame.from_dict(sample_dict)

        # Update types
        for entry in self.SCHEMA_SAMPLES.schema_entries:
            df_samples[entry.name] = df_samples[entry.name].astype(entry.type)

        return df_samples

    def add_categorical_column_to_label_data(
        self,
        label_data: pd.DataFrame,
        label_names: tuple[str, ...] = DEFAULT_LABEL_NAMES,
        label_bins=(-0.1, 0.25, 0.5, 0.75, 1.1),
    ) -> pd.DataFrame:
        """
        Adds categorical column labels to data.
        Requires knowledge of the number of categories beforehand.
        """

        col: str = self.SCHEMA_FULL.PROBABILITY.name
        lblcat: str = self.SCHEMA_FULL.LABEL_CATEGORY.name

        categories = sorted(list(set(label_data.loc[:, col])))
        if len(label_names) != (n_labels_found := len(categories)):
            raise ValueError(
                f"Number of categorical values doesn't equal number "
                f"of different encountered labels ({n_labels_found})."
            )

        if len(label_bins) != len(label_names) + 1:
            raise ValueError(f"Number of bins must be number of label names + 1.")

        label_data[lblcat] = pd.cut(label_data.loc[:, col], bins=label_bins, labels=label_names)

        return label_data


class DataSetConfigWineDetector(ImageDataSetConfig):

    SCHEMA_LABELS: ClassVar[SchemaLabelsWineDetector] = SchemaLabelsWineDetector()
    SCHEMA_SAMPLES: ClassVar[SchemaSamplesWineDetector] = SchemaSamplesWineDetector()
    SCHEMA_FULL: ClassVar[SchemaFullWineDetector] = SchemaFullWineDetector()

    def __init__(
        self,
        name: str = "Wine Detector Dataset",
        dataset_cls: Type[DefectDetectionDataSet] = None,
        sample_dirs: os.PathLike | Collection[os.PathLike] = Path(DIR_DATASETS),
        label_path: os.PathLike = Path("data", "winequality-red.csv"),
        schema_samples: SchemaSamplesELPV = SCHEMA_SAMPLES,
        schema_labels: SchemaLabelsELPV = SCHEMA_LABELS,
        sample_type_desc: str = "wine quality data",
        controller_cls: Optional[Type[TUIControllerDataSet]] = None,
        downloader: Optional[Type[DataSetDownloader]] = DataSetDownloader,
        default_filter_query_str: Optional[str] = None,
    ):
        """
        Provides configuration to load the Wine Detector dataset for training a defect detection model.

        :param name: name/id used to identify this dataset (type).
        :param dataset_cls: (optional) Dataset class to use (defaults to None -> ImageDataSetELPV).
        :param sample_dirs: (optional) One ore more path-like object(s) pointing to a directory with sample files.
        :param label_path: (optional) A path-like object pointing to corresponding label file.
        :param schema_samples: (optional) SchemaSamples (column specification) object declaring column names and types
            for the samples in this dataset. Defaults to cls.SCHEMA_SAMPLES.
        :param schema_labels: (optional) SchemaLabels (column specification) object declaring column names and types
            for the labels in this dataset. Defaults to cls.SCHEMA_LABELS.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        :param controller_cls: (optional) class which handles interactions with the TUI (Default = None).
        :param downloader: (optional) class with a 'download' method to download
            the required data (Default = DataSetDownloaderWineDetector)
        :param default_filter_query_str: (optional) Default query string to filter data with (Default = None).
        """

        super().__init__(
            name=name,
            sample_dirs=sample_dirs,
            schema_samples=schema_samples,
            label_path=label_path,
            schema_labels=schema_labels,
            sample_type_desc=sample_type_desc,
            downloader=downloader,
            controller_cls=controller_cls,
            dataset_cls=dataset_cls,
            default_filter_query_str=default_filter_query_str,
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
        log.warn(f"{type(self).__name__} 'label_data' not implemented")
        warnings.warn(f"{type(self).__name__} 'label_data' not implemented")
        return pd.DataFrame()

    @cached_property
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into DataFrame format.
        This step is seperate from loading the label data as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        """
        log.warn(f"{type(self).__name__} 'sample_data' not implemented")
        warnings.warn(f"{type(self).__name__} 'sample_data' not implemented")
        return pd.DataFrame()
