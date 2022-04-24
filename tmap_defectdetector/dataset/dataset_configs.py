"""
Add your dataset configurations here by extending DataSetConfig
and defining Schemas (see ELPV dataset below for example).
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Collection

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import cv2 as cv

from tmap_defectdetector.dataset.base.dataset_configs_base import (
    ImageDatasetConfig,
)
from tmap_defectdetector.dataset.base.schemas_base import (
    ColName,
    SchemaDefectData,
)
from tmap_defectdetector.dataset.schemas import (
    SchemaLabelsELPV,
    SchemaSamplesELPV,
    SchemaFullELPV,
)

from tmap_defectdetector.logger import log
from tmap_defectdetector.pathconfig.paths import DIR_DATASETS


class DataSetConfigELPV(ImageDatasetConfig):

    SCHEMA_LABELS: ClassVar[SchemaLabelsELPV] = SchemaLabelsELPV()
    SCHEMA_SAMPLES: ClassVar[SchemaSamplesELPV] = SchemaSamplesELPV()
    SCHEMA_FULL: ClassVar[SchemaFullELPV] = SchemaFullELPV()

    def __init__(
        self,
        sample_dirs: os.PathLike
        | Collection[os.PathLike] = (DIR_DATASETS / "dataset-elpv" / "images",),
        label_path: os.PathLike = Path(DIR_DATASETS / "dataset-elpv" / "labels.csv"),
        schema_samples: SchemaSamplesELPV = SCHEMA_SAMPLES,
        schema_labels: SchemaLabelsELPV = SCHEMA_LABELS,
        sample_type_desc: str = "solar panel sample image",
    ):
        """
        Provides configuration to load the ELPV dataset for training a defect detection model.

        :param sample_dirs: One ore more path-like object(s) pointing to a directory with sample files.
        :param label_path: A path-like object pointing to corresponding label file.
        :param schema_samples: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param schema_labels: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        """
        super().__init__(
            sample_dirs=sample_dirs,
            schema_samples=schema_samples,
            label_path=label_path,
            schema_labels=schema_labels,
            sample_type_desc=sample_type_desc,
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

        # Add column with ids identifying which labels belong to which samples (in this case the path is used)
        label_df[DataSetConfigELPV.SCHEMA_LABELS.LABEL_SAMPLE_ID.name] = [
            str(Path(p).name) for p in label_df[type(self).SCHEMA_LABELS.LABEL_PATH.name]
        ]

        # Update types
        for col in type(self).SCHEMA_LABELS:
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

        sample_dict: dict[ColName, list] = {
            colname: [] for colname in self.SCHEMA_SAMPLES.to_dict()
        }
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
