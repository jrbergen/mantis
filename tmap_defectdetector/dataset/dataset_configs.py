from __future__ import annotations

import os
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Iterable, ClassVar, Type

import numpy as np
import pandas as pd
from pandas import DataFrame

from tmap_defectdetector.dataset.dataset_config_base import (
    DataSetConfig,
    ColSchemaType,
    merge_colschemas,
    ColType,
    ColSchema,
    ColSchemaDefectData,
)
from tmap_defectdetector.logger import log


class _ImageDataset(ColSchema):
    """
    Used to specify the schema (column names and data types) for a
    defect dataset/labelset/sampleset pertaining image samples specifically.
    """

    MIRROR_AXIS: ColType = np.uint8, "Number representing mirrored state w.r.t original image."
    ROT_DEG: ColType = np.int16, "Rotation of image in degrees w.r.t. original image."
    TRANSL_X: ColType = np.int16, "Translation of Y-pixel positions w.r.t. original image."
    TRANSL_Y: ColType = np.int16, "Translation of X-pixel positions w.r.t. original image."


ImageDataSet = merge_colschemas(ColSchemaDefectData, _ImageDataset)


class ColSchemaLabelsELPV(ColSchemaDefectData):
    """Specifies schema for ELPV label data."""

    LABEL_FILENAME: ColType = str, "Name of the file which stores a row's data labels, as string."
    LABEL_PATH: ColType = str, "Full path of the file which stores a row's data label, as string."
    TYPE: ColType = (
        "category",
        "Represents the type of photovoltaic panel (monocrystalline/polycrystalline).",
    )
    PROBABILITY: ColType = np.float64, "Expresses the degree of defectiveness."


class ColSchemaSamplesELPV(ImageDataSet):
    """Specifies schema for ELPV sample data."""

    SAMPLE_PATH: ColType = str, "Full path of the file which stores the data labels, as string."
    SAMPLE: ColType = (
        np.ndarray,
        "A numpy array representing the image of the actual photovoltaic panel.",
    )


ColSpecFullELPV = merge_colschemas([ImageDataSet, ColSchemaLabelsELPV, ColSchemaSamplesELPV])


class DataSetConfigELPV(DataSetConfig):

    LABEL_SCHEMA: ClassVar[ColSchemaLabelsELPV] = ColSchemaLabelsELPV
    SAMPLE_SCHEMA = ClassVar[ColSchemaSamplesELPV] = ColSchemaSamplesELPV
    FULL_SCHEMA = ClassVar[ColSchemaSamplesELPV] = ColSchemaSamplesELPV

    def __init__(
        self,
        sample_path_s: os.PathLike | Iterable[os.PathLike],
        label_path: os.PathLike,
        sample_col_schema: ColSchema = ColSchemaSamplesELPV,
        label_col_schema: ColSchema = ColSchemaLabelsELPV,
        sample_type_desc: str = "solar panel sample image",
    ):
        """
        Provides configuration to load the ELPV dataset for training a defect detection model.

        :param sample_path_s: One ore more path-like object(s) pointing to a sample file.
        :param label_path: A path-like object pointing to corresponding label file.
        :param sample_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param label_col_schema: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        """
        super().__init__(
            sample_path_s=sample_path_s,
            sample_col_schema=sample_col_schema,
            label_path=label_path,
            label_col_schema=label_col_schema,
            sample_type_desc=sample_type_desc,
        )

    @cached_property
    def full_dataset(self) -> DataFrame:
        """Merges sample and label DataFrames into one coherent whole."""
        # return self.
        df_labels = self.label_data
        df_samples = self.sample_data

    @cached_property
    def label_data(self) -> DataFrame:
        """Provides way to load labels for a specific dataset into DataFrame format."""

        lpath = self.label_path
        if not lpath.exists():
            raise FileNotFoundError(f"Couldn't find CSV file to load labels from: {str(lpath)!r}.")

        # np.genfromtxt reads the dataset from the ELPV dataset's labels.csv according to its format.
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

        # Make column with path to label file
        label_df[DataSetConfigELPV.LABEL_SCHEMA.LABEL_FILENAME.name] = [lpath] * len(label_df)

        # Add column with ids identifying which labels belong to which samples (in this case the path is used)
        label_df[DataSetConfigELPV.LABEL_SCHEMA.LABEL_SAMPLE_ID.name] = [
            str(Path(p).name) for p in label_df[type(self).LABEL_SCHEMA.LABEL_FILENAME.name]
        ]

        # Update types
        for col in type(self).LABEL_SCHEMA:
            label_df[col.name] = label_df[col.name].astype(col.type)

        log.info(f"Read ELPV sample labels from file: {str(lpath)}.")
        return label_df

    @cached_property
    def sample_data(self) -> DataFrame:
        """Provides way to load samples for a specific dataset into DataFrame format."""
        return None
        # # Make sure data paths is a list
        # if isinstance(sample_path_s, Path):
        #     sample_path_s = [sample_path_s]
        #
        # # Concatenate label sets
        # labels = functools.reduce(
        #     operator.add, [SampleLabelsELPV.from_csv(lp) for lp in label_paths]
        # )
        #
        # # Load each image with Pillow and put it in a list (tqdm adds progress bar)
        # sample_objs: dict[str, list] = {col: [] for col in cls.SAMPLE_COLUMNS}
        #
        # load_img_func = str if load_images_as_path_strings else cv.imread
        # for sample_id, file in enumerate(
        #     tqdm(
        #         sample_paths,
        #         desc="Loading samples (images)...",
        #         total=len(sample_paths),
        #         unit=f" {cls.DEFAULT_DATASET_UNITS}",
        #     )
        # ):
        #     sample_objs[cls.SAMPLE_COLUMNS.SAMPLE].append(load_img_func(str(file.resolve())))
        #     sample_objs[cls.SAMPLE_COLUMNS.PATH].append(str(file.resolve()))
        #     sample_objs[cls.SAMPLE_COLUMNS.FILENAME].append(str(file.name))
        #
        # sample_objs[cls.SAMPLE_COLUMNS.LABEL_SAMPLE_ID] = sample_objs[cls.SAMPLE_COLUMNS.FILENAME]
        #
        # # Specify column names and corresponding datatypes for image manipulation relevant after dataset amplification.
        # amplification_info_colname_coltypes = {
        #     cls.SAMPLE_COLUMNS.MIRROR_AXIS: np.uint8,
        #     cls.SAMPLE_COLUMNS.ROT_DEG: np.uint16,
        #     cls.SAMPLE_COLUMNS.TRANSLATION_X: np.int32,
        #     cls.SAMPLE_COLUMNS.TRANSLATION_Y: np.int32,
        # }
        # default_property_vals = [0] * len(sample_objs[cls.SAMPLE_COLUMNS.SAMPLE])
        # for column_name, _ in amplification_info_colname_coltypes.items():
        #     sample_objs[column_name] = default_property_vals
        #
        # # Create dataframe with images
        # samples_dframe = pd.DataFrame.from_dict(sample_objs)
        #
        # # Initialize data amplification metadata (specify types)
        # for column_name, tgt_dtype in amplification_info_colname_coltypes.items():
        #     samples_dframe[column_name] = samples_dframe[column_name].astype(tgt_dtype)
        #
        # # Instantiate ImageDataSet
        # return cls(samples=samples_dframe, labels=labels)


if __name__ == "__main__":
    a = DataSetConfigELPV(
        sample_path_s=[
            Path(Path.home(), ".tmapdd", "datasets", "dataset-elpv", "images", "cell0001.png")
        ],
        label_path=Path(Path.home(), ".tmapdd", "datasets", "dataset-elpv", "labels.csv"),
    )
    print(f"Temporary breakpoint in {__name__}")
