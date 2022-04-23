from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from tmap_defectdetector.dataset.dataset_config_base import DataSetConfig, ColumnSpec

class ImageDataset(ColSpec):
    """
    Used to specify the column names and data types for a defect dataset/labelset/sampleset.
    A user-defined mapping whose keys are also accessible via the '.' operator; e.g. ColSpec.COLNAME.name
    """

    LABEL_SAMPLE_ID: ColType = str
    """
    Column with equal entries for sample and label data so 
    sample and label data can be merged in the right order.
    """

IMAGE_DATASET_DEFAULT_COLSPEC = ColumnSpec(
    sample_label_id=str,
    mirror_axis=np.uint8,
    rotation=np.int16,
    translation_x=np.int16,
    translation_y=np.int16
)


class DataSetConfigELPV(DataSetConfig):

    ELPV_LABEL_COLSPEC = ColumnSpec(
        label_filename="category",
        label_path=str,
        path=str,
        type="category",
        probability=np.float64,
    )
    ELPV_SAMPLE_COLSPEC = ColumnSpec(
        sample_path=str,
        sample=np.ndarray,
    )

    ELPV_FULL_COLSPEC = IMAGE_DATASET_DEFAULT_COLSPEC + ELPV_LABEL_COLSPEC + ELPV_SAMPLE_COLSPEC

    def __init__(
        self,
        sample_path_s: os.PathLike | Iterable[os.PathLike],
        label_path: os.PathLike,
        sample_col_spec: ColumnSpec = ELPV_SAMPLE_COLSPEC,
        label_col_spec: ColumnSpec = ELPV_LABEL_COLSPEC,
        sample_type_desc: str = "solar panel sample image",
    ):
        """
        Provides configuration to load the ELPV dataset for training a defect detection model.

        :param sample_path_s: One ore more path-like object(s) pointing to a sample file.
        :param sample_col_spec: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param label_path: A path-like object pointing to corresponding label file.
        :param label_col_spec: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        """
        super().__init__(
            sample_path_s, sample_col_spec, label_path, label_col_spec, sample_type_desc
        )
        self.sample_path_s = (
            sample_path_s if isinstance(sample_path_s, Iterable) else [sample_path_s]
        )
        self.label_path = label_path

        self.sample_col_spec: ColumnSpec = sample_col_spec
        self.label_col_spec: ColumnSpec = label_col_spec

        self.sample_type_desc: str = sample_type_desc

    def load_full_dataset(self) -> DataFrame:
        """Merges sample and label DataFrames into one coherent whole."""

        # Make sure label paths is a list
        if isinstance(label_paths, Path):
            label_paths = [label_paths]

    def load_label_data(
        self,
        label_path: Path = ,
        label_column_spec: Optional[ColumnSpec] = None,
    ) -> DataFrame:
        """Provides way to load labels for a specific dataset into DataFrame format."""
        if not isinstance(label_path, Iterable):
            label_path = [label_path]

        for lpath in label_path:
            if not lpath.exists():
                raise FileNotFoundError(
                    f"Couldn't find CSV file to load labels from: {str(label_path)!r}."
                )

        # np.genfromtxt reads the dataset from the ELPV dataset's labels.csv according to its format.
        dframes = []
        for lpath in label_path:
            dframes.append(pd.DataFrame(
                np.genfromtxt(
                    lpath,
                    dtype=["|U19", "<f8", "|U4"],
                    names=[self.ELPV_LABEL_COLSPEC.path,
                           self.ELPV_LABEL_COLSPEC.probability,
                           self.ELPV_LABEL_COLSPEC.type],
                    encoding="utf-8",
                    )
                )
            )

        labels = pd.concatenate(dframes, axis=1)
        # Make column with path to label file
        labels[self.ELPV_LABEL_COLSPEC.filename] =  [label_path] * len(labels)

        # Add column with ids identifying which labels belong to which samples (in this case the path is used)
        labels[self.ELPV_FULL_COLSPEC.sample_label_id] = [str(Path(p).name) for p in labels[self.ELPV_FULL_COLSPEC.label_filename]]

        # Update types
        for name, dtype_ in cls.DFRAME_COLS:
            labels[name] = labels[name].astype(dtype_)

        log.info(f"Read ELPV sample labels from csv file: {str(csv_path)}.")
        return cls(label_dframe=labels)
    def load_sample_data(
        self, sample_path_s: Path | Iterable[Path], sample_colspec: ColumnSpec
    ) -> DataFrame:
        """Provides way to load samples for a specific dataset into DataFrame format."""
        # Make sure data paths is a list
        if isinstance(sample_path_s, Path):
            sample_path_s = [sample_path_s]

        # Concatenate label sets
        labels = functools.reduce(
            operator.add, [SampleLabelsELPV.from_csv(lp) for lp in label_paths]
        )

        # Load each image with Pillow and put it in a list (tqdm adds progress bar)
        sample_objs: dict[str, list] = {col: [] for col in cls.SAMPLE_COLUMNS}

        load_img_func = str if load_images_as_path_strings else cv.imread
        for sample_id, file in enumerate(
            tqdm(
                sample_paths,
                desc="Loading samples (images)...",
                total=len(sample_paths),
                unit=f" {cls.DEFAULT_DATASET_UNITS}",
            )
        ):
            sample_objs[cls.SAMPLE_COLUMNS.SAMPLE].append(load_img_func(str(file.resolve())))
            sample_objs[cls.SAMPLE_COLUMNS.PATH].append(str(file.resolve()))
            sample_objs[cls.SAMPLE_COLUMNS.FILENAME].append(str(file.name))

        sample_objs[cls.SAMPLE_COLUMNS.LABEL_SAMPLE_ID] = sample_objs[cls.SAMPLE_COLUMNS.FILENAME]

        # Specify column names and corresponding datatypes for image manipulation relevant after dataset amplification.
        amplification_info_colname_coltypes = {
            cls.SAMPLE_COLUMNS.MIRROR_AXIS: np.uint8,
            cls.SAMPLE_COLUMNS.ROTATION: np.uint16,
            cls.SAMPLE_COLUMNS.TRANSLATION_X: np.int32,
            cls.SAMPLE_COLUMNS.TRANSLATION_Y: np.int32,
        }
        default_property_vals = [0] * len(sample_objs[cls.SAMPLE_COLUMNS.SAMPLE])
        for column_name, _ in amplification_info_colname_coltypes.items():
            sample_objs[column_name] = default_property_vals

        # Create dataframe with images
        samples_dframe = pd.DataFrame.from_dict(sample_objs)

        # Initialize data amplification metadata (specify types)
        for column_name, tgt_dtype in amplification_info_colname_coltypes.items():
            samples_dframe[column_name] = samples_dframe[column_name].astype(tgt_dtype)

        # Instantiate ImageDataSet
        return cls(samples=samples_dframe, labels=labels)
