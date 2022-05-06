"""Contains baseclass & concrete implementations for DataSets and more specialized types thereof."""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TypeAlias, Union, cast

import pandas as pd
from numpy import ndarray
from textual.app import App
from tqdm import tqdm

from mantis import DIR_TMP, DIR_DATASETS
from mantis.dataset.base.dataset_configs_base import FALLBACK_LABEL_CATEGORY
from mantis.dataset.base.datasets_base import DefectDetectionDataSetImages
from mantis.dataset.dataset_configs import DataSetConfigELPV
from mantis.dataset.downloaders import DataSetDownloaderELPV
from mantis.dataset.schemas import SchemaLabelsELPV
from mantis.logger import log
from mantis.model.base import SplitDataSet
from mantis.path_helpers import open_directory_with_filebrowser

ImageCollection: TypeAlias = list[ndarray] | tuple[ndarray, ...]
Translation: TypeAlias = Union[tuple[float, float] | tuple[int, int]]


class ImageDataSetELPV(DefectDetectionDataSetImages):
    def __init__(self, dataset_cfg: DataSetConfigELPV):
        """
        ImageDataSet specific for the ELPV photovoltaic cell defectg dataset.
        (See https://github.com/zae-bayern/elpv-dataset for original dataset).
        """
        super().__init__(dataset_cfg=dataset_cfg)

    @property
    def dataset_cfg(self) -> DataSetConfigELPV:
        if not isinstance((cfg := super().dataset_cfg), DataSetConfigELPV):
            raise TypeError(
                f"Expected dataset configuration of type {DataSetConfigELPV.__name__}, got {type(cfg)}."
            )
        return cast(DataSetConfigELPV, cfg)

    @classmethod
    def run(
        cls, app: App, save_and_open_amplified_dataset: bool = False, **dataset_cfg_kwargs
    ) -> ImageDataSetELPV:
        """
        Performs an example run which (down)loads the ELPV defect image dataset,
        amplifies it with mirroring, rotations, and translations, and then optionally
        shows it .

        :param save_and_open_amplified_dataset: (optional) flag to indicate whether to save
            the example amplified dataset as images to a temporary directory.
            Can take quite some time and space(default = False)
        :param dataset_cfg_kwargs: (optional) keyword arguments passed
            to dataset configuration.
        """
        # Initialize the dataset downloader and download the ELPV dataset from its git repository.
        downloader = DataSetDownloaderELPV()
        downloader.download()  # The dataset is downloaded to %LOCALAPPDATA%/.tmapdd/datasets/dataset-elpv/ (on Windows)

        # Initialize/load the ELPV dataset using the ELPV dataset configuration.
        elpv_dataset_config = DataSetConfigELPV(**dataset_cfg_kwargs)
        dataset = cls(dataset_cfg=elpv_dataset_config)

        # Filter dataset -> use only the polycrystalline solarpanels w/ type 'poly'.
        dataset.filter(query=f"{SchemaLabelsELPV().TYPE.name}=='poly'")

        # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
        # put preprocessing methods in the ImageDataSet class itself later.
        dataset.amplify_data()

        # Specify and create a temporary directory to save our (amplified) image dataset.
        # Then open it in your OS's default filebrowser
        # Warning; can take a long time and quite a lot of storage space depending
        # on the number of samples in the dataset as well as the size of the accompanied images.
        if save_and_open_amplified_dataset:
            new_data_dir = Path(
                DIR_TMP,
                f"tmap_defectdetector_dataset_{datetime.utcnow().strftime('%Y_%m_%d_T%H%M%SZ')}",
            )
            new_data_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_images(new_data_dir)
            open_directory_with_filebrowser(new_data_dir)

    def move_data_categorical_subdirs(
        self,
        label_names: tuple[str, ...] = DataSetConfigELPV.DEFAULT_LABEL_NAMES,
        validation: bool = True,
        validation_frac: float = 0.65,
    ) -> None:
        """Moves images to categories for tensorflow ImageDataGenerator's categorical data processing"""
        lblcat: str = self.dataset_cfg.SCHEMA_FULL.LABEL_CATEGORY.name
        pathcol: str = self.dataset_cfg.SCHEMA_FULL.SAMPLE_PATH.name
        id_col: str = self.dataset_cfg.SCHEMA_FULL.LABEL_SAMPLE_ID.name
        # pathcol_cat: str = self.dataset_cfg.SCHEMA_FULL.SAMPLE_PATH_CATEGORY.name

        if validation:
            log.info(
                f"Splitting dataset into training "
                f"({validation_frac * 100:.2f}%) and test ({(1 - validation_frac) * 100:.2f}%) parts."
            )
            train_set: pd.DataFrame = self.data.sample(frac=validation_frac)
            validate_set: pd.DataFrame = pd.concat([self.data, train_set]).drop_duplicates(
                subset=[id_col], keep=False
            )

            categorized_train = Path(DIR_DATASETS, "dataset-elpv/categories/train")
            categorized_validate = Path(DIR_DATASETS, "dataset-elpv/categories/validate")
            for cur_set, dir_ in zip((train_set, validate_set), (categorized_train, categorized_validate)):

                dir_.mkdir(parents=True, exist_ok=True)
                for lbl in label_names:
                    Path(dir_, lbl).mkdir(parents=True, exist_ok=True)

                for lbl in label_names:
                    curdf = cur_set[cur_set[lblcat].str.match(lbl)]
                    paths = curdf.loc[:, pathcol].tolist()
                    tot = len(paths)
                    for path in tqdm(
                        paths, desc=f"Copying dataset category {lbl!r} to its category dir...", total=tot
                    ):
                        tgtpath = Path(dir_, lbl, Path(path).name)
                        shutil.copy2(Path(path).resolve(), tgtpath)
        else:
            categorized = Path(DIR_DATASETS, "dataset-elpv/categories")
            categorized.mkdir(parents=True, exist_ok=True)
            for lbl in label_names:
                Path(categorized, lbl).mkdir(parents=True, exist_ok=True)

            for lbl in label_names:
                curdf = self.data[self.data[lblcat] == lbl]
                paths = curdf.loc[:, pathcol].tolist()
                tot = len(paths)
                for path in tqdm(
                    paths, desc=f"Copying dataset category {lbl!r} to its category dir...", total=tot
                ):

                    tgtpath = Path(categorized, lbl, Path(path).name)
                    if not tgtpath.exists():
                        shutil.copy2(Path(path).resolve(), tgtpath)
