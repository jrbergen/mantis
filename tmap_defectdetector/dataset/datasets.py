"""Contains baseclass & concrete implementations for DataSets and more specialized types thereof."""
from __future__ import annotations


import functools
import operator
import inspect
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import cv2 as cv
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig, ImageDatasetConfig
from tmap_defectdetector.dataset.base.datasets_base import DefectDetectionDataSetImages
from tmap_defectdetector.dataset.dataset_configs import DataSetConfigELPV
from tmap_defectdetector.logger import log


ImageCollection: TypeAlias = list[ndarray] | tuple[ndarray, ...]
Translation: TypeAlias = tuple[float, float] | tuple[int, int]


class ImageDataSetELPV(DefectDetectionDataSetImages):
    def __init__(self, dataset_config: DataSetConfigELPV):
        """
        ImageDataSet specific for the ELPV photovoltaic cell defectg dataset.
        (See https://github.com/zae-bayern/elpv-dataset for original dataset).

        """
        super().__init__(dataset_config=dataset_config)
