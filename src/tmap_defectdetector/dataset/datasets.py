"""Contains baseclass & concrete implementations for DataSets and more specialized types thereof."""
from __future__ import annotations
from typing import TypeAlias, Union

from numpy import ndarray

from src.tmap_defectdetector.dataset.base.datasets_base import DefectDetectionDataSetImages
from src.tmap_defectdetector.dataset.dataset_configs import DataSetConfigELPV


ImageCollection: TypeAlias = list[ndarray] | tuple[ndarray, ...]
Translation: TypeAlias = Union[tuple[float, float] | tuple[int, int]]


class ImageDataSetELPV(DefectDetectionDataSetImages):
    def __init__(self, dataset_cfg: DataSetConfigELPV):
        """
        ImageDataSet specific for the ELPV photovoltaic cell defectg dataset.
        (See https://github.com/zae-bayern/elpv-dataset for original dataset).
        """
        super().__init__(dataset_cfg=dataset_cfg)
