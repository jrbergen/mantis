from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

from pandas import DataFrame

from tmap_defectdetector.dataset.datasets import ImageDataSetELPV


class DefectDetectionModel:
    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset: ImageDataSetELPV) -> DefectDetectionModel:
        return NotImplemented


class SplitDataSet(NamedTuple):
    train: DataFrame
    test: DataFrame
