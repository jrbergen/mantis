from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple, TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from mantis.dataset.datasets import ImageDataSetELPV


class DefectDetectionModel:
    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset: ImageDataSetELPV) -> DefectDetectionModel:
        return NotImplemented


class SplitDataSet(NamedTuple):
    train: DataFrame
    test: DataFrame
