from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, Callable, Collection, Optional, ClassVar, NamedTuple

import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame

from tmap_defectdetector.dataset.base.dataset_configs_base import ImageDataSetConfig
from tmap_defectdetector.dataset.base.datasets_base import DefectDetectionDataSetImages
from tmap_defectdetector.dataset.datasets import ImageDataSetELPV
from tmap_defectdetector.logger import log
from tmap_defectdetector.model import GPU_AVAILABLE
from tmap_defectdetector.model.base import SplitDataSet, DefectDetectionModel

TensorFlowMetric: TypeAlias = Collection[str | Callable[[float, float], any] | tf.keras.metrics.Metric]
TensorFlowLossFunction: TypeAlias = tf.keras.losses.Loss | str
TensorFlowPredictions: TypeAlias = list[ndarray]


class ValidImgType(Enum):
    BINARY: str = "binary"
    GRAYSCALE: str = "grayscale"
    RGB: str = "rgb"


@dataclass
class CNNModelConfig:

    n_epochs: int = 1024 if GPU_AVAILABLE else 64
    training_frac: float = 0.65
    n_nodes_layer2: int = 128
    n_nodes_layer3: int = 10
    activation_func_id: str = "relu"
    optimizer: str = "adam"
    metrics: TensorFlowMetric = ("accuracy",)
    loss_function: TensorFlowLossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _img_type: ValidImgType = ValidImgType.BINARY

    @property
    def img_type(self) -> ValidImgType:
        return self._img_type

    @img_type.setter
    def img_type(self, img_type: ValidImgType):
        if not isinstance(img_type, ValidImgType):
            raise ValueError(
                f"Invalid image type for parameter 'img_type'. Valid: {', '.join(x.name for x in ValidImgType)}"
            )
        self._img_type = img_type


class CNNModel(DefectDetectionModel):

    FLOAT_COMP_TOL: ClassVar[float] = 1e-8
    """Float comparison error tolerance."""

    def __init__(
        self,
        dataset: DefectDetectionDataSetImages,
        training_frac: float = 0.65,
        model_config: CNNModelConfig = CNNModelConfig(),
    ):
        """
        Convolutional Neural Network model for image-based defect detection.

        :param dataset: DefectDetectionDataSetImages (derived) instance containing
            labeled samples for training.
        :param training_frac: (optional) fraction of data to use
            for training vs. test/validation data (Default = .65).
        :param model_config: (optional) container storing configuration for the CNN Model
            (Default = CNNModelConfig instance).
        """
        self.training_frac: float = training_frac

        self.dataset: DefectDetectionDataSetImages = dataset

        self.training_data: DataFrame = pd.DataFrame()
        self.test_data: DataFrame = pd.DataFrame()

        self.cfg: ImageDataSetConfig = self.dataset.dataset_cfg
        self.id_col: str = self.cfg.SCHEMA_LABELS.LABEL_SAMPLE_ID.name
        self.model_config: CNNModelConfig = model_config

        self._grayscale: bool = False

    @property
    def data(self) -> pd.DataFrame:
        """Returns the full dataset (samples + labels) as pandas DataFrame."""
        return self.dataset.data

    def init_training_and_test_sets(self):
        self.training_data, self.test_data = self._get_training_and_test_sets()
        if not (len(self.training_data) + len(self.test_data)) == len(self.data):
            raise RuntimeError("Splitting of data into test/training sets resulted in information loss.")

    def _get_training_and_test_sets(self, frac: Optional[float] = None) -> SplitDataSet:
        """
        Splits the dataset into training and test/validation parts according to specified fraction.

        :param frac: fraction of data to use for training (Default = None -> self.training_frac attribute)
        """
        if frac is None:
            frac = self.training_frac
        log.info(f"Splitting dataset into training ({frac*100:.2f}%) and test ({(1-frac)*100:.2f}%) parts.")
        return SplitDataSet(
            train=(img_train := self.data.sample(frac=frac)),
            test=pd.concat([self.data, img_train]).drop_duplicates(subset=[self.id_col], keep=False),
        )

    @classmethod
    def from_dataset(cls, dataset: ImageDataSetELPV) -> CNNModel:
        raise NotImplementedError(f"{cls.__name__}'s factory method not yet implemented.")

    def amplify_data(self, mirror_axes: tuple[int, int, int, int] = (1, 2, 3, 4)) -> None:
        """Amplifies data for current selected dataset"""
        self.dataset.amplify_data(mirror_axes=mirror_axes)

    def filter(self, query: str) -> None:
        self.dataset.filter(query=query)


class ImageShape(NamedTuple):
    rows: int
    cols: int


class ModelEvaluation(NamedTuple):
    loss: float
    accuracy: float
    probability_model: tf.keras.Sequential
    predictions: TensorFlowPredictions
