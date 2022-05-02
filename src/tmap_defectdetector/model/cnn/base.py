from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias, Callable, Collection, Optional, ClassVar, NamedTuple

import pandas as pd
import tensorflow as tf
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator
from numpy import ndarray
from pandas import DataFrame

from tmap_defectdetector import DIR_APP, DIR_TMP
from tmap_defectdetector.dataset.base.dataset_configs_base import ImageDataSetConfig
from tmap_defectdetector.dataset.base.datasets_base import DefectDetectionDataSetImages
from tmap_defectdetector.dataset.datasets import ImageDataSetELPV
from tmap_defectdetector.logger import log
from tmap_defectdetector.model import GPU_AVAILABLE
from tmap_defectdetector.model.base import SplitDataSet, DefectDetectionModel

TensorFlowMetric: TypeAlias = Collection[str | Callable[[float, float], any] | tf.keras.metrics.Metric]
TensorFlowLossFunction: TypeAlias = tf.keras.losses.Loss | str
TensorFlowLossFunctionSequence: TypeAlias = tuple[TensorFlowLossFunction, ...] | list[TensorFlowLossFunction]
TensorFlowActivationFunction: TypeAlias = str
TensorFlowActivationFunctionSequence: TypeAlias = (
    tuple[TensorFlowActivationFunction, ...] | list[TensorFlowLossFunction]
)
TensorFlowPredictions: TypeAlias = list[ndarray]
DenseLayerNodeCounts: TypeAlias = tuple[int | None, ...] | list[int | None]
ConvLayerNodeCounts: TypeAlias = tuple[int, ...] | list[int]


class ValidImgType(Enum):
    BINARY: str = "binary"
    GRAYSCALE: str = "grayscale"
    RGB: str = "rgb"


class ConvLayerArgs(NamedTuple):
    nodes: int
    n_colors: int
    activation: TensorFlowActivationFunction


class DenseLayerArgs(NamedTuple):
    nodes: int
    activation: TensorFlowActivationFunction


@dataclass
class CNNModelConfig:

    n_epochs: int = 1024 if GPU_AVAILABLE else 64
    training_frac: float = 0.65
    layer_spec: tuple[Conv2D | MaxPooling2D | Dense, ...] = (
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(4, activation="sigmoid"),
    )
    n_nodes_conv_layers: ConvLayerNodeCounts = (128,)
    activation_funcs_conv: TensorFlowActivationFunctionSequence = ("relu",)
    dense_layer_spec: DenseLayerNodeCounts = (DenseLayerArgs(64, "relu"), DenseLayerArgs(10, None))
    activation_funcs_dense: TensorFlowActivationFunctionSequence = ("relu", None)
    activation_func_id: str = "relu"
    optimizer: str = "adam"
    metrics: TensorFlowMetric = ("accuracy",)
    loss_function: TensorFlowLossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _img_type: ValidImgType = ValidImgType.BINARY

    @property
    def n_colors(self) -> int:
        match self._img_type:
            case ValidImgType.BINARY | ValidImgType.GRAYSCALE:
                return 1
            case ValidImgType.RGB:
                return 3
            case _:
                raise ValueError(
                    f"Unrecognized _img_type type; "
                    f"expected {type(ValidImgType).__name__}, got {type(self._img_type)}."
                )

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


class ConvNetModel(Model):
    def __init__(self, modelcfg: CNNModelConfig):
        super().__init__()
        mcfg = modelcfg
        if mcfg.n_nodes_conv_layers:
            self.conv_layers: list[Conv2D] = [
                Conv2D(n_nodes, mcfg.n_colors, activation=mcfg.activation_funcs_conv[ii])
                for ii, n_nodes in enumerate(mcfg.n_nodes_conv_layers)
            ]

        self.dense_layers: list[Dense] = [
            Dense(n_nodes, activation=mcfg.activation_funcs_dense[ii])
            for ii, n_nodes in enumerate(mcfg.n_nodes_dense_layers)
        ]
        self.flatten = Flatten()

    def call(self, inputs, *args, **kwargs):

        x = self.conv_layers[0](inputs)

        if len(inputs) > 1:
            for conv_layer in self.conv_layers[1:]:
                x = conv_layer(x)
        x = self.flatten(x)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return x

    @classmethod
    def from_modelconfig(cls, modelcfg: CNNModelConfig):
        return cls(modelcfg=modelcfg)


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
