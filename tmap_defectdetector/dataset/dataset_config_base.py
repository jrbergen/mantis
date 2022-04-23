"""
Contains classes for handling data schemas (column name- and type specifications),
and utility functions to e.g. combine them or build an empty DataFrame out of a schema.
This file also contains baseclasses for dataset configuration.
"""

from __future__ import annotations

import inspect
import itertools
import os
import string
from abc import abstractmethod, ABC
from collections import UserDict
from dataclasses import dataclass
from enum import unique, Enum
from pathlib import Path
from types import DynamicClassAttribute
from typing import Iterable, Optional, TypeAlias, Iterator, ClassVar, Type

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.base import ExtensionDtype


ColType: TypeAlias = ExtensionDtype | np.dtype | type | str


class ColEnum(Enum):
    @DynamicClassAttribute
    def type(self):
        """
        Alias for an enum's 'value' attribute which makes sense in the context of
        column name -> type mappings.
        """
        if self.value is str:
            return pd.StringDtype()
        return self.value


@unique
class ColSpec(ColEnum):
    """
    Used to specify the column names and data types for a generic data
    storage object which has a schema (e.g. a pandas DataFrame).
    """


class ColSpecDefectData(ColSpec):
    """
    Used to specify the column names and data types for a defect dataset/labelset/sampleset.
    A user-defined mapping whose keys are also accessible via the '.' operator; e.g. ColSpec.COLNAME.name
    """

    LABEL_SAMPLE_ID: ColType = str
    """
    Column with equal entries for sample and label data so 
    sample and label data can be merged in the right order.
    """


def combine_colspecs(
    colspec_a: ColType | Type[ColSpec], colspec_b: ColType | Type[ColSpec]
) -> Type[ColSpec]:
    """Combines"""
    if not (issubclass(colspec_a, ColEnum) and issubclass(colspec_b, ColEnum)):  # type: ignore
        raise TypeError(
            f"Can only combine {ColEnum.__name__} objects, "
            f"got {type(colspec_a)!r} and {type(colspec_b)!r}."
        )

    for member in colspec_a:  # type: ignore
        if member.name in colspec_b.__members__ and member.type != (  # type: ignore
            memb_type := getattr(colspec_b, member.name).type  # type: ignore
        ):
            raise TypeError(
                f"Cannot add {ColSpec.__name__} objects with overlapping keys but different type values "
                f"to prevent accidental overwrites:\n Name at issue: {member.name!r} "
                f"with different types {member.type!r} and {memb_type!r}"
            )
    return ColSpec(  # type: ignore
        ColSpec.__name__,
        {at.name: at.type for at in set(itertools.chain(colspec_a, colspec_b))},  # type: ignore
        module=__name__,
    )


def new_df_from_colspec(colspec: ColSpec | Type[ColSpec]) -> DataFrame:
    """Creates an empty DataFrame with the column names and types as specified by for this ColumnSpec instance."""
    return pd.DataFrame({col.name: pd.Series(dtype=col.type) for col in colspec})  # type: ignore


class ColumnSpec(UserDict):
    """
    Used to specify the column names and data types for a dataset/labelset/sampleset.
    A user-defined mapping whose keys are also accessible via the '.' operator; e.g. ColumnSpec.COLNAME.name
    """

    _DEFAULT_ID_COLUMN = "LABEL_SAMPLE_ID"
    _DEFAULT_ID_COLUMN_TYPE = int

    def __init__(self, /, **kwargs):
        self._data_attr_was_created = False
        super().__init__(**kwargs)
        setattr(self, self._DEFAULT_ID_COLUMN, self._DEFAULT_ID_COLUMN_TYPE)

    def spawn_empty_dframe(self) -> DataFrame:
        """Creates an empty DataFrame with the column names and types as specified by for this ColumnSpec instance."""
        return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in self})

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.data)

    @property
    def types(self) -> tuple[ColumnType, ...]:
        return tuple(self.data.values())

    def __setitem__(self, key, value):
        if key == "_data_attr_was_created":
            self.__dict__["_data_attr_was_created"] = value
        elif key == "data" and self._data_attr_was_created:
            raise ValueError("'data' key is reserved...")
        elif key == "data":
            self.__dict__["data"] = value
            self._data_attr_was_created = True
        else:
            if not isinstance(key, str):
                raise TypeError("Column name must be a string")

            # Restrict keys such that they always work in
            if key.startswith("_") or not all(
                s in string.ascii_letters + string.digits + "_" for s in key
            ):
                raise ValueError(
                    f"Invalid key: {key!r}. Can only set alphanumeric names \n"
                    " which may contain underscores but cannot start with one."
                )
            # Make sure that if we pass string as type that it becomes a proper
            # pandas string datatype and doesn't degrade to 'object' datatype.
            if value is str:
                value = pd.StringDtype()

            # We only allow assignment of types
            if not isinstance(value, (ExtensionDtype, np.dtype, str, type)):
                raise TypeError(
                    "ColumnSpec attributes (which represent column names) \n"
                    "can only be assigned types: np.dtype/type/str/pandas.core.base.ExtensionDtype. \n"
                    f"Got: {value}."
                )
            self.data[key] = value

    def __iter__(self) -> Iterator[tuple[str, ColumnType]]:
        yield from self.data.items()

    def __getattr__(self, item):
        if isinstance(item, str) and item in self.data.keys():
            return item
        return self.data[item]

    def __setattr__(self, key, value):
        self.__setitem__(key=key, value=value)

    def __add__(self, other: ColumnSpec) -> ColumnSpec:
        if not isinstance(other, ColumnSpec):
            return NotImplemented
        self.data |= other.data
        return self

    def __sub__(self, other: ColumnSpec) -> ColumnSpec:
        if not isinstance(other, ColumnSpec):
            return NotImplemented
        for key in other.data:
            del self.data[key]
        return self


class DataSetConfig(ABC):
    def __init__(
        self,
        sample_path_s: os.PathLike | Iterable[os.PathLike],
        sample_col_spec: ColSpec,
        label_path: os.PathLike,
        label_col_spec: ColSpec,
        sample_type_desc: str = "sample",
    ):
        """
        Provides ways to load a training dataset's samples and labels into a DataFrame.

        :param sample_path_s: One ore more path-like object(s) pointing to a sample file.
        :param sample_col_spec: ColSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param label_path: One ore more path-like object(s) pointing to corresponding label files.
        :param label_col_spec: ColSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample".
        """
        self.sample_path_s = (
            sample_path_s if isinstance(sample_path_s, Iterable) else [sample_path_s]
        )
        self.label_path = Path(label_path)

        self.sample_col_spec: ColSpec = sample_col_spec
        self.label_col_spec: ColSpec = label_col_spec

        self.sample_type_desc: str = sample_type_desc

    @property
    def column_spec(self) -> ColSpec:
        """
        ColSpec object containing all column names and data types
        for the dataframe containing samples + labels
        """
        return self.sample_col_spec + self.label_col_spec

    @property
    def label_columns(self) -> tuple[str, ...]:
        return self.label_col_spec.columns

    @property
    def sample_columns(self) -> tuple[str, ...]:
        return self.sample_col_spec.columns

    @abstractmethod
    def load_full_dataset(self) -> DataFrame:
        """Merges sample and label DataFrames into one coherent whole."""
        ...

    @abstractmethod
    def load_label_data(
        self,
        label_path_or_paths: Optional[Path | Iterable[Path]],
        label_column_spec: Optional[ColSpec] = None,
    ) -> pd.DataFrame:
        """Provides way to load labels for a specific dataset into DataFrame format."""
        ...

    @abstractmethod
    def load_sample_data(
        self, data_path_or_paths: Path | Iterable[Path], data_colspec: ColSpec
    ) -> DataFrame:
        """Provides way to load samples for a specific dataset into DataFrame format."""
        ...

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({", ".join(f'{k}={v}' for k, v in vars(self).items())})"""


class DefectDetectionDataSet:

    DEFAULT_DATASET_UNITS: ClassVar[str] = " samples"

    def __init__(self, sample_and_label_data: DataFrame):
        """Baseclass for a generic dataset used for training a defect detection model."""
        self._data_original: DataFrame = sample_and_label_data
        self._data_filtered: DataFrame = sample_and_label_data.copy()

    @classmethod
    def from_dataset_configuration(cls, dataset_cfg: DataSetConfig) -> DefectDetectionDataSet:
        """
        Instantiates a DefectDetectionDataSet (derived) object based on a dataset configuration
        (i.e. DataSetConfig object).

        :param dataset_cfg: DataSetConfig derived object specifying
            how/where to read in the data labels and samples.
        """
        return cls(sample_and_label_data=dataset_cfg.load_full_dataset())

    def amplify_data(self):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case the training data comprises images.
        Subclasses should implement this method.
        """
        raise NotImplementedError(
            f"method {inspect.currentframe().f_code.co_name} not implemented for baseclass."
        )

    def reset(self) -> None:
        """Resets dataset to initialization state (before any filters were applied)."""
        self._data_filtered = self._data_original

    @property
    def data(self) -> DataFrame:
        """
        Returns a dataframe w/ _filtered_ samples and labels if any filter has been applied.
        To reset to the unfiltered version, call the `reset()` method.
        Note that this is a read-only property.
        """
        return self._data_filtered

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data})"
