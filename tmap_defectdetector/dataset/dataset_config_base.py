"""
Contains classes for handling data schemas (column name- and type specifications),
and utility functions to e.g. combine them or build an empty DataFrame out of a schema.
This file also contains baseclasses for dataset configuration.
"""

from __future__ import annotations

import inspect
import itertools
import os
import sys
from abc import abstractmethod, ABC
from functools import cached_property


from pathlib import Path
from types import DynamicClassAttribute
from typing import Iterable, Optional, TypeAlias, ClassVar, Type, Collection, TypeVar, TYPE_CHECKING

from aenum import extend_enum

if TYPE_CHECKING:
    from enum import (
        unique,
        Enum,
    )  # MyPy doesn't support advancedenum typechecks, so for typechecking use std version.
else:
    from aenum import unique, Enum
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.base import ExtensionDtype


ColType: TypeAlias = ExtensionDtype | np.dtype | type | str


class ColEnum(Enum):
    """Enum storing column name(s) + data type(s) for a (DataFrame) schema."""

    # Allows docstrings for Enum members using the `aenum` library (same author as stdlib's Enum module btw).
    _init_ = "value __doc__"

    @DynamicClassAttribute
    def type(self):
        """
        Alias for an enum's 'value' attribute which makes
        sense in the context of column name -> type mappings.
        """
        if self.value is str:
            return pd.StringDtype()
        return self.value


@unique
class ColSchema(ColEnum):
    """
    Used to specify the schema (column names and data types) for a generic data
    storage object which has a schema (e.g. a pandas DataFrame).
    """


ColSchemaType: TypeAlias = ColSchema | Type[ColSchema]


ColSchemaDefectData = extend_enum(
    enumeration=ColSchema,
    name="ColSchemaDefectData",
    _init_="value __doc__",
    __doc__="Used to the schema (column names and data types) for a defect dataset/labelset/sampleset which has a schema (e.g. a pandas DataFrame)",
)
# value=str,
# __doc__=,
# )
# )

#
# )


print(f"Temporary breakpoint in {__name__}")
#
# LABEL_SAMPLE_ID: ColType = str
# """
# Column with equal entries for sample and label data so
# sample and label data can be merged in the right order.
# """
#


def merge_colschemas(*colschemas: Collection[ColSchemaType]) -> ColSchemaType:
    """
    Merges multiple ColSchema schemas into one.

    :raises TypeError:
        - If ColSchemas contain overlapping names with different values/types assigned.
        - If an objects is passed in the 'colspecs' argument which is not a subclass of ColEnum.
        - If an empty collection is passed to the 'colspecs' argument.
    """
    # Make our Collection instance an iterable and a list if it isn't one already
    if not isinstance(colschemas, Iterable) or not isinstance(colschemas, list):
        colschemas = list(colschemas)

    # Check that we won't reach the recursion limit
    if len(colschemas) > (reclim := sys.getrecursionlimit()):
        raise ValueError(
            f"Number of column schemas may not exceed the recursion limit (={reclim})."
        )

    # Check that the next schema we're trying to merge is of the proper type.
    for colspec in colschemas:
        if not issubclass(colspec, ColEnum):  # type: ignore
            raise TypeError(
                f"Can only combine {ColEnum.__name__} objects; got a {type(colspec)!r} type."
            )

    # Raise an error if the supplied Collection of schemas is empty.
    if not colschemas:
        raise TypeError(f"Tried to merge empty {ColSchema.__name__} collection...")

    # Return the schema unaltered if only a single one was passed.
    if len(colschemas) == 1:
        return colschemas[0]

    # Get the first to schema's to merge them, and pack the remaining ones in a tuple called 'remaining'
    # for the next recursion.
    colschema_a, colschema_a, *remaining = colschemas

    # Check that we don't have duplicate keys
    for member in colschema_a:  # type: ignore
        if member.name in colschema_a.__members__ and member.type != (  # type: ignore
            memb_type := getattr(colschema_a, member.name).type  # type: ignore
        ):
            raise TypeError(
                f"Cannot add {ColSchema.__name__} objects with overlapping keys but different type values "
                f"to prevent accidental overwrites:\n Name at issue: {member.name!r} "
                f"with different types {member.type!r} and {memb_type!r}"
            )

    # Merge the first two schema's into a new schema
    new_colschema = ColSchema(  # type: ignore
        ColSchema.__name__,
        {at.name: at.type for at in set(itertools.chain(colschema_a, colschema_a))},  # type: ignore
        module=__name__,
    )

    if remaining:  # If unmerged schemas remain, merge them with the newly created one.
        return merge_colschemas(*[new_colschema, *remaining])
    else:  # Otherwise return the merged schemas
        return new_colschema


def new_df_from_colspec(colspec: ColSchemaType) -> DataFrame:
    """Creates an empty DataFrame with the column names and types as specified by for this ColumnSpec instance."""
    return pd.DataFrame({col.name: pd.Series(dtype=col.type) for col in colspec})  # type: ignore


# Make sure _ColSchemaDefectData subclasses are recognized by static type checker as valid by binding a typevar to it.
ColSchemaDefectDataType = TypeVar("ColSchemaDefectDataType", bound=ColSchemaDefectData)


class DataSetConfig(ABC):

    _CACHED_PROPERTIES: tuple[str, ...] = ("full_dataset", "label_data", "sample_data")
    """Properties which need to be (re)loaded if an attribute changes change."""

    def __init__(
        self,
        sample_path_s: os.PathLike | Iterable[os.PathLike],
        sample_col_schema: ColSchemaDefectDataType,
        label_path: os.PathLike,
        label_col_schema: ColSchemaDefectDataType,
        sample_type_desc: str = "sample",
    ):
        """
        Provides ways to load a training dataset's samples and labels into a DataFrame.

        :param sample_path_s: One ore more path-like object(s)
            pointing to a sample file.
        :param sample_col_schema: ColSchemaDefectData object representing
            column schema (column names + dtypes) for the DataFrame to
            be created which will contain the samples.
        :param label_path: One ore more path-like object(s)
            pointing to corresponding label files.
        :param label_col_schema: ColSchemaDefectData object representing
            column schema (column names + dtypes) for the DataFrame to
            be created which will contain the samples.
        :param sample_type_desc: (optional) description for this kind
            of sample (default = "sample").
        """
        self.sample_path_s = (
            sample_path_s if isinstance(sample_path_s, Iterable) else [sample_path_s]
        )
        self.label_path = Path(label_path)

        self.sample_col_schema: ColSchemaType = sample_col_schema
        self.label_col_schema: ColSchemaType = label_col_schema

        self.sample_type_desc: str = sample_type_desc

    @cached_property
    @abstractmethod
    def full_dataset(self) -> DataFrame:
        """Merges sample and label DataFrames into one coherent whole."""
        ...

    @cached_property
    @abstractmethod
    def label_data(self) -> pd.DataFrame:
        """
        Provides way to load labels for a specific dataset into
        a DataFrame using the label path passed to the constructor.
        """
        ...

    @cached_property
    @abstractmethod
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into
        a DataFrame using the label path passed to the constructor.
        """
        ...

    @property
    def column_schema(self) -> ColSchema | Type[ColSchema]:
        """
        ColSchema object containing all column names and data types
        for the dataframe containing samples + labels.
        """
        return merge_colschemas(*[self.sample_col_schema, self.label_col_schema])

    @property
    def label_colnames(self) -> tuple[str, ...]:
        """Retrieve column names for the label data."""
        return tuple(self.label_col_schema.keys())  # type: ignore

    @property
    def sample_colnames(self) -> tuple[str, ...]:
        """Retrieve column names for the sample data."""
        return tuple(self.sample_col_schema.keys())  # type: ignore

    @property
    def label_coltypes(self) -> tuple[str, ...]:
        """Retrieve column datatypes for the label data."""
        return tuple(self.label_col_schema.values())  # type: ignore

    @property
    def sample_coltypes(self) -> tuple[str, ...]:
        """Retrieve column datatypes for the sample data."""
        return tuple(self.sample_col_schema.values())  # type: ignore

    def __setattr__(self, key, value):
        # Override setattr such that cached properties are reset if the relevant attributes have changed
        self.__dict__[key] = value
        for cached_attrname in type(self)._CACHED_PROPERTIES:
            if cached_attrname in self.__dict__:
                self.__dict__.pop(cached_attrname, None)

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({", ".join(f'{k}={v}' for k, v in vars(self).items())})"""


class DefectDetectionDataSet:

    DEFAULT_DATASET_UNITS: ClassVar[str] = " samples"

    def __init__(self, sample_and_label_data: DataFrame):
        """Baseclass for a generic dataset used for training a defect detection model."""
        self._data_original: DataFrame = sample_and_label_data
        self._data_filtered: DataFrame = sample_and_label_data.copy()

    @classmethod
    def from_dataset_config(cls, dataset_cfg: DataSetConfig) -> DefectDetectionDataSet:
        """
        Instantiates a DefectDetectionDataSet (derived) object based on a dataset configuration
        (i.e. DataSetConfig object).

        :param dataset_cfg: DataSetConfig derived object specifying
            how/where to read in the data labels and samples.
        """
        return cls(sample_and_label_data=dataset_cfg.full_dataset())

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
