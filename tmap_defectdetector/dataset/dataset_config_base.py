"""
Contains classes for handling data schemas (column name- and type specifications),
and utility functions to e.g. combine them or build an empty DataFrame out of a schema.
This file also contains baseclasses for dataset configuration.
"""

from __future__ import annotations

import inspect
import os
import string
import warnings
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property


from pathlib import Path
from typing import (
    Iterable,
    TypeAlias,
    ClassVar,
    TypeVar,
    Iterator,
    cast,
    Mapping,
    Collection,
)

import numpy as np
from pandas import DataFrame
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype


ColType: TypeAlias = ExtensionDtype | np.dtype | type | str


class ColName(str):
    """
    Subclass of str used for column names,
    restricted to contain only ascii letters + digits + underscore.
    """

    _ALLOWED_CHARS: str = string.ascii_letters + string.digits + "_"

    def __new__(cls, s: str) -> ColName:

        if any(substr not in cls._ALLOWED_CHARS for substr in s):
            raise ValueError(
                f"Column name may only consist of ASCII letters, digits, and underscores, but got {s!r}."
            )
        return str.__new__(cls, s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class SchemaEntry:

    __slots__ = ("_name", "_type", "docstring")

    def __init__(self, name: ColName | str, type_: ColType, docstring: str = ""):
        """
        Dataclass storing a single column name + data type combination (i.e. a single schema entry).

        :param name: ColName or string instance representing column name.
        :param type_: column type for this schema entry (ExtensionDtype | np.dtype | type | str)
        :param docstring: (optional) doctsring describing this schema entry and/or its purpose.
        """
        self._name, self._type = "", ""
        self.name: ColName = name if type(name) is ColName else ColName(str(name))
        self.type: ColType = type_
        self.docstring: str = docstring

    @property
    def name(self) -> ColName:
        return self._name

    @name.setter
    def name(self, name: ColName) -> None:
        if not isinstance(name, (str, ColName)):
            raise TypeError(f"Name must be of type str or {ColName.__name__}.")
        elif not type(name) is ColName:
            self._name = ColName(name)
        else:
            self._name = name

    @property
    def type(self) -> ColType:
        return self._type

    @type.setter
    def type(self, type_: ColType):
        if type_ is np.ndarray:
            warnings.warn(
                "\nTried to use numpy.ndarray as an explicit data type for a schema entry. \n"
                f"Reverting to {object.__name__!r} instead for compatibility with pandas DataFrames.",
                UserWarning,
            )
            type_ = object
        if not isinstance(type_, (ExtensionDtype, np.dtype, type, str)):
            raise TypeError(
                f"Type attribute must be of type: ExtensionDtype | np.dtype | type | str."
            )
        self._type = type_

    @property
    def __doc__(self) -> str:
        return self.docstring if self.docstring else type(self).__doc__

    def __eq__(self, other: SchemaEntry | object) -> bool:
        if isinstance(other, SchemaEntry):
            return self.name == other.name and self.type == other.type
        return NotImplemented

    def __repr__(self) -> str:
        if hasattr(self._type, "__name__"):
            typedesc = self._type.__name__
        elif isinstance(self._type, str):
            typedesc = f"{self._type!r}"
        else:
            typedesc = self._type.__repr__()
        return f"{type(self).__name__}(name={self._name!r}, type={typedesc}, docstring={repr(self.docstring)})"


@dataclass
class ColSchema:
    def __init__(self, colentries: Mapping[str, SchemaEntry]):
        """
        Used to specify the schema (column names and data types) for a generic data
        storage object which has a schema (e.g. a pandas DataFrame).

        :param colentries: mapping/dictionary of attribute names as keys
            with associated SchemaEntry objects as values.
        """
        self.add_entries(**colentries)

    def add_entries(self, **colentries) -> None:
        """
        Adds SchemaEntry entries to this schema
        (duplicate column names or existing attribute names are not allowed).

        :param **colentries: keyword arguments with SchemaEntry objects as values
        """
        encountered_attrnames = set()
        encountered_colnames = set()
        for attrname, entry in colentries.items():
            if isinstance(entry, SchemaEntry):
                if attrname in encountered_attrnames:
                    raise ValueError(
                        f"Tried to set ColSchema with duplicate attribute name: {attrname!r}"
                    )
                if entry.name in encountered_colnames:
                    raise ValueError(
                        f"Tried to set ColSchema with duplicate column name: {entry.name!r}"
                    )
                setattr(self, attrname, entry)
                encountered_colnames.add(entry.name)
                encountered_attrnames.add(attrname)
            else:
                warnings.warn(
                    f"Tried to add column entry of other type than {SchemaEntry.__name__}: ignored.",
                    UserWarning,
                )

    @property
    def schema_entries(self) -> Iterator[SchemaEntry]:
        """Yields all SchemaEntry object contained by this ColSchema instance."""
        yield from self

    @property
    def columns(self) -> Iterator[ColName]:
        """Yields column names for this schema."""
        for colentry in self:
            yield colentry.name

    @property
    def types(self) -> Iterator[ColType]:
        """Yields column types for this schema."""
        for colentry in self:
            yield colentry.type

    def values(self) -> Iterator[ColType]:
        """
        Yields column types for this schema.
        Alias for 'types' property (but as function instead of property to be
        consistent with common use of 'values()')
        """
        yield from self.types

    def items(self) -> Iterator[tuple[ColName, ColType]]:
        """Yields 2-tuples of all contained column names and types."""
        yield from zip(self.columns, self.types)

    def to_dict(self) -> dict[ColName, ColType]:
        """Returns dictionary of column names and corresponding column types."""
        return {entry.name: entry.type for entry in self.schema_entries}

    def to_new_dframe(self) -> DataFrame:
        """
        Creates an empty DataFrame with the column names and types
        as specified by this ColSchema instance.
        """
        return pd.DataFrame({col.name: pd.Series(dtype=col.type) for col in self.schema_entries})

    def __getitem__(self, colname: ColName | str) -> SchemaEntry:
        """
        Tries to return a SchemaEntry with the provided column name.

        :param colname: ColName instance
        :raise KeyError: if no SchemaEntry exists with column name $colname in this instance.

        """
        for colentry in self:
            if colentry.name == colname:
                return colentry
        raise KeyError(f"No {SchemaEntry.__name__} found with column name {colname!r}.")

    def __iter__(self) -> Iterator[SchemaEntry]:
        """Yields SchemaEntry instances defined in this ColSchema instance."""
        for attrname in dir(self):
            col_entry: SchemaEntry
            if not attrname.startswith("_") and isinstance(
                (col_entry := getattr(self, attrname)), SchemaEntry
            ):
                yield col_entry

    def __len__(self) -> int:
        """Returns number of SchemaEntry instances defined by this ColSchema instance."""
        return len(list(self.schema_entries))

    def __hash__(self) -> int:
        return hash(f"{key}={str(value)}_{hash(value)}" for key, value in self.items())

    def __eq__(self, other: ColSchema | object) -> bool:
        if not isinstance(other, ColSchema):
            return NotImplemented
        return all(other_entry in self.schema_entries for other_entry in other.schema_entries)

    def __or__(self, other: ColSchema) -> ColSchema:
        """
        Creates union of two ColSchema instances (union is equivalent to addition for this class).

        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical column names.
        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical attribute names.
        """
        return self.__add__(other)

    def __add__(self, other: ColSchema) -> ColSchema:
        """
        Adds two ColSchema instances (addition is equivalent to union for this class).

        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical column names.
        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical attribute names.
        """
        if not isinstance(other, ColSchema):
            return NotImplemented
        schema_dict_self = {k: v for k, v in vars(self).items() if isinstance(v, SchemaEntry)}
        schema_dict_other = {k: v for k, v in vars(other).items() if isinstance(v, SchemaEntry)}
        for attr_self, entry_self in schema_dict_self.items():
            for attr_other, entry_other in schema_dict_other.items():
                if (
                    attr_self == attr_other or entry_self.name == entry_other.name
                ) and entry_self.type != entry_other.type:
                    raise ValueError(
                        f"Cannot combine/merge/add schemas containing duplicate keys/attribute names but "
                        f"different values, to prevent accidental overwrites."
                    )
        schema_dict_self.update(**schema_dict_other)
        return ColSchema(**schema_dict_self)  # type: ignore

    def __repr__(self) -> str:
        return f"{ColSchema.__name__}({', '.join(f'{k}={v}' for k, v in vars(self).items() if isinstance(v, SchemaEntry))})"


@dataclass(repr=False)
class ColSchemaDefectData(ColSchema):
    """
    Used to specify the schema (column names and data types) for a
    defect detection dataset (e.g. a pandas DataFrame).
    """

    LABEL_SAMPLE_ID: SchemaEntry = SchemaEntry(
        name="LABEL_SAMPLE_ID",
        type_=str,
        docstring=(
            "Column with equal entries for sample and label data, "
            "such that sample and label data can be merged in "
            "the right order."
        ),
    )


# Make sure _ColSchemaDefectData subclasses are recognized by static type checker as valid by binding a typevar to it.
ColSchemaDefectDataType = TypeVar("ColSchemaDefectDataType", bound=ColSchemaDefectData)


class DataSetConfig(ABC):

    _CACHED_PROPERTIES: tuple[str, ...] = ("full_dataset", "label_data", "sample_data")
    """Properties which need to be (re)loaded if an attribute changes change."""

    def __init__(
        self,
        sample_dirs: os.PathLike | Collection[os.PathLike],
        sample_col_schema: ColSchemaDefectDataType,
        label_path: os.PathLike,
        label_col_schema: ColSchemaDefectDataType,
        sample_type_desc: str = "sample",
    ):
        """
        Provides ways to load a training dataset's samples and labels into a DataFrame.

        :param sample_dirs: :param sample_dirs: One ore more path-like object(s)
            pointing to a directory with sample files.
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
        self.sample_dirs: list[Path] = (
            list(Path(d) for d in sample_dirs)
            if isinstance(sample_dirs, Iterable)
            else [Path(sample_dirs)]
        )
        self.label_path = Path(label_path)

        self.sample_col_schema: ColSchema = sample_col_schema
        self.label_col_schema: ColSchema = label_col_schema

        self.sample_type_desc: str = sample_type_desc

    @classmethod
    def file_is_sample(cls, file: Path) -> bool:
        """
        Function used to check if a file can be identified as / considered a sample for this dataset.
        Defaults to just checking whether a Path indeed points to a file. Override in extending classes
        to perform more comprehensive checks.

        :param file: Path to (potential) sample file.
        """
        return file.is_file()

    @cached_property
    @abstractmethod
    def full_dataset(self) -> DataFrame:
        """
        Merges sample and label DataFrames into one coherent whole.
        _Must_ be implemented by subclasses.
        """
        ...

    @cached_property
    @abstractmethod
    def label_data(self) -> pd.DataFrame:
        """
        Provides way to load labels for a specific dataset into
        a DataFrame using the label path passed to the constructor.
        _Must_ be implemented by subclasses.
        """
        ...

    @cached_property
    @abstractmethod
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into
        a DataFrame using the label path passed to the constructor.
        _Must_ be implemented by subclasses.
        """
        ...

    @property
    def column_schema(self) -> ColSchema:
        """
        ColSchema object containing all column names and data types
        for the dataframe containing samples + labels.
        """
        return self.label_col_schema | self.sample_col_schema

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
        return cls(sample_and_label_data=dataset_cfg.full_dataset)

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
