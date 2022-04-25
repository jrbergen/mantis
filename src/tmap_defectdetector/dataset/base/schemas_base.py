"""
Contains baseclasses with which to build schemas (i.e. column name and type mappings) for new datasets.
"""

from __future__ import annotations

import string
import warnings
from dataclasses import dataclass
from typing import TypeAlias, Mapping, Iterator, TypeVar, Type

import numpy as np
import pandas as pd
from pandas import DataFrame
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
        Supports addition of documentation strings / descriptions to the entries.

        :param name: ColName or string instance representing column name.
        :param type_: column type for this schema entry (ExtensionDtype | np.dtype | type | str)
        :param docstring: (optional) doctsring describing this schema entry and/or its purpose.
        """
        self._name: ColName = ColName("")
        self._type: ColType = ""
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
        return str(self.docstring) if self.docstring else str(type(self).__doc__)

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


SchemaDerived = TypeVar("SchemaDerived")
"""Type variable to indicate some variable schema derived type."""


@dataclass
class Schema:
    def __init__(self, colentries: Mapping[str, SchemaEntry]):
        """
        Used to specify the schema (column names and data types) for a generic data
        storage object which has a schema (e.g. a pandas DataFrame).

        :param colentries: mapping/dictionary of attribute names as keys
            with associated SchemaEntry objects as values.
        """
        self.add_entries(colentries)

    def add_entries(self, colentries: Mapping[str, SchemaEntry]) -> None:
        """
        Adds SchemaEntry entries to this schema
        (duplicate column names or existing attribute names are not allowed).

        :param colentries: a mapping of attribute names to SchemaEntry objects.
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

    def __eq__(self, other: Schema | object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return all(other_entry in self.schema_entries for other_entry in other.schema_entries)

    def __or__(self, other: Schema) -> Schema:
        """
        Creates union of two ColSchema instances (union is equivalent to addition for this class).

        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical column names.
        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical attribute names.
        """
        return self.__add__(other)

    def __add__(self, other: Schema) -> Schema:
        """
        Adds/merges two ColSchema instances (addition is equivalent to union for this class).

        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical column names.
        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical attribute names.
        """
        return Schema(colentries=self._merged_schemas_to_attrdict(other))

    def _merged_schemas_to_attrdict(self, other: Schema):
        """
        Adds/merges two ColSchema instances (addition is equivalent to union for this class).

        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical column names.
        :raises ValueError: if non-identical SchemaEntry members of the schemas have identical attribute names.
        """
        if not isinstance(other, Schema):
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
        return schema_dict_self

    def combine_with_schema(
        self, other_schema: Schema, target_schema_type: Type[SchemaDerived]
    ) -> SchemaDerived:
        new_schema_dict: dict[str, SchemaEntry] = self._merged_schemas_to_attrdict(other_schema)
        if not issubclass(target_schema_type, Schema):
            raise TypeError(f"Target schema must be a subclass of {target_schema_type.__name__}.")
        # !NOTE: This 'SchemaDerived' typevar approach seems wrong; find how this should be done.
        target_schema_type: Type[Schema]  # type: ignore
        return target_schema_type(colentries=new_schema_dict)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"{Schema.__name__}("
            + ", ".join(f"{k}={v}" for k, v in vars(self).items() if isinstance(v, SchemaEntry))
            + ")"
        )


@dataclass(repr=False)
class SchemaDefectData(Schema):
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
SchemaDefectDataType = TypeVar("SchemaDefectDataType", bound=SchemaDefectData)


@dataclass(repr=False)
class SchemaLabels(SchemaDefectData):
    """Specifies schema for label data."""

    pass


@dataclass(repr=False)
class SchemaSamples(SchemaDefectData):
    """
    Used to specify the schema (column names and data types) for sample data.
    """

    SAMPLE_PATH: SchemaEntry = SchemaEntry(
        "SAMPLE_PATH",
        str,
        docstring="Full path of the file which stores the sample data, as string.",
    )
    SAMPLE: SchemaEntry = SchemaEntry(
        "SAMPLE",
        object,  # sould be of multiple types; use 'object' type for basecalass to be safe.
        docstring="An entry representing a data sample",
    )


@dataclass(repr=False)
class SchemaFull(SchemaLabels, SchemaSamples):
    """Specifies schema for a dataset's label _and_ sample data."""

    pass


@dataclass(repr=False)
class SchemaSamplesImageData(SchemaSamples):
    """
    Used to specify the schema (column names and data types) for a
    defect dataset/labelset/sampleset pertaining image samples specifically.
    """

    MIRROR_AXIS: SchemaEntry = SchemaEntry(
        "MIRROR_AXIS",
        np.uint8,
        docstring="Number representing mirrored state w.r.t original image.",
    )
    ROT_DEG: SchemaEntry = SchemaEntry(
        "ROT_DEG", np.int16, docstring="Rotation of image in degrees w.r.t. original image."
    )
    TRANSL_X: SchemaEntry = SchemaEntry(
        "TRANSL_X", np.int16, docstring="Translation of Y-pixel positions w.r.t. original image."
    )
    TRANSL_Y: SchemaEntry = SchemaEntry(
        "TRANSL_Y", np.int16, docstring="Translation of X-pixel positions w.r.t. original image."
    )
    SAMPLE: SchemaEntry = SchemaEntry(
        "SAMPLE",
        object,  # should be an np.ndarray but use 'object' as type for compatibility with pandas DataFrame.
        docstring="A numpy array representing the sample image.",
    )


@dataclass(repr=False)
class SchemaFullImageData(SchemaLabels, SchemaSamplesImageData):
    """Specifies schema for an image dataset's label _and_ sample data."""

    pass
