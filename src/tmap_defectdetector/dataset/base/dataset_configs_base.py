"""
Contains classes for handling data schemas (column name- and type specifications),
and utility functions to e.g. combine them or build an empty DataFrame out of a schema.
This file also contains baseclasses for dataset configuration.
"""
from __future__ import annotations

import os
from abc import abstractmethod, ABC
from functools import cached_property


from pathlib import Path
from typing import (
    Iterable,
    Iterator,
    Collection,
    ClassVar,
    Optional,
    Callable,
    Type,
    TYPE_CHECKING,
)

from pandas import DataFrame
import pandas as pd


from tmap_defectdetector.dataset.base.schemas_base import (
    Schema,
    SchemaSamplesImageData,
    SchemaLabels,
    SchemaFullImageData,
    SchemaSamples,
    SchemaFull,
)
from tmap_defectdetector.logger import log

if TYPE_CHECKING:
    from tmap_defectdetector.dataset.base.downloaders_base import DataSetDownloader
    from tmap_defectdetector.controllers.base import TUIControllerDataSet
    from tmap_defectdetector.dataset.base.datasets_base import (
        DefectDetectionDataSetImages,
        DefectDetectionDataSet,
    )

FALLBACK_LABEL_CATEGORY: str = "UNDEFINED"


class DataSetConfig(ABC):

    _CACHED_PROPERTIES: tuple[str, ...] = ("full_dataset", "label_data", "sample_data")
    """Properties which need to be (re)loaded if an attribute changes change."""

    SCHEMA_LABELS: ClassVar[SchemaLabels] = SchemaLabels()
    SCHEMA_SAMPLES: ClassVar[SchemaSamples] = SchemaSamples()

    def __init__(
        self,
        name: str,
        dataset_cls: Optional[Type[DefectDetectionDataSet]],
        sample_dirs: os.PathLike | Collection[os.PathLike],
        schema_samples: SchemaSamples,
        label_path: os.PathLike,
        schema_labels: SchemaLabels,
        sample_type_desc: str = "sample",
        controller_cls: Optional[Type[TUIControllerDataSet]] = None,
        downloader: Optional[Type[DataSetDownloader]] = None,
        default_filter_query_str: Optional[str] = None,
    ):
        """
        Provides ways to load a training dataset's samples and labels into a DataFrame.

        :param name: name/id used to identify this dataset (type).
        :param dataset_cls: class derived from DefectDetectionDataSet
        :param sample_dirs: One ore more path-like object(s)
            pointing to a directory with sample files.
        :param schema_samples: SchemaSamples (derived) object representing
            column schema (column names + dtypes) for the DataFrame to
            be created which will contain the samples.
        :param label_path: One ore more path-like object(s)
            pointing to corresponding label files.
        :param schema_labels: SchemaLabels (derived) object representing
            column schema (column names + dtypes) for the DataFrame to
            be created which will contain the samples.
        :param sample_type_desc: (optional) description for this kind
            of sample (default = "sample").
        :param controller_cls: (optional) class which handles interactions with the TUI (Default = None).
        :param downloader: (optional) class with a 'download' method to download
            the required data (Default = None)
        :param default_filter_query_str: (optional) Default query string to filter data with.
        """
        self.name: str = name
        self.sample_dirs: list[Path] = (
            list(Path(d) for d in sample_dirs) if isinstance(sample_dirs, Iterable) else [Path(sample_dirs)]
        )
        self.label_path = Path(label_path)

        self.schema_samples: Schema = schema_samples
        self.schema_labels: Schema = schema_labels

        self.sample_type_desc: str = sample_type_desc
        self.downloader_cls: Optional[Type[DataSetDownloader]] = downloader
        self.downloader: Optional[DataSetDownloader] = None
        self.dataset_cls: Type[DefectDetectionDataSet] = dataset_cls
        self.controller_cls: Optional[Type[TUIControllerDataSet]] = controller_cls
        self.default_filter_query_str: Optional[str] = default_filter_query_str

    def init_downloader(self, **downloader_kwargs) -> None:
        if self.downloader_cls is None:
            log("Couldn't initialize unspecified downloader.")
        else:
            self.downloader = self.downloader_cls(**downloader_kwargs)

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
    def schema_full(self) -> SchemaFull:
        """
        ColSchema object containing all column names and data types
        for the dataframe containing samples + labels.
        """
        return self.schema_labels.combine_with_schema(self.schema_samples, target_schema_type=SchemaFull)

    @property
    def label_colnames(self) -> tuple[str, ...]:
        """Retrieve column names for the label data."""
        return tuple(self.schema_labels.keys())  # type: ignore

    @property
    def sample_colnames(self) -> tuple[str, ...]:
        """Retrieve column names for the sample data."""
        return tuple(self.schema_samples.keys())  # type: ignore

    @property
    def label_coltypes(self) -> tuple[str, ...]:
        """Retrieve column datatypes for the label data."""
        return tuple(self.schema_labels.values())  # type: ignore

    @property
    def sample_coltypes(self) -> tuple[str, ...]:
        """Retrieve column datatypes for the sample data."""
        return tuple(self.schema_samples.values())  # type: ignore

    def __setattr__(self, key, value) -> None:
        # Override setattr such that cached properties are reset if the relevant attributes have changed
        self.__dict__[key] = value
        for cached_attrname in type(self)._CACHED_PROPERTIES:
            if cached_attrname in self.__dict__:
                self.__dict__.pop(cached_attrname, None)

    def __repr__(self) -> str:
        return f"""{type(self).__name__}({", ".join(f'{k}={v}' for k, v in vars(self).items())})"""


class ImageDataSetConfig(DataSetConfig):
    SCHEMA_LABELS: ClassVar[SchemaLabels] = SchemaLabels()
    SCHEMA_SAMPLES: ClassVar[SchemaSamplesImageData] = SchemaSamplesImageData()

    _RASTER_IMG_EXTENSIONS: set[str] = {
        ".tif",
        ".tiff",
        ".png",
        ".bmp",
        ".raw",
        ".jfif",
        ".jif",
        ".jfi",
        ".jpe",
        ".jpeg",
        ".jpg",
        ".dib",
        ".gif",
        ".webp",
        ".arw",
        ".cr2",
        ".nrw",
        ".k25",
        ".heif",
        ".heic",
    }
    """Valid/expected possible extensions for raster images."""

    def __init__(
        self,
        name: str,
        dataset_cls: Optional[Type[DefectDetectionDataSetImages]],
        sample_dirs: os.PathLike | Collection[os.PathLike],
        label_path: os.PathLike,
        schema_samples: SchemaSamplesImageData = SCHEMA_SAMPLES,
        schema_labels: SchemaLabels = SCHEMA_LABELS,
        sample_type_desc: str = "solar panel sample image",
        controller_cls: Optional[Type[TUIControllerDataSet]] = None,
        downloader: Optional[Type[DataSetDownloader]] = None,
        default_filter_query_str: Optional[str] = None,
    ):
        """
        Provides configuration to load an image dataset for training a defect detection model.

        :param name: this dataset's name.
        :param dataset_cls: class derived from DefectDetectionDataSetImages
        :param sample_dirs: One ore more path-like object(s) pointing to a directory with sample files.
        :param label_path: A path-like object pointing to corresponding label file.
        :param schema_samples: ColumnSpec (column specification) object declaring column names and types
            for the samples in this dataset.
        :param schema_labels: ColumnSpec (column specification) object declaring column names and types
            for the labels in this dataset.
        :param sample_type_desc: (optional) description of this kind of sample (default = "sample").
        :param controller_cls: (optional) class which handles interactions with the TUI (Default = None).
        :param downloader: (optional) class with a 'download' method to download
            the required data (Default = None)
        :param default_filter_query_str: (optional) Default query string to filter data with.
        """
        super().__init__(
            name=name,
            dataset_cls=dataset_cls,
            sample_dirs=sample_dirs,
            schema_samples=schema_samples,
            label_path=label_path,
            schema_labels=schema_labels,
            sample_type_desc=sample_type_desc,
            downloader=downloader,
            controller_cls=controller_cls,
            default_filter_query_str=default_filter_query_str,
        )

    @classmethod
    def file_is_sample(cls, file: Path) -> bool:
        """Checks if potential sample file is an image assuming it has a (correct) extension."""
        return file.is_file() and file.suffix.lower() in ImageDataSetConfig._RASTER_IMG_EXTENSIONS

    def get_sample_paths(
        self,
        filechecker_function: Optional[Callable[[Path], bool]] = None,
        glob_pat: str = "*.*",
        recursive: bool = True,
    ) -> Iterator[Path]:
        """Iterates over sample paths

        :param filechecker_function: (optional) callable which takes a Path and returns whether
            it is a valid sample file for this dataset (default = None -> uses this
            class its 'file_is_sample' classmethod).
        :param glob_pat: (optional) string specifying the glob pattern to search files with (default = "*.*").
        :param recursive: (optional) bool indicating whether to search subdirectories of the sample directory
            recursively (default = True).
        """

        _filechecker_function: Callable[[Path], bool] = (
            type(self).file_is_sample if filechecker_function is None else filechecker_function
        )

        for sample_dir in self.sample_dirs:
            sample_dir = Path(sample_dir)
            sample_path_iterator = sample_dir.rglob(glob_pat) if recursive else sample_dir.glob(glob_pat)

            for potential_sample_file in sample_path_iterator:
                if _filechecker_function(potential_sample_file):
                    yield potential_sample_file

    @cached_property
    def full_dataset(self) -> DataFrame:
        """
        Merges sample and label DataFrames into one coherent whole.
        """
        try:
            label_data: DataFrame = self.label_data
            sample_data: DataFrame = self.sample_data
        except NotImplementedError as err:
            raise NotImplementedError(
                "Cannot obtain 'full_dataset' if 'label_data' "
                "and 'sample_data' properties are not implemented."
            ) from err

        # Check that column name and type used for label & sample ID are equal.
        if self.SCHEMA_LABELS.LABEL_SAMPLE_ID != self.SCHEMA_SAMPLES.LABEL_SAMPLE_ID:
            raise ValueError(
                f"Cannot merge label data with sample data: label schema entry "
                f"{self.SCHEMA_LABELS.LABEL_SAMPLE_ID.name!r} is not congruent with "
                f"{self.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name!r}."
            )

        # Check that label data and sample data are of equal length
        # as labels <-> samples needs to be a one-to-one mapping.
        if (n_labels := len(label_data)) != (n_samples := len(sample_data)):
            raise ValueError(
                "Number of samples must equal number of passed labels. "
                f"(got {n_labels} labels and {n_samples} samples...)."
            )

        # Identical columns present in both label and sample data to merge on.
        on_column: str = self.SCHEMA_LABELS.LABEL_SAMPLE_ID.name

        log.info("Merging label and sample datasets...")
        full_df = label_data.merge(sample_data, how="outer", on=on_column, sort=True)

        # Move ID column to the front and return dataframe
        return full_df[[on_column] + [col for col in full_df.columns if col != on_column]]

    @abstractmethod
    @cached_property
    def label_data(self) -> DataFrame:
        """
        Provides way to load labels for a specific dataset into DataFrame format.
        This step is seperate from loading the samples as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        _Must_ be implemented for extending classes.
        """
        raise NotImplementedError("Not implemented for baseclass")

    @abstractmethod
    @cached_property
    def sample_data(self) -> DataFrame:
        """
        Provides way to load samples for a specific dataset into DataFrame format.
        This step is seperate from loading the label data as the datasets encountered
        during this project commonly have labels and samples in different formats.
        The loading result is cached in memory until an attribute of this configuration
        is changed.
        _Must_ be implemented for extending classes.
        """
        raise NotImplementedError("Not implemented for baseclass")

    @property
    def schema_full(self) -> SchemaFullImageData:
        """
        ColSchemaFullImageData object containing all column names and data types
        for the dataframe containing samples + labels for an image dataset.
        """
        return self.schema_labels.combine_with_schema(
            self.schema_samples, target_schema_type=SchemaFullImageData
        )
