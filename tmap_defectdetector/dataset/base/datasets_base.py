from __future__ import annotations

import inspect
from typing import ClassVar, cast

from pandas import DataFrame, Series

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig, ImageDatasetConfig
from tmap_defectdetector.logger import log


class DefectDetectionDataSet:

    DEFAULT_DATASET_UNITS: ClassVar[str] = "samples"

    def __init__(self, dataset_config: DataSetConfig):
        """
        Construct a DefectDetectionDataSet (derived) object from
        a dataset configuration (i.e. DataSetConfig object).

        :param dataset_config: DataSetConfig derived object
            specifying how/where to read in the data labels
            and samples using the relevant schemas.
        """
        self._data_original: DataFrame = dataset_config.full_dataset
        self._data_filtered: DataFrame = self._data_original.copy()
        self._dataset_config: DataSetConfig = dataset_config
        log.info(f"Initialized {type(self).__name__!r} dataset.")

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
    def labels(self) -> DataFrame | Series:
        return self._data_filtered.loc[:, self.dataset_config.schema_labels.columns]

    @property
    def samples(self) -> DataFrame | Series:
        return self._data_filtered.loc[:, self.dataset_config.schema_samples.columns]

    @property
    def dataset_config(self) -> DataSetConfig:
        """
        Returns the dataset configuration object used when this dataset was initialized.
        Read-only.
        """
        return self._dataset_config

    @property
    def data(self) -> DataFrame:
        """
        Returns a dataframe w/ _filtered_ samples and labels if any filter has been applied.
        To reset to the unfiltered version, call the `reset()` method.
        Note that this is a read-only property.
        """
        return self._data_filtered

    def filter(self, query: str):
        """
        Filters data based on a pandas query string.

        :param query: string specifying Pandas query operation.
        """
        try:
            self._data_filtered = self.data.query(query)
        except Exception as err:
            raise ValueError(
                f"Invalid filter query string or other error occured"
                f" during execution of query: {query!r}."
            ) from err

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data})"


class DefectDetectionDataSetImages(DefectDetectionDataSet):

    DEFAULT_DATASET_UNITS: ClassVar[str] = "sample images"

    def __init__(self, dataset_config: ImageDatasetConfig):
        """
        Construct a DefectDetectionDataSetImages (derived) object from
        a dataset configuration for image data (i.e. ImageDataSetConfig object).

        :param dataset_config: ImageDataSetConfig derived object
            specifying how/where to read in the data labels and
            samples using the relevant schemas.
        """
        super().__init__(dataset_config=dataset_config)

    @property
    def dataset_config(self) -> ImageDatasetConfig:
        if not isinstance((cfg := super().dataset_config), ImageDatasetConfig):
            raise TypeError(
                f"Expected dataset configuration of type {ImageDatasetConfig.__name__}, got {type(cfg)}."
            )
        return cast(ImageDatasetConfig, cfg)

    @property
    def images(self) -> Series:
        """
        Returns the image data samples in a pandas.Series.
        """
        return self.samples.loc[:, self.dataset_config.SAMPLE_SCHEMA.SAMPLE]

    def amplify_data(self):
        """
        Performs operations which effectively increase the image dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case of image data.
        Subclasses can implement this method.
        """
        log.info("Amplyfing dataset...")
        self.add_mirror_images()
        self.add_translated_images()
        self.add_rotated_images()

    def asdf(self):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case the training data comprises images.

        :param mirror_axes: (optional) tuple specifying which
            (one or more) of the 4 symmetry axes for a square we should mirror over.
            1: mirror over horizontal axis
            2: mirror over vertical axis
            3: mirror over diagonal axis from top-left to bottom-right
            4: mirror over diagonal axis from bottom-left to top-right
        """

        # Make sure any non-loaded images are loaded now.
        # self._load_any_nonloaded_images()

        # mirror_kwargs = kwrags.
        # self._mirror(images)

        log.info("Adding mirrored images to dataset")
        self.add_mirror_images(mirror_axes=mirror_axes)

        log.info("Adding translated images to dataset (not really, not implemented yet...)")
        log.info("Adding rotated images to dataset (not really, not implemented yet...)")
        log.info("Adding superimposed images to dataset (not really, not implemented yet...)")

    def add_translated_images(self):
        ...

    def add_rotated_images(self):
        ...

    def add_mirror_images(self, mirror_axes: tuple[int, ...] = (1, 2, 3, 4)):
        """
        Adds mirrored version of images
        :param mirror_axes: (optional) tuple specifying which
            (one or more) of the 4 symmetry axes for a square we should mirror over.
            1: mirror over horizontal axis
            2: mirror over vertical axis
            3: mirror over diagonal axis from top-left to bottom-right
            4: mirror over diagonal axis from bottom-left to top-right
        """
        ...
        # non_mirrored_entries = self.data[self.data[self.MIRROR_AXIS] == 0]
        #
        # print(f"Temporary breakpoint in {__name__}")
        # # df = pd.DataFrame(
        # #     columns=self.data.columns,n
        # #     dtype={c: dt for c, dt in zip(self.data.columns, self.data.dtypes)},
        # # )
        # # for mirror_axis in mirror_axes:
        # #     match mirror_axis:
        # #         case 1:
        #
        # print(f"Temporary breakpoint in {__name__}")
