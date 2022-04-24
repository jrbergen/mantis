from __future__ import annotations

import inspect
import re
import shutil
import warnings
from pathlib import Path
from typing import ClassVar, cast, Collection


import pandas as pd
from pandas import DataFrame, Series
import cv2 as cv
from tqdm import tqdm

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig, ImageDatasetConfig
from tmap_defectdetector.image.mirrors import (
    mirror_horizontal,
    mirror_vertical,
    mirror_diag_bottomleft_topright,
    mirror_diag_topleft_bottomright,
)
from tmap_defectdetector.image.rotations import rotate_square
from tmap_defectdetector.image.translations import translate_image
from tmap_defectdetector.logger import log


class DefectDetectionDataSet:

    DEFAULT_DATASET_UNITS: ClassVar[str] = "samples"

    def __init__(self, dataset_cfg: DataSetConfig):
        """
        Construct a DefectDetectionDataSet (derived) object from
        a dataset configuration (i.e. DataSetConfig object).

        :param dataset_cfg: DataSetConfig derived object
            specifying how/where to read in the data labels
            and samples using the relevant schemas.
        """
        self._data_original: DataFrame = dataset_cfg.full_dataset
        self._data_filtered: DataFrame = self._data_original.copy()
        self._dataset_cfg: DataSetConfig = dataset_cfg
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
        return self._data_filtered.loc[:, self.dataset_cfg.schema_labels.columns]

    @property
    def samples(self) -> DataFrame | Series:
        return self._data_filtered.loc[:, self.dataset_cfg.schema_samples.columns]

    @property
    def dataset_cfg(self) -> DataSetConfig:
        """
        Returns the dataset configuration object used when this dataset was initialized.
        Read-only.
        """
        return self._dataset_cfg

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
    IDNUM_REX: re.Pattern = re.compile(
        r"(?P<prefix>.*?)(?P<img_id>\d+)(?P<suffix>.*)"
    )  # *(?P<extension>\.\D?)?$")

    def __init__(self, dataset_cfg: ImageDatasetConfig):
        """
        Construct a DefectDetectionDataSetImages (derived) object from
        a dataset configuration for image data (i.e. ImageDataSetConfig object).

        :param dataset_cfg: ImageDataSetConfig derived object
            specifying how/where to read in the data labels and
            samples using the relevant schemas.
        """
        super().__init__(dataset_cfg=dataset_cfg)

    @property
    def dataset_cfg(self) -> ImageDatasetConfig:
        if not isinstance((cfg := super().dataset_cfg), ImageDatasetConfig):
            raise TypeError(
                f"Expected dataset configuration of type {ImageDatasetConfig.__name__}, got {type(cfg)}."
            )
        return cast(ImageDatasetConfig, cfg)

    @property
    def images(self) -> Series:
        """
        Returns the image data samples in a pandas.Series.
        """
        return self.samples.loc[:, self.dataset_cfg.SCHEMA_SAMPLES.SAMPLE.name]

    def amplify_data(self, mirror_axes: tuple[int, ...] = (1, 2, 3, 4)):
        """
        Performs operations which effectively increase the dataset size
        as to reduce overfitting problems / allow for a more generalizable
        model.
        This can be done by e.g. by mirroring, rotating, translating,
        or applying filters in case of image data.

        :param mirror_axes: (optional) tuple specifying which
            (one or more) of the 4 symmetry axes for a square we should mirror over.
            1: mirror over horizontal axis
            2: mirror over vertical axis
            3: mirror over diagonal axis from top-left to bottom-right
            4: mirror over diagonal axis from bottom-left to top-right
        """
        log.info("Amplyfing dataset...")
        self.add_translated_images()
        self.add_mirror_images(mirror_axes=mirror_axes)
        self.add_rotated_images()

    def add_superimposed_images(self) -> None:
        log.info("Skipped image superimposition; superimposing images is not yet implemented...")

    def max_sample_id(self, df: DataFrame) -> int:
        """Gets the heighest numeric identifier contained in any of the sample IDs as an integer.

        :param df: DataFrame containing at least a column with sample IDs
            (convertable to strings containing a single integer numeric ID).
        """
        max_id: int = 0
        for cur_id in df[self.dataset_cfg.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name].tolist():
            try:
                cur_id = self.IDNUM_REX.match(str(cur_id))["img_id"]  # type: ignore
            except (KeyError, AttributeError, IndexError, TypeError) as err:
                raise ValueError(
                    f"A numeric identifier could not be found for sample ID: {str(cur_id)!r}."
                    "Currently every sample must have a (partially) numeric ID."
                ) from err
            if (cur_id := int(cur_id)) > max_id:
                max_id = cur_id
        return max_id

    def max_sample_id_number_zfill(self, df: DataFrame) -> int:
        """
        Gets largest length of numeric part of sample ids
        (e.g. to determine amount of zero-padding required).

        :param df: DataFrame containing at least a column with sample IDs
            (convertable to strings containing a single integer numeric ID).
        """
        max_len: int = 0
        for cur_id in df[self.dataset_cfg.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name].tolist():
            try:
                cur_len = len(self.IDNUM_REX.match(str(cur_id))["img_id"])  # type: ignore
            except (KeyError, AttributeError, IndexError, TypeError) as err:
                raise ValueError(
                    f"A numeric identifier could not be found for sample ID: {str(cur_id)!r}."
                    "Currently every sample must have a (partially) numeric ID."
                ) from err
            if cur_len > max_len:
                max_len = cur_len
        return max_len

    def _update_ids_for_dataset_addition(
        self, df_to_add: DataFrame, cur_max_sample_id: int, zero_padding: int
    ) -> DataFrame:
        """
        Updates/generates distinct sample IDs for new (amplified) data.

        :param df_to_add: new data to be concatenated as DataFrame
        :param cur_max_sample_id: current maximum sample ID in the dataset
            (if multiple new dataframes are generated without directly being added
            to the dataframe returned by self.data, make sure to manually keep track/increase
            the maximum sample id with each dataframe to add after a first call to 'max_sample_id'.
        :param zero_padding: determines how many numbers the numeric part of the id string will comprise.
        """
        id_col: str = self.dataset_cfg.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name

        # Get regex Match objects with prefix, img_id, and suffix groups.
        matches = df_to_add[id_col].apply(lambda s: self.IDNUM_REX.match(s)).to_list()

        # Make new ids with the same prefix and suffix as the matches, but a new ID number.
        log.info("Generating and adding new sample IDs for (newly generated?) data.")
        new_ids = [
            f"{curmatch['prefix']}{str(newid).zfill(zero_padding)}{curmatch['suffix']}"
            for curmatch, newid in zip(
                matches, range(cur_max_sample_id + 1, cur_max_sample_id + 1 + len(df_to_add))
            )
        ]
        df_to_add.loc[:, id_col] = new_ids
        return df_to_add

    def add_translated_images(
        self,
        translations: Collection[tuple[float | int, float | int]] = (
            (0.25, 0.25),
            (0.25, -0.25),
            (-0.25, -0.25),
            (-0.25, 0.25),
        ),
    ) -> None:
        """
        Adds translated version of images to dataset.

        :param translations: (optional) Collection of 2-tuples specifying
            translations in the x/y directions. If 2-tuples contain fractions in
            the domain [0, 1], the image is translated in the corresponding direction
            with that fraction of the total iage as pixels (result is rounded).
            If integers are passed the image is translated by that amount of pixels
            (modulo the image width/height).
             (default = ((0.25, 0.25), (0.25, -0.25), (-0.25, -0.25), (-0.25, 0.25)))
        :raises ValueError: if translations contain fractional values < -1.0 or > 1.0.

        .. note ::
            The positive x direction == left->right, and the positive y direction == top-> bottom.
        """
        if not isinstance(translations, Collection):
            raise TypeError(
                f"Translations must be a 'Collection', i.e. must define "
                f"__sized__, __iter__, and __contains__ methods. Got type {type(translations)}."
            )

        for translation in translations:
            if not isinstance(translation, Collection):
                raise ValueError("")
            if len(translation) != 2:
                raise ValueError(
                    f"Translations must be 2-tuples, got tuple of length ({str(translation)!r}."
                )
        translations = tuple((tr[0], tr[1]) for tr in translations)

        trans_col_x: str = self.dataset_cfg.SCHEMA_SAMPLES.TRANSL_X.name
        trans_col_y: str = self.dataset_cfg.SCHEMA_SAMPLES.TRANSL_Y.name

        img_col: str = self.dataset_cfg.SCHEMA_SAMPLES.SAMPLE.name

        cur_max_sample_id = self.max_sample_id(self.data)
        zero_padding: int = self.max_sample_id_number_zfill(self.data)
        non_translated_entries: pd.DataFrame = self.data[
            (self.data[trans_col_x] == 0) & (self.data[trans_col_y] == 0)
        ]

        translated_dfs: list[DataFrame] = []
        with tqdm(
            total=len(non_translated_entries) * len(translations),
            bar_format="{postfix[0]} (x={postfix[1][x]}, y={postfix[1][y]})",
            postfix=[
                "Applying translation to dataset",
                dict(x="(initializing)", y="(initializing)"),
            ],
        ) as pbar:

            for trans_x, trans_y in translations:
                log.info(f"Applying translation to dataset (x={trans_x}, y={trans_y}).")
                pbar.postfix[1]["x"], pbar.postfix[1]["y"] = trans_x, trans_y
                cur_df: DataFrame = non_translated_entries.copy()
                cur_df[img_col].apply(
                    lambda img: translate_image(img, translation_x=trans_x, translation_y=trans_y)
                )
                cur_df.loc[:, trans_col_x] = [trans_x] * len(cur_df)
                cur_df.loc[:, trans_col_y] = [trans_y] * len(cur_df)

                # Update sample IDs
                cur_df = self._update_ids_for_dataset_addition(
                    df_to_add=cur_df, cur_max_sample_id=cur_max_sample_id, zero_padding=zero_padding
                )
                cur_max_sample_id += len(cur_df)
                translated_dfs.append(cur_df)

        log.info("Concatenating DataFrames (with new translated images)...")
        self._data_filtered = pd.concat([self.data, *translated_dfs], axis=0)
        log.info(f"Translation of {type(self).__name__}'s samples completed.")
        log.info(f"Dataset now contains of {len(self.data)} samples.")

    def add_rotated_images(self, rotations: tuple[int, ...] = (90, 180, 270)) -> None:
        """
        Adds rotated version of images to data.

        :param rotations: (optional) tuple specifying which rotations should be applied.
            Rotations are restricted to multiples of 90 degrees (default = (90, 180, 270)).

        """
        for rot in rotations:
            if rot % 90 != 0:
                raise ValueError(f"Rotations are restricted to 90 degree multiples, got {rot}.")
        rotations = tuple(set(rot % 360 for rot in rotations))
        rot_col: str = self.dataset_cfg.SCHEMA_SAMPLES.ROT_DEG.name

        non_rotated_entries: pd.DataFrame = self.data[self.data[rot_col] == 0]

        rotated_dfs: list[DataFrame] = []
        img_col: str = self.dataset_cfg.SCHEMA_SAMPLES.SAMPLE.name
        cur_max_sample_id = self.max_sample_id(self.data)
        zero_padding: int = self.max_sample_id_number_zfill(self.data)

        with tqdm(
            total=len(non_rotated_entries) * len(rotations),
            bar_format="{postfix[0]} ({postfix[1][desc]} degrees)",
            postfix=["Rotating dataset", dict(desc="(initializing)")],
        ) as pbar:

            for angle in rotations:
                log.info(f"Rotating dataset ({angle} degrees).")
                pbar.postfix[1]["desc"] = angle
                cur_df: DataFrame = non_rotated_entries.copy()
                cur_df[img_col].apply(lambda img: rotate_square(img, angle))
                cur_df.loc[:, rot_col] = [angle] * len(cur_df)

                # Update sample IDs
                cur_df = self._update_ids_for_dataset_addition(
                    df_to_add=cur_df, cur_max_sample_id=cur_max_sample_id, zero_padding=zero_padding
                )
                cur_max_sample_id += len(cur_df)
                rotated_dfs.append(cur_df)

        log.info("Concatenating DataFrames (with new rotated images)...")
        self._data_filtered = pd.concat([self.data, *rotated_dfs], axis=0)
        log.info(f"Rotation of {type(self).__name__}'s samples completed.")
        log.info(f"Dataset now contains of {len(self.data)} samples.")

    def add_mirror_images(self, mirror_axes: tuple[int, ...] = (1, 2, 3, 4)) -> None:
        """
        Adds mirrored version of images to dataset.

        :param mirror_axes: (optional) tuple specifying which (one or more)
            of the 4 symmetry axes for a square we should mirror over (default = (1, 2, 3, 4)).
            1: mirror over horizontal axis.
            2: mirror over vertical axis.
            3: mirror over diagonal axis from top-left to bottom-right.
            4: mirror over diagonal axis from bottom-left to top-right.
        """
        mirror_col: str = self.dataset_cfg.SCHEMA_SAMPLES.MIRROR_AXIS.name
        cur_max_sample_id = self.max_sample_id(self.data)
        zero_padding: int = self.max_sample_id_number_zfill(self.data)

        non_mirrored_entries: pd.DataFrame = self.data[self.data[mirror_col] == 0]

        mirr_axis_descr: dict[int, str] = {
            0: "horizontally",
            1: "vertically",
            2: "diagonally (axis spans top-left -> bottom-right)",
            3: "diagonally (axis spans bottom-left -> top-right)",
        }

        mirrored_dfs: list[DataFrame] = []
        img_col: str = self.dataset_cfg.SCHEMA_SAMPLES.SAMPLE.name
        with tqdm(
            total=len(non_mirrored_entries) * len(mirror_axes),
            bar_format="{postfix[0]} {postfix[1][desc]}",
            postfix=["Mirroring dataset", dict(desc="(initializing)")],
        ) as pbar:

            for axis in mirror_axes:
                try:
                    log.info(f"Mirroring dataset {mirr_axis_descr[axis]}...")
                    pbar.postfix[1]["desc"] = mirr_axis_descr[axis]
                except KeyError:
                    pass

                cur_df = non_mirrored_entries.copy()
                match axis:
                    case 1:  # Mirror horizontally
                        cur_df.loc[:, img_col] = cur_df[img_col].apply(mirror_horizontal)
                    case 2:
                        cur_df.loc[:, img_col] = cur_df[img_col].apply(mirror_vertical)
                    case 3:
                        cur_df.loc[:, img_col] = cur_df[img_col].apply(
                            mirror_diag_topleft_bottomright
                        )
                    case 4:
                        cur_df.loc[:, img_col] = cur_df[img_col].apply(
                            mirror_diag_bottomleft_topright
                        )
                    case _:
                        raise ValueError(
                            f"Mirror axes values are restricted to domain [1, 4], got {axis}."
                        )

                # Update sample IDs
                cur_df = self._update_ids_for_dataset_addition(
                    df_to_add=cur_df, cur_max_sample_id=cur_max_sample_id, zero_padding=zero_padding
                )
                cur_max_sample_id += len(cur_df)
                mirrored_dfs.append(cur_df)

        log.info("Concatenating DataFrames (with new mirrored images)...")
        self._data_filtered = pd.concat([self.data, *mirrored_dfs], axis=0)
        log.info(f"Mirroring of {type(self).__name__}'s samples completed.")
        log.info(f"Dataset now contains of {len(self.data)} samples.")

    def save_images(
        self, target_directory: Path, leave_free_space_bytes: int = 500_000_000
    ) -> None:
        """
        Saves currently filtered/amplified set of images to disk.

        :param target_directory: Path to target directory.
            Will be created (including any parents) if it doesn't exist.
        :param leave_free_space_bytes: (optional) if the remaining space on the target
            directory's filesystem gets below this number (+- about 500x the average image size),
            the saving process is aborted. (default = 500_000_000 (i.e. 500MB).
        """
        # id_col: str = self.dataset_cfg.SCHEMA_SAMPLES.LABEL_SAMPLE_ID.name
        # matches = self.data[id_col].apply(lambda s: self.IDNUM_REX.match(s))
        warnings.warn(
            "\nSaving an amplified dataset as image can take quite a while\n"
            "as well as use a lot of disk/memory space depending on the images sizes.",
            UserWarning,
        )

        sample_col: str = self.dataset_cfg.SCHEMA_SAMPLES.SAMPLE.name
        for iirow, row in enumerate(
            tqdm(
                self.data.itertuples(),
                desc=f"Saving images to {target_directory}...",
                total=len(self.data),
            )
        ):
            savepath = Path(target_directory, row.LABEL_SAMPLE_ID)
            if (
                iirow % 500 == 0
                and shutil.disk_usage(target_directory).free < leave_free_space_bytes
            ):
                # Don't check space every loop as it's a relatively expensive check.
                raise IOError(
                    f"Minimum required remaining free space on target filesystem (disk/memory) exceeded: {leave_free_space_bytes/1024**2:.2f}MiB"
                )
            cv.imwrite(str(savepath.resolve()), getattr(row, sample_col))
