"""The main file containing the program's entrypoint."""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from tmap_defectdetector.compatibility_checks import version_check
from tmap_defectdetector.config.paths import DIR_DATASETS
from tmap_defectdetector.dataset_downloaders import DatasetDownloaderELPV

from tmap_defectdetector.datasets import ImageDataSetELPV
from tmap_defectdetector.image_helpers import file_is_image
from tmap_defectdetector.logger import log


def cli():
    ...


def main():
    version_check()

    # Download the ELPV dataset from its git repository
    DatasetDownloaderELPV().download(Path(DIR_DATASETS, "dataset-elpv"))
    # The dataset is downloaded to %LOCALAPPDATA%/.tmapdd/datasets/dataset-elpv/ (on Windows)

    # We first find the image paths and label file Path(s) before we construct the ImageDataSet.
    # This should probably be made part of a method in the DatasetDownloader which gives the label/data paths.
    label_paths = [Path(DIR_DATASETS, "dataset-elpv", "labels.csv")]

    image_dirs = [Path(DIR_DATASETS, p) for p in [Path("dataset-elpv", "images")]]

    log.info("Loading images...")
    data_paths = []
    # Currently this crudely finds all image files in any of the (sub)directories in the image_dirs variable.
    for imgdir in image_dirs:
        files = tuple(imgdir.rglob("*.*"))
        for file in tqdm(files, desc="Finding images...", unit="images", total=len(files)):
            if file.is_file() and file_is_image(file):
                data_paths.append(file)

    # Now we know where the label path(s) and the data (image file) paths are, so we can construct the ImageDataSet.
    dataset = ImageDataSetELPV.from_paths(data_paths=data_paths, label_paths=label_paths)

    # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    # put preprocessing methods in the ImageDataSet class itself later.
    dataset.amplify_data()

    log.info("All done!")


if __name__ == "__main__":
    main()
