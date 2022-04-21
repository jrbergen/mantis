"""The main file containing the program's entrypoint."""
from __future__ import annotations

from pathlib import Path

from tmap_defectdetector.compatibility_checks import version_check
from tmap_defectdetector.dataset_downloaders import DatasetDownloaderELPV

from tmap_defectdetector.datasets import ImageDataSetELPV
from tmap_defectdetector.image_helpers import file_is_image
from tmap_defectdetector.logger import log


def cli():
    ...


def main():
    version_check()

    # Initialize the dataset downloader and download the ELPV dataset from its git repository.
    downloader = DatasetDownloaderELPV()
    downloader.download()  # The dataset is downloaded to %LOCALAPPDATA%/.tmapdd/datasets/dataset-elpv/ (on Windows)

    # We first find the image paths in the image dataset directories and accept the files
    # which comply with our anonymous (lambda) function check.
    sample_files = downloader.get_data_files(
        filechecker_function=lambda p: Path.is_file(p) and file_is_image(p)
    )
    label_files = downloader.label_paths

    # Now we know where the label path(s) and the data (image file) paths are, we can construct the ImageDataSet.
    dataset = ImageDataSetELPV.from_paths(data_paths=sample_files, label_paths=label_files)

    # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    # put preprocessing methods in the ImageDataSet class itself later.
    dataset.amplify_data()

    log.info("All done!")


if __name__ == "__main__":
    main()
