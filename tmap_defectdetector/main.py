from __future__ import annotations


from tmap_defectdetector.compatibility_checks import version_check
from tmap_defectdetector.dataset_loading import DatasetDownloaderELPV
from tmap_defectdetector.datasets import ImageDataSet
from tmap_defectdetector.logger import log


def cli():
    ...


def main():
    version_check()

    # Just an example for now demonstrating dataset downloading, this will not be in main eventually
    dataset_downloader = DatasetDownloaderELPV()
    dataset_downloader.download()

    example_unfinished_imagedataset_obj = ImageDataSet()
    log.info("All done!")


if __name__ == "__main__":
    main()
