from __future__ import annotations


import sys
import warnings

from tmap_defectdetector.dataset_loading import DatasetDownloaderELPV
from tmap_defectdetector.datasets import ImageDataSet
from tmap_defectdetector.logger import log


def version_check(min_py_version: tuple[int, ...] = (3, 10, 0)) -> None:
    """
    Check Python version compatibility. This should be done by a package installer (e.g. pip, poetry, flit)
    automatically, but it may be nice to check this in case the script is run as standalone (although
    nesting the main package it in a src subdir should prevent that IIRC).
    """
    if not (cur_pyversion := sys.version_info) >= min_py_version or True:
        warnings.warn(
            f"You are running Python version {'.'.join(str(x) for x in cur_pyversion[:3])}. "
            f"Although this may work, use version {'.'.join(str(x) for x in min_py_version)} or higher "
            "for optimal compatibility.",
            category=UserWarning,
        )
    log.debug(f"Python version OK: {cur_pyversion!r}")


def cli():
    ...


def main():
    version_check()

    # Just an example for now demonstrating dataset downloading, this will not be in main eventually
    dataset_downloader = DatasetDownloaderELPV()
    dataset_downloader.download()

    example_unfinished_imagedataset_obj = ImageDataSet()
    log.info('All done!')


if __name__ == "__main__":
    main()
