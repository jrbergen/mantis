"""The main file containing the program's entrypoint."""
from __future__ import annotations

from tmap_defectdetector.compatibility_checks import version_check
from tmap_defectdetector.dataset.datasets import ImageDataSetELPV
from tmap_defectdetector.dataset.downloaders import DatasetDownloaderELPV

from tmap_defectdetector.dataset.dataset_configs import DataSetConfigELPV
from tmap_defectdetector.dataset.schemas import SchemaLabelsELPV

from tmap_defectdetector.logger import log


def cli():
    ...


def example_elpv():
    # Initialize the dataset downloader and download the ELPV dataset from its git repository.
    downloader = DatasetDownloaderELPV()
    downloader.download()  # The dataset is downloaded to %LOCALAPPDATA%/.tmapdd/datasets/dataset-elpv/ (on Windows)

    # Initialize/load the ELPV dataset using the ELPV dataset configuration.
    elpv_dataset_config = DataSetConfigELPV()
    dataset = ImageDataSetELPV(dataset_config=elpv_dataset_config)

    # Filter dataset -> use only the polycrystalline solarpanels w/ type 'poly'.
    dataset.filter(query=f"{SchemaLabelsELPV().TYPE.name}=='poly'")

    # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    # put preprocessing methods in the ImageDataSet class itself later.
    dataset.amplify_data()


def main():
    version_check()
    example_elpv()
    log.info("All done!")


if __name__ == "__main__":
    main()
