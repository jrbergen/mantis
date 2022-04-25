"""The main file containing the program's entrypoint."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.tmap_defectdetector.compatibility_checks import version_check
from src.tmap_defectdetector.dataset.datasets import ImageDataSetELPV
from src.tmap_defectdetector.dataset.downloaders import DataSetDownloaderELPV

from src.tmap_defectdetector.dataset.dataset_configs import DataSetConfigELPV
from src.tmap_defectdetector.dataset.schemas import SchemaLabelsELPV

from src.tmap_defectdetector.logger import log
from src.tmap_defectdetector.pathconfig.path_helpers import open_directory_with_filebrowser
from src.tmap_defectdetector.pathconfig.paths import DIR_TMP


def cli():
    """CLI is not yet implemented."""
    ...


def example_elpv(save_and_open_amplified_dataset: bool = True):
    """
    Performs an example run which (down)loads the ELPV defect image dataset,
    amplifies it with mirroring, rotations, and translations, and then optionally
    shows it .

    :param save_and_open_amplified_dataset: (optional) flag to indicate whether to save
        the example amplified dataset as images to a temporary directory.
        Can take quite some time and space(default = False)
    """
    # Initialize the dataset downloader and download the ELPV dataset from its git repository.
    downloader = DataSetDownloaderELPV()
    downloader.download()  # The dataset is downloaded to %LOCALAPPDATA%/.tmapdd/datasets/dataset-elpv/ (on Windows)

    # Initialize/load the ELPV dataset using the ELPV dataset configuration.
    elpv_dataset_config = DataSetConfigELPV()
    dataset = ImageDataSetELPV(dataset_cfg=elpv_dataset_config)

    # Filter dataset -> use only the polycrystalline solarpanels w/ type 'poly'.
    dataset.filter(query=f"{SchemaLabelsELPV().TYPE.name}=='poly'")

    # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    # put preprocessing methods in the ImageDataSet class itself later.
    dataset.amplify_data()

    # Specify and create a temporary directory to save our (amplified) image dataset.
    # Then open it in your OS's default filebrowser
    # Warning; can take a long time and quite a lot of storage space depending
    # on the number of samples in the dataset as well as the size of the accompanied images.
    if save_and_open_amplified_dataset:
        new_data_dir = Path(
            DIR_TMP,
            f"tmap_defectdetector_dataset_{datetime.utcnow().strftime('%Y_%m_%d_T%H%M%SZ')}",
        )
        new_data_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_images(new_data_dir)
        open_directory_with_filebrowser(new_data_dir)


def main():
    version_check()
    example_elpv()
    log.info("All done!")


if __name__ == "__main__":
    main()
