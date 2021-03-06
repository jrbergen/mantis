"""The main file containing the program's entrypoint."""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path


from mantis.compatibility_checks import version_check
from mantis.dataset.base.dataset_configs_base import DataSetConfig
from mantis.dataset.datasets import ImageDataSetELPV

from mantis.dataset.dataset_configs import DataSetConfigELPV, DataSetConfigWineDetector
from mantis.dataset.downloaders import DataSetDownloaderELPV
from mantis.dataset.schemas import SchemaLabelsELPV
from mantis.model.cnn.elpv_vgg16 import elpv_vgg16

from mantis.path_helpers import open_directory_with_filebrowser
from mantis import DIR_TMP, TEXTUAL_LOGPATH, APP_NAME
from mantis.tui.app import MantisTui


def cli():
    """CLI is not yet implemented."""
    ...


def gui():
    """GUI is not yet implemented."""
    ...


def tui():
    """TUI entrypoint for Mantis Defect Detector."""
    os.environ["PYTHONASYNCIODEBUG"] = "1"
    dataset_configs: list[DataSetConfig] = [
        DataSetConfigELPV(),
        DataSetConfigWineDetector(name="Wine Detector Dataset (Not yet implemented)"),
    ]
    app = MantisTui()
    app.run(title="Defect Detector", log=TEXTUAL_LOGPATH, dataset_configs=dataset_configs)


def example_elpv(save_and_open_amplified_dataset: bool = False):
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
    downloader.download()  # The dataset is downloaded to %LOCALAPPDATA%/.mantis/datasets/dataset-elpv/ (on Windows)

    # Clear terminal
    os.system("clear") if platform.system() == "Linux" else os.system("cls")

    # Initialize/load the ELPV dataset using the ELPV dataset configuration.
    elpv_dataset_config = DataSetConfigELPV()
    dataset = ImageDataSetELPV(dataset_cfg=elpv_dataset_config)

    # Filter dataset -> use only the polycrystalline solarpanels w/ type 'poly'.
    dataset.filter(query=f"{SchemaLabelsELPV().TYPE.name}=='poly'")

    # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    # put preprocessing methods in the ImageDataSet class itself later.
    # dataset.amplify_data()
    dataset.move_data_categorical_subdirs()
    # dataset.save_categorical()

    # Specify model configuration here manually for now
    # model_config = CNNModelConfig(
    #     n_epochs=1024 if GPU_AVAILABLE else 64,
    # )

    # Construct and train model
    # model = CNNModelELPV(dataset, model_config=model_config)
    #
    # model.full_run_from_dataset(dataset=dataset)

    # Specify and create a temporary directory to save our (amplified) image dataset.
    # Then open it in your OS's default filebrowser
    # Warning; can take a long time and quite a lot of storage space depending
    # on the number of samples in the dataset as well as the size of the accompanied images.
    if save_and_open_amplified_dataset:
        new_data_dir = Path(
            DIR_TMP,
            f"{APP_NAME}_dataset_{datetime.utcnow().strftime('%Y_%m_%d_T%H%M%SZ')}",
        )
        new_data_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_images(new_data_dir)
        open_directory_with_filebrowser(new_data_dir)


def main():
    version_check()

    if platform.system() == "Linux" or any(s in sys.argv[1:] for s in ("--tui", "-tui", "-t", "--t")):
        tui()
    else:
        example_elpv()
        for iiarg, arg in enumerate(sys.argv[1:]):

            if arg in ("-e", "--epochs", "--e"):
                try:
                    epochs = int(sys.argv[1:][iiarg + 1])
                except (IndexError, TypeError):
                    print("Epoch argument not castable to int?")
                    raise ValueError("Epoch argument not castable to int?")
                break
        else:
            epochs = 50

        elpv_vgg16(epochs=epochs)


if __name__ == "__main__":
    main()
