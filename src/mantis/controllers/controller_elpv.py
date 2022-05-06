from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mantis import DIR_TMP
from mantis.controllers.base import TUIControllerDataSet

from mantis.dataset.dataset_configs import DataSetConfigELPV
from mantis.dataset.datasets import ImageDataSetELPV
from mantis.dataset.schemas import SchemaLabelsELPV
from mantis.path_helpers import open_directory_with_filebrowser

if TYPE_CHECKING:
    from mantis.dataset.base.dataset_configs_base import DataSetConfig


class TUIControllerELPV(TUIControllerDataSet):
    def __init__(
        self,
        dataset_cfg: DataSetConfig = None,
        default_dataset_filter_query: str = f"{SchemaLabelsELPV().TYPE.name}=='poly'",
    ):
        if dataset_cfg is None:
            dataset_cfg = DataSetConfigELPV()
        super().__init__(
            dataset_cfg=dataset_cfg,
            default_dataset_filter_query=default_dataset_filter_query,
        )
        self.dataset: ImageDataSetELPV

    def save_amplified_dataset(self, open_in_filebrowser: bool = True):
        if hasattr(self.dataset, "save_images"):
            new_data_dir = Path(
                DIR_TMP,
                f"{APP_NAME}_dataset_{datetime.utcnow().strftime('%Y_%m_%d_T%H%M%SZ')}",
            )
            new_data_dir.mkdir(parents=True, exist_ok=True)
            self.dataset.save_images(new_data_dir)
            if open_in_filebrowser:
                open_directory_with_filebrowser(new_data_dir)

    #
    # @classmethod
    # def run(cls) -> DataSetELPV:
    #     """
    #     Performs an example run which (down)loads the ELPV defect image dataset,
    #     amplifies it with mirroring, rotations, and translations, and then optionally
    #     shows it .
    #
    #     :param save_and_open_amplified_dataset: (optional) flag to indicate whether to save
    #         the example amplified dataset as images to a temporary directory.
    #         Can take quite some time and space(default = False)
    #     """
    #     # Initialize the dataset downloader and download the ELPV dataset from its git repository.
    #
    #     # Initialize/load the ELPV dataset using the ELPV dataset configuration.
    #     elpv_dataset_config = cls()
    #     dataset = ImageDataSetELPV(dataset_cfg=elpv_dataset_config)
    #
    #     # Filter dataset -> use only the polycrystalline solarpanels w/ type 'poly'.
    #     dataset.filter(query=f"{SchemaLabelsELPV().TYPE.name}=='poly'")
    #
    #     # Here comes the preprocessing step (we could e.g. make a ImageDataSetPreProcessor class/function or perhaps
    #     # put preprocessing methods in the ImageDataSet class itself later.
    #     dataset.amplify_data()
    #
    #     # Specify and create a temporary directory to save our (amplified) image dataset.
    #     # Then open it in your OS's default filebrowser
    #     # Warning; can take a long time and quite a lot of storage space depending
    #     # on the number of samples in the dataset as well as the size of the accompanied images.
