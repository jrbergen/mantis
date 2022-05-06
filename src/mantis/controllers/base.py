from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, Optional

from textual.message import Message
from textual.message_pump import MessagePump
from textual.reactive import Reactive

from mantis.dataset.base.dataset_configs_base import DataSetConfig
from mantis.dataset.base.datasets_base import DefectDetectionDataSet
from mantis.dataset.base.downloaders_base import DataSetDownloader
from mantis.logger import log
from mantis.tui.events import StatusUpdate
from mantis.tui.widgets.menus import ActionSelectionMenu, ActionButtonInfo


class ActionController(MessagePump, ABC):
    pass


class TUIControllerDataSet(ActionController, ABC):

    status: Reactive = Reactive("Initializing...")

    def __init__(
        self,
        dataset_cfg: DataSetConfig,
        default_dataset_filter_query: str = "",
    ):

        super().__init__(parent=self.app)
        self.set_parent(self.app)

        self.config: DataSetConfig = dataset_cfg
        self.downloader: Optional[DataSetDownloader] = self.config.downloader
        self.dataset_cls: Type[DefectDetectionDataSet] = self.config.dataset_cls
        self.dataset: Optional[DefectDetectionDataSet] = None
        self.default_dataset_filter_query: str = default_dataset_filter_query

    async def load_dataset(self):
        await self.app.post_message(StatusUpdate(self, "Initializing downloader..."))
        self.config.init_downloader()
        await self.app.post_message(
            StatusUpdate(self, "Downloading dataset... (be patient; can currently cause UI to hang)")
        )
        self.downloader.download()
        await self.app.post_message(StatusUpdate(self, "Initializing dataset..."))
        self.dataset = self.dataset_cls(dataset_cfg=self.config)
        if self.default_dataset_filter_query:
            await self.app.post_message(
                StatusUpdate(self, f"Applying filter to dataset: {self.default_dataset_filter_query!r}")
            )
            self.dataset.filter(self.default_dataset_filter_query)
        StatusUpdate(self, f"Finished loading data... Awaiting further input.")

    async def amplify_dataset(self):
        if self.dataset is None:
            log.info("Dataset not initialized; loading dataset before amplification...")
            await self.load_dataset()
        StatusUpdate(self, f"Amplifying data... (be patient; can currently cause UI to hang)")
        self.dataset.amplify_data()

    async def train_sequence(self):
        await self.load_dataset()
        await self.amplify_dataset()
        await self.train()

    async def train(self):
        StatusUpdate(self, f"Started training model for dataset {type(self.dataset).__name__}!")
        raise NotImplementedError("training not implemented")

    async def benchmark(self):
        ...

    def build_menu(self) -> ActionSelectionMenu:
        return ActionSelectionMenu(
            menu_items_and_callbacks={
                ActionButtonInfo("Load, amplify & train", self.train_sequence),
                ActionButtonInfo("Load", self.load_dataset),
                ActionButtonInfo("Amplify dataset", self.amplify_dataset),
                ActionButtonInfo("Benchmark", self.benchmark),
            },
            name=f"Choose option (dataset={self.config.name}):",
        )
