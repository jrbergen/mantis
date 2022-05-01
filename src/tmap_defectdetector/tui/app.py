from __future__ import annotations

import asyncio
from fractions import Fraction
from typing import Optional, Type, Collection, TYPE_CHECKING

from rich.panel import Panel
from textual.driver import Driver
from textual.layouts.grid import GridLayout


from textual.app import App

from textual.widgets import Footer, Static

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig
from tmap_defectdetector.tui.events import (
    NextItem,
    PreviousItem,
    ActivateSelected,
    StartRun,
    StatusUpdate,
)
from tmap_defectdetector.tui.widgets.header import CustomHeader
from tmap_defectdetector.tui.widgets.menus import DataSetSelectionMenu, SelectionMenu
from tmap_defectdetector.tui.widgets.infopanel import InfoPanel
from tmap_defectdetector.tui.widgets.statusvars import Status

if TYPE_CHECKING:
    from tmap_defectdetector.controllers.base import TUIControllerDataSet


class MantisTui(App):

    _GOLDEN_RATIO = Fraction((1 + 5**0.5) / 2)
    grid: GridLayout

    def __init__(
        self,
        dataset_configs: Optional[Collection[DataSetConfig]] = None,
        screen: bool = True,
        driver_class: Type[Driver] | None = None,
        log: str = "",
        log_verbosity: int = 1,
        title: str = "Mantis Defect Detector",
    ):
        self.dataset_configs: list[DataSetConfig] = list(dataset_configs) if dataset_configs else []
        self.menu: Optional[SelectionMenu] = DataSetSelectionMenu.from_dataset_configs(self.dataset_configs)

        self.header = CustomHeader(style="bold red on black")
        self.infopanel = InfoPanel(name="Info panel")
        self.footer: Footer = Footer()
        self.active_controller: Optional[TUIControllerDataSet] = None

        self.running_status = self._FILTER_STATUS

        super().__init__(
            screen=screen,
            driver_class=driver_class,
            log=log,
            log_verbosity=log_verbosity,
            title=title,
        )

    async def on_load(self):
        await self.bind("q", "quit", "Quit")
        await self.bind("up", "up", "Prev. option")
        await self.bind("down", "down", "Next option")
        await self.bind("enter", "enter", "Select")

    async def action_up(self):
        await self.menu.post_message(NextItem(self))

    async def action_down(self):
        await self.menu.post_message(PreviousItem(self))

    async def action_enter(self):
        if self.focused is not None:
            await self.focused.post_message(ActivateSelected(self))

    async def on_mount(self):
        n_status_rows = 1
        n_status_cols = 1

        self.grid: GridLayout = await self.view.dock_grid()

        self.grid.add_column("left", fraction=self._GOLDEN_RATIO.denominator)
        self.grid.add_column("right", repeat=n_status_cols, fraction=self._GOLDEN_RATIO.numerator)

        self.grid.add_row("top", max_size=10, min_size=10)
        self.grid.add_row("center", repeat=n_status_rows, min_size=24)
        self.grid.add_row("bot", max_size=3)

        areas = dict(
            footer="left-start|right-end,bot",
            # logo='left-start|left-end,top',
            running_status="left-start|left-end,center-start|center-end",
            header="left-start|right-end,top",
            menu="left-start|left-end,center-start|center-end",
            status="right-start|right-end,center-start|center-end",
        )
        self.grid.add_areas(**areas)

        self.grid.place(
            header=self.header,
            # logo=Static(Text(LOGO3, no_wrap=True, style="frame bold red on black")),
            status=self.infopanel,
            running_status=self.running_status,
            menu=self.menu,
            footer=self.footer,
        )
        self.running_status.visible = False
        if self.menu is not None:
            await self.menu.focus()
            await self.refresh_menu()

    async def refresh_menu(self):
        await self.menu.post_message(NextItem(self))
        await self.menu.post_message(PreviousItem(self))

    async def on_status_update(self, event: StatusUpdate):
        self.infopanel.status = event.action

    async def on_start_run(self, event: StartRun):
        # self.dataset_menu.enabled = True

        curcfg: DataSetConfig = self.dataset_configs[self.menu.idx]

        self.menu: DataSetSelectionMenu
        self.infopanel.label_file = str(curcfg.label_path)

        # Hide + disable menu
        self.menu.enabled = False
        self.menu.refresh()
        self.header.refresh()
        self.footer.refresh()

        # Hide everything for download prompt.
        self.menu.visible = False
        self.header.visible = False
        self.footer.visible = False
        self.infopanel.visible = False
        await asyncio.sleep(0.5)
        # self.dataset_menu.current.style = "blihnk red on yellow"
        self.infopanel.status = Status.DATASET_DOWNLOADING
        self.menu: DataSetSelectionMenu

        # Start donwload
        downloader = curcfg.downloader_cls()
        downloader.download()

        # Re-enable panels after download completed.
        self.header.visible = True
        self.running_status.visible = True
        self.footer.visible = True
        self.infopanel.visible = True

        # Initialize dataset and update panel info
        dataset = curcfg.dataset_cls(dataset_cfg=curcfg)
        self.infopanel.dataset_size = self.infopanel.n_input_files = len(dataset.samples)

        # Filter the data
        self.infopanel.status = Status.DATASET_FILTERING
        self.grid.place(running_status=self._FILTER_STATUS)
        dataset.filter(curcfg.default_filter_query_str)
        self.infopanel.dataset_size = len(dataset.samples)

        # Data amplification
        self.infopanel.status = Status.DATASET_AMPLIFICATION
        self.grid.place(running_status=self._AMPFLIF_STATUS)
        dataset.amplify_data()

        self.infopanel.status = Status.DATASET_AMPLIFICATION
        self.infopanel.dataset_size = len(dataset.samples)

        # Train dataset (not yet ipmlemented)
        self.infopanel.status = Status.TRAINING
        self.grid.place(running_status=self._TRAIN_STATUS_MOCK)
        await asyncio.sleep(7)

        # Done
        self.infopanel.status = Status.DONE
        self.grid.place(running_status=self._DONE_STATUS_ALPHAVERSION)
        self.running_status.visible = False
        self.running_status.hidden = True
        self.menu.visible = True
        self.menu.enabled = True
        self.refresh()
        self.grid.require_update()

    _FILTER_STATUS = Static(
        Panel(
            "[blink yellow2]Waiting for data filtration[/] to finish...\n"
            "(Progress cannot be displayed as dataset amplification is not "
            "implemented asynchronously currently; this shouldn't take more than "
            "a few minutes max.)."
        ),
    )

    _AMPFLIF_STATUS = Static(
        Panel(
            "[blink magenta2]Waiting for data amplification[/] to finish...\n"
            "(Progress cannot be displayed as dataset amplification is not "
            "implemented asynchronously currently; this shouldn't take more than "
            "a few minutes max.)."
        ),
    )

    _TRAIN_STATUS_MOCK = Static(
        Panel(
            "[blink cyan2]Waiting for model training[/] to complete...\n"
            "(Training is not yet implemented; waiting a few seconds to simulate actual training)"
        ),
    )

    _TRAIN_STATUS = Static(
        Panel("[blink cyan2]Waiting for model training [/] to complete...\n"),
    )

    _DONE_STATUS = Static(
        Panel("[green3]Model training was completed[/]\n"),
    )
    _DONE_STATUS_ALPHAVERSION = Static(
        Panel(
            "[green3]Model training was completed \n "
            "(press Q to quit; rest of the functionality has not yet been implemented)[/]\n"
        ),
    )
