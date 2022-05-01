from __future__ import annotations

import asyncio
import os
from fractions import Fraction
from typing import Optional, Type, Collection, Union, Protocol, TYPE_CHECKING

from textual.driver import Driver
from textual.events import Event
from textual.layouts.grid import GridLayout

__version__ = "0.0.1"

from textual.app import App

from textual.widgets import Footer


from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig
from tmap_defectdetector.tui.events import (
    NextItem,
    PreviousItem,
    ActivateSelected,
    Disable,
    Enable,
    StartRun,
    StatusUpdate,
)
from tmap_defectdetector.tui.widgets.header import CustomHeader
from tmap_defectdetector.tui.widgets.menus import DataSetSelectionMenu, ActionSelectionMenu, SelectionMenu
from tmap_defectdetector.tui.widgets.infopanel import InfoPanel


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
        self.menu: Optional[SelectionMenu] = self.dataset_menu

        self.header = CustomHeader(style="bold red on black")
        self.infopanel = InfoPanel(name="Info panel")
        self.footer: Footer = Footer()
        self.active_controller: Optional[TUIControllerDataSet] = None

        super().__init__(
            screen=screen,
            driver_class=driver_class,
            log=log,
            log_verbosity=log_verbosity,
            title=title,
        )

    @property
    def dataset_menu(self):
        return DataSetSelectionMenu.from_dataset_configs(self.dataset_configs)

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

        self.grid.add_column("left", fraction=self._GOLDEN_RATIO.denominator, min_size=24)
        self.grid.add_column("right", repeat=n_status_cols, fraction=self._GOLDEN_RATIO.numerator)

        self.grid.add_row("top", max_size=10, min_size=10)
        self.grid.add_row("center", repeat=n_status_rows, min_size=24)
        self.grid.add_row("bot", max_size=3)

        areas = dict(
            footer="left-start|right-end,bot",
            # logo='left-start|left-end,top',
            header="left-start|right-end,top",
            menu="left-start|left-end,center-start|center-end",
            status="right-start|right-end,center-start|center-end",
        )
        self.grid.add_areas(**areas)

        self.grid.place(
            header=self.header,
            # logo=Static(Text(LOGO3, no_wrap=True, style="frame bold red on black")),
            status=self.infopanel,
            menu=self.menu,
            footer=self.footer,
        )
        if self.menu is not None:
            await self.menu.focus()
            await self.refresh_menu()

    async def refresh_menu(self):
        await self.menu.post_message(NextItem(self))
        await self.menu.post_message(PreviousItem(self))

    async def on_status_update(self, event: StatusUpdate):
        self.infopanel.phase = event.action

    async def on_start_run(self, event: StartRun):

        curcfg: DataSetConfig = self.dataset_configs[self.menu.idx]
        self.dataset_menu.enabled = False

        if curcfg.controller_cls is None:
            raise NotImplementedError(
                f"Current dataset configuration {DataSetConfig.__name__!r} has no assigned TUI controller..."
            )
        controller = curcfg.controller_cls(dataset_cfg=curcfg)

        menu = controller.build_menu()

        self.menu = menu
        self.dataset_menu.visible = False
        self.grid.place(menu=self.menu)
        await self.refresh_menu()

        # self.menu = ActionSelectionMenu()
        # print(f"Temporary breakpoint in {__name__}")
        # dset_opt = self.dataset_menu.current.config
        # sender = event.sender
        # if hasattr(sender, "dset_class"):
        #
        #     sender: DataSetSelectionItem
        #     config_cls: Type[DataSetConfig] = event.sender.dset_class
        #     cfg = config_cls()
        #
        #     menu = ActionSelectionMenu({
        #         'Start training': DataSet.from.
        #     })
        #
        # else:
        #     self.log(f"{type(event).__name__} event contained no dataset_class")
        #
        # self.menu.visible = False

        # await asyncio.sleep(3)
        # await self.menu.post_message(Enable(self))
