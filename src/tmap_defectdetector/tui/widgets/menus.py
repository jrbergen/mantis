from __future__ import annotations

from typing import Union, Collection, Optional, Awaitable, Callable, NamedTuple

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from textual.reactive import Reactive
from textual.widget import Widget

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig
from tmap_defectdetector.tui.events import ActivateSelected
from tmap_defectdetector.tui.widgets.selection_item import MenuItem, DataSetSelectionItem

ChoosableRenderableType = Union[RenderableType, MenuItem]


class SelectionMenu(Widget):

    enabled = Reactive(True)
    can_focus = True

    def __init__(self, options: Optional[Collection[MenuItem]] = None, name: str = "Selection menu"):

        self._idx = 0
        self.options = []
        if options:
            self.add_options(options)
        super().__init__(name=name)

    def update_children_ownership(self):
        for option in self.options:
            if hasattr(option, "set_parent"):
                option.set_parent(self)

    def add_options(self, options: Collection[MenuItem] = tuple()):
        self.options += list(options)
        self.update_children_ownership()

    @property
    def menu(self) -> Panel:
        table = Table.grid(expand=True)
        for iiselec, selection in enumerate(self.options):
            table.add_row(selection)
        return Panel(table, title=self.name)

    def render(self) -> RenderableType:
        return self.menu

    async def action_up(self):
        await self.previous()

    async def action_down(self):
        await self.next()

    async def on_activate_selected(self, event: ActivateSelected):
        if hasattr(self.current, "on_activate_selected"):
            await self.current.on_activate_selected(event)
        else:
            self.log(
                f"WARNING: tried to call 'activate_selected' on current obj: "
                f"{repr(self.current)}, but it has no 'activate_selected' method. Did nothing."
            )

    @property
    def current(self) -> MenuItem:
        return self.options[self.idx]

    @current.setter
    def current(self, current: MenuItem) -> None:
        if not isinstance(current, MenuItem):
            raise TypeError(f"Current can only be set with {MenuItem.__name__} instances.")
        if current in self.options:
            self.idx = self.options.index(current)
        else:
            self.reset()

    def reset(self) -> None:
        self.idx = 0

    def watch_enabled(self):
        for option in self.options:
            option.enable() if self.enabled else option.disable()
        self.table.animate("layout_offset_x", 0 if self.enabled else -40)

    def previous(self) -> MenuItem:
        self.idx += 1
        new_option = self.options[self.idx]
        if new_option.enabled:
            return self.options[self.idx]
        elif any(option.enabled for option in self.options):
            return self.previous()
        else:
            self.idx -= 1
            return self.options[self.idx]

    def next(self) -> MenuItem:
        self.idx -= 1
        new_option = self.options[self.idx]
        if new_option.enabled:
            return self.options[self.idx]
        elif any(option.enabled for option in self.options):
            return self.next()
        else:
            self.idx += 1
            return self.options[self.idx]

    @property
    def idx(self) -> int:
        return self._idx

    @idx.setter
    def idx(self, value: int) -> None:
        if value > len(self.options) - 1:
            self._idx = 0
        elif value < 0:
            self._idx = len(self.options) - 1
        else:
            self._idx = value

    async def watch_enabled(self, enabled: bool):
        if enabled:
            for option in self.options:
                self.log(f"Enabled: {self.name}")
                option.enable()
        else:
            self.log(f"Disabled: {self.name}")
            for option in self.options:
                option.disable()

    def on_next_item(self):
        self.current.selected = False
        self.next().selected = True
        self.refresh()

    def on_previous_item(self):
        self.current.selected = False
        self.previous().selected = True
        self.refresh()

    async def on_hide(self) -> None:
        self.visible = False

    async def on_show(self) -> None:
        self.visible = True

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            return NotImplemented
        return self.options[idx]

    def __iter__(self) -> MenuItem:
        yield from self.options

    def __next__(self) -> MenuItem:
        return self.next()

    def __len__(self) -> int:
        return len(self.options)


class DataSetSelectionMenu(SelectionMenu):
    def __init__(
        self, options: Optional[Collection[MenuItem]] = None, name: str = "Select Dataset to work with"
    ):
        super().__init__(options=options, name=name)

    @classmethod
    def from_dataset_configs(cls, dataset_configs: Collection[DataSetConfig]) -> DataSetSelectionMenu:
        return DataSetSelectionMenu(
            [
                DataSetSelectionItem(label=cfg.name, name=cfg.name, dset_class=type(cfg))
                for cfg in dataset_configs
            ]
        )


class ActionButtonInfo(NamedTuple):
    name: str
    callable: Callable[[], Awaitable]
    enabled: bool = True


class ActionSelectionMenu(SelectionMenu):
    def __init__(
        self,
        menu_items_and_callbacks: Collection[ActionButtonInfo],
        name: str = "Choose option",
    ):
        options = [
            MenuItem(label=abi.name, callback=abi.callable, enabled=abi.enabled)
            for abi in menu_items_and_callbacks
        ]
        super().__init__(options=options, name=name)
