from __future__ import annotations

from typing import Protocol, TYPE_CHECKING, NamedTuple, Callable, Awaitable, Type

from rich.console import RenderableType
from textual import events, log
from textual._types import MessageTarget

from mantis.dataset.base.dataset_configs_base import DataSetConfig
from mantis.tui.events import ActivateSelected, StartRun, Disable, Enable
from mantis.tui.widgets.button import StylableButton


class Selectable(Protocol):
    def choose(self) -> None:
        ...


class SelectionState(NamedTuple):
    enabled: bool
    selected: bool


async def default_menu_callback():
    log("Activated button without overriden 'on_activate_selected' method; does nothing.")


class MenuItem(StylableButton, Selectable, MessageTarget):
    def __init__(
        self,
        label: RenderableType,
        name: str = "",
        selected: bool = False,
        enabled: bool = True,
        callback: Callable[[], Awaitable] = default_menu_callback,
    ):
        super().__init__(label=label, name=name, selected=selected, enabled=enabled)
        self._state_before_disabled: SelectionState = self.state
        self.callback = callback

    @property
    def state(self) -> SelectionState:
        return SelectionState(enabled=self.enabled, selected=self.selected)

    @property
    def label_selected(self) -> str:
        if hasattr(self.parent, "size"):
            spacer = round((self.parent.size.width - len(self.label) - self.gutter.width * 2) / 2) - 2
            if spacer < 0:
                spacer = 1
        else:
            spacer = 1
        return f">{spacer*' '}{self.label}{(spacer + 1)*' '}"

    async def on_enable(self, event: Enable):
        self.enable()

    async def on_disable(self, event: Disable):
        self.disable()

    def disable(self):
        self.enabled, self.selected = False, self._state_before_disabled.selected
        self.refresh()
        self.log(f"Disabled: {self.name}")

    def enable(self):
        self._state_before_disabled = SelectionState(enabled=self.enabled, selected=self.selected)
        self.enabled = True
        self.refresh()
        self.log(f"Enabled: {self.name}")

    async def on_click(self, event: events.Click):
        self.log(f"Clicked: {self.name}")
        if self.enabled and not self.selected:
            self.selected = True
        await self.on_activate_selected(ActivateSelected(self))

    async def on_activate_selected(self, event: ActivateSelected):
        await self.callback()


class DataSetSelectionItem(MenuItem):
    def __init__(
        self,
        label: RenderableType,
        name: str,
        dset_class: Type[DataSetConfig] = None,
        selected: bool = False,
        enabled: bool = True,
    ):
        self.dset_class: Type[DataSetConfig] = dset_class
        super().__init__(label=label, name=name, selected=selected, enabled=enabled)

    async def on_activate_selected(self, event: ActivateSelected):
        self.log(f"Started run: for dataset {self.name}")
        if self.selected and self.enabled:
            await self.parent.post_message(StartRun(self))
