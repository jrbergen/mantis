from __future__ import annotations

from rich.console import RenderableType
from rich.style import StyleType
from textual import events
from textual.reactive import Reactive
from textual.widget import Widget
from textual.widgets._button import ButtonRenderable, ButtonPressed

from tmap_defectdetector.tui.colors import TERM_BG_DEFAULT, TERM_BG_SELECTED, FG_DISABLED, TERM_BG_DISABLED


class Button(Widget):
    def __init__(
        self,
        label: RenderableType,
        name: str | None = None,
        style: StyleType = "white on dark_blue",
    ):
        super().__init__(name=name)
        self.name = name or str(label)
        self.button_style = style

        self.label = label

    label: Reactive[RenderableType] = Reactive("")
    focussed: Reactive[bool] = False

    def render(self) -> RenderableType:
        return ButtonRenderable(self.label, style=self.button_style)

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default().stop()
        await self.emit(ButtonPressed(self))


class StylableButton(Widget):
    _STYLE_DEFAULT: str = f"green on {TERM_BG_DEFAULT}"
    _STYLE_SELECTED: str = f"red on {TERM_BG_SELECTED}"  # "bright_red on light_pink1"
    _STYLE_DISABLED: str = f"{FG_DISABLED} on {TERM_BG_DISABLED}"  # "gray93 on black"

    def __init__(
        self,
        label: RenderableType,
        name: str | None = None,
        selected: bool = False,
        enabled: bool = True,
        style_default: str = _STYLE_DEFAULT,
        style_selected: str = _STYLE_SELECTED,
        style_disabled: str = _STYLE_DISABLED,
    ):
        super().__init__(name=name)
        self.name = name or str(label)
        self.label = label
        self.selected = selected
        self.enabled = enabled
        self.style_default = style_default
        self.style_selected = style_selected
        self.style_disabled = style_disabled

    selected = Reactive(False)
    enabled = Reactive(True)
    label: Reactive[RenderableType] = Reactive("")

    @property
    def label_selected(self) -> str:
        return f"> {self.label}"

    def render(self) -> RenderableType:
        if self.enabled:
            label = self.label_selected if self.selected else self.label
            return ButtonRenderable(label, style=self.style_selected if self.selected else self.style_default)
        else:
            return ButtonRenderable(self.label, style=self.style_disabled)

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default().stop()
        await self.emit(ButtonPressed(self))
