from __future__ import annotations

from typing import Optional

from rich import box as boxtypes
from rich.box import Box
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from textual import events
from textual.events import Event
from textual.reactive import Reactive
from textual.widget import Widget
from textual.widgets import Header, Footer

from tmap_defectdetector.tui.styles import Styles
from tmap_defectdetector.tui.events import DeSelectEvent, SelectEvent

available_widgets = [
    "Button",
    "ButtonPressed",
    "DirectoryTree",
    "FileClick",
    "Footer",
    "Header",
    "Placeholder",
    "ScrollView",
    "Static",
    "TreeClick",
    "TreeControl",
    "TreeNode",
    "NodeID",
]


class TUIStrings:
    TITLE = f"Defect Detector for TMAP course"


class DefaultWidget(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    selected: Reactive[bool] = Reactive(False)
    selectable: Reactive[bool] = Reactive(False)

    def __init__(
        self,
        name: str,
        style: Optional[Style] = None,
        style_selected: Optional[Style] = None,
        subtitle: Optional[str | Text] = None,
        box: Box = boxtypes.ROUNDED,
        width: int = 10,
        height: int = 5,
        start_selected: bool = False,
    ):

        self.subtitle_: Optional[str | Text] = subtitle
        self.box_style: Box = box
        self.style_default: Optional[Style] = Styles.DEFAULT if style is None else style
        self.style_selected: Optional[Style] = Styles.SELECTED if style_selected is None else style_selected
        self.width = width
        self.height = height

        super().__init__(name=name)

        if start_selected and self.selectable and not self.selected:
            self.selected = not self.selected

    def render(self) -> Panel:
        return Panel(
            # Text(self.name, justify="center"),
            Text("SELECTED" if self.selected else "NOT_SELECTED", justify="center"),
            style=(self.style_selected if self.selected else self.style_default),
            subtitle=self.subtitle_,
            box=self.box_style,
        )

    async def on_event(self, event: Event):
        self.log(f"{type(self).__name__} got event: {event.__repr__()!r}")
        match event:
            case DeSelectEvent():
                self.has_focus = False
            case SelectEvent():
                self.has_focus = True
            case _:
                pass

    async def on_enter(self) -> None:
        self.selected = True

    async def on_leave(self) -> None:
        self.selected = False

    async def on_click(self, event) -> None:
        self.selected = True


class NonReactiveHeader(Header):

    selectable: Reactive[bool] = False

    async def on_click(self, event: events.Click):
        self.tall = False


class DefaultFooter(Footer):
    selectable: Reactive[bool] = False


class FileSelectionWidget(DefaultWidget):
    def __init__(self, name: str):
        super().__init__(name=name)
