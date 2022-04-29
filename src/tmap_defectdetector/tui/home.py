"""Implements the starting screen for the TUI (text-based User Interface)."""
import sys
from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING, Collection

from rich.console import RenderResult, ConsoleOptions, Console
from rich.text import Text
from textual.app import App
from textual import events
from textual.events import Blur, Focus
from textual.layouts.grid import GridLayout
from textual.reactive import Reactive
from textual.views import GridView
from textual.widget import Widget
from textual.widgets import Placeholder, Header, Footer, Button

from tmap_defectdetector.dataset.base.dataset_configs_base import DataSetConfig
from tmap_defectdetector.tui.styles import Styles
from tmap_defectdetector.tui.widgets import DefaultWidget, NonReactiveHeader, DefaultFooter

if TYPE_CHECKING:
    from tmap_defectdetector.dataset.base.datasets_base import DefectDetectionDataSet


try:
    from pyfiglet import Figlet
except ImportError:
    print("Please install pyfiglet to run this example")
    raise


class FigletText:
    """A renderable to generate figlet text that adapts to fit the container."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Build a Rich renderable to render the Figlet text."""
        size = min(options.max_width / 2, options.max_height)
        if size < 4:
            yield Text(self.text, style="bold")
        else:
            if size < 7:
                font_name = "mini"
            elif size < 8:
                font_name = "small"
            elif size < 10:
                font_name = "standard"
            else:
                font_name = "big"
            font = Figlet(font=font_name, width=options.max_width)
            yield Text(font.renderText(self.text).rstrip("\n"), style="bold")


def default_abort_prodedure():
    print("Aborting & quitting Defect Detector.")
    sys.exit(0)


class Selector(Widget):

    active: bool = False


class DefectDetectorTUI(GridView):
    TITLE_BAR_FRAC: int = 2
    STATUS_BAR_FRAC: int = 1
    INTERACTIVE_PART_FRAC: int = 8 - TITLE_BAR_FRAC - STATUS_BAR_FRAC

    selected_button: Reactive[tuple[int, int]] = Reactive((0, 0))

    def __init__(
        self,
        name: str = "Defect Detector TUI",
        abort_procedure: Callable = default_abort_prodedure,
        dataset_configs: Collection[DataSetConfig] = tuple(),
        *args,
        **kwargs,
    ):

        self.name = name
        self.abort_procedure: Callable = abort_procedure
        self.dataset_configs: Collection[DataSetConfig] = dataset_configs
        self.widgets: list[list[DefaultWidget]] = [[]]

        self._selected: Optional[DefaultWidget] = None

        self._selx: int = 0
        self._sely: int = 0
        # self._selx, self._sely = self.get_first_selectable_xy()
        self.log(f"Found first selectable at x,y: {self._selx, self._sely}")
        self.dset_buttons: dict[str, Button] = dict()

        if not self.widgets:
            raise ValueError("Must add widgets to app!")

        super().__init__(*args, name=name, **kwargs)

    def watch_selected_button(self) -> None:
        for iix, widgetlist in enumerate(self.widgets):
            for iiy, widget in enumerate(widgetlist):
                if (iix, iiy) != self.selected_button:
                    widget.selected = False
                else:
                    widget.selected = True

    # def get_first_selectable_xy(self) -> tuple[int, int]:
    #     for iix, widgetlist in enumerate(self.widgets):
    #         for iiy, widget in enumerate(widgetlist):
    #             if widget.selectable:
    #                 return (iix, iiy)
    #     return (0, 0)

    async def on_mount(self) -> None:

        self.selected_dataset = 0

        def make_dataset_button(text: str, style: str):
            return Button(FigletText(text), style=style, name=text)

        self.dset_buttons = {
            name: make_dataset_button(name, Styles.DEFAULT)
            for name in [cfg.name for cfg in self.dataset_configs]
        }
        # Set basic grid settings
        self.grid.set_gap(2, 1)
        self.grid.set_gutter(1)
        self.grid.set_align("center", "center")

        self.header = NonReactiveHeader(tall=False)
        self.footer = Footer()

        # Create rows / columns / areas
        self.grid.add_column("col", max_size=30, repeat=8)
        self.grid.add_row(
            "header",
            max_size=3,
        )
        self.grid.add_row("spacer", max_size=1)
        self.grid.add_row("datasets", max_size=5, repeat=len(self.dataset_configs))
        self.grid.add_row("footer", max_size=1)
        self.grid.add_areas(
            header="col1-start|col8-end,header",
            datasets="col3-start|col6-end,datasets",
            footer="col1-start|col8-end,footer",
        )
        # Place out widgets in to the layout
        self.grid.place(header=self.header)
        self.grid.place(*self.buttons.values(), header=self.header, footer=Footer())
        # self.grid.
        # await self.build_mainview()
        # for irow, widgetlist in enumerate(self.widgets):
        #     for icol, widget in enumerate(widgetlist):
        #         match widget:
        #             case Header() | NonReactiveHeader():
        #                 await self.view.dock(widget, edge="top", size=3)
        #             case Footer() | DefaultFooter():
        #                 await self.view.dock(widget, edge="bottom", size=1)
        #             case _:
        #                 await self.view.dock(widget, edge="top", size=widget.height)

    async def build_mainview(self):
        """Make a simple grid arrangement."""
        # self.add_widget(NonReactiveHeader(), y=0)
        self.add_widget(DefaultWidget("home_window"), x=0, y=1)
        self.add_widget(DefaultWidget("home_window2"), x=0, y=2)
        self.add_widget(Footer())

        for widgetlist in self.widgets:
            for widget in widgetlist:
                widget.set_parent(self)

        await self.widgets[0][1].focus()
        print(f"Temporary breakpoint in {__name__}")

        # self.grid = await self.view.dock_grid(edge="left", name="grid")
        #
        # self.grid.add_column(fraction=1, name="left", min_size=70)
        # self.grid.add_column(size=60, name="center")
        # self.grid.add_column(fraction=1, name="right")
        #
        # self.grid.add_row(fraction=1, name="top", size=2)
        # self.grid.add_row(fraction=1, name="middle", min_size=20)
        # self.grid.add_row(fraction=1, name="bottom")
        #
        # self.grid.add_areas(
        #     area1="left-start|right-end,top",
        #     area2="left-start|center-end,middle",
        #     area3="center-start|right-end,middle",
        #     area4="left-start|right-end,bottom",
        # )
        #
        # self.grid.place(
        #     area1=NonReactiveHeader(),
        #     area2=DefaultWidget(name="Interactive part", style=Styles.DEFAULT),
        #     area3=DefaultWidget(name="Interactive part", style=Styles.DEFAULT),
        #     area4=Footer(),
        # )

    def add_widget(self, widget: Widget, x: Optional[int] = None, y: Optional[int] = None):
        if x is None and y is None:
            self.widgets[0].append(widget)
        elif y is None:
            self.widgets[x].append(widget)
        elif x is None:
            self.widgets[0].insert(y, widget)
        else:
            self.widgets[x].insert(y, widget)

    async def on_key(self, event):
        """Delegates keypress events."""

        for key in "q", "escape", "ctrl+z":
            await self.bind(key, "abort", "Quit")

        await self.bind("left", "left")
        await self.bind("right", "right")
        await self.bind("up", "up")
        await self.bind("down", "down")

        match event.key:
            case "q" | "escape" | "ctrl+z":
                self.abort_procedure()
            case "up":
                await self.action_up()
            case "down":
                await self.action_down()
            case "left":
                await self.action_left()
            case "right":
                await self.action_right()

            case _:
                self.log(f"Key pressed: {event.key!r}")

    @property
    def active_widget(self) -> Widget:
        w = self.widgets[self.sel_y][self.sel_x]
        # sely = self._sely % len(self.widgets)
        # w = self.widgets[sely][self._selx % len(self.widgets[sely])]
        self.log(f"Active widget = {w.__repr__()}")
        return w

    async def action_abort(self):
        """Defines what gets called when the 'abort' action is invoked via a binding."""
        await self.abort_procedure()

    async def action_left(self):
        await self.active_widget.on_event(Blur(sender=self))

        if self._selx > 0:
            left_widget = self.widgets[self._sely][self._selx - 1]
            if left_widget.selectable:
                self._selx -= 1
        await self.active_widget.on_event(Focus(sender=self))

    async def action_right(self):
        await self.active_widget.on_event(Blur(sender=self))
        self.log("Blurred: right")
        if self._selx < len(self.widgets):
            right_widget = self.widgets[self._sely][self._selx + 1]
            if right_widget.selectable:
                self._selx -= 1
            self._selx += 1
        await self.active_widget.on_event(Focus(sender=self))

    async def action_up(self):
        await self.active_widget.on_event(Blur(sender=self))
        self.log("Blurred: up")
        if self._sely > 0:
            above_widget = self.widgets[self._sely][self._selx - 1]
            if above_widget.selectable:
                self._sely -= 1
            self._sely -= 1
        await self.active_widget.on_event(Focus(sender=self))

    async def action_down(self):
        await self.active_widget.on_event(Blur(sender=self))
        if self._sely < len(self.widgets):
            below_widget = self.widgets[self._sely + 1][self._selx]
            if below_widget.selectable:
                self._sely += 1
        await self.active_widget.on_event(Focus(sender=self))


class DefectDetectorTUIApp(App):
    """Defect detector application."""

    def __init__(self, dataset_configs: Collection[DataSetConfig] = tuple(), *args, **kwargs):
        self.dataset_cfgs: list[DataSetConfig] = list(dataset_configs)
        super().__init__(*args, **kwargs)

    async def on_mount(self) -> None:
        await self.view.dock(DefectDetectorTUI(dataset_configs=self.dataset_cfgs))
