from __future__ import annotations

from pathlib import Path

import rich
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.reactive import Reactive
from textual.widget import Widget
import tensorflow as tf

from mantis import TEXTUAL_LOGPATH, FILE_LOG, DIR_APP
from mantis.tui.widgets.statusvars import Status


@rich.repr.auto(angular=False)
class InfoPanel(Widget, can_focus=False):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    style: Reactive[str] = Reactive("")
    height: Reactive[int | None] = Reactive(None)
    status: Reactive[Text] = Reactive(Status.DATASET_SELECTION)
    epoch: Reactive[int] = Reactive(0)

    n_input_files: Reactive[int] = Reactive(0)
    dataset_size: Reactive[int] = Reactive(0)
    application_dir: Reactive[str] = Reactive(DIR_APP)
    log_filepath: Reactive[str] = Reactive(FILE_LOG)
    label_file: Reactive[str] = Reactive("<Undefined>")

    def __init__(self, *, name: str | None, height: int | None = None) -> None:
        super().__init__(name=name)
        self.height = height
        self.epoch: int = 0
        self.n_input_files: int = 0
        self.status: Text = Status.DATASET_SELECTION
        self.built_w_gpu_support: str = tf.test.is_built_with_gpu_support()
        self.label_file: str = "<Undefined>"

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "has_focus", self.has_focus, False
        yield "mouse_over", self.mouse_over, False

    def render(self) -> RenderableType:
        info = Table.grid(padding=(0, 0), expand=True)

        info.add_column(justify="left")

        info.add_row(Text("Status: ", style="bold red") + self.status)
        info.add_row(Text("\n", style="None"))
        info.add_row(f"[underline bright_blue]Training progress")
        info.add_row(f"[bright_green]  Epoch: [/] [bright_yellow]{self.epoch}")
        info.add_row(Text("", style="None"))

        info.add_row(f"[underline bright_blue]Dataset info")
        info.add_row(f"[bright_green]  Dataset size: [bright_yellow] {self.dataset_size}")
        info.add_row(Text(style="None"))

        info.add_row(f"[underline bright_blue]Input info")
        info.add_row(f"[bright_green]  No. input files: [bright_yellow] {self.n_input_files}")
        info.add_row(f"[bright_green]  Label file: [bright_yellow] {self.label_file}")
        info.add_row(Text(style="None"))

        info.add_row(f"[underline bright_blue]Application info")
        info.add_row(f"[bright_green]  Application directory: [bright_yellow] {self.application_dir}")
        info.add_row(f"[bright_green]  Application Log file:  [bright_yellow] {self.log_filepath}")
        info.add_row(f"[bright_green]  TUI log file:  [bright_yellow] {TEXTUAL_LOGPATH}")
        info.add_row(Text(style="None"))

        info.add_row(f"[underline bright_blue]System info")

        info.add_row(
            f"[bright_green]  Tensorflow GPU support:  [bright_yellow] {str(self.built_w_gpu_support)}"
        )
        info.add_row(Text(style="None"))

        return Panel(info, title="[bright_blue]Info[/]")

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False
