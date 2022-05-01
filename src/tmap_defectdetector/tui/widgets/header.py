from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from textual.widgets import Header

from tmap_defectdetector.tui.widgets.statusvars import LOGO


class CustomHeader(Header):
    def render(self) -> RenderableType:
        header_table = Table.grid(padding=(0, 0), expand=True)
        header_table.style = self.style
        header_table.add_column(justify="full", ratio=0, width=8, overflow="ignore")
        header_table.add_column("clock", justify="right", width=1)
        header_table.add_row(LOGO, self.get_clock() if self.clock else "")
        header: RenderableType
        header = Panel(header_table, style=self.style) if self.tall else header_table
        return header
