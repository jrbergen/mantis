"""Implements the starting screen for the TUI (text-based User Interface)."""
import sys
from typing import Optional, Callable

from textual.app import App
from textual import events
from textual.widgets import Placeholder

def default_abort_prodedure():
    print("Aborting & quitting Defect Detector.")
    sys.exit(0)

class DefectDetectorTUI(App):
    TITLE_BAR_FRAC: int = 2
    STATUS_BAR_FRAC: int = 1
    INTERACTIVE_PART_FRAC: int = 8 - TITLE_BAR_FRAC - STATUS_BAR_FRAC

    def __init__(self, abort_procedure: Callable = default_abort_prodedure, *args, **kwargs):
        self.abort_procedure: Callable = abort_procedure
        super().__init__(*args, **kwargs)

    async def on_mount(self, event: events.Mount) -> None:
        """Create a grid with auto-arranging cells."""

        grid = await self.view.dock_grid()

        grid.add_row("title_bar", fraction=self.TITLE_BAR_FRAC, max_size=10)
        grid.add_row("status_bar_frac", fraction=self.STATUS_BAR_FRAC, max_size=10)
        grid.add_row("interactive_part_frac", fraction=self.INTERACTIVE_PART_FRAC, max_size=10)
        grid.add_column("col", fraction=1, max_size=20)
        grid.add_row("row", fraction=1, max_size=10)
        grid.set_repeat(True, True)
        grid.add_areas(center="col-2-start|col-4-end,row-2-start|row-3-end")
        grid.set_align("stretch", "center")

        placeholders = [Placeholder() for _ in range(20)]
        grid.place(*placeholders, center=Placeholder())

    async def on_key(self, event):
        if event.key == 'q':
            self.abort_procedure()