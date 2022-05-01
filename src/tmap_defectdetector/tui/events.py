from __future__ import annotations

from textual.events import Event, Action


class StatusUpdate(Action):
    pass


class ActivateSelected(Event, bubble=False):
    pass


class StartRun(Event, bubble=True):
    pass


class StartTrain(Event, bubble=True):
    pass


class StartBenchmark(Event, bubble=True):
    pass


class ChangeParameters(Event, bubble=True):
    pass


class Enable(Event, bubble=False):
    pass


class Disable(Event, bubble=False):
    pass


class NextItem(Event):
    pass


class PreviousItem(Event):
    pass
