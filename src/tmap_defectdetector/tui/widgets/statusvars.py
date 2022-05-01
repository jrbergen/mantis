from __future__ import annotations

from enum import Enum

from rich.text import Text

_fg = "red"

LOGO = r"""
  __  __             _   _       _____        __          _     _____       _            _             
 |  \/  |           | | (_)     |  __ \      / _|        | |   |  __ \     | |          | |            
 | \  / | __ _ _ __ | |_ _ ___  | |  | | ___| |_ ___  ___| |_  | |  | | ___| |_ ___  ___| |_ ___  _ __ 
 | |\/| |/ _` | '_ \| __| / __| | |  | |/ _ \  _/ _ \/ __| __| | |  | |/ _ \ __/ _ \/ __| __/ _ \| '__|
 | |  | | (_| | | | | |_| \__ \ | |__| |  __/ ||  __/ (__| |_  | |__| |  __/ ||  __/ (__| || (_) | |   
 |_|  |_|\__,_|_| |_|\__|_|___/ |_____/ \___|_| \___|\___|\__| |_____/ \___|\__\___|\___|\__\___/|_|   
"""


class Status(Enum):
    INITIALIZING: Text = Text(
        "Initializing",
        style="blink dodger_blue2",
    )
    DATASET_SELECTION: Text = Text(
        "Awaiting dataset selection",
        style="blink deep_sky_blue1",
    )
    DATASET_LOADING: Text = Text("Downloading dataset", style="blink yellow2")
    DATASET_DOWNLOADING: Text = Text("Loading dataset", style="blink yellow2")
    DATASET_AMPLIFICATION: Text = Text("Dataset amplificiation", style="blink yellow2")
    TRAINING: Text = Text("Model training", style="blink yellow2")
    DONE: Text = Text("Done", style="bold green")
