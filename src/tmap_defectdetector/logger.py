"""Contains logging related functionality."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

from tmap_defectdetector.pathconfig.paths import FILE_LOG


LOG_LEVEL: str | int = "debug" if "-debug" in sys.argv[1:] or "--debug" in sys.argv[1:] else logging.INFO

LOG_WRITEMODE: str = "a"
"""'w' for overwriting the log each session, 'a' for appending."""

# LOG_FORMAT: str = '%(name)s - %(levelname)s — %(message)s'
LOG_FORMAT_TTY: str = "%(levelname)s — PID: %(process)d - %(message)s"
"""Format string for log messages (terminal)"""

LOG_FORMAT_FILE: str = "%(asctime)s — %(levelname)s — PID: %(process)d — %(message)s"
"""Format string for log messages (log file)"""

LOG_MAX_SIZE_BYTES: int = 10_000_000
"""Maximum size on disk for log files before they are rotated."""

LOG_SHOW_PROCESS_ID: bool = False


def _initialize_logger(log_name: str = __name__) -> logging.Logger:
    """
    Initializes logger (by default for the current module).

    :param log_name: (optional str) name for logger instance (defaults to module name).

    :returns: logging.Logger object
    """

    if not (log_dir := FILE_LOG.parent).exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    logger = logging.getLogger(log_name)

    # Handler for terminal/console stdout
    console_handler = RichHandler(rich_tracebacks=True, log_time_format="%Y-%m-%dT%H:%M:%S%z")
    console_format = logging.Formatter(
        f"{'[PID=%(process)d]' if LOG_SHOW_PROCESS_ID else ''}%(module)s.%(funcName)s: %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Handler for file-based log
    file_format = f"{'[PID=%(process)d]' if LOG_SHOW_PROCESS_ID else ''}%(module)s.%(funcName)s: %(message)s"
    file_handler = RotatingFileHandler(
        FILE_LOG,
        mode=LOG_WRITEMODE,
        maxBytes=LOG_MAX_SIZE_BYTES,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(file_format))
    logger.addHandler(file_handler)

    return logger


log = _initialize_logger()
