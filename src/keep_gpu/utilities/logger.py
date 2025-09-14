import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from colorlog import ColoredFormatter
except ImportError:
    ColoredFormatter = None  # fallback if colorlog not available


def _parse_log_level(env_value: Optional[str], default: Optional[int]) -> Optional[int]:
    """
    Parse log level from string. Return None if disabled.

    - If env_value is in {"", "no", "0"}, disable the logger.
    - Else, try to parse as logging level. Fallback to default.
    """
    if env_value is None:
        return default
    if env_value.strip().lower() in {"", "no", "0"}:
        return None
    return getattr(logging, env_value.strip().upper(), default)


def _build_console_handler(level: int) -> logging.Handler:
    """Create a colored console handler with filename:lineno."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = "%(log_color)s%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    if ColoredFormatter:
        formatter = ColoredFormatter(
            fmt,
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(fmt, "%H:%M:%S")
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(
    name: str, level: int, log_dir: str | Path = "logs"
) -> logging.Handler:
    """Create a file handler with filename:lineno."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = log_dir / f"{name}_{timestamp}.log"

    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setLevel(level)
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s:%(filename)s:%(lineno)d  %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    return handler


def setup_logger(
    name: str = "keep_gpu",
    default_console_level: Optional[int] = logging.INFO,
    default_file_level: Optional[int] = None,
) -> logging.Logger:
    """
    Set up a logger with configurable console and file handlers.

    Environment variables:
    - CONSOLE_LOG_LEVEL: INFO, DEBUG, WARNING, ERROR, or 'no'/''/0 to disable
    - FILE_LOG_LEVEL: same as above
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_is_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)  # master switch: keep open

    # Decide levels or disable
    console_env = os.getenv("CONSOLE_LOG_LEVEL")
    file_env = os.getenv("FILE_LOG_LEVEL")
    console_level = _parse_log_level(console_env, default_console_level)
    file_level = _parse_log_level(file_env, default_file_level)

    if console_level is not None:
        logger.addHandler(_build_console_handler(console_level))
    if file_level is not None:
        logger.addHandler(_build_file_handler(name, file_level))

    logger.propagate = False
    logger._is_configured = True  # type: ignore
    return logger
