"""
Structured JSON logger for the QYX_MedTech backend.
Uses Python's built-in logging with a clean format.
"""

import logging
import sys
from app.config import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    return logger
