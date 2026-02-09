"""Utility functions and logging setup."""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Use loguru if available, otherwise stdlib logging
try:
    from loguru import logger

    def setup_logging(config: dict = None) -> None:
        log_config = (config or {}).get("logging", {})
        level = log_config.get("level", "INFO")
        log_file = log_config.get("log_file", "./logs/quantbot.log")
        rotation = log_config.get("rotation", "10 MB")
        logger.remove()
        logger.add(sys.stdout, level=level,
                    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}",
                    colorize=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, level="DEBUG",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
                    rotation=rotation, retention="30 days", compression="zip")

except ImportError:
    # Fallback to stdlib logging
    logger = logging.getLogger("quantbot")

    def setup_logging(config: dict = None) -> None:
        log_config = (config or {}).get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        log_file = log_config.get("log_file", "./logs/quantbot.log")

        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
                                       datefmt="%H:%M:%S")
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"))
        logger.addHandler(fh)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"
