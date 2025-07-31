# ───────────────────────── src/manilafolder/logging_utils.py ─────────────────────────
"""
Logging utilities with rotating file handler.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logger(config: Optional[Config] = None) -> logging.Logger:
    """Set up a rotating file logger for error tracking.

    Args:
        config: Configuration object, uses defaults if None

    Returns:
        Configured logger instance

    Raises:
        OSError: If log file cannot be created
    """
    if config is None:
        config = Config()

    logger = logging.getLogger("manilafolder")

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.ERROR)

    # Create log directory if it doesn't exist
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.log_file,
        maxBytes=config.max_log_size,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )

    # Set up formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def log_error(
    message: str, exception: Optional[Exception] = None, config: Optional[Config] = None
) -> None:
    """Log an error message with optional exception details.

    Args:
        message: Error message to log
        exception: Optional exception to include in log
        config: Configuration object, uses defaults if None
    """
    logger = setup_logger(config)

    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)
