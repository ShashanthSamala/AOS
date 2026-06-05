# AURA-OS — Kernel Layer
# logger.py — Structured logging for AURA-OS
# Author: Samala Shashanth | Project: AURA-OS

import logging
import os


def get_logger(name, config=None):
    """
    Create a configured logger instance.

    Args:
        name: Logger name (e.g. 'kernel', 'perception')
        config: Optional config dict with logging settings

    Returns:
        logging.Logger instance
    """
    cfg = config or {}
    log_cfg = cfg.get("logging", {})

    level_str = log_cfg.get("level", "INFO").upper()
    log_file = log_cfg.get("log_file", "logs/aura.log")

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"aura.{name}")

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level_str, logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot write to log file {log_file}: {e}")

    return logger
