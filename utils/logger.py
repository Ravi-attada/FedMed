"""
Member 5 - Ravi (Integration & QA)
Task: Centralized logging utility shared across all modules
"""
import logging
import os
from datetime import datetime

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name, level=logging.INFO):
    """
    Returns a configured logger writing to console and daily log file.
    Args:
        name: Logger name (e.g., 'FedMedServer', 'Node-1')
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'fedmed_{datetime.now().strftime("%Y%m%d")}.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
