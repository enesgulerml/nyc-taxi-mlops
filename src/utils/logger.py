import logging
import os
import sys

from src.config import LOG_FILE_PATH


def get_logger(name):
    # IF THERE IS NO LOG FOLDER, CREATE IT
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")

    # 1. WRITE TO FILE
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. WRITE TO CONSOLE
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
