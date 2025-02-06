import logging
import os
from src.config import load_config

config = load_config()
report_path = config.get("report_path", "report.txt")

def setup_logger(log_file=report_path):
    """Function for configuring a logger that saves logs to a file."""

    # Create logger
    logger = logging.getLogger('titanic_analysis')

    # Avoid adding multiple handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Set up the file handler to write logs to a file
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Set up formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(file_handler)

    return logger
