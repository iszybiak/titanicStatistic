import logging
from venv import logger

from fontTools.misc.cython import returns


def setup_logger(log_file='report.txt'):
    """Function for configuring a logger that saves logs to a file."""

    logger = logging.getLogger('titanic_analysis')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger