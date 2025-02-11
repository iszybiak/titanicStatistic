import logging
import os
from src.config import load_config


def setup_logger():
    """Configures a logger that saves logs to a file, with error handling."""
    try:
        # Load configuration
        config = load_config()
        report_path = config.get("report_path", "report.txt")
    except Exception as e:
        report_path = "report.txt"  # Default path if config fails
        print(f"Warning: Failed to load config. Using default log file: {report_path}. Error: {e}")

    try:
        # Ensure directory exists
        log_dir = os.path.dirname(report_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        report_path = "report.txt"  # Fallback to default

    # Create logger
    logger = logging.getLogger("titanic_analysis")

    if not logger.hasHandlers():  # Avoid duplicate handlers
        logger.setLevel(logging.DEBUG)

        try:
            # Set up file handler
            file_handler = logging.FileHandler(report_path, mode="a")
            file_handler.setLevel(logging.DEBUG)

            # Set up formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler: {e}")

    return logger
