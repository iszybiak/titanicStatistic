import json
import logging

# Set up logger
logger = logging.getLogger('config_loader')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(config_path="../config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        logger.info(f"Configuration loaded successfully from {config_path}.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        raise ValueError(f"Configuration file {config_path} not found.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise ValueError(f"Error decoding JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while loading configuration: {e}")
        raise ValueError(f"Error loading configuration: {e}")
