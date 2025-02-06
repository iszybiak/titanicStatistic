import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file and returns a DataFrame.

    Args:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded data.

    Raises:
    - FileNotFoundError: If the file doesn't exist.
    - ValueError: If the file isn't a CSV or can't be read.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.lower().endswith('.csv'):
        raise ValueError("File is not a CSV.")

    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
