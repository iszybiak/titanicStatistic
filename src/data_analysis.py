import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.logger import setup_logger

logger = setup_logger()


def data_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the given data, performs basic exploration, visualizes missing values,
    identifies outliers, and generates a summary report.

    Args:
    - data (pd.DataFrame): The input dataset for analysis.
    - report_path (str): Path where the summary report will be saved.

    Returns:
    - pd.DataFrame: The original dataset with no changes.

    Raises:
    - ValueError: If the input is not a DataFrame or if required columns are missing.
    - FileNotFoundError: If the report directory does not exist.
    """
    # Ensure input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Basic exploration
    try:
        data_describe = data.describe()
        logger.info("Data Summary:\n")
        logger.info(data_describe.to_string() + "\n\n")
    except Exception as e:
        logger.error(f"Error during basic exploration: {e}")
        raise ValueError("Error during basic exploration.")

    # Visualizing missing values
    try:
        missing_values = data.isnull().sum()
        logger.info("Missing Values:\n")
        logger.info(missing_values.to_string() + "\n\n")
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Value Map")
        plt.show()
    except Exception as e:
        logger.error(f"Error visualizing missing values: {e}")
        raise ValueError("Error visualizing missing values.")

    # Outlier analysis
    try:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=data[['Age', 'Fare']])
        plt.title("Boxplot of Age and Fare")
        plt.show()

        # Identifying outliers using IQR
        if 'Age' not in data.columns or 'Fare' not in data.columns:
            raise ValueError("Data must contain 'Age' and 'Fare' columns for outlier analysis.")

        Q1 = data[['Age', 'Fare']].quantile(0.25)
        Q3 = data[['Age', 'Fare']].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data[['Age', 'Fare']] < (Q1 - 1.5 * IQR)) | (data[['Age', 'Fare']] > (Q3 + 1.5 * IQR))).sum()
        logger.info(f"Number of outliers: {outliers}")
    except Exception as e:
        logger.error(f"Error during outlier analysis: {e}")
        raise ValueError("Error during outlier analysis.")

    # Visualizing correlations and distributions
    try:
        sns.pairplot(data.select_dtypes(include=[np.number]))
        plt.show()
    except Exception as e:
        logger.error(f"Error visualizing distributions: {e}")
        raise ValueError("Error visualizing distributions.")

    logger.info(f"Data analysis completed")

    return data
