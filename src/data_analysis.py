import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.logger import setup_logger

logger = setup_logger()

def data_analysis(data: pd.DataFrame, report_path) -> pd.DataFrame:

    # Basic Exploration
    data_info = data.describe()
    logger.info(data.head())
    logger.info(data.info())
    logger.info(data_info)

    # Visualization of missing values
    missing_values = data.isnull().sum()
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Value Map")
    plt.show()

    # Outlier Analysis
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data[['Age', 'Fare']])
    plt.title("Wykres pudełkowy wieku i ceny biletu")
    plt.show()

    # Identification of outliers
    Q1 = data[['Age', 'Fare']].quantile(0.25)
    Q3 = data[['Age', 'Fare']].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data[['Age', 'Fare']] < (Q1 - 1.5 * IQR)) | (data[['Age', 'Fare']] > (Q3 + 1.5 * IQR))).sum()
    logger.info("Number of outliers:" + outliers)

    # Visualization of schedules
    sns.pairplot(data.select_dtypes(include=[np.number]))
    plt.show()

    with open(report_path, "w") as report_file:
        report_file.write("Data Summary:\n")
        report_file.write(data_info.to_string() + "\n\n")
        report_file.write("Missing Values:\n")
        report_file.write(missing_values.to_string() + "\n\n")

    logger.info("Data analysis completed. Results saved to report.txt")

    return data

