import pandas as pd
import os

from fontTools.misc.cython import returns

from src.logger import setup_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = setup_logger()

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(df.info())
    return pd.read_csv(filepath)

def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Checks for missing values and fills them with appropriate values.
        From the report I received information about missing individual data:
        INFO - Missing values:  Age         177
                                Embarked      2
        The code below focuses on completing this data.
    """
    # Checking for missing data in the entire set
    missing_data = data.isnull().sum()
    logger.info(f'Missing values: {missing_data[missing_data > 0]}')

    if missing_data.any():
        if 'Age' in missing_data.columns:
            # Filling in the missing 'Age', average Age.
            data.fillna(data['Age'].mean(), inplace=True)
        if 'Embarked' in missing_data.columns:
            # Filling in the missing 'Embarked', value that appears most often in Embarked.
            data.fillna(data['Embarked'].mode()[0], inplace=True)
        logger.info("The missing data has been filled in.")
    else:
        logger.info("The collection is complete. It has no missing data")
    return data

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Coding of categorical variables"""
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    return data

def split_data(data: pd.DataFrame, target_column: str):
    """Division of data into features (X) and labels (y)"""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(filepath: str, target: str):
    """Processes data and returns training and test sets."""
    data = load_data(filepath)
    data = handle_missing_data(data)
    data = encode_categorical_features(data)
    X_train, X_test, y_train, y_test = split_data(data, target)
    return X_train, X_test, y_train, y_test

# if __name__ == '__main__':
#     train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
#     X_train, X_test, y_train, y_test = preprocess(train_file, 'Survived')







