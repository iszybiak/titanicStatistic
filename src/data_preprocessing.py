import pandas as pd
import os
from src.logger import setup_logger
from sklearn.model_selection import train_test_split

logger = setup_logger()

# def load_data():
train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
df = pd.read_csv(train_file)
print(df.info())
#     return pd.read_csv(train_file)


def handle_missing_data(data):
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
        # Filling in the missing 'Age', average Age.
        df.fillna(df['Age'].mean(), inplace=True)
        # Filling in the missing 'Embarked', value that appears most often in Embarked.
        df.fillna(df['Embarked'].mode()[0], inplace=True)
        logger.info("The missing data has been filled in.")
    else:
        logger.info("The collection is complete. It has no missing data")
    return data


def encode_categorical_features(data):
    """Coding of categorical variables"""
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    return data

def split_data(data, target_column='Survived'):
    """Division of data into features (X) and labels (y)"""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)










