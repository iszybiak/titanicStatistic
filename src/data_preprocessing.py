import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import setup_logger
from sklearn.model_selection import train_test_split
from src.data_loader import load_data

logger = setup_logger()

def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Checks for missing values and fills them with appropriate values."""

    # Completing numerical values with mean
    num_imputer = SimpleImputer(strategy="mean")
    data[['Age', 'Fare']] = num_imputer.fit_transform(data[['Age', 'Fare']])

    # Completing categorical values with the most frequently occurring value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[['Embarked']] = cat_imputer.fit_transform(data[['Embarked']])

    logger.info("Missing data have been corrected.")
    return data

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """ Dd"""
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Mapa upraszczająca tytuły
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Don': 'Mr', 'Rev': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Col': 'Rare',
        'Capt': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Lady': 'Mrs', 'Countess': 'Mrs', 'Dona': 'Mrs'
    }

    # Zamiana tytułów na uproszczone
    data['Title'] = data['Title'].map(title_map).fillna('Rare')

    # Removing the original column
    data.drop(columns=['Name'], inplace=True)

    return data

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Coding of categorical variables using OneHotEncoder"""
    categorical_features = ['Sex', 'Embarked', 'Title']
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore")

    encoded_data = encoder.fit_transform(data[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

    data = data.drop(columns=categorical_features).reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1)

    logger.info("Categorical variables have been encoded.")

    return data

def scale_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    """ Scale ... """
    numeric_features = ['Age', 'Fare']
    scaler = StandardScaler()

    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    logger.info("Numeric variables have been scaled.")
    return data


def split_data(data: pd.DataFrame, target_column: str):
    """Division of data into features (X) and labels (y)"""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(filepath: str, target: str, data_type: str):
    """Processes data and returns appropriate dataset based on type."""
    data = load_data(filepath)
    data = handle_missing_data(data)
    data = extract_features(data)
    data = encode_categorical_features(data)
    data = scale_numeric_features(data)

    if data_type == 'train':
        X_train, X_test, y_train, y_test = split_data(data, target)
        return X_train, X_test, y_train, y_test
    elif data_type == 'test':
        return data.drop(columns=[target], errors='ignore')
    else:
        raise ValueError("Invalid data_type. Choose 'train' or 'test'.")


# if __name__ == '__main__':
#     X_train, X_test, y_train, y_test = preprocess('../data/train.csv', 'Survived')






