import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import setup_logger
from src.data_loader import load_data

# Setup logger
logger = setup_logger()


def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in the data with appropriate values."""
    try:
        # Completing numerical values with mean
        num_imputer = SimpleImputer(strategy="mean")
        data[['Age', 'Fare']] = num_imputer.fit_transform(
            data[['Age', 'Fare']])

        # Completing categorical values with the most frequently occurring
        # value
        cat_imputer = SimpleImputer(strategy="most_frequent")
        data[['Embarked']] = cat_imputer.fit_transform(data[['Embarked']])

        logger.info("Missing data has been handled.")
        return data
    except Exception as e:
        logger.error(f"Error handling missing data: {e}")
        raise


def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extracts and simplifies features from the data."""
    try:
        if 'Name' not in data.columns:
            raise ValueError("Missing 'Name' column in data.")

        data['Title'] = data['Name'].str.extract(
            r' ([A-Za-z]+)\.', expand=False)

        # Simplifying titles
        title_map = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Don': 'Mr', 'Rev': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Col': 'Rare',
            'Capt': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Lady': 'Mrs', 'Countess': 'Mrs', 'Dona': 'Mrs'
        }

        data['Title'] = data['Title'].map(title_map).fillna('Rare')

        # Dropping the original 'Name' column
        data.drop(columns=['Name'], inplace=True)

        return data
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical features using OneHotEncoder."""
    try:
        categorical_features = ['Sex', 'Embarked', 'Title']

        if not all(feature in data.columns for feature in categorical_features):
            raise ValueError(
                f"Missing one or more categorical features: {categorical_features}")

        encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
        encoded_data = encoder.fit_transform(
            data[categorical_features]).toarray()
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_features))

        # Dropping original categorical columns and adding encoded columns
        data = data.drop(columns=categorical_features).reset_index(drop=True)
        data = pd.concat([data, encoded_df], axis=1)

        logger.info("Categorical features have been encoded.")
        return data
    except Exception as e:
        logger.error(f"Error encoding categorical features: {e}")
        raise


def scale_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    """Scales numeric features using StandardScaler."""
    try:
        numeric_features = ['Age', 'Fare']

        if not all(feature in data.columns for feature in numeric_features):
            raise ValueError(
                f"Missing one or more numeric features: {numeric_features}")

        scaler = StandardScaler()
        data[numeric_features] = scaler.fit_transform(data[numeric_features])

        logger.info("Numeric features have been scaled.")
        return data
    except Exception as e:
        logger.error(f"Error scaling numeric features: {e}")
        raise


def split_data(data: pd.DataFrame, target_column: str):
    """Splits data into features (X) and target (y), then returns train-test split."""
    try:
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data.")

        X = data.drop(columns=[target_column])
        y = data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def preprocess(filepath: str, target: str, data_type: str):
    """Processes data and returns appropriate dataset based on the data type ('train' or 'test')."""
    try:
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
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
