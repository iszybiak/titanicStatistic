import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import handle_missing_data, extract_features, encode_categorical_features


def test_handle_missing_data():
    data = pd.DataFrame({
        "Age": [22, None, 35, 40, None],
        "Fare": [7.25, 71.83, None, 8.05, 15.50],
        "Embarked": ["S", "", "C", "Q", "S"]
    })

    processed_data = handle_missing_data(data)

    assert processed_data.isnull().sum().sum() == 0
    assert processed_data["Age"].dtype == "float64"
    assert processed_data["Fare"].dtype == "float64"
    assert processed_data["Embarked"].dtype == "object"


def test_extract_features():
    data = pd.DataFrame({"Name": ["Kelly, Mr. James", "Dodge, Dr. Washington", "Daniels, Miss. Sarah"]})
    processed_data = extract_features(data)

    assert "Title" in processed_data.columns
    assert processed_data["Title"].tolist() == ["Mr", "Rare", "Miss"]
    assert "Name" not in processed_data.columns  # Powinna być usunięta


def test_encode_categorical_features():
    data = pd.DataFrame({"Sex": ["male", "female"], "Embarked": ["S", "C"], "Title": ["Mr", "Miss"]})
    processed_data = encode_categorical_features(data)

    assert "Sex_male" in processed_data.columns
    assert "Embarked_S" in processed_data.columns
    assert "Title_Mr" in processed_data.columns
