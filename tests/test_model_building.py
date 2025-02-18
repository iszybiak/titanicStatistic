import pickle
import pandas as pd
import sys
import os
from src.model_building import create_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def test_pipeline(tmp_path):
    data = pd.DataFrame({
        "Age": [22, 35, 40, 50, 29],
        "Fare": [7.25, 71.83, 8.05, 15.50, 23.00],
        "Sex_male": [1, 0, 1, 1, 0],
        "Embarked_S": [1, 0, 1, 1, 0]
    })

    labels = pd.Series([0, 1, 0, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.5

    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as model_file:
        pickle.dump(pipeline, model_file)

    assert model_path.exists()
