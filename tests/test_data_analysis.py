import pandas as pd
import sys
import os
from src.data_analysis import data_analysis
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def test_data_analysis(tmp_path):
    data = pd.DataFrame({
        "Age": [22, 35, 40, 50, 29],
        "Fare": [7.25, 71.83, 8.05, 15.50, 23.00],
        "Embarked": ["S", "C", "Q", "S", "C"]
    })

    processed_data = data_analysis(data)

    assert "Age" in processed_data.columns
    assert "Fare" in processed_data.columns
