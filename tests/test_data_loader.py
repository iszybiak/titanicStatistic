import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data

def test_load_data(tmp_path):
    test_file = tmp_path / "test_data.csv"
    test_file.write_text("Name,Age,Fare,Embarked\nJohn,22,7.25,S\nAlice,35,71.83,C")

    df = load_data(str(test_file))

    assert df.shape == (2, 4)  # Powinny być 2 wiersze i 4 kolumny
    assert list(df.columns) == ["Name", "Age", "Fare", "Embarked"]


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
