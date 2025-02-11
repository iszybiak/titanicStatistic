import pytest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config


def test_load_config(tmp_path):
    test_config_file = tmp_path / "config.json"
    test_config_file.write_text(json.dumps({"data_path": "data.csv", "model_path": "model.pkl"}))

    config = load_config(str(test_config_file))

    assert config["data_path"] == "data.csv"
    assert config["model_path"] == "model.pkl"


def test_load_config_invalid_json(tmp_path):
    test_config_file = tmp_path / "invalid_config.json"
    test_config_file.write_text("{invalid_json}")  # Niepoprawna składnia JSON

    with pytest.raises(ValueError):
        load_config(str(test_config_file))
