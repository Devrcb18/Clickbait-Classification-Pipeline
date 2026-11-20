from pathlib import Path
import joblib
import pandas as pd
import pytest
import yaml

#--------------------------- yaml and config existence ---------------------
@pytest.fixture(scope="session")
def root()-> Path:
    return Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def params(root)-> dict:
    with open(root / "params.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

@pytest.fixture(scope="session")
def raw_data(root)-> pd.DataFrame:
    return pd.read_csv(root / "data_files" / "raw" / "clickbait_title_classification.csv")
