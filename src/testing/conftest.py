from pathlib import Path
import joblib
import pandas as pd
import pytest
import yaml
import numpy as np
from scipy.sparse import load_npz

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

@pytest.fixture(scope="session")
def artifacts(root):
    model = joblib.load(root / "models" / "clickbait_xgb.joblib")
    x_test = load_npz(root / "data_files" / "processed" / "x_test.npz")
    y_test = np.load(root / "data_files" / "processed" / "y_test.npy")
    return (x_test, y_test), model