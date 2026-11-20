import pytest
import yaml
from pathlib import Path

#--------------------------- yaml and config existence ---------------------
@pytest.fixture(scope="session")
def root()-> Path:
    return Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def params(root)-> dict:
    with open(root / "params.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def test_params_file_exists(root):
    assert (root / "params.yaml").is_file()

def test_params_structure(params):
    required_sections = ["data_ingestion", "feature_engineering", "model_building"]
    for section in required_sections:
        assert section in params, f"Missing section: {section}"

def test_data_ingestion_params(params):
    di = params["data_ingestion"]
    assert "test_size" in di
    assert isinstance(di["test_size"], (int, float))
    assert 0 < di["test_size"] < 1
    assert "file_path" in di
    assert isinstance(di["file_path"], str)

def test_feature_engineering_params(params):
    fe = params["feature_engineering"]
    assert "max_features" in fe
    assert isinstance(fe["max_features"], int)
    assert fe["max_features"] > 0

def test_model_building_params(params):
    mb = params["model_building"]
    assert "max_depth_upper_limit" in mb
    assert "max_depth_lower_limit" in mb
    assert "learning_rate_upper_limit" in mb
    assert "learning_rate_lower_limit" in mb
    assert "num_estimator_choices" in mb
    assert "eval_metric" in mb
    assert "random_state" in mb
    
    assert mb["max_depth_lower_limit"] < mb["max_depth_upper_limit"]
    assert mb["learning_rate_lower_limit"] < mb["learning_rate_upper_limit"]
    assert isinstance(mb["num_estimator_choices"], list)
    assert len(mb["num_estimator_choices"]) > 0
    assert all(isinstance(n, int) and n > 0 for n in mb["num_estimator_choices"])
    assert mb["random_state"] == 42
    assert mb["eval_metric"] == "logloss"