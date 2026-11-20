import pytest
from pathlib import Path
from src.models.model_building import load_params 
from xgboost import XGBClassifier
#-------------------model testing for parameters data type----------
def test_model_parameters_have_expected_types(artifacts):
    _, model = artifacts
    model_params = model.get_params()
    assert isinstance(model_params["max_depth"], int)
    assert isinstance(model_params["learning_rate"], float)
    assert isinstance(model_params["n_estimators"], int)
    assert model_params["random_state"] == 42

#----------------- model checking -----------------------------
def test_model_is_xgboost_classifier(artifacts):
    _, model = artifacts
    assert isinstance(model, XGBClassifier)

# ----------------- model artifact existence ----------------------    
def test_file_saving():
    model_path =Path("models/clickbait_xgb.joblib")
    assert model_path.is_file()
    vectorizer_path = Path("models/clickbait_xgb.joblib")
    assert vectorizer_path.is_file()
    
    
    