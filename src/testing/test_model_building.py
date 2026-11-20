import pytest
from pathlib import Path
from xgboost import XGBClassifier

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
    
    
    