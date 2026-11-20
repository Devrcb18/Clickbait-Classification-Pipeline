import pytest
from pathlib import Path
from xgboost import XGBClassifier

#----------------- model checking -----------------------------
def test_model_is_xgboost_classifier(artifacts):
    _, model = artifacts
    assert isinstance(model, XGBClassifier)

    
    
    