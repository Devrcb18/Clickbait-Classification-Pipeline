from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse, load_npz

EXPECTED_COLUMNS = {"title", "clickbait"}
PROBABILITY_TOLERANCE = 1e-6


@pytest.mark.parametrize(
    "relative_path",
    [
        "params.yaml",
        "dvc.yaml",
        "data_files/raw/clickbait_title_classification.csv",
        "data_files/processed/x_train.npz",
        "data_files/processed/x_test.npz",
        "data_files/processed/y_train.npy",
        "data_files/processed/y_test.npy",
        "models/tfidf_vectorizer.joblib",
        "models/clickbait_xgb.joblib",
    ],
)
def test_project_files_exist(root: Path, relative_path: str):
    assert (root / relative_path).is_file(), f"Missing project file: {relative_path}"


def test_raw_data_loads(raw_data):
    assert not raw_data.empty
    assert EXPECTED_COLUMNS.issubset(raw_data.columns)
    assert raw_data["title"].notna().all()
    assert raw_data["clickbait"].isin([0, 1]).all()
    assert set(raw_data["clickbait"].unique()) == {0, 1}


def test_raw_data_is_balanced(raw_data):
    counts = raw_data["clickbait"].value_counts(normalize=True)
    assert (counts > 0.4).all()


def test_params_are_valid(params: dict):
    assert 0 < params["data_ingestion"]["test_size"] < 1
    assert params["feature_engineering"]["max_features"] > 0
    building = params["model_building"]
    assert building["max_depth_lower_limit"] < building["max_depth_upper_limit"]
    assert building["learning_rate_lower_limit"] < building["learning_rate_upper_limit"]
    assert building["random_state"] == 42


def test_processed_splits_load(root: Path, params: dict, raw_data):
    x_train = load_npz(root / "data_files" / "processed" / "x_train.npz")
    x_test = load_npz(root / "data_files" / "processed" / "x_test.npz")
    y_train = np.load(root / "data_files" / "processed" / "y_train.npy")
    y_test = np.load(root / "data_files" / "processed" / "y_test.npy")

    assert issparse(x_train) and issparse(x_test)
    assert x_train.shape[1] == x_test.shape[1] == params["feature_engineering"]["max_features"]
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    assert x_train.shape[0] + x_test.shape[0] == len(raw_data)

    test_size = params["data_ingestion"]["test_size"]
    assert x_test.shape[0] / len(raw_data) == pytest.approx(test_size, abs=0.01)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_vectorizer_and_model_load(artifacts, params: dict):
    vectorizer, model = artifacts
    assert hasattr(vectorizer, "transform")
    assert hasattr(model, "predict_proba")
    assert len(vectorizer.get_feature_names_out()) == params["feature_engineering"]["max_features"]


def test_model_returns_valid_probabilities(artifacts):
    vectorizer, model = artifacts
    matrix = vectorizer.transform(["You Won't Believe What This Celebrity Did Next"])
    probabilities = model.predict_proba(matrix)

    assert probabilities.shape == (1, 2)
    assert probabilities.sum(axis=1) == pytest.approx(1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_separates_clickbait_from_plain_titles(artifacts):
    vectorizer, model = artifacts
    titles = [
        "You Won't Believe What This Celebrity Did Next",
        "Scientists publish findings on deep-sea coral restoration",
    ]
    clickbait_probabilities = model.predict_proba(vectorizer.transform(titles))[:, 1]
    assert clickbait_probabilities[0] > 0.5
    assert clickbait_probabilities[0] > clickbait_probabilities[1]


def test_transform_handles_unseen_text(artifacts):
    vectorizer, _ = artifacts
    matrix = vectorizer.transform(["zzzz unknown token sdkjfhsdkjfh"])
    assert matrix.shape[0] == 1
    assert matrix.nnz >= 0


def test_streamlit_app_renders_without_errors(root: Path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(root / "src" / "api" / "main.py"), default_timeout=120)
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Analyze", "Batch scoring", "Model overview"]
    assert app.button
