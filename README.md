# Clickbait Classifier

An end-to-end NLP based ML project that detects clickbait headlines using **TF-IDF features** and an **XGBoost** classifier, with a **Streamlit** interface for single-title and batch scoring.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange.svg)](https://xgboost.readthedocs.io/)
[![DVC](https://img.shields.io/badge/DVC-pipeline-13ADC7.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-20B2AA.svg)](https://dagshub.com/Devrcb18/fraud-detection-project-)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/aryansinha2003/click-bait-title-classification)

---

## Overview

Clickbait headlines are engineered to attract clicks rather than inform. This project trains a binary text classifier on 32,000 labeled headlines and exposes it through a polished Streamlit app that can:

- Score a single headline and show the clickbait probability with an interpretable confidence meter.
- Highlight the highest-weight TF-IDF terms for the input text.
- Keep a short in-session history of recent predictions.
- Score a CSV of headlines in batch with an adjustable decision threshold and CSV export.
- Show the training configuration and model details.

The pipeline is reproducible end to end with **DVC**, and experiments are tracked with **MLflow** (via DagsHub).

---
## Dagshub Link <https://dagshub.com/Devrcb18/fraud-detection-project->
## Dataset

| Property | Value |
| --- | --- |
| File | <https://www.kaggle.com/datasets/aryansinha2003/click-bait-title-classification> |
| Rows | 32,000 |
| Columns | `title` (headline text), `clickbait` (0 = not clickbait, 1 = clickbait) |
| Class balance | 16,001 negative / 15,999 positive |
| Mean title length | ~54 characters |
| Train/test split | 80% / 20% (random state 42) |

The raw CSV is versioned with DVC rather than committed to Git.

---

## Model Pipeline

1. **Ingestion** (`src/data/ingestion.py`)
   - Loads the raw CSV, splits it into train/test sets, fits a `TfidfVectorizer` (max 5,000 features), and saves the vectorizer plus sparse splits to `data_files/processed/` and `models/`.
2. **Model building** (`src/models/model_building.py`)
   - Loads the processed splits and runs a Hyperopt TPE search (30 evaluations) over XGBoost `max_depth`, `learning_rate`, and `n_estimators`.
   - Trains the final XGBoost classifier with the best parameters, logs params/metrics/artifacts to MLflow, and saves `models/clickbait_xgb.joblib`.
3. **Serving** (`src/api/main.py`)
   - Streamlit app that loads the saved vectorizer and model and performs inference.

### Evaluation (saved model on hold-out split)

| Metric | Score |
| --- | --- |
| Accuracy | 0.896 |
| F1 score | 0.890 |
| Precision | 0.946 |
| Recall | 0.840 |

---

## Project Structure

```
Clickbait classification/
├── src/
│   ├── api/main.py              # Streamlit application
│   ├── data/ingestion.py        # Data loading, split, TF-IDF, save
│   └── models/model_building.py # Hyperopt + XGBoost + MLflow
├── data_files/
│   ├── raw/                     # DVC-tracked source CSV
│   └── processed/               # Sparse feature matrices and labels
├── models/                      # Vectorizer and trained model artifacts
├── logs/                        # Pipeline logs
├── .streamlit/config.toml       # App theme
├── params.yaml                  # Pipeline and model configuration
├── dvc.yaml / dvc.lock          # Reproducible pipeline definition
├── requirements.txt             # Runtime/app dependencies
├── Dockerfile                   # Container image for the app
└── clickbait-classification.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git, Docker (optional), and DVC for the reproducible pipeline
- Access to the configured DagsHub remote if you want the DVC-tracked data and MLflow tracking

### Install

```bash
git clone <repository-url>
cd "Clickbait classification"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` covers the Streamlit app and inference. To run the full training pipeline, also install:

```bash
pip install mlflow dagshub hyperopt scipy matplotlib seaborn
```

### Fetch data and reproduce the pipeline

```bash
dvc pull          # fetch DVC-tracked raw data and artifacts
dvc repro         # run ingestion -> model building
```

You can also run stages directly:

```bash
python src/data/ingestion.py
python src/models/model_building.py
```

MLflow tracking is configured to the project's DagsHub repository; set DagsHub credentials in your environment if prompted.

### Run the app

```bash
streamlit run src/api/main.py
```

Open <http://localhost:8501>.

---

## Run with Docker

```bash
docker build -t clickbait-app .
docker run -p 8501:8501 clickbait-app
```

The image includes the trained artifacts from `models/`, so build it after running the pipeline (or pulling via `dvc pull`). A health check is exposed at `/_stcore/health`.

---

## Configuration

Model and pipeline settings live in `params.yaml`:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `data_ingestion.test_size` | 0.20 | Hold-out fraction |
| `feature_engineering.max_features` | 5000 | TF-IDF vocabulary cap |
| `model_building.max_depth_lower_limit` | 3 | Hyperopt lower bound |
| `model_building.max_depth_upper_limit` | 10 | Hyperopt upper bound |
| `model_building.learning_rate_lower_limit` | 0.0003 | Log-uniform lower bound |
| `model_building.learning_rate_upper_limit` | 0.3 | Log-uniform upper bound |
| `model_building.num_estimator_choices` | `[3, 5, 7, 10]` | Candidate estimator counts |
| `model_building.eval_metric` | `logloss` | XGBoost evaluation metric |
| `model_building.random_state` | 42 | Reproducibility seed |

Changing parameters and running `dvc repro` re-executes only the affected stages.

---

## App Details

The Streamlit app has three tabs:

- **Analyze** — enter/paste a headline or pick a sample; get a color-coded verdict (likely clickbait / borderline / likely not clickbait), probability meter, confidence, and top weighted TF-IDF terms.
- **Batch scoring** — upload a CSV containing a `title` column; tune the decision threshold, review summary metrics, and download predictions.
- **Model overview** — vocabulary size, estimator count, pipeline summary, and raw configuration.

Predictions are probabilistic estimates and are meant to support, not replace, editorial judgement.

---

## Tech Stack

- **scikit-learn** — TF-IDF vectorization, metrics, splitting
- **XGBoost** — gradient-boosted classifier
- **Hyperopt** — TPE hyperparameter optimization
- **MLflow + DagsHub** — experiment tracking and remote storage
- **DVC** — data and pipeline versioning
- **Streamlit** — interactive UI
- **Docker** — reproducible deployment

---

## Notes

- The model artifacts in `models/` are generated outputs; rerun the pipeline if they are absent.
- The app degrades gracefully with an explanatory message if `models/tfidf_vectorizer.joblib` or `models/clickbait_xgb.joblib` is missing.
- Logs from pipeline runs are written to `logs/data_ingestion.log`.
