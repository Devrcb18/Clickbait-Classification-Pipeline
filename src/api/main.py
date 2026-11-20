from pathlib import Path
import html

import joblib
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "clickbait_xgb.joblib"
PARAMS_PATH = ROOT / "params.yaml"

SAMPLES = [
    "You Won't Believe What This Celebrity Did Next",
    "Government announces new policy on renewable energy subsidies",
    "This One Weird Trick Will Make You Rich Overnight",
    "Scientists publish findings on deep-sea coral restoration",
    "Doctors Hate Her! Discover the Secret They're Hiding",
]

st.set_page_config(
    page_title="Clickbait Classifier",
    page_icon="\U0001F9F2",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1100px; }
        .hero {
            background: linear-gradient(120deg, #1f2933 0%, #323f4b 55%, #3e4c59 100%);
            border-radius: 18px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.4rem;
            color: #f5f7fa;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -0.5px; }
        .hero p { margin: 0.4rem 0 0; color: #cbd2d9; font-size: 1rem; max-width: 720px; }
        .hero .tag {
            display: inline-block;
            margin-bottom: 0.7rem;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            color: #ffd166;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .result-card {
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin: 1rem 0;
            background: #ffffff;
            border: 1px solid #e4e7eb;
            border-left: 6px solid #9aa5b1;
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.06);
        }
        .result-card.high { border-left-color: #e12d39; }
        .result-card.mid { border-left-color: #f0b429; }
        .result-card.low { border-left-color: #27ab83; }
        .result-verdict { font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }
        .result-card.high .result-verdict { color: #e12d39; }
        .result-card.mid .result-verdict { color: #b07d00; }
        .result-card.low .result-verdict { color: #21806a; }
        .result-title { font-size: 1.15rem; font-weight: 600; color: #1f2933; margin: 0.35rem 0 0.9rem; }
        .meter { height: 12px; border-radius: 999px; background: #eef1f5; overflow: hidden; }
        .meter-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
        .meter-fill.high { background: linear-gradient(90deg, #f29b9b, #e12d39); }
        .meter-fill.mid { background: linear-gradient(90deg, #ffd166, #f0b429); }
        .meter-fill.low { background: linear-gradient(90deg, #8eedc7, #27ab83); }
        .result-meta { margin-top: 0.75rem; color: #616e7c; font-size: 0.9rem; }
        .result-meta strong { color: #1f2933; }
        .chips { margin-top: 0.4rem; }
        .chip {
            display: inline-block;
            margin: 0 0.35rem 0.35rem 0;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: #eef1f5;
            color: #3e4c59;
            font-size: 0.8rem;
        }
        .chip span { color: #9aa5b1; }
        .footer-note { color: #9aa5b1; font-size: 0.8rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


@st.cache_data(show_spinner=False)
def load_params():
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def predict_titles(vectorizer, model, titles):
    matrix = vectorizer.transform(titles)
    probabilities = model.predict_proba(matrix)[:, 1]
    return matrix, probabilities


def top_terms(vectorizer, matrix, limit=6):
    row = matrix.max(axis=0).toarray().ravel()
    names = vectorizer.get_feature_names_out()
    ranked = row.argsort()[-limit:][::-1]
    return [(names[i], float(row[i])) for i in ranked if row[i] > 0]


def verdict_for(probability):
    if probability >= 0.66:
        return "high", "Likely clickbait"
    if probability >= 0.33:
        return "mid", "Borderline"
    return "low", "Likely not clickbait"


def render_result(title, probability, terms):
    level, verdict = verdict_for(probability)
    percent = probability * 100
    confidence = max(probability, 1 - probability) * 100
    chips = "".join(
        f'<span class="chip">{html.escape(term)} <span>{weight:.2f}</span></span>'
        for term, weight in terms
    )
    chips_block = f'<div class="chips">{chips}</div>' if chips else ""
    st.markdown(
        f"""
        <div class="result-card {level}">
            <div class="result-verdict">{verdict}</div>
            <div class="result-title">{html.escape(title)}</div>
            <div class="meter"><div class="meter-fill {level}" style="width: {percent:.1f}%"></div></div>
            <div class="result-meta">
                Clickbait probability <strong>{percent:.1f}%</strong>
                &nbsp;&middot;&nbsp; model confidence <strong>{confidence:.1f}%</strong>
            </div>
            {chips_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_artifacts_missing():
    st.error(
        "Model artifacts were not found. Generate them first with `dvc repro` "
        "(or run `python src/data/ingestion.py` and `python src/models/model_building.py`)."
    )
    missing = [str(path.relative_to(ROOT)) for path in (VECTORIZER_PATH, MODEL_PATH) if not path.exists()]
    st.caption("Missing: " + ", ".join(f"`{path}`" for path in missing))
    st.stop()


def single_tab(vectorizer, model):
    st.subheader("Analyze a headline")
    st.caption("Paste a title and the TF-IDF + XGBoost model will score how clickbait it looks.")

    if "title_input" not in st.session_state:
        st.session_state.title_input = ""

    st.write("**Try an example**")
    columns = st.columns(len(SAMPLES))
    for column, sample in zip(columns, SAMPLES):
        label = sample if len(sample) <= 22 else sample[:21] + "\u2026"
        if column.button(label, key=f"sample_{sample}", help=sample, width="stretch"):
            st.session_state.title_input = sample

    title = st.text_area(
        "Headline or article title",
        key="title_input",
        height=100,
        placeholder="e.g. You Won't Believe What Happened Next",
    )

    if st.button("Analyze headline", type="primary", width="stretch"):
        cleaned = title.strip()
        if not cleaned:
            st.warning("Enter a headline first.")
            return
        matrix, probabilities = predict_titles(vectorizer, model, [cleaned])
        probability = float(probabilities[0])
        render_result(cleaned, probability, top_terms(vectorizer, matrix))

        history = st.session_state.setdefault("history", [])
        history.insert(0, {"Title": cleaned, "Clickbait probability": round(probability, 4)})
        del history[10:]

    history = st.session_state.get("history", [])
    if history:
        with st.expander("Recent predictions", expanded=False):
            st.dataframe(
                pd.DataFrame(history),
                width="stretch",
                hide_index=True,
                column_config={
                    "Clickbait probability": st.column_config.ProgressColumn(
                        "Clickbait probability",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    )
                },
            )


def batch_tab(vectorizer, model):
    st.subheader("Score a batch of headlines")
    st.caption("Upload a CSV with a `title` column to classify every row at once.")

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return

    title_column = next((column for column in frame.columns if column.strip().lower() == "title"), None)
    if title_column is None:
        st.error("The CSV needs a column named `title`.")
        return

    titles = frame[title_column].fillna("").astype(str)
    valid = titles.str.strip().str.len() > 0
    if not valid.any():
        st.warning("No non-empty titles found in the file.")
        return

    matrix, probabilities = predict_titles(vectorizer, model, titles[valid].tolist())
    results = frame.loc[valid].copy()
    results["clickbait_probability"] = probabilities.round(4)

    threshold = st.slider("Clickbait decision threshold", 0.05, 0.95, 0.5, 0.05)
    results["verdict"] = ["Clickbait" if p >= threshold else "Not clickbait" for p in probabilities]

    total = len(results)
    flagged = int((results["clickbait_probability"] >= threshold).sum())
    metric_columns = st.columns(3)
    metric_columns[0].metric("Titles scored", total)
    metric_columns[1].metric("Flagged as clickbait", flagged)
    metric_columns[2].metric("Flagged share", f"{flagged / total:.0%}" if total else "0%")

    st.dataframe(
        results[[title_column, "clickbait_probability", "verdict"]],
        width="stretch",
        hide_index=True,
        column_config={
            "clickbait_probability": st.column_config.ProgressColumn(
                "Clickbait probability",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            title_column: st.column_config.TextColumn("Title", width="large"),
        },
    )

    st.download_button(
        "Download results as CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="clickbait_predictions.csv",
        mime="text/csv",
        width="stretch",
    )


def model_tab(vectorizer, model):
    st.subheader("Model overview")
    params = load_params()
    feature_count = len(vectorizer.get_feature_names_out())
    model_params = model.get_params()

    columns = st.columns(3)
    columns[0].metric("TF-IDF features", f"{feature_count:,}")
    columns[1].metric("Vocabulary cap", params.get("feature_engineering", {}).get("max_features", "n/a"))
    columns[2].metric("Estimators", model_params.get("n_estimators", "n/a"))

    st.markdown("**Inference pipeline**")
    st.markdown(
        "1. Headline is cleaned into a single text field.\n"
        "2. The trained TF-IDF vectorizer converts it to a sparse feature vector.\n"
        "3. The XGBoost classifier returns a clickbait probability, thresholded for the verdict."
    )

    st.markdown("**Training configuration**")
    st.json(params)
    st.caption(
        "Training metrics and artifacts are tracked with MLflow; the DVC pipeline rebuilds "
        "the vectorizer and model from `params.yaml`."
    )


def main():
    st.markdown(
        """
        <div class="hero">
            <span class="tag">NLP &middot; TF-IDF + XGBoost</span>
            <h1>Clickbait Classifier</h1>
            <p>Score headlines for clickbait risk with the trained XGBoost model from this project.
            Analyze a single title or upload a CSV for batch scoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        render_artifacts_missing()

    vectorizer, model = load_artifacts()

    with st.sidebar:
        st.header("Artifacts")
        st.success(f"Vectorizer loaded\n\n`models/{VECTORIZER_PATH.name}`")
        st.success(f"Model loaded\n\n`models/{MODEL_PATH.name}`")
        params = load_params()
        test_size = params.get("data_ingestion", {}).get("test_size")
        if test_size is not None:
            st.info(f"Train/test split: {1 - float(test_size):.0%} / {float(test_size):.0%}")
        st.divider()
        st.caption(
            "Source: `src/api/main.py`\n\n"
            "Run locally:\n\n"
            "`streamlit run src/api/main.py`\n\n"
            "Container:\n\n"
            "`docker build -t clickbait-app .`\n\n"
            "`docker run -p 8501:8501 clickbait-app`"
        )

    analyze_tab, batch_tab_view, model_tab_view = st.tabs(
        ["Analyze", "Batch scoring", "Model overview"]
    )
    with analyze_tab:
        single_tab(vectorizer, model)
    with batch_tab_view:
        batch_tab(vectorizer, model)
    with model_tab_view:
        model_tab(vectorizer, model)

    st.markdown(
        '<div class="footer-note">Predictions are probabilistic estimates from a TF-IDF + XGBoost '
        "classifier and should support, not replace, editorial judgement.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
