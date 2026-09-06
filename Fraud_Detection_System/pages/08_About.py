import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.metric_cards import render_info_card, render_badges

render_sidebar("About")

st.markdown("## About This Project")
render_info_card(
    "<b>Fraud Detection Pipeline</b> — Project 2 of the DecodeLabs Data Science "
    "Industrial Training Kit (Supervised Learning track).<br><br>"
    "<b>Goal:</b> Build and tune a classification model to identify fraudulent "
    "transactions in a highly imbalanced dataset, using a leak-free pipeline."
)

st.markdown("##### Tech Stack")
render_badges([
    "Python", "Streamlit", "scikit-learn", "imbalanced-learn (SMOTE)",
    "Plotly", "Pandas", "NumPy", "Joblib",
])

st.markdown("##### Architecture")
st.markdown(
    """
    `Streamlit UI (pages/ + components/)` → `Preprocessing (utils/preprocessing.py)`
    → `Imbalance handling (utils/smote_handler.py)`
    → `Model layer (utils/model_trainer.py — imblearn Pipeline: Logistic Regression / Random Forest)`
    → `Evaluation (utils/evaluator.py — Precision, Recall, ROC-AUC)`
    → `Prediction (utils/predictor.py)` → `Persistence (models/*.pkl)`
    """
)

st.markdown("##### Dataset")
st.markdown(
    """
    Designed for the **Credit Card Fraud Detection** dataset (Kaggle:
    `mlg-ulb/creditcardfraud`) — 284,807 transactions, 492 fraud cases (0.17%).
    Download `creditcard.csv` and place it in `data/raw/`.
    If it isn't found, the app automatically falls back to a synthetic
    dataset with the same schema so it always stays runnable.
    """
)

st.markdown("##### Key Engineering Decisions")
st.markdown(
    """
    - SMOTE and scaling live **inside** an `imblearn.pipeline.Pipeline`, fit only on training folds — no leakage.
    - Models are compared using **Precision, Recall, F1 and ROC-AUC**, never raw accuracy (99.8% accuracy means nothing on this dataset).
    - `GridSearchCV` with `StratifiedKFold` tunes hyperparameters without ever touching the held-out test set.
    - `utils/` holds pure logic with no Streamlit dependency (unit-testable), `components/` holds the Streamlit rendering — a standard separation for production ML apps.
    - Trained pipelines persist to `models/*.pkl`, so Predictions still works after an app restart without retraining.
    """
)

st.markdown("---")
st.caption("Built by Kinza Arshad · Data Science Undergraduate, KFUEIT · DecodeLabs Batch 2026")
