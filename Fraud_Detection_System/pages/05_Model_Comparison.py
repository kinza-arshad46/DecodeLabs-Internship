import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.charts import render_roc_comparison
from utils.model_trainer import save_best_model

render_sidebar("Model Comparison")

st.markdown("## Model Comparison")

if len(st.session_state.trained_models) < 1:
    st.warning("Train at least one model on the **Model Training** page first.")
else:
    rows = []
    metrics_for_roc = {}
    for name, result in st.session_state.trained_models.items():
        m = result["metrics"]
        metrics_for_roc[name] = m
        rows.append({
            "Metric": name,
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1-Score": round(m["f1"], 3),
            "ROC-AUC": round(m["roc_auc"], 3),
            "False Positives": m["false_positives"],
            "False Negatives": m["false_negatives"],
        })
    comp_df = pd.DataFrame(rows).set_index("Metric").T

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("##### Metric Table")
        st.dataframe(comp_df, use_container_width=True)
    with col2:
        st.markdown("##### ROC Curve Comparison")
        render_roc_comparison(metrics_for_roc)

    best_model_name = max(
        st.session_state.trained_models,
        key=lambda n: st.session_state.trained_models[n]["metrics"]["roc_auc"],
    )
    st.success(f"🏆 Best model by ROC-AUC: **{best_model_name}**")

    if st.button("💾 Save best model as models/best_model.pkl"):
        best_result = st.session_state.trained_models[best_model_name]
        path = save_best_model(best_result["model"], best_model_name)
        st.success(f"Saved to `{path}`")
