import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.metric_cards import render_badges, render_info_card

render_sidebar("Dashboard")

col_text, col_icon = st.columns([2.2, 1])
with col_text:
    st.caption("Project 2 · Supervised Learning")
    st.markdown("# Fraud Detection Pipeline")
    st.markdown(
        "Build and tune classification models to detect fraudulent "
        "transactions in a highly imbalanced dataset."
    )
    render_badges(["SMOTE", "Logistic Regression", "Random Forest", "Precision", "Recall", "ROC-AUC"])
    st.write("")
    if st.button("Start Pipeline →", type="primary"):
        st.switch_page("pages/02_Data_Overview.py")

with col_icon:
    render_info_card(
        '<div style="text-align:center; padding-top:20px; padding-bottom:20px;">'
        '<span style="font-size:4rem;">🖥️🔒</span></div>'
    )

st.write("")
c1, c2, c3, c4 = st.columns(4)
feature_cards = [
    (c1, "🧮", "Imbalanced Data Handling", "Using SMOTE"),
    (c2, "🧠", "Multiple Models Training", "Logistic Regression & Random Forest"),
    (c3, "📈", "Proper Evaluation Metrics", "Precision, Recall & ROC-AUC"),
    (c4, "🎛️", "Hyperparameter Tuning", "Find the best model"),
]
for col, icon, title, desc in feature_cards:
    with col:
        render_info_card(
            f'<div style="font-size:1.4rem;">{icon}</div>'
            f'<div style="font-weight:700; margin-top:6px;">{title}</div>'
            f'<div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">{desc}</div>'
        )

st.write("")
st.markdown("#### How the pipeline works")
steps = [
    "1. Load Data", "2. EDA", "3. Preprocessing", "4. Handle Imbalance",
    "5. Train Models", "6. Evaluate", "7. Compare & Tune", "8. Predict",
]
cols = st.columns(len(steps))
for c, s in zip(cols, steps):
    with c:
        render_info_card(f"<div style='text-align:center; font-size:0.8rem;'>{s}</div>")
