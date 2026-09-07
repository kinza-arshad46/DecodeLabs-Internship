"""
components/sidebar.py
------------------------
Renders the custom sidebar navigation (replacing Streamlit's default
multipage nav, which is hidden via CSS - see assets/style/style.css).

This module ALSO initializes the shared session state (dataset, trained
models, threshold) the first time any page runs. Streamlit only executes
app.py's top-level code once, at the root URL - when a user navigates
directly to a page under pages/, only that page's script runs. Since every
page calls render_sidebar() first, this is the one place guaranteed to run
on every page load, making it the natural spot for shared setup.
"""

import os
import streamlit as st

from utils.data_loader import load_data

CSS_PATH = os.path.join("assets", "style", "style.css")


def _load_css():
    # Re-injected on every page run (not cached in session_state) because
    # Streamlit re-renders each page's DOM from scratch on navigation.
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PAGES = [
    ("pages/01_Dashboard.py", "🏠", "Dashboard"),
    ("pages/02_Data_Overview.py", "📊", "Data Overview"),
    ("pages/03_Preprocessing.py", "🧪", "Preprocessing"),
    ("pages/04_Model_Training.py", "🧠", "Model Training"),
    ("pages/05_Model_Comparison.py", "⚖️", "Model Comparison"),
    ("pages/06_Predictions.py", "🔮", "Predictions"),
    ("pages/07_Settings.py", "⚙️", "Settings"),
    ("pages/08_About.py", "ℹ️", "About"),
]

SESSION_DEFAULTS = {
    "df": None,
    "is_real_data": False,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "smote_X": None, "smote_y": None,
    "trained_models": {},
    "threshold": 0.5,
    "pred_input": {},
}


def init_session_state():
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val.copy() if isinstance(val, dict) else val

    if st.session_state.df is None:
        df, is_real = load_data()
        st.session_state.df = df
        st.session_state.is_real_data = is_real


def render_sidebar(active_label: str):
    _load_css()
    init_session_state()

    with st.sidebar:
        st.markdown("### 🛡️ Fraud Detection Pipeline")
        st.caption("Project 2 — Supervised Learning")
        st.markdown("---")

        for path, icon, label in PAGES:
            if label == active_label:
                st.markdown(
                    f"<div style='background:#1e293b; border-radius:8px; padding:8px 12px; "
                    f"margin-bottom:2px; font-weight:600;'>{icon} &nbsp; {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(path, label=label, icon=icon)

        st.markdown("---")
        n_trained = len(st.session_state.trained_models)
        st.caption(f"Models trained: {n_trained}/2")
        if not st.session_state.is_real_data:
            st.caption("⚠️ Using synthetic demo data")
        else:
            st.caption("✅ Using real dataset")
        st.caption("Built by Kinza Arshad · KFUEIT")
