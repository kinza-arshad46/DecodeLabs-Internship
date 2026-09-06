"""
app.py
--------
Streamlit app entry point.

This file only handles global page config and then hands off to
pages/01_Dashboard.py, which is the actual landing page. All feature
pages live under pages/ and share state via components/sidebar.py's
init_session_state().

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/01_Dashboard.py")
