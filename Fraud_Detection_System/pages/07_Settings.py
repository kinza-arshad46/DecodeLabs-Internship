import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar, SESSION_DEFAULTS

render_sidebar("Settings")
df = st.session_state.df

st.markdown("## Settings")

st.markdown("##### Classification Threshold")
st.session_state.threshold = st.slider(
    "Probability above which a transaction is flagged as fraud",
    0.05, 0.95, st.session_state.threshold, 0.05,
)
st.caption(
    "Lower threshold → catches more fraud but more false alarms. "
    "Higher threshold → fewer false alarms but may miss fraud."
)

st.markdown("---")
st.markdown("##### Data")
st.write(f"Active dataset: {'Real (data/raw/)' if st.session_state.is_real_data else 'Synthetic demo data'}")
st.write(f"Rows loaded: {len(df):,}")

st.markdown("---")
st.markdown("##### Reset")
if st.button("🔄 Reset app state (clear trained models & splits)"):
    for key in ["X_train", "X_test", "y_train", "y_test", "smote_X", "smote_y", "trained_models", "pred_input"]:
        default_val = SESSION_DEFAULTS[key]
        st.session_state[key] = default_val.copy() if isinstance(default_val, dict) else default_val
    st.success("App state cleared.")
    st.rerun()
