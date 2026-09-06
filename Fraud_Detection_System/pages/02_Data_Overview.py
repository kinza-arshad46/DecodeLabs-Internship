import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.metric_cards import render_metric_row
from components.charts import (
    render_class_distribution,
    render_amount_distribution,
    render_correlation_heatmap,
    render_top_correlated_table,
)
from utils.data_loader import dataset_summary

render_sidebar("Data Overview")
df = st.session_state.df

st.markdown("## Data Overview")
if not st.session_state.is_real_data:
    st.info(
        "Real dataset not found in `data/raw/` — showing a synthetic demo dataset "
        "with the same schema and imbalance ratio. See the About page for how to "
        "plug in the real Kaggle dataset."
    )

summary = dataset_summary(df)
render_metric_row([
    ("Total Transactions", f"{summary['total']:,}", ""),
    ("Fraudulent Transactions", f"{summary['fraud']:,} ({summary['fraud_pct']}%)", "danger"),
    ("Normal Transactions", f"{summary['normal']:,} ({summary['normal_pct']}%)", "success"),
    ("Features", f"{summary['features']}", ""),
    ("Missing Values", f"{summary['missing']}", ""),
])

st.write("")
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Class Distribution")
    render_class_distribution(df)
with col2:
    st.markdown("##### Transaction Amount Distribution")
    render_amount_distribution(df)

col3, col4 = st.columns([1.4, 1])
with col3:
    st.markdown("##### Feature Correlation Heatmap")
    render_correlation_heatmap(df)
with col4:
    st.markdown("##### Top Correlated Features with Class")
    render_top_correlated_table(df)
