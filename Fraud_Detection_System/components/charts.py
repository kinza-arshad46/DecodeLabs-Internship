"""
components/charts.py
-----------------------
Thin Streamlit rendering wrappers. All the actual figure-building logic
lives in utils/visualization.py (kept Streamlit-free so it's unit
testable) - this module just calls st.plotly_chart on the returned Figure.
"""

import streamlit as st

from utils import visualization as viz


def render_class_distribution(df):
    st.plotly_chart(viz.class_distribution_donut(df), use_container_width=True)


def render_amount_distribution(df):
    st.plotly_chart(viz.amount_distribution_histogram(df), use_container_width=True)


def render_correlation_heatmap(df):
    st.plotly_chart(viz.correlation_heatmap(df), use_container_width=True)


def render_top_correlated_table(df, top_n: int = 8):
    top_df = viz.top_correlated_features(df, top_n=top_n)
    st.dataframe(top_df, use_container_width=True, hide_index=True, height=380)


def render_smote_bar(y_series, labels: dict, colors: dict):
    st.plotly_chart(viz.smote_class_bar(y_series, labels, colors), use_container_width=True)


def render_roc_comparison(models_metrics: dict):
    st.plotly_chart(viz.roc_curve_comparison(models_metrics), use_container_width=True)
