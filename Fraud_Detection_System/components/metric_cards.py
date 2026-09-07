"""
components/metric_cards.py
-----------------------------
Styled metric cards and badges, used across the Dashboard, Data Overview
and Model Training pages.
"""

import streamlit as st


def metric_card_html(label: str, value: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value{tone_class}">{value}</div>
    </div>
    """


def render_metric_row(metrics: list):
    """
    metrics: list of (label, value, tone) tuples. tone is "", "danger",
    "success" or "info".
    """
    cols = st.columns(len(metrics))
    for col, (label, value, tone) in zip(cols, metrics):
        col.markdown(metric_card_html(label, value, tone), unsafe_allow_html=True)


def badge_html(text: str) -> str:
    return f'<span class="badge">{text}</span>'


def render_badges(items: list):
    st.markdown("".join(badge_html(b) for b in items), unsafe_allow_html=True)


def render_info_card(content_html: str):
    st.markdown(f'<div class="info-card">{content_html}</div>', unsafe_allow_html=True)
