"""
components/navbar.py
-----------------------
A small reusable "page header" shown at the top of every page - keeps the
title/subtitle/caption styling consistent without repeating markdown in
every page file.
"""

import streamlit as st


def render_page_header(title: str, subtitle: str = None, caption: str = None):
    if caption:
        st.caption(caption)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(subtitle)
