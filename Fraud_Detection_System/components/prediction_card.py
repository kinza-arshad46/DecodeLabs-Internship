"""
components/prediction_card.py
--------------------------------
Renders the fraud/normal result card and probability breakdown on the
Predictions page.
"""

import streamlit as st


def render_prediction_result(is_fraud: bool, probability: float, model_name: str, threshold: float):
    if is_fraud:
        st.markdown(
            f"""
            <div class="fraud-alert">
                <h3 style="color:#f87171; margin:0;">🚨 FRAUD</h3>
                <p style="color:#fca5a5;">Probability: {probability:.2f} ({probability*100:.1f}%)</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Model used: {model_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="safe-alert">
                <h3 style="color:#4ade80; margin:0;">✅ NORMAL</h3>
                <p style="color:#86efac;">Probability of fraud: {probability:.2f} ({probability*100:.1f}%)</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Model used: {model_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("**Probability Breakdown**")
    st.progress(float(probability), text=f"Fraud probability: {probability*100:.1f}%")
    st.caption(f"Decision threshold: {threshold:.2f} (change this on the Settings page)")
