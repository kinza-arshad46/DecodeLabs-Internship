import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.prediction_card import render_prediction_result
from utils.model_trainer import load_model
from utils.predictor import build_input_row, predict_transaction

render_sidebar("Predictions")

st.markdown("## Make Prediction")

available_models = dict(st.session_state.trained_models)

# Fall back to models saved on disk in an earlier session, if none are
# trained in this session yet - this is what makes the app usable across
# app restarts without retraining every time.
if not available_models:
    for name in ["Logistic Regression", "Random Forest"]:
        disk_model = load_model(name)
        if disk_model is not None:
            available_models[name] = {"model": disk_model, "metrics": None, "best_params": {}}
    if available_models:
        st.info("Loaded previously trained model(s) from `models/*.pkl` (no models trained in this session yet).")

if not available_models or st.session_state.X_train is None:
    st.warning(
        "Train at least one model on the **Model Training** page first "
        "(preprocessing must also have been run, so feature columns are known)."
    )
else:
    model_name = st.selectbox("Model to use", list(available_models.keys()))
    model = available_models[model_name]["model"]
    feature_cols = list(st.session_state.X_train.columns)

    if not st.session_state.pred_input:
        st.session_state.pred_input = {c: 0.0 for c in feature_cols}

    col_fill1, col_fill2, _ = st.columns([1, 1, 2])
    with col_fill1:
        if st.button("🎲 Load random test transaction"):
            sample = st.session_state.X_test.sample(1).iloc[0]
            st.session_state.pred_input = sample.to_dict()
    with col_fill2:
        if st.button("🚩 Load random fraud example"):
            fraud_idx = st.session_state.y_test[st.session_state.y_test == 1].index
            if len(fraud_idx) > 0:
                sample = st.session_state.X_test.loc[np.random.choice(fraud_idx)]
                st.session_state.pred_input = sample.to_dict()
            else:
                st.info("No fraud examples in the current test split.")

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown("##### Transaction Details")
        time_val = st.number_input("Time (s)", value=float(st.session_state.pred_input.get("Time", 0.0)))
        amount_val = st.number_input("Amount", value=float(st.session_state.pred_input.get("Amount", 0.0)))
        v_cols = st.columns(3)
        v_values = {}
        for i, col_name in enumerate([c for c in feature_cols if c.startswith("V")][:3]):
            with v_cols[i]:
                v_values[col_name] = st.number_input(
                    col_name, value=float(st.session_state.pred_input.get(col_name, 0.0)), key=f"main_{col_name}"
                )

        with st.expander("Advanced: enter remaining V4–V28 features"):
            remaining_v = [c for c in feature_cols if c.startswith("V")][3:]
            adv_cols = st.columns(4)
            for i, col_name in enumerate(remaining_v):
                with adv_cols[i % 4]:
                    v_values[col_name] = st.number_input(
                        col_name, value=float(st.session_state.pred_input.get(col_name, 0.0)), key=f"adv_{col_name}"
                    )

        predict_clicked = st.button("Predict", type="primary")

    with col_right:
        st.markdown("##### Prediction Result")
        if predict_clicked:
            values = {"Time": time_val, "Amount": amount_val, **v_values}
            input_df = build_input_row(feature_cols, values)
            is_fraud, probability = predict_transaction(model, input_df, threshold=st.session_state.threshold)
            render_prediction_result(is_fraud, probability, model_name, st.session_state.threshold)
        else:
            st.info("Fill in the transaction details and click **Predict**.")
