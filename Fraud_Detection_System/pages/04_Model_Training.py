import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from utils.model_trainer import train_model, save_model
from utils.evaluator import evaluate_model

render_sidebar("Model Training")

st.markdown("## Model Training")

if st.session_state.X_train is None:
    st.warning("Run **Preprocessing** first so training/test data exists.")
else:
    col_a, col_b, col_c = st.columns([1.3, 1, 1])
    with col_a:
        selected_models = st.multiselect(
            "Select models",
            ["Logistic Regression", "Random Forest"],
            default=["Logistic Regression", "Random Forest"],
        )
    with col_b:
        use_tuning = st.toggle("Hyperparameter tuning", value=True)
    with col_c:
        cv_folds = st.selectbox("Cross validation (k-fold)", [3, 5, 10], index=1)

    col_d, col_e = st.columns(2)
    with col_d:
        quick_grid = st.checkbox("Fast mode (smaller search grid, quicker training)", value=True)
    with col_e:
        auto_save = st.checkbox("Save trained models to models/*.pkl", value=True)

    if st.button("🚀 Train Selected Models", type="primary"):
        for model_name in selected_models:
            with st.spinner(f"Training {model_name}..."):
                best_model, best_params = train_model(
                    model_name,
                    st.session_state.X_train, st.session_state.y_train,
                    cv_folds=cv_folds, use_grid_search=use_tuning, quick_grid=quick_grid,
                )
                metrics = evaluate_model(best_model, st.session_state.X_test, st.session_state.y_test)
                st.session_state.trained_models[model_name] = {
                    "model": best_model, "best_params": best_params, "metrics": metrics,
                }
                if auto_save:
                    save_model(best_model, model_name)
        st.success("Training complete. See results below, or open Model Comparison.")

    if st.session_state.trained_models:
        cols = st.columns(len(st.session_state.trained_models))
        for col, (name, result) in zip(cols, st.session_state.trained_models.items()):
            with col:
                params_html = "".join(
                    f"<li>{k.replace('classifier__','')}: {v}</li>"
                    for k, v in result["best_params"].items()
                )
                m = result["metrics"]
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b>{name}</b>
                            <span class="badge" style="color:#4ade80; border-color:#14532d;">Trained</span>
                        </div>
                        <p style="color:#94a3b8; font-size:0.8rem; margin-top:10px;">Best Parameters</p>
                        <ul style="font-size:0.85rem; margin-top:0;">{params_html}</ul>
                        <p style="color:#94a3b8; font-size:0.8rem; margin-top:10px;">Test Performance</p>
                        <div style="display:flex; gap:18px;">
                            <div><b>{m['precision']:.2f}</b><br><span style="color:#94a3b8; font-size:0.75rem;">Precision</span></div>
                            <div><b>{m['recall']:.2f}</b><br><span style="color:#94a3b8; font-size:0.75rem;">Recall</span></div>
                            <div><b>{m['roc_auc']:.2f}</b><br><span style="color:#94a3b8; font-size:0.75rem;">ROC-AUC</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
