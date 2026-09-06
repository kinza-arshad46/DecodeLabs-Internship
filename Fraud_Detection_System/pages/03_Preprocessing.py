import streamlit as st

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from components.sidebar import render_sidebar
from components.metric_cards import render_info_card
from components.charts import render_smote_bar
from utils.preprocessing import clean_data, split_data
from utils.smote_handler import apply_smote
from utils.data_loader import dataset_summary, save_processed_split

render_sidebar("Preprocessing")
df = st.session_state.df

st.markdown("## Preprocessing & SMOTE")
st.caption("Split first, then resample — SMOTE is applied only to the training fold to avoid data leakage.")

test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

if st.button("Run Preprocessing", type="primary"):
    with st.spinner("Cleaning data, splitting, and previewing SMOTE..."):
        cleaned_df, clean_report = clean_data(df)
        X_train, X_test, y_train, y_test = split_data(cleaned_df, test_size=test_size)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test

        X_res, y_res = apply_smote(X_train, y_train)
        st.session_state.smote_X, st.session_state.smote_y = X_res, y_res

        processed_path = save_processed_split(X_train, X_test, y_train, y_test)
    st.success(f"Preprocessing complete. Processed data saved to `{processed_path}`.")

if st.session_state.y_train is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Before SMOTE (Training Data)")
        render_smote_bar(
            st.session_state.y_train,
            {0: "Normal", 1: "Fraud"},
            {"Normal": "#3b82f6", "Fraud": "#f87171"},
        )
    with col2:
        st.markdown("##### After SMOTE (Training Data)")
        render_smote_bar(
            st.session_state.smote_y,
            {0: "Normal (Original)", 1: "Fraud (Synthetic)"},
            {"Normal (Original)": "#3b82f6", "Fraud (Synthetic)": "#f87171"},
        )
    with col3:
        st.markdown("##### Preprocessing Steps")
        summary = dataset_summary(df)
        render_info_card(
            f"<p>✅ <b>Missing Value Check</b><br><span style='color:#94a3b8; font-size:0.85rem;'>"
            f"{summary['missing']} missing values found</span></p>"
            f"<p>✅ <b>Feature Scaling</b><br><span style='color:#94a3b8; font-size:0.85rem;'>"
            f"StandardScaler applied inside the model pipeline (Logistic Regression only)</span></p>"
            f"<p>✅ <b>Train/Test Split</b><br><span style='color:#94a3b8; font-size:0.85rem;'>"
            f"{int((1-test_size)*100)}% Training, {int(test_size*100)}% Testing (stratified)</span></p>"
            f"<p>✅ <b>SMOTE Applied</b><br><span style='color:#94a3b8; font-size:0.85rem;'>"
            f"Balanced training data only — test set kept untouched</span></p>"
        )
else:
    st.warning("Click **Run Preprocessing** to split the data and preview SMOTE.")
