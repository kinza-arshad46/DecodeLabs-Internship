"""
Interactive Streamlit Dashboard – Customer Segmentation Explorer
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_raw_data, clean_and_engineer
from src.preprocessing import FeatureScaler, PCAReducer
from src.clustering import OptimalKFinder, CustomerClusterer
from src.personas import map_centroids_to_original, assign_persona_names, build_persona_profiles
from src.visualization import plot_clusters_3d_plotly

st.set_page_config(
    page_title="DecodeLabs | Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare():
    raw = load_raw_data("data/marketing_campaign.csv")
    df = clean_and_engineer(raw)
    return df


def main():
    st.markdown('<p class="main-header">🎯 Customer Segmentation Studio</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">DecodeLabs Project 3 · Unsupervised Learning · PCA + K-Means + Business Personas</p>',
        unsafe_allow_html=True,
    )

    df = load_and_prepare()

    # Sidebar controls
    st.sidebar.header("⚙️ Pipeline Controls")
    variance_thr = st.sidebar.slider("PCA Variance Threshold", 0.80, 0.99, 0.95, 0.01)
    k_min, k_max = st.sidebar.slider("K Range for Search", 2, 12, (2, 10))
    force_k = st.sidebar.number_input("Force K (0 = auto)", 0, 12, 0)

    run_btn = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn or st.session_state.results is None:
        with st.spinner("Running Scale → PCA → Optimal-K → Clustering → Personas ..."):
            # Scale
            scaler = FeatureScaler()
            X_scaled = scaler.fit_transform(df)

            # PCA
            pca = PCAReducer(variance_threshold=variance_thr)
            X_pca = pca.fit_transform(X_scaled)

            # Optimal K
            finder = OptimalKFinder(k_range=range(k_min, k_max + 1))
            results_df = finder.evaluate(X_pca)
            optimal_k = force_k if force_k > 0 else finder.recommended_k_

            # Cluster
            clusterer = CustomerClusterer(n_clusters=optimal_k)
            clusterer.fit(X_pca)
            labels = clusterer.labels_

            # Personas
            centroids_orig = map_centroids_to_original(clusterer.centroids_pca_, pca, scaler)
            centroids_df = pd.DataFrame(centroids_orig, columns=df.columns)
            personas = assign_persona_names(centroids_df, list(df.columns))
            profile = build_persona_profiles(df, labels, list(df.columns))

            st.session_state.results = {
                "df": df,
                "X_pca": X_pca,
                "labels": labels,
                "centroids_pca": clusterer.centroids_pca_,
                "centroids_df": centroids_df,
                "personas": personas,
                "profile": profile,
                "results_df": results_df,
                "optimal_k": optimal_k,
                "pca": pca,
                "finder": finder,
            }

    res = st.session_state.results

    # ---- KPI Row ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(res['df']):,}")
    col2.metric("Features (engineered)", len(res["df"].columns))
    col3.metric("PCA Components", res["pca"].n_components_selected_)
    col4.metric("Optimal K", res["optimal_k"])

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Diagnostics", "🗺️ Cluster Map", "👥 Personas", "📈 Profiles", "📁 Data"
    ])

    with tab1:
        st.subheader("Elbow Method & Silhouette Score")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res["results_df"]["k"], y=res["results_df"]["wcss"],
            mode="lines+markers", name="WCSS", line=dict(color="#2c3e50", width=3)
        ))
        fig.add_vline(x=res["finder"].best_k_elbow_, line_dash="dash", line_color="red",
                      annotation_text=f"Elbow K={res['finder'].best_k_elbow_}")
        fig.update_layout(title="Elbow Method", xaxis_title="K", yaxis_title="WCSS", height=400)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=res["results_df"]["k"], y=res["results_df"]["silhouette"],
            mode="lines+markers", name="Silhouette", line=dict(color="#27ae60", width=3)
        ))
        fig2.add_vline(x=res["finder"].best_k_silhouette_, line_dash="dash", line_color="red",
                       annotation_text=f"Best Silhouette K={res['finder'].best_k_silhouette_}")
        fig2.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(res["results_df"].style.highlight_max(subset=["silhouette"], color="lightgreen"),
                     use_container_width=True)

    with tab2:
        st.subheader("Interactive Cluster Visualization")
        if res["X_pca"].shape[1] >= 3:
            fig3d = plot_clusters_3d_plotly(
                res["X_pca"], res["labels"], res["centroids_pca"],
                title=f"3D PCA Projection (K={res['optimal_k']})"
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            fig2d = px.scatter(
                x=res["X_pca"][:, 0], y=res["X_pca"][:, 1],
                color=res["labels"].astype(str),
                title="2D PCA Projection",
                labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(fig2d, use_container_width=True)

        sizes = pd.Series(res["labels"]).value_counts().sort_index()
        st.bar_chart(sizes)

    with tab3:
        st.subheader("Strategic Persona Matrix")
        for cid, p in sorted(res["personas"].items()):
            with st.expander(f"{p['emoji']}  Cluster {cid}: {p['name']}", expanded=True):
                st.write(p["description"])
                st.markdown("**Recommended Actions**")
                for a in p["actions"]:
                    st.markdown(f"- {a}")

                # Key stats
                if cid in res["centroids_df"].index:
                    c = res["centroids_df"].loc[cid]
                    m1, m2, m3, m4 = st.columns(4)
                    if "Age" in c:
                        m1.metric("Avg Age", f"{c['Age']:.0f}")
                    if "Income" in c:
                        m2.metric("Avg Income", f"${c['Income']:,.0f}")
                    if "Total_Spent" in c:
                        m3.metric("Total Spent", f"${c['Total_Spent']:,.0f}")
                    if "Total_Purchases" in c:
                        m4.metric("Purchases", f"{c['Total_Purchases']:.1f}")

    with tab4:
        st.subheader("Cluster Statistical Profiles")
        st.dataframe(res["profile"], use_container_width=True)

        # Simple comparison radar-like view with selected features
        key_feats = [c for c in ["Age", "Income", "Total_Spent", "Recency", "Total_Purchases"] if c in res["df"].columns]
        if key_feats:
            means = res["df"].assign(Cluster=res["labels"]).groupby("Cluster")[key_feats].mean()
            fig = px.bar(means.reset_index().melt(id_vars="Cluster"),
                         x="Cluster", y="value", color="variable", barmode="group",
                         title="Mean Feature Values by Cluster")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Segmented Customer Data (sample)")
        display_df = res["df"].copy()
        display_df["Cluster"] = res["labels"]
        display_df["Persona"] = display_df["Cluster"].map(
            {cid: p["name"] for cid, p in res["personas"].items()}
        )
        st.dataframe(display_df.head(200), use_container_width=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Segmented Dataset (CSV)",
            csv,
            "customers_segmented.csv",
            "text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
