"""
utils/visualization.py
------------------------
Pure chart-building functions. Each function takes data and returns a
plotly Figure - none of them call Streamlit. The `components/charts.py`
module wraps these with `st.plotly_chart(...)` for rendering. Keeping the
figure logic separate from the rendering call makes it easy to unit test
(see tests/test_pipeline.py) and easy to reuse (e.g. exporting a chart to
a notebook or a report).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e5e7eb",
)


def class_distribution_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Class"].value_counts().rename({0: "Normal", 1: "Fraud"})
    fig = px.pie(
        names=counts.index, values=counts.values, hole=0.6,
        color=counts.index,
        color_discrete_map={"Normal": "#3b82f6", "Fraud": "#f87171"},
    )
    fig.update_layout(**DARK_LAYOUT, legend=dict(orientation="h"))
    return fig


def amount_distribution_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Amount", nbins=60, log_x=True)
    fig.update_traces(marker_color="#60a5fa")
    fig.update_layout(**DARK_LAYOUT, bargap=0.05)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**DARK_LAYOUT, height=420)
    return fig


def top_correlated_features(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    corr_with_class = df.corr(numeric_only=True)["Class"].drop("Class")
    top = corr_with_class.reindex(corr_with_class.abs().sort_values(ascending=False).index).head(top_n)
    out = top.reset_index()
    out.columns = ["Feature", "Correlation"]
    out["Correlation"] = out["Correlation"].round(3)
    return out


def smote_class_bar(y_series, labels: dict, colors: dict) -> go.Figure:
    counts = pd.Series(y_series).value_counts().rename(labels)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=colors)
    fig.update_layout(**DARK_LAYOUT, showlegend=False, xaxis_title="", yaxis_title="Count")
    return fig


def roc_curve_comparison(models_metrics: dict) -> go.Figure:
    """
    models_metrics: {model_name: metrics_dict} where metrics_dict has 'fpr',
    'tpr', 'roc_auc' keys (as returned by evaluator.evaluate_model).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#ef4444"), name="Random Guess (AUC = 0.50)",
    ))
    colors = ["#4ade80", "#60a5fa", "#f59e0b"]
    for i, (name, m) in enumerate(models_metrics.items()):
        fig.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f"{name} (AUC = {m['roc_auc']:.2f})",
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig
