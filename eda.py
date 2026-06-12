"""
eda.py
======
Exploratory Data Analysis for the Telco Churn dataset.
Returns Plotly figures that can be displayed in Streamlit or saved as HTML.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ─────────────────────────────────────────────────────────────────
STAY_COLOR  = "#4CC9F0"   # cyan-blue  — stayed
CHURN_COLOR = "#F72585"   # vivid pink — churned
BG_COLOR    = "#0F1117"
GRID_COLOR  = "#1E2130"
TEXT_COLOR  = "#E0E0E0"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _dark_layout(**kwargs):
    layout = LAYOUT_BASE.copy()
    layout.update(kwargs)
    return layout


# ── 1. Churn Rate Donut ──────────────────────────────────────────────────────
def plot_churn_rate(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing overall churn vs. retained."""
    col = "Churn" if "Churn" in df.columns else None
    if col is None:
        return go.Figure()

    if df[col].dtype == int or df[col].dtype == float:
        counts = df[col].value_counts().rename({0: "Stayed", 1: "Churned"})
    else:
        counts = df[col].value_counts().rename({"No": "Stayed", "Yes": "Churned"})

    labels = list(counts.index)
    values = list(counts.values)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker_colors=[STAY_COLOR, CHURN_COLOR],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} customers<extra></extra>",
    ))
    fig.update_layout(
        title="Overall Churn Rate",
        showlegend=True,
        **_dark_layout(height=360)
    )
    return fig


# ── 2. Churn by Contract Type ────────────────────────────────────────────────
def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    if "Contract" not in df.columns:
        return go.Figure()

    col = "Churn"
    if df[col].dtype in [int, float]:
        grp = df.groupby("Contract")[col].mean().reset_index()
        grp.columns = ["Contract", "ChurnRate"]
        grp["ChurnRate"] *= 100
    else:
        grp = (
            df.groupby(["Contract", col])
            .size().unstack(fill_value=0)
            .assign(ChurnRate=lambda x: x.get("Yes", 0) / x.sum(axis=1) * 100)
            .reset_index()[["Contract", "ChurnRate"]]
        )

    grp = grp.sort_values("ChurnRate", ascending=True)

    fig = px.bar(
        grp, x="ChurnRate", y="Contract", orientation="h",
        color="ChurnRate",
        color_continuous_scale=["#4CC9F0", "#F72585"],
        labels={"ChurnRate": "Churn Rate (%)"},
        text=grp["ChurnRate"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Churn Rate by Contract Type",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        **_dark_layout(height=320)
    )
    return fig


# ── 3. Tenure Distribution ───────────────────────────────────────────────────
def plot_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    if "tenure" not in df.columns:
        return go.Figure()

    col = "Churn"
    stayed  = df[df[col] == 0]["tenure"] if df[col].dtype in [int, float] else df[df[col] == "No"]["tenure"]
    churned = df[df[col] == 1]["tenure"] if df[col].dtype in [int, float] else df[df[col] == "Yes"]["tenure"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=stayed, name="Stayed",
        marker_color=STAY_COLOR, opacity=0.75,
        nbinsx=30, histnorm="percent"
    ))
    fig.add_trace(go.Histogram(
        x=churned, name="Churned",
        marker_color=CHURN_COLOR, opacity=0.75,
        nbinsx=30, histnorm="percent"
    ))
    fig.update_layout(
        barmode="overlay",
        title="Tenure Distribution: Stayed vs. Churned",
        xaxis_title="Tenure (months)",
        yaxis_title="% of Group",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **_dark_layout(height=360)
    )
    return fig


# ── 4. Monthly Charges vs. Churn ─────────────────────────────────────────────
def plot_monthly_charges(df: pd.DataFrame) -> go.Figure:
    if "MonthlyCharges" not in df.columns:
        return go.Figure()

    col = "Churn"
    stayed  = df[df[col] == 0]["MonthlyCharges"] if df[col].dtype in [int, float] else df[df[col] == "No"]["MonthlyCharges"]
    churned = df[df[col] == 1]["MonthlyCharges"] if df[col].dtype in [int, float] else df[df[col] == "Yes"]["MonthlyCharges"]

    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=stayed,  name="Stayed",
        side="negative", line_color=STAY_COLOR,
        fillcolor="rgba(76,201,240,0.2)", meanline_visible=True,
    ))
    fig.add_trace(go.Violin(
        y=churned, name="Churned",
        side="positive", line_color=CHURN_COLOR,
        fillcolor="rgba(247,37,133,0.2)", meanline_visible=True,
    ))
    fig.update_layout(
        title="Monthly Charges Distribution",
        yaxis_title="Monthly Charges ($)",
        violinmode="overlay",
        yaxis=dict(gridcolor=GRID_COLOR),
        **_dark_layout(height=360)
    )
    return fig


# ── 5. Churn by Internet Service ─────────────────────────────────────────────
def plot_churn_by_internet(df: pd.DataFrame) -> go.Figure:
    if "InternetService" not in df.columns:
        return go.Figure()

    col = "Churn"
    if df[col].dtype in [int, float]:
        grp = df.groupby("InternetService")[col].mean().reset_index()
        grp.columns = ["InternetService", "ChurnRate"]
        grp["ChurnRate"] *= 100
    else:
        grp = (
            df.groupby(["InternetService", col])
            .size().unstack(fill_value=0)
            .assign(ChurnRate=lambda x: x.get("Yes", 0) / x.sum(axis=1) * 100)
            .reset_index()[["InternetService", "ChurnRate"]]
        )

    fig = px.bar(
        grp, x="InternetService", y="ChurnRate",
        color="InternetService",
        color_discrete_sequence=[STAY_COLOR, CHURN_COLOR, "#7209B7"],
        labels={"ChurnRate": "Churn Rate (%)"},
        text=grp["ChurnRate"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Churn Rate by Internet Service",
        showlegend=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        **_dark_layout(height=320)
    )
    return fig


# ── 6. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric features against churn."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or "Churn" not in num_df.columns:
        return go.Figure()

    corr = num_df.corr()[["Churn"]].drop("Churn", errors="ignore")
    corr = corr.sort_values("Churn", ascending=False)

    fig = px.bar(
        corr.reset_index(),
        x="Churn", y="index", orientation="h",
        color="Churn",
        color_continuous_scale=["#4CC9F0", "#7209B7", "#F72585"],
        labels={"Churn": "Correlation with Churn", "index": "Feature"},
    )
    fig.update_layout(
        title="Feature Correlation with Churn",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        **_dark_layout(height=420)
    )
    return fig


# ── 7. Payment Method Churn ──────────────────────────────────────────────────
def plot_churn_by_payment(df: pd.DataFrame) -> go.Figure:
    if "PaymentMethod" not in df.columns:
        return go.Figure()

    col = "Churn"
    if df[col].dtype in [int, float]:
        grp = df.groupby("PaymentMethod")[col].mean().reset_index()
        grp.columns = ["PaymentMethod", "ChurnRate"]
        grp["ChurnRate"] *= 100
    else:
        grp = (
            df.groupby(["PaymentMethod", col])
            .size().unstack(fill_value=0)
            .assign(ChurnRate=lambda x: x.get("Yes", 0) / x.sum(axis=1) * 100)
            .reset_index()[["PaymentMethod", "ChurnRate"]]
        )

    grp = grp.sort_values("ChurnRate", ascending=False)
    fig = px.pie(
        grp, names="PaymentMethod", values="ChurnRate",
        color_discrete_sequence=["#F72585", "#7209B7", "#4361EE", "#4CC9F0"],
        hole=0.45,
    )
    fig.update_layout(
        title="Churn Distribution by Payment Method",
        **_dark_layout(height=360)
    )
    return fig


# ── Summary stats ─────────────────────────────────────────────────────────────
def get_eda_summary(df: pd.DataFrame) -> dict:
    """Return key EDA statistics as a dict for display."""
    col = "Churn"
    total = len(df)

    if df[col].dtype in [int, float]:
        churned = int(df[col].sum())
    else:
        churned = int((df[col] == "Yes").sum())

    churn_rate = churned / total * 100

    stats = {
        "total_customers":  total,
        "churned":          churned,
        "retained":         total - churned,
        "churn_rate_pct":   round(churn_rate, 2),
        "avg_tenure":       round(df["tenure"].mean(), 1) if "tenure" in df.columns else None,
        "avg_monthly":      round(df["MonthlyCharges"].mean(), 2) if "MonthlyCharges" in df.columns else None,
    }
    return stats
