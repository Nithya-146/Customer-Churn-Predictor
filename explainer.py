"""
explainer.py
============
SHAP-based explainability for the Telco Churn classifiers.
Generates global summary plots and per-customer waterfall/force plots.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import io
import base64

# ── Colors ───────────────────────────────────────────────────────────────────
BG_COLOR   = "#0F1117"
GRID_COLOR = "#1E2130"
TEXT_COLOR = "#E0E0E0"
POS_COLOR  = "#F72585"   # positive SHAP (pushes toward churn)
NEG_COLOR  = "#4CC9F0"   # negative SHAP (pushes toward staying)


# ── Explainer Factory ─────────────────────────────────────────────────────────
def build_explainer(model, X_background: pd.DataFrame, model_type: str = "tree"):
    """Build a SHAP TreeExplainer (fast, exact for tree-based models)."""
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, X_background)
    else:
        explainer = shap.Explainer(model, X_background)
    return explainer


def get_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for dataset X. Returns 2D array (n_samples, n_features)."""
    sv = explainer.shap_values(X)
    sv = np.array(sv)
    # RF TreeExplainer may return shape (n_samples, n_features, n_classes)
    if sv.ndim == 3:
        return sv[:, :, 1]
    # Legacy list format [class0_arr, class1_arr]
    if isinstance(sv, list):
        return np.array(sv[1])
    return sv


# ── Global Summary (Plotly bar) ───────────────────────────────────────────────
def plot_shap_summary_bar(shap_values, feature_names: list,
                          top_n: int = 15, title: str = "SHAP Feature Importance") -> go.Figure:
    """Global mean |SHAP| bar chart."""
    sv = np.array(shap_values)
    if sv.ndim == 3:          # RF: (n_samples, n_features, n_classes) -> pick class 1
        sv = sv[:, :, 1]
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)
    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": mean_abs,
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#7209B7"], [0.5, POS_COLOR], [1.0, "#FFD700"]],
        labels={"Importance": "Mean |SHAP Value|"},
    )
    fig.update_layout(
        title=title,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=10, r=20, t=50, b=40),
        height=440,
    )
    return fig


# ── SHAP Beeswarm via Matplotlib → base64 PNG ────────────────────────────────
def plot_shap_beeswarm_img(shap_values: np.ndarray, X: pd.DataFrame,
                           top_n: int = 15) -> str:
    """Return a base64-encoded PNG of the SHAP beeswarm/dot plot."""
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    BG_COLOR,
        "text.color":        TEXT_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        max_display=top_n,
        show=False,
        plot_size=None,
        color_bar_label="Feature Value",
    )
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close("all")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Per-customer Waterfall (Plotly) ──────────────────────────────────────────
def plot_shap_waterfall(shap_values_row: np.ndarray, feature_names: list,
                        feature_values: pd.Series, base_value: float,
                        top_n: int = 10) -> go.Figure:
    """Interactive waterfall chart for a single customer prediction."""
    sv   = pd.Series(shap_values_row, index=feature_names)
    fv   = feature_values.reindex(feature_names)

    # Top N by absolute magnitude
    top_idx  = sv.abs().nlargest(top_n).index
    sv_top   = sv[top_idx]
    fv_top   = fv[top_idx]

    # Sort from most positive to most negative
    order   = sv_top.sort_values(ascending=True).index
    sv_ord  = sv_top[order]
    fv_ord  = fv_top[order]

    labels  = [f"{feat}<br><sub>val={val:.2f}</sub>"
               for feat, val in zip(order, fv_ord)]
    colors  = [POS_COLOR if v > 0 else NEG_COLOR for v in sv_ord.values]

    final_val = base_value + sv.sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sv_ord.values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sv_ord.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#555")

    fig.update_layout(
        title=f"SHAP Explanation — Churn Probability: {final_val:.1%} "
              f"(base: {base_value:.1%})",
        xaxis_title="SHAP Value (impact on churn probability)",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=10, r=20, t=60, b=40),
        height=420,
    )
    return fig


# ── SHAP Dependence Plot (Plotly) ─────────────────────────────────────────────
def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame,
                         feature: str, feature_names: list) -> go.Figure:
    """SHAP dependence plot for a selected feature."""
    if feature not in feature_names:
        return go.Figure()

    idx = feature_names.index(feature)
    sv  = shap_values[:, idx]
    fv  = X[feature].values if feature in X.columns else X.iloc[:, idx].values

    fig = px.scatter(
        x=fv, y=sv,
        color=sv,
        color_continuous_scale=[[0, NEG_COLOR], [0.5, "#7209B7"], [1, POS_COLOR]],
        labels={"x": feature, "y": "SHAP Value", "color": "SHAP"},
        opacity=0.65,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title=f"SHAP Dependence — {feature}",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    return fig


# ── Human-readable SHAP reasons ───────────────────────────────────────────────
def get_churn_reasons(shap_values_row: np.ndarray, feature_names: list,
                      top_n: int = 5) -> list[dict]:
    """Return top-N reasons (feature + direction + magnitude) as plain text."""
    sv = pd.Series(shap_values_row, index=feature_names)
    top_pos = sv.nlargest(top_n)   # push TOWARD churn
    top_neg = sv.nsmallest(top_n)  # push AWAY from churn

    reasons = []
    for feat, val in top_pos.items():
        if val > 0.005:
            reasons.append({
                "feature":   feat,
                "direction": "increases churn risk",
                "magnitude": abs(val),
                "positive":  True,
            })
    for feat, val in top_neg.items():
        if val < -0.005:
            reasons.append({
                "feature":   feat,
                "direction": "reduces churn risk",
                "magnitude": abs(val),
                "positive":  False,
            })

    reasons.sort(key=lambda r: r["magnitude"], reverse=True)
    return reasons[:top_n]
