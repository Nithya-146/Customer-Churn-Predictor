"""
model_trainer.py
================
Train Random Forest and XGBoost classifiers on the Telco churn dataset.
Handles class imbalance with SMOTE and compares models via ROC-AUC.
"""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, accuracy_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colors ───────────────────────────────────────────────────────────────────
RF_COLOR  = "#4CC9F0"
XGB_COLOR = "#F72585"
BG_COLOR  = "#0F1117"
GRID_COLOR = "#1E2130"
TEXT_COLOR = "#E0E0E0"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── SMOTE Oversampling ────────────────────────────────────────────────────────
def apply_smote(X_train, y_train, seed: int = 42):
    """Apply SMOTE to handle class imbalance."""
    print(f"Before SMOTE - Class distribution: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=seed, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE - Class distribution: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


# ── Model Training ────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, seed: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("   [OK] Random Forest trained.")
    return rf


def train_xgboost(X_train, y_train, seed: int = 42) -> XGBClassifier:
    """Train an XGBoost classifier."""
    print("Training XGBoost...")
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    print("   [OK] XGBoost trained.")
    return xgb


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Return a dict of evaluation metrics for one model."""
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_proba)
    acc         = accuracy_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm          = confusion_matrix(y_test, y_pred)

    print(f"\n[{model_name}] Results:")
    print(f"   ROC-AUC  : {roc_auc:.4f}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return {
        "name":    model_name,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "f1":      f1,
        "fpr":     fpr,
        "tpr":     tpr,
        "cm":      cm,
        "y_pred":  y_pred,
        "y_proba": y_proba,
    }


# ── Plotly Figures ────────────────────────────────────────────────────────────
def _dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    base.update(kwargs)
    return base


def plot_roc_curves(rf_metrics: dict, xgb_metrics: dict) -> go.Figure:
    """Side-by-side ROC curves for both models."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rf_metrics["fpr"], y=rf_metrics["tpr"],
        mode="lines", name=f"Random Forest (AUC={rf_metrics['roc_auc']:.3f})",
        line=dict(color=RF_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=xgb_metrics["fpr"], y=xgb_metrics["tpr"],
        mode="lines", name=f"XGBoost      (AUC={xgb_metrics['roc_auc']:.3f})",
        line=dict(color=XGB_COLOR, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random Classifier",
        line=dict(color="#555", dash="dash"),
    ))

    fig.update_layout(
        title="ROC Curves — Random Forest vs. XGBoost",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **_dark_layout(height=400)
    )
    return fig


def plot_confusion_matrices(rf_metrics: dict, xgb_metrics: dict) -> go.Figure:
    """Side-by-side confusion matrices."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Random Forest", "XGBoost"]
    )

    for col_idx, metrics in enumerate([rf_metrics, xgb_metrics], start=1):
        cm     = metrics["cm"]
        labels = ["Stayed", "Churned"]
        color  = RF_COLOR if col_idx == 1 else XGB_COLOR

        annotations = []
        z_text = [[str(cm[i][j]) for j in range(2)] for i in range(2)]

        fig.add_trace(
            go.Heatmap(
                z=cm, x=labels, y=labels,
                text=z_text, texttemplate="%{text}",
                colorscale=[[0, BG_COLOR], [1, color]],
                showscale=False,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title="Confusion Matrices",
        **_dark_layout(height=360)
    )
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15) -> go.Figure:
    """Bar chart of top-N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    color = RF_COLOR if "Forest" in model_name else XGB_COLOR

    fig = px.bar(
        x=top.values, y=top.index, orientation="h",
        color=top.values,
        color_continuous_scale=["#7209B7", color],
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(
        title=f"{model_name} — Top {top_n} Feature Importances",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        **_dark_layout(height=420)
    )
    return fig


def plot_metrics_comparison(rf_metrics: dict, xgb_metrics: dict) -> go.Figure:
    """Grouped bar chart comparing key metrics."""
    metrics_names  = ["ROC-AUC", "Accuracy", "F1 Score"]
    rf_vals  = [rf_metrics["roc_auc"], rf_metrics["accuracy"], rf_metrics["f1"]]
    xgb_vals = [xgb_metrics["roc_auc"], xgb_metrics["accuracy"], xgb_metrics["f1"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Random Forest", x=metrics_names, y=rf_vals,
        marker_color=RF_COLOR, text=[f"{v:.3f}" for v in rf_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="XGBoost", x=metrics_names, y=xgb_vals,
        marker_color=XGB_COLOR, text=[f"{v:.3f}" for v in xgb_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — Key Metrics",
        yaxis=dict(range=[0, 1.12], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **_dark_layout(height=380)
    )
    return fig


# ── Save / Load ───────────────────────────────────────────────────────────────
def save_models(rf_model, xgb_model, preprocessor, feature_names: list):
    joblib.dump(rf_model,     MODELS_DIR / "rf_model.pkl")
    joblib.dump(xgb_model,    MODELS_DIR / "xgb_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"\n[SAVE] Models saved to '{MODELS_DIR}/'")


def load_models():
    rf_model      = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb_model     = joblib.load(MODELS_DIR / "xgb_model.pkl")
    preprocessor  = joblib.load(MODELS_DIR / "preprocessor.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return rf_model, xgb_model, preprocessor, feature_names


def models_exist() -> bool:
    return all([
        (MODELS_DIR / "rf_model.pkl").exists(),
        (MODELS_DIR / "xgb_model.pkl").exists(),
        (MODELS_DIR / "preprocessor.pkl").exists(),
        (MODELS_DIR / "feature_names.pkl").exists(),
    ])
