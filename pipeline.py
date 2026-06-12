"""
pipeline.py
===========
End-to-end training pipeline:
  Load data -> EDA -> SMOTE -> Train RF + XGBoost -> Evaluate -> SHAP -> Save
Run this script once to train and persist models before launching the Streamlit app.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import get_train_test
from model_trainer import (
    apply_smote, train_random_forest, train_xgboost,
    evaluate_model, save_models,
    plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_metrics_comparison,
)
from eda import get_eda_summary, plot_churn_rate, plot_churn_by_contract
from explainer import build_explainer, get_shap_values, plot_shap_summary_bar


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def banner(msg: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def run_pipeline(csv_path: str | None = None):
    start = time.time()

    # Step 1: Load & Preprocess Data
    banner("STEP 1 - Loading & Preprocessing Data")
    X_train, X_test, y_train, y_test, feature_names, preprocessor, raw_df = \
        get_train_test(csv_path)

    print(f"   Training samples : {len(X_train):,}")
    print(f"   Test samples     : {len(X_test):,}")
    print(f"   Features         : {len(feature_names)}")

    # Step 2: EDA Summary
    banner("STEP 2 - Exploratory Data Analysis")
    stats = get_eda_summary(raw_df)
    print(f"   Total customers  : {stats['total_customers']:,}")
    print(f"   Churned          : {stats['churned']:,}  ({stats['churn_rate_pct']:.1f}%)")
    print(f"   Avg Tenure       : {stats['avg_tenure']} months")
    print(f"   Avg Monthly Bill : ${stats['avg_monthly']}")

    fig_churn = plot_churn_rate(raw_df)
    fig_churn.write_html(str(ARTIFACTS_DIR / "eda_churn_rate.html"))
    plot_churn_by_contract(raw_df).write_html(str(ARTIFACTS_DIR / "eda_churn_by_contract.html"))
    print("   [OK] EDA charts saved to artifacts/")

    # Step 3: SMOTE Oversampling
    banner("STEP 3 - SMOTE Oversampling")
    X_res, y_res = apply_smote(X_train, y_train)

    # Step 4: Train Models
    banner("STEP 4 - Training Models")
    rf_model  = train_random_forest(X_res, y_res)
    xgb_model = train_xgboost(X_res, y_res)

    # Step 5: Evaluate
    banner("STEP 5 - Model Evaluation")
    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    winner = "Random Forest" if rf_metrics["roc_auc"] >= xgb_metrics["roc_auc"] else "XGBoost"
    print(f"\n   Best ROC-AUC: {winner}")

    plot_roc_curves(rf_metrics, xgb_metrics).write_html(str(ARTIFACTS_DIR / "roc_curves.html"))
    plot_confusion_matrices(rf_metrics, xgb_metrics).write_html(str(ARTIFACTS_DIR / "confusion_matrices.html"))
    plot_metrics_comparison(rf_metrics, xgb_metrics).write_html(str(ARTIFACTS_DIR / "metrics_comparison.html"))
    plot_feature_importance(rf_model, feature_names, "Random Forest").write_html(str(ARTIFACTS_DIR / "rf_feature_importance.html"))
    plot_feature_importance(xgb_model, feature_names, "XGBoost").write_html(str(ARTIFACTS_DIR / "xgb_feature_importance.html"))
    print("   [OK] Evaluation charts saved to artifacts/")

    # Step 6: SHAP Explainability
    banner("STEP 6 - SHAP Explainability")

    background_size = min(100, len(X_train))
    bg_idx       = np.random.choice(len(X_train), background_size, replace=False)
    X_background = pd.DataFrame(X_train).iloc[bg_idx]
    X_test_df    = pd.DataFrame(X_test, columns=feature_names)

    print("   Computing SHAP values for XGBoost...")
    xgb_explainer   = build_explainer(xgb_model, X_background)
    xgb_shap_values = get_shap_values(xgb_explainer, X_test_df)
    plot_shap_summary_bar(xgb_shap_values, feature_names,
        title="XGBoost SHAP - Global Feature Importance"
    ).write_html(str(ARTIFACTS_DIR / "xgb_shap_summary.html"))

    print("   Computing SHAP values for Random Forest...")
    rf_explainer   = build_explainer(rf_model, X_background)
    rf_shap_values = get_shap_values(rf_explainer, X_test_df)
    plot_shap_summary_bar(rf_shap_values, feature_names,
        title="Random Forest SHAP - Global Feature Importance"
    ).write_html(str(ARTIFACTS_DIR / "rf_shap_summary.html"))
    print("   [OK] SHAP charts saved to artifacts/")

    # Step 7: Save Models
    banner("STEP 7 - Saving Models")
    save_models(rf_model, xgb_model, preprocessor, feature_names)

    elapsed = time.time() - start
    banner(f"Pipeline Complete in {elapsed:.1f}s")
    print("   Run: streamlit run app.py\n")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(csv_path=csv_arg)
