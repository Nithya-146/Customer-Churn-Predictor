"""
data_loader.py
==============
Handles dataset loading, cleaning, and preprocessing for the
Telco Customer Churn pipeline.
"""

import pandas as pd
import numpy as np
import os
import io
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Telco dataset — embedded fallback (600 synthetic rows that mirror real data)
# ---------------------------------------------------------------------------
def _generate_synthetic_telco(n=600, seed=42) -> pd.DataFrame:
    """Generate a realistic synthetic Telco Churn dataset when the real one
    is unavailable.  Distributions are calibrated to match the IBM Telco CSV.
    """
    rng = np.random.default_rng(seed)

    n_customers = n
    customer_ids = [f"0000-{i:05d}" for i in range(n_customers)]

    gender        = rng.choice(["Male", "Female"], n_customers)
    senior        = rng.choice([0, 1], n_customers, p=[0.84, 0.16])
    partner       = rng.choice(["Yes", "No"],   n_customers, p=[0.48, 0.52])
    dependents    = rng.choice(["Yes", "No"],   n_customers, p=[0.30, 0.70])
    tenure        = rng.integers(0, 73, n_customers)
    phone_service = rng.choice(["Yes", "No"],   n_customers, p=[0.90, 0.10])

    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n_customers)
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_customers, p=[0.34, 0.44, 0.22]
    )

    def internet_addon(base_p=0.44):
        return np.where(
            internet_service == "No", "No internet service",
            rng.choice(["Yes", "No"], n_customers, p=[base_p, 1 - base_p])
        )

    online_security   = internet_addon(0.29)
    online_backup     = internet_addon(0.34)
    device_protection = internet_addon(0.34)
    tech_support      = internet_addon(0.29)
    streaming_tv      = internet_addon(0.38)
    streaming_movies  = internet_addon(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_customers, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n_customers, p=[0.59, 0.41])
    payment_method    = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly_charges = (
        20 +
        (internet_service == "Fiber optic") * rng.uniform(30, 50, n_customers) +
        (internet_service == "DSL") * rng.uniform(10, 30, n_customers) +
        (phone_service == "Yes") * rng.uniform(5, 20, n_customers) +
        rng.normal(0, 5, n_customers)
    ).clip(18, 120).round(2)

    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n_customers)).clip(0)
    total_charges = np.where(tenure == 0, 0.0, total_charges).round(2)

    # Churn probability driven by real-world signals
    churn_logit = (
        -2.0
        + 1.8  * (contract == "Month-to-month")
        - 0.8  * (contract == "Two year")
        + 0.9  * (internet_service == "Fiber optic")
        + 1.2  * (payment_method == "Electronic check")
        - 0.05 * tenure
        + 0.01 * monthly_charges
        - 0.5  * (online_security == "Yes")
        - 0.4  * (tech_support == "Yes")
        + 0.6  * (paperless_billing == "Yes")
        + rng.normal(0, 0.5, n_customers)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn      = np.where(rng.random(n_customers) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID":         customer_ids,
        "gender":             gender,
        "SeniorCitizen":      senior,
        "Partner":            partner,
        "Dependents":         dependents,
        "tenure":             tenure,
        "PhoneService":       phone_service,
        "MultipleLines":      multiple_lines,
        "InternetService":    internet_service,
        "OnlineSecurity":     online_security,
        "OnlineBackup":       online_backup,
        "DeviceProtection":   device_protection,
        "TechSupport":        tech_support,
        "StreamingTV":        streaming_tv,
        "StreamingMovies":    streaming_movies,
        "Contract":           contract,
        "PaperlessBilling":   paperless_billing,
        "PaymentMethod":      payment_method,
        "MonthlyCharges":     monthly_charges,
        "TotalCharges":       total_charges,
        "Churn":              churn,
    })
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset.

    Priority:
    1. ``csv_path`` argument (explicit file path)
    2. ``WA_Fn-UseC_-Telco-Customer-Churn.csv`` in the project directory
    3. Synthetic fallback (so the pipeline always works)
    """
    search_paths = [
        csv_path,
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "telco_churn.csv",
    ]

    for path in search_paths:
        if path and os.path.exists(path):
            print(f"[OK] Loaded dataset from: {path}")
            df = pd.read_csv(path)
            return df

    print("[WARN] No local dataset found - generating synthetic Telco data (600 rows).")
    return _generate_synthetic_telco(n=700)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data: fix types, handle missing values."""
    df = df.copy()

    # TotalCharges can be ' ' in the real dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID — not a feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model performance."""
    df = df.copy()

    # Avg monthly spend
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Number of add-on services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumAddons"] = df[addon_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    # Tenure bucket
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 100],
        labels=["0-12m", "13-24m", "25-48m", "49m+"]
    ).astype(str)

    return df


def encode_and_scale(df: pd.DataFrame):
    """Encode categoricals, scale numerics.

    Returns
    -------
    X : pd.DataFrame   — encoded feature matrix
    y : pd.Series      — binary target
    feature_names : list[str]
    preprocessor : dict  — {"encoders": {...}, "scaler": scaler}
    """
    df = df.copy()
    target = "Churn"
    y = df.pop(target)

    # ---- Encode binary Yes/No columns ----
    binary_cols = [c for c in df.columns
                   if df[c].dtype == object and set(df[c].unique()) <= {"Yes", "No"}]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # ---- Label-encode remaining categoricals ----
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ---- Scale numeric columns ----
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
    num_cols = [c for c in num_cols if c in df.columns]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    preprocessor = {"encoders": encoders, "scaler": scaler,
                    "num_cols": num_cols, "binary_cols": binary_cols,
                    "cat_cols": cat_cols}

    return df, y, list(df.columns), preprocessor


def get_train_test(csv_path: str | None = None, test_size: float = 0.2, seed: int = 42):
    """Full preprocessing pipeline: load → clean → engineer → encode → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, preprocessor, raw_df
    """
    raw_df      = load_raw_data(csv_path)
    clean_df    = clean_data(raw_df.copy())
    engineered  = feature_engineer(clean_df)
    X, y, feature_names, preprocessor = encode_and_scale(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, preprocessor, clean_df
