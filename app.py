"""
app.py — Streamlit Customer Churn Prediction Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0F1117; color: #E0E0E0; }

.metric-card {
    background: linear-gradient(135deg, #1A1D2E 0%, #16192A 100%);
    border: 1px solid #2A2D3E;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.78rem; color: #888; margin-top: 2px; }

.churn-high {
    background: linear-gradient(135deg,#2D0A16,#1A0A14);
    border: 2px solid #F72585;
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.churn-low {
    background: linear-gradient(135deg,#0A1A2D,#0A141A);
    border: 2px solid #4CC9F0;
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.churn-title { font-size: 1.1rem; font-weight: 600; margin-bottom: .4rem; }
.churn-pct { font-size: 3.2rem; font-weight: 800; }
.churn-high .churn-pct { color: #F72585; }
.churn-low  .churn-pct { color: #4CC9F0; }

.reason-pos {
    background: rgba(247,37,133,.12);
    border-left: 4px solid #F72585;
    border-radius: 8px; padding: .6rem .9rem; margin-bottom: .4rem;
}
.reason-neg {
    background: rgba(76,201,240,.10);
    border-left: 4px solid #4CC9F0;
    border-radius: 8px; padding: .6rem .9rem; margin-bottom: .4rem;
}
.reason-text { font-size: .85rem; margin: 0; }

div[data-testid="stSidebar"] { background: #111320; border-right: 1px solid #1E2130; }
.stButton>button {
    background: linear-gradient(135deg,#7209B7,#F72585);
    color: white; border: none; border-radius: 10px;
    padding: .6rem 1.8rem; font-weight: 600; font-size: .95rem;
    width: 100%; transition: opacity .2s;
}
.stButton>button:hover { opacity: .85; }
h1,h2,h3 { color: #F0F0F0; }
.stTabs [data-baseweb="tab"] { color: #888; }
.stTabs [aria-selected="true"] { color: #F72585; border-bottom: 2px solid #F72585; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    rf  = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")
    pre = joblib.load(MODELS_DIR / "preprocessor.pkl")
    fn  = joblib.load(MODELS_DIR / "feature_names.pkl")
    return rf, xgb, pre, fn


def preprocess_input(raw: dict, preprocessor: dict, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Add engineered features
    tenure = raw["tenure"]
    monthly = raw["MonthlyCharges"]
    total   = raw["TotalCharges"]
    df["AvgMonthlySpend"] = total / tenure if tenure > 0 else monthly

    addon_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                  "TechSupport","StreamingTV","StreamingMovies"]
    df["NumAddons"] = sum(1 for c in addon_cols if raw.get(c) == "Yes")

    df["TenureBucket"] = pd.cut(
        [tenure], bins=[-1,12,24,48,100],
        labels=["0-12m","13-24m","25-48m","49m+"]
    ).astype(str)[0]

    # Encode binary
    for col in preprocessor["binary_cols"]:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    # Label encode
    for col, le in preprocessor["encoders"].items():
        if col in df.columns:
            val = df[col].astype(str).iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])

    # Scale numeric
    num_cols = [c for c in preprocessor["num_cols"] if c in df.columns]
    df[num_cols] = preprocessor["scaler"].transform(df[num_cols])

    # Align to feature_names
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def models_ready():
    return all((MODELS_DIR / f).exists() for f in
               ["rf_model.pkl","xgb_model.pkl","preprocessor.pkl","feature_names.pkl"])


# ── Sidebar — Customer Profile ─────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown("## 👤 Customer Profile")
    st.sidebar.markdown("---")

    tenure   = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total    = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0,
                                       float(monthly * tenure), step=10.0)

    st.sidebar.markdown("#### 📋 Demographics")
    gender   = st.sidebar.selectbox("Gender", ["Male","Female"])
    senior   = st.sidebar.checkbox("Senior Citizen")
    partner  = st.sidebar.selectbox("Partner", ["Yes","No"])
    deps     = st.sidebar.selectbox("Dependents", ["Yes","No"])

    st.sidebar.markdown("#### 📞 Services")
    phone    = st.sidebar.selectbox("Phone Service", ["Yes","No"])
    multi    = st.sidebar.selectbox("Multiple Lines",
                ["Yes","No","No phone service"])
    internet = st.sidebar.selectbox("Internet Service",
                ["Fiber optic","DSL","No"])

    st.sidebar.markdown("#### 🔒 Add-ons")
    sec   = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
    bkp   = st.sidebar.selectbox("Online Backup",   ["Yes","No","No internet service"])
    dev   = st.sidebar.selectbox("Device Protection",["Yes","No","No internet service"])
    tech  = st.sidebar.selectbox("Tech Support",     ["Yes","No","No internet service"])
    tv    = st.sidebar.selectbox("Streaming TV",     ["Yes","No","No internet service"])
    movies= st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    st.sidebar.markdown("#### 💳 Billing")
    contract = st.sidebar.selectbox("Contract",
        ["Month-to-month","One year","Two year"])
    paperless= st.sidebar.selectbox("Paperless Billing", ["Yes","No"])
    payment  = st.sidebar.selectbox("Payment Method", [
        "Electronic check","Mailed check",
        "Bank transfer (automatic)","Credit card (automatic)"
    ])

    return {
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "gender": gender, "SeniorCitizen": 1 if senior else 0,
        "Partner": partner, "Dependents": deps,
        "PhoneService": phone, "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": bkp,
        "DeviceProtection": dev, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem">
      <h1 style="font-size:2.4rem;background:linear-gradient(135deg,#4CC9F0,#7209B7,#F72585);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        📡 Customer Churn Predictor
      </h1>
      <p style="color:#888;margin-top:.4rem">
        End-to-end ML pipeline · Random Forest + XGBoost · SHAP Explainability
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ready():
        st.error("⚠️ Models not found. Run `python pipeline.py` first, then refresh.")
        st.code("python pipeline.py", language="bash")
        _show_training_tab()
        return

    rf, xgb, preprocessor, feature_names = load_models()
    raw = sidebar_inputs()

    tabs = st.tabs(["🎯 Predict", "📊 EDA", "🤖 Model Comparison", "🔍 SHAP Global"])

    # ── TAB 1: Predict ─────────────────────────────────────────────────────────
    with tabs[0]:
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            predict_clicked = st.button("🔮 Predict Churn")

        if predict_clicked:
            try:
                X_input = preprocess_input(raw, preprocessor, feature_names)
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                return

            rf_prob  = rf.predict_proba(X_input)[0][1]
            xgb_prob = xgb.predict_proba(X_input)[0][1]
            ensemble = (rf_prob + xgb_prob) / 2

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            for col, name, prob, color in [
                (c1, "Random Forest", rf_prob, "#4CC9F0"),
                (c2, "XGBoost",       xgb_prob,"#F72585"),
                (c3, "Ensemble",      ensemble, "#7209B7"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <p class="metric-label">{name}</p>
                      <p class="metric-value" style="color:{color}">{prob:.1%}</p>
                      <p class="metric-label">churn probability</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            label   = "HIGH CHURN RISK ⚠️" if ensemble >= 0.5 else "LOW CHURN RISK ✅"
            css_cls = "churn-high" if ensemble >= 0.5 else "churn-low"
            st.markdown(f"""
            <div class="{css_cls}">
              <div class="churn-title">{label}</div>
              <div class="churn-pct">{ensemble:.1%}</div>
              <div style="color:#888;font-size:.85rem;margin-top:.3rem">
                Ensemble prediction (RF + XGBoost average)
              </div>
            </div>""", unsafe_allow_html=True)

            # SHAP explanation
            st.markdown("### 🔍 Why this prediction?")
            try:
                import shap
                from explainer import build_explainer, get_shap_values, \
                    plot_shap_waterfall, get_churn_reasons

                explainer    = build_explainer(xgb, X_input)
                shap_vals    = get_shap_values(explainer, X_input)
                base_val     = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(base_val[-1])

                reasons = get_churn_reasons(shap_vals[0], feature_names)

                col_r, col_w = st.columns([1, 1.6])
                with col_r:
                    st.markdown("#### Top Churn Drivers")
                    for r in reasons:
                        css = "reason-pos" if r["positive"] else "reason-neg"
                        icon = "🔴" if r["positive"] else "🟢"
                        st.markdown(f"""
                        <div class="{css}">
                          <p class="reason-text">
                            {icon} <b>{r['feature']}</b> — {r['direction']}
                            <span style="float:right;color:#aaa">{r['magnitude']:.3f}</span>
                          </p>
                        </div>""", unsafe_allow_html=True)

                with col_w:
                    fig = plot_shap_waterfall(
                        shap_vals[0], feature_names,
                        X_input.iloc[0], base_val
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")

        else:
            st.info("👈 Configure the customer profile in the sidebar, then click **Predict Churn**.")

    # ── TAB 2: EDA ─────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📊 Exploratory Data Analysis")
        try:
            from data_loader import load_raw_data, clean_data
            from eda import (get_eda_summary, plot_churn_rate,
                             plot_churn_by_contract, plot_tenure_distribution,
                             plot_monthly_charges, plot_churn_by_internet,
                             plot_correlation_heatmap, plot_churn_by_payment)

            with st.spinner("Loading dataset for EDA…"):
                raw_df   = load_raw_data()
                clean_df = clean_data(raw_df.copy())
                stats    = get_eda_summary(raw_df)

            m1,m2,m3,m4 = st.columns(4)
            for col, label, val, color in [
                (m1,"Total Customers", f"{stats['total_customers']:,}", "#4CC9F0"),
                (m2,"Churned",         f"{stats['churned']:,}",         "#F72585"),
                (m3,"Churn Rate",      f"{stats['churn_rate_pct']}%",   "#7209B7"),
                (m4,"Avg Tenure",      f"{stats['avg_tenure']}mo",      "#4CC9F0"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <p class="metric-value" style="color:{color}">{val}</p>
                      <p class="metric-label">{label}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            with r1c1: st.plotly_chart(plot_churn_rate(raw_df),            use_container_width=True)
            with r1c2: st.plotly_chart(plot_churn_by_contract(raw_df),     use_container_width=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1: st.plotly_chart(plot_tenure_distribution(raw_df),   use_container_width=True)
            with r2c2: st.plotly_chart(plot_monthly_charges(raw_df),       use_container_width=True)

            r3c1, r3c2 = st.columns(2)
            with r3c1: st.plotly_chart(plot_churn_by_internet(raw_df),     use_container_width=True)
            with r3c2: st.plotly_chart(plot_churn_by_payment(raw_df),      use_container_width=True)

            st.plotly_chart(plot_correlation_heatmap(clean_df), use_container_width=True)

        except Exception as e:
            st.error(f"EDA error: {e}")

    # ── TAB 3: Model Comparison ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🤖 Model Performance Comparison")
        try:
            from data_loader import get_train_test
            from model_trainer import (apply_smote, evaluate_model,
                                       plot_roc_curves, plot_confusion_matrices,
                                       plot_metrics_comparison, plot_feature_importance)

            with st.spinner("Evaluating models on test set…"):
                X_tr, X_te, y_tr, y_te, fn, prep, _ = get_train_test()
                X_res, y_res = apply_smote(X_tr, y_tr)
                rf_m  = evaluate_model(rf,  X_te, y_te, "Random Forest")
                xgb_m = evaluate_model(xgb, X_te, y_te, "XGBoost")

            st.plotly_chart(plot_metrics_comparison(rf_m, xgb_m), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(plot_roc_curves(rf_m, xgb_m),          use_container_width=True)
            with c2: st.plotly_chart(plot_confusion_matrices(rf_m, xgb_m),   use_container_width=True)

            c3, c4 = st.columns(2)
            with c3: st.plotly_chart(plot_feature_importance(rf,  fn, "Random Forest"), use_container_width=True)
            with c4: st.plotly_chart(plot_feature_importance(xgb, fn, "XGBoost"),       use_container_width=True)

        except Exception as e:
            st.error(f"Model comparison error: {e}")

    # ── TAB 4: SHAP Global ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 🔍 SHAP Global Explainability")
        st.markdown("""
        > **SHAP (SHapley Additive exPlanations)** assigns each feature a contribution
        > to each prediction — grounded in cooperative game theory.
        > The global view shows which features matter most *on average* across all customers.
        """)

        model_choice = st.radio("Select model", ["XGBoost","Random Forest"], horizontal=True)

        try:
            from data_loader import get_train_test
            from explainer import build_explainer, get_shap_values, \
                plot_shap_summary_bar, plot_shap_dependence

            with st.spinner("Computing SHAP values (this may take ~30 sec)…"):
                X_tr, X_te, y_tr, y_te, fn, _, _ = get_train_test()
                X_te_df = pd.DataFrame(X_te, columns=fn)
                n_bg    = min(80, len(X_tr))
                bg      = pd.DataFrame(X_tr).sample(n_bg, random_state=42)

                chosen_model = xgb if model_choice == "XGBoost" else rf
                expl = build_explainer(chosen_model, bg)
                sv   = get_shap_values(expl, X_te_df)

            st.plotly_chart(plot_shap_summary_bar(sv, fn, title=f"{model_choice} — SHAP Feature Importance"),
                            use_container_width=True)

            st.markdown("#### SHAP Dependence Plot")
            feat = st.selectbox("Feature", fn, index=fn.index("tenure") if "tenure" in fn else 0)
            st.plotly_chart(plot_shap_dependence(sv, X_te_df, feat, fn), use_container_width=True)

        except Exception as e:
            st.error(f"SHAP error: {e}")


def _show_training_tab():
    st.markdown("### 🚀 Run the Training Pipeline")
    st.markdown("Execute the following command in your terminal:")
    st.code("python pipeline.py", language="bash")
    st.markdown("This will:\n- Load / generate the dataset\n- Apply SMOTE\n- Train RF + XGBoost\n- Save SHAP artifacts\n- Persist models to `models/`")


if __name__ == "__main__":
    main()
