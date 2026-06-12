# 📡 Customer Churn Prediction Pipeline

An end-to-end ML pipeline for telecom customer churn prediction with SHAP explainability.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
The app auto-downloads a sample of the Telco Customer Churn dataset.  
Or place your `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

### 3. Run the Training Pipeline
```bash
python pipeline.py
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
Customer Churn Predictor/
├── app.py                  # Streamlit dashboard
├── pipeline.py             # End-to-end training pipeline
├── data_loader.py          # Dataset loading & preprocessing
├── eda.py                  # Exploratory Data Analysis
├── model_trainer.py        # Random Forest + XGBoost training
├── explainer.py            # SHAP explainability module
├── requirements.txt
└── models/                 # Saved model artifacts
    ├── rf_model.pkl
    ├── xgb_model.pkl
    └── preprocessor.pkl
```

## 🧠 DS Concept: SHAP Explainability
SHAP (SHapley Additive exPlanations) values bridge the gap between model accuracy
and real-world explainability. Each prediction is decomposed into feature contributions,
answering "why did the model predict churn?" — crucial for business decisions.

## 📊 Pipeline Steps
1. **Data Loading** — Telco Customer Churn dataset
2. **EDA** — Churn rate, contract types, tenure distributions
3. **SMOTE** — Handle class imbalance via oversampling
4. **Modeling** — Random Forest + XGBoost comparison
5. **SHAP** — Feature importance & individual explanations
6. **Streamlit App** — Interactive churn predictor
