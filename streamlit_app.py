import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from utils.feature_engineering import feature_engineering as fe
from utils.predict import predict
from utils.explainer import explain_prediction


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.markdown("""
    <style>
    h1, h3 {
        text-align: center;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## Fraud Detection Dashboard")

st.markdown("### Enter Transaction Details")
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", ['Online', 'POS', 'ATM', 'Transfer'])
    location = st.selectbox("Location", ['Abuja', 'Lagos', 'Ibadan', 'Kano', 'Port Harcourt'])
    device_type = st.selectbox("Device Type", ['Mobile', 'ATM Machine', 'POS Terminal', 'Web'])
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0)

with col2:
    risk_score = st.number_input("Risk Score", min_value=0.0)
    is_foreign = st.checkbox("Foreign Transaction")
    is_high_risk_country = st.checkbox("High Risk Country")
    previous_fraud = st.checkbox("Previous Fraud Flag")
    txn_date = st.date_input("Transaction Date")
    txn_time = st.time_input("Transaction Time")


user_input = {
    "transaction_type": transaction_type,
    "location": location,
    "device_type": device_type,
    "transaction_amount": transaction_amount,
    "is_foreign_transaction": int(is_foreign),
    "is_high_risk_country": int(is_high_risk_country),
    "previous_fraud_flag": int(previous_fraud),
    "risk_score": risk_score,
    "transaction_time": datetime.combine(txn_date, txn_time)
}


if st.button("Run Prediction", type="primary"):
    df_input = pd.DataFrame([user_input])
    mod_data = fe(df_input)

    prediction = predict(mod_data)
    label = "Fraudulent 📛" if prediction == 1 else "Legitimate ✅"
    color = "#d62728" if prediction == 1 else "#2ca02c"

    st.markdown(f"<h3 style='color:{color}; text-align: center;'>Transaction {label}</h3>", unsafe_allow_html=True)


    st.markdown("### Feature Importance")
    _, shap_values = explain_prediction(mod_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bar Plot**")
        fig_bar, _ = plt.subplots(figsize=(5, 2.5))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig_bar)

    with col2:
        st.markdown("**Waterfall Plot**")
        fig_waterfall, _ = plt.subplots(figsize=(5, 2.5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_waterfall)

    st.caption("Use SHAP values to interpret how each feature contributed to this prediction.")

    st.markdown("### Weekly Fraud Trend")

    df_history = pd.read_excel("utils/data/fintech_sample_fintech_transactions (003).csv(4).xlsx")
    df_history["transaction_time"] = pd.to_datetime(df_history["transaction_time"])
    fraud_df = df_history[df_history["label_code"] == 1]
    weekly_fraud = fraud_df.resample("W", on="transaction_time").size()
    weekly_fraud.index = weekly_fraud.index.to_series().dt.strftime("Week %U, %b %d")

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(weekly_fraud.index, weekly_fraud.values, marker='o', color='#d62728', linewidth=2)
    ax.set_xlabel("Week", fontsize=7)
    ax.set_ylabel("Fraud Count", fontsize=7)
    ax.tick_params(axis='x', rotation=30, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)
