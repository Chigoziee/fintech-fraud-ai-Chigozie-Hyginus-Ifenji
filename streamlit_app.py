import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from utils.feature_engineering import feature_engineering as fe
from utils.predict import predict
from utils.explainer import explain_prediction


# Page config
st.set_page_config(page_title="🔍 Fraud Detection Dashboard", layout="wide")

# Page title
st.markdown("<h1 style='text-align: center; color: red;'> Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Layout input into two columns
left, right = st.columns(2)

with left:
    st.markdown("### Transaction Details")
    transaction_type = st.selectbox("Transaction Type", ['Online', 'POS', 'ATM', 'Transfer'])
    location = st.selectbox("Location", ['Abuja', 'Lagos', 'Ibadan', 'Kano', 'Port Harcourt'])
    device_type = st.selectbox("Device Type", ['Mobile', 'ATM Machine', 'POS Terminal', 'Web'])
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, )

with right:
    st.markdown("### Risk Flags & Time")
    risk_score = st.number_input("Risk Score", min_value=0.0)
    is_foreign = st.checkbox("Is Foreign Transaction?")
    is_high_risk_country = st.checkbox("Is High Risk Country?")
    previous_fraud = st.checkbox("Previous Fraud Flag")
    txn_date = st.date_input("Transaction Date")
    txn_time = st.time_input("Transaction Time")

# Combine inputs
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


if st.button("Run Prediction"):
    df_input = pd.DataFrame([user_input])
    mod_data = fe(df_input)

    prediction = predict(mod_data)
    label = "Fraud" if prediction == 1 else "Not Fraud"
    color = "red" if prediction == 1 else "green"

    st.markdown(f"<h3 style='color:{color}'>Prediction: {label}</h3>", unsafe_allow_html=True)

    # SHAP explanation
    st.markdown("### SHAP Feature Importance")
    _, shap_values = explain_prediction(mod_data)

    tab1, tab2 = st.tabs(["Bar Plot", "Waterfall Plot"])

    with tab1:
        fig_bar, _ = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig_bar)

    with tab2:
        fig_waterfall, _ = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_waterfall)

    st.info("Interpret SHAP values to understand what contributed most to this prediction.")
