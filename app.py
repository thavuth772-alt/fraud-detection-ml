import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detector", layout="centered")

st.title("💳 Fraud Detection System")

# Load model + columns
model = joblib.load("xgb_model.pkl")
cols = joblib.load("columns.pkl")

st.subheader("Enter Transaction Details")

amount = st.number_input("💰 Transaction Amount", min_value=0.0)
hour = st.slider("⏰ Transaction Time (Hour)", 0, 23)

foreign = st.radio("🌍 Is this a foreign transaction?", ["No", "Yes"])
foreign = 1 if foreign == "Yes" else 0

mismatch = st.radio("📍 Location mismatch detected?", ["No", "Yes"])
mismatch = 1 if mismatch == "Yes" else 0

trust = st.slider("📱 Device Safety Level", 0.0, 1.0)

velocity = st.slider("🔄 Transactions in last 24 hours", 0, 50)

age = st.number_input("👤 Cardholder Age", min_value=18)

# BUTTON
if st.button("Check Transaction"):

    data = pd.DataFrame([[amount, hour, foreign, mismatch, trust, velocity, age]],
        columns=[
            "amount",
            "transaction_hour",
            "foreign_transaction",
            "location_mismatch",
            "device_trust_score",
            "velocity_last_24h",
            "cardholder_age"
        ])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    st.write(f"Fraud Probability: {prob:.2f}")

    if pred == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe")
