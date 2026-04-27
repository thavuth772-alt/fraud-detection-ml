import streamlit as st
import pandas as pd
import joblib

st.title("💳 Fraud Detection App")

model = joblib.load("xgb_model.pkl")

# Inputs
amount = st.number_input("Amount")
hour = st.number_input("Transaction Hour")
foreign = st.selectbox("Foreign Transaction", [0,1])
mismatch = st.selectbox("Location Mismatch", [0,1])
trust = st.slider("Device Trust Score", 0.0, 1.0)
velocity = st.number_input("Velocity Last 24h")
age = st.number_input("Cardholder Age")

if st.button("Predict"):
    data = pd.DataFrame([[amount, hour, foreign, mismatch, trust, velocity, age]],
                        columns=["amount","transaction_hour","foreign_transaction",
                                 "location_mismatch","device_trust_score",
                                 "velocity_last_24h","cardholder_age"])
    
    pred = model.predict(data)
    
    if pred[0] == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Legit Transaction")
