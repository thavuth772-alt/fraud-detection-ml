import streamlit as st
import pandas as pd
import joblib

st.title("💳 Fraud Detection App")

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preds = model.predict(data)
    data["Prediction"] = preds
    st.write(data.head())
