import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor")

# Load model dan data pendukung
model = joblib.load("brf_model.pkl")
features = joblib.load("model_features.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🎓 Student Performance Predictor")

# Input user
def user_input():
    data = {}
    for col in features:
        data[col] = st.number_input(col, step=1.0)
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    label = le.inverse_transform([prediction])[0]
    st.success(f"Prediksi Status Siswa: **{label}**")
