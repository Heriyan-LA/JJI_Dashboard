import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("brf_model.pkl")
    features = joblib.load("model_features.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, features, le

model, features, le = load_model()

st.title("🎓 Student Performance Predictor")

def user_input():
    data = {}
    for feat in features:
        data[feat] = st.number_input(f"{feat}", value=0.0)
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Prediksi"):
    pred = model.predict(input_df)[0]
    label = le.inverse_transform([pred])[0]
    st.success(f"📌 Hasil Prediksi: **{label}**")
