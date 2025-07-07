import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model dan scaler
model = joblib.load('model/model_rf.pkl')
scaler = joblib.load('model/scaler.pkl')

# fitur yg harus diinput dari user
important_features = [
    'Age_at_enrollment',
    'Admission_grade',
    'Gender',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade'
]

# tampilkan fitur all 
all_features = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
    'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
    'Fathers_occupation', 'Admission_grade', 'Displaced', 'Educational_special_needs',
    'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment',
    'International', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

# nilai default 
default_values = {
    'Marital_status': 1,
    'Application_mode': 1,
    'Application_order': 1,
    'Course': 9999,
    'Daytime_evening_attendance': 1,
    'Previous_qualification': 1,
    'Previous_qualification_grade': 120.0,
    'Nacionality': 1,
    'Mothers_qualification': 1,
    'Fathers_qualification': 1,
    'Mothers_occupation': 1,
    'Fathers_occupation': 1,
    'Displaced': 0,
    'Educational_special_needs': 0,
    'Scholarship_holder': 0,
    'International': 0,
    'Curricular_units_1st_sem_credited': 0,
    'Curricular_units_1st_sem_enrolled': 6,
    'Curricular_units_1st_sem_evaluations': 6,
    'Curricular_units_1st_sem_approved': 5,
    'Curricular_units_1st_sem_without_evaluations': 0,
    'Curricular_units_2nd_sem_credited': 0,
    'Curricular_units_2nd_sem_enrolled': 6,
    'Curricular_units_2nd_sem_evaluations': 6,
    'Curricular_units_2nd_sem_approved': 5,
    'Curricular_units_2nd_sem_without_evaluations': 0,
    'Unemployment_rate': 10.0,
    'Inflation_rate': 1.0,
    'GDP': 1.5
}

# main 
st.set_page_config(page_title="Dashboard Informatif Status Mahasiswa", layout="centered")
st.title("Prediksi Status Mahasiswa")
st.markdown("Isi data utama berikut, untuk fitur lain diisi secara otomatis.")

# Form input user
with st.form("prediction_form"):
    age = st.slider("Usia saat mendaftar", 16, 60, 20)
    admission = st.slider("Nilai masuk (0–200)", 0.0, 200.0, step=0.1)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
    debtor = st.selectbox("Debtor (utang akademik)?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    fees = st.selectbox("Apakah bayar SPP tepat waktu?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    grade1 = st.slider("Rata-rata nilai semester 1", 0.0, 20.0, step=0.1)
    grade2 = st.slider("Rata-rata nilai semester 2", 0.0, 20.0, step=0.1)
    submitted = st.form_submit_button("🔍 Prediksi Status")

# Proses prediksi
if submitted:
    # Inisialisasi semua fitur
    full_input = [0.0] * len(all_features)

    # Isi input user
    full_input[all_features.index('Age_at_enrollment')] = age
    full_input[all_features.index('Admission_grade')] = admission
    full_input[all_features.index('Gender')] = gender
    full_input[all_features.index('Debtor')] = debtor
    full_input[all_features.index('Tuition_fees_up_to_date')] = fees
    full_input[all_features.index('Curricular_units_1st_sem_grade')] = grade1
    full_input[all_features.index('Curricular_units_2nd_sem_grade')] = grade2

    # Isi fitur lainnya dengan nilai default
    for feat, val in default_values.items():
        if feat not in important_features:
            full_input[all_features.index(feat)] = val

    # Scaling dan prediksi
    input_scaled = scaler.transform([full_input])
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Output prediksi
    st.subheader("📢 Hasil Prediksi")
    if prediction == "Dropout":
        st.error("Mahasiswa diprediksi akan *Dropout*.")
    elif prediction == "Enrolled":
        st.info("Saat ini mahasiswa masih *Terdaftar*.")
    elif prediction == "Graduate":
        st.success("Selamat... Mahasiswa *Lulus*.")

    # Visualisasi probabilitas
    st.subheader("📊 Probabilitas Tiap Kelas")
    labels = model.classes_
    fig, ax = plt.subplots()
    ax.bar(labels, proba, color='steelblue')
    ax.set_ylim([0, 1])
    for i, v in enumerate(proba):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)
