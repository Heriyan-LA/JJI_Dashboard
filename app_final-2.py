import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model dan scaler
try:
    model = joblib.load('model/model_rf.pkl')
    scaler = joblib.load('model/scaler.pkl')
except FileNotFoundError:
    st.error("File model tidak ditemukan! Pastikan model ada di folder 'model/'")
    st.stop()

# Fitur penting untuk input user
important_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Admission_grade',
    'Age_at_enrollment',
    'Debtor',
    'Daytime_evening_attendance',
    'Scholarship_holder',
    'Previous_qualification_grade',
    'Unemployment_rate'
]

# Semua fitur model
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

# Nilai default untuk fitur non-penting
default_values = {
    'Marital_status': 1,
    'Application_mode': 1,
    'Application_order': 0,
    'Course': 9500,  # Nilai yang lebih realistis
    'Previous_qualification': 1,
    'Nacionality': 1,
    'Mothers_qualification': 1,
    'Fathers_qualification': 1,
    'Mothers_occupation': 1,
    'Fathers_occupation': 1,
    'Displaced': 0,
    'Educational_special_needs': 0,
    'Gender': 0,
    'International': 0,
    'Curricular_units_1st_sem_credited': 0,
    'Curricular_units_1st_sem_enrolled': 0,
    'Curricular_units_1st_sem_evaluations': 0,
    'Curricular_units_1st_sem_approved': 0,
    'Curricular_units_1st_sem_without_evaluations': 0,
    'Curricular_units_2nd_sem_credited': 0,
    'Curricular_units_2nd_sem_enrolled': 0,
    'Curricular_units_2nd_sem_evaluations': 0,
    'Curricular_units_2nd_sem_approved': 0,
    'Curricular_units_2nd_sem_without_evaluations': 0,
    'Inflation_rate': 1.4,
    'GDP': 2.1
}

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Status Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# Header
st.title("🎓 Prediksi Status Akademik Mahasiswa")
st.markdown("""
**Isi data utama berikut untuk prediksi status mahasiswa.**
Fitur lain akan diisi dengan nilai default secara otomatis.
""")

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("ℹ️ Panduan Penggunaan")
    st.markdown("""
    1. Isi semua field pada form utama
    2. Klik tombol **Prediksi Status**
    3. Lihat hasil prediksi dan probabilitas
    """)
    st.divider()
    st.subheader("Keterangan Fitur")
    st.markdown("""
    - **Nilai Semester**: Rata-rata nilai (skala 0-20)
    - **Status Pembayaran**: Ketepatan pembayaran biaya kuliah
    - **Nilai Masuk**: Nilai seleksi masuk (skala 0-200)
    - **Status Utang**: Memiliki tunggakan biaya akademik
    - **Attendance**: Jenis kelas yang diikuti
    """)

# Form input user
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        grade1 = st.slider(
            "Nilai Rata-rata Semester 1 (0-20)",
            0.0, 20.0, 10.0, 0.1,
            help="Nilai akademik semester pertama"
        )
        grade2 = st.slider(
            "Nilai Rata-rata Semester 2 (0-20)",
            0.0, 20.0, 10.0, 0.1,
            help="Nilai akademik semester kedua"
        )
        admission = st.slider(
            "Nilai Masuk (0-200)", 
            0.0, 200.0, 120.0, 0.1,
            help="Nilai seleksi masuk perguruan tinggi"
        )
        previous = st.slider(
            "Nilai Prestasi Akademik Sebelumnya (0-200)", 
            0.0, 200.0, 120.0, 0.1,
            help="Nilai prestasi akademik sebelum masuk"
        )
        
    with col2:
        age = st.slider(
            "Usia Saat Mendaftar", 
            16, 60, 20,
            help="Usia saat pertama kali mendaftar kuliah"
        )
        unemployment = st.slider(
            "Tingkat Pengangguran (%)", 
            0.0, 20.0, 5.0, 0.1,
            help="Faktor ekonomi makro wilayah"
        )
        attendance = st.selectbox(
            "Jenis Kelas", 
            options=[1, 0], 
            format_func=lambda x: "Kelas Sore" if x == 1 else "Kelas Siang",
            help="Jadwal perkuliahan yang diikuti"
        )
        fees = st.selectbox(
            "Pembayaran Tepat Waktu?", 
            options=[1, 0], 
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
        debtor = st.selectbox(
            "Memiliki Utang Akademik?", 
            options=[1, 0], 
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
        scholarship = st.selectbox(
            "Penerima Beasiswa?", 
            options=[1, 0], 
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
    
    submitted = st.form_submit_button("🔍 Prediksi Status", use_container_width=True)

# Proses prediksi
if submitted:
    # Inisialisasi array input
    full_input = np.zeros(len(all_features))
    
    # Mapping input user ke fitur
    user_inputs = {
        'Curricular_units_2nd_sem_grade': grade2,
        'Curricular_units_1st_sem_grade': grade1,
        'Tuition_fees_up_to_date': fees,
        'Admission_grade': admission,
        'Age_at_enrollment': age,
        'Debtor': debtor,
        'Daytime_evening_attendance': attendance,
        'Scholarship_holder': scholarship,
        'Previous_qualification_grade': previous,
        'Unemployment_rate': unemployment
    }
    
    # Isi nilai dari user
    for feature, value in user_inputs.items():
        idx = all_features.index(feature)
        full_input[idx] = value
    
    # Isi nilai default untuk fitur lainnya
    for feature, default_value in default_values.items():
        if feature not in user_inputs:
            idx = all_features.index(feature)
            full_input[idx] = default_value
    
    # Transformasi dan prediksi
    try:
        input_scaled = scaler.transform(full_input.reshape(1, -1))
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        st.stop()

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")
    result_col = st.columns(3)
    
    with result_col[0]:
        if prediction == "Dropout":
            st.error("## Dropout")
            st.metric("Probabilitas", f"{proba[0]*100:.1f}%")
        else:
            st.write("## Dropout")
            st.metric("Probabilitas", f"{proba[0]*100:.1f}%")
    
    with result_col[1]:
        if prediction == "Enrolled":
            st.info("## Terdaftar")
            st.metric("Probabilitas", f"{proba[1]*100:.1f}%")
        else:
            st.write("## Terdaftar")
            st.metric("Probabilitas", f"{proba[1]*100:.1f}%")
    
    with result_col[2]:
        if prediction == "Graduate":
            st.success("## Lulus")
            st.metric("Probabilitas", f"{proba[2]*100:.1f}%")
        else:
            st.write("## Lulus")
            st.metric("Probabilitas", f"{proba[2]*100:.1f}%")
    
    # Visualisasi probabilitas
    st.subheader("Distribusi Probabilitas Status")
    labels = model.classes_
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, proba, color=['#FF4B4B', '#1C83E1', '#00C897'])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Probabilitas")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tambah label nilai
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.03, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', 
                ha='left', va='center')
    
    st.pyplot(fig)
    
    # Rekomendasi berdasarkan hasil
    st.subheader("🔍 Analisis Rekomendasi")
    if prediction == "Dropout":
        st.warning("""
        **Rekomendasi Intervensi:**
        - Tingkatkan bimbingan akademik
        - Evaluasi masalah finansial
        - Berikan dukungan mental/konseling
        - Tinjau beban studi
        """)
    elif prediction == "Enrolled":
        st.info("""
        **Rekomendasi:**
        - Pertahankan performa akademik
        - Monitor perkembangan semester berikutnya
        - Identifikasi mata kuliah bermasalah
        """)
    else:
        st.success("""
        **Rekomendasi:**
        - Persiapkan kelulusan tepat waktu
        - Berikan informasi lanjutan studi/karier
        - Evaluasi keberhasilan program studi
        """)

# Footer
st.divider()
st.caption("© 2025 Dashboard Prediksi Status Mahasiswa - Sistem Pendukung Keputusan Akademik")
