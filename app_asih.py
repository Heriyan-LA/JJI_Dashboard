import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load('model/model_rf.pkl')
scaler = joblib.load('model/scaler.pkl')

# Fitur penting
important_features = [
    'Age_at_enrollment',
    'Admission_grade',
    'Gender',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade'
]

# Semua fitur
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

# Nilai default
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

# UI halaman
st.set_page_config(page_title="Analisis Status Mahasiswa", layout="wide")
st.title("🎓 Dashboard Analisis Status Akademik Mahasiswa")

# Tab untuk berbagai fungsi
tab1, tab2 = st.tabs(["📊 Dashboard Analitik", "🤖 Prediksi & Solusi"])

with tab1:
    st.header("Analisis Karakteristik Siswa Dropout (DO)")
    st.markdown("""
    Visualisasi ini menunjukkan karakteristik umum siswa yang melakukan Dropout (DO) berdasarkan dataset pendidikan.
    """)
    
    # Generate sample data for visualization (in real app, this would come from actual dataset)
    np.random.seed(42)
    sample_size = 200
    
    # Create synthetic data for visualization
    data = pd.DataFrame({
        'Age_at_enrollment': np.random.randint(18, 35, size=sample_size),
        'Admission_grade': np.random.normal(120, 20, sample_size).clip(60, 200),
        'Gender': np.random.choice([0, 1], sample_size, p=[0.45, 0.55]),
        'Tuition_fees_up_to_date': np.random.choice([0, 1], sample_size, p=[0.7, 0.3]),
        'Curricular_units_1st_sem_grade': np.random.normal(10, 4, sample_size).clip(0, 20),
        'Curricular_units_2nd_sem_grade': np.random.normal(9, 4, sample_size).clip(0, 20),
        'Status': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], sample_size, p=[0.4, 0.3, 0.3])
    })
    
    # Filter dropout students
    dropout_data = data[data['Status'] == 'Dropout']
    
    # Layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Usia Siswa DO")
        fig, ax = plt.subplots()
        sns.histplot(dropout_data['Age_at_enrollment'], bins=10, kde=True, ax=ax, color='#FF6B6B')
        ax.set_xlabel("Usia")
        ax.set_ylabel("Jumlah Siswa")
        ax.set_title("Distribusi Usia Mahasiswa DO")
        st.pyplot(fig)
        
        st.subheader("Perbandingan Nilai Semester")
        fig, ax = plt.subplots()
        semester_means = dropout_data[['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']].mean()
        ax.bar(['Semester 1', 'Semester 2'], semester_means, color=['#4ECDC4', '#FF6B6B'])
        ax.set_ylabel("Rata-rata Nilai")
        ax.set_title("Perbandingan Nilai Semester Siswa DO")
        ax.set_ylim(0, 20)
        for i, v in enumerate(semester_means):
            ax.text(i, v + 0.5, f"{v:.1f}", ha='center')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Status Pembayaran Siswa DO")
        payment_counts = dropout_data['Tuition_fees_up_to_date'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(payment_counts, labels=['Tepat Waktu', 'Terlambat'], 
               colors=['#4ECDC4', '#FF6B6B'], autopct='%1.1f%%')
        ax.set_title("Proporsi Ketepatan Pembayaran")
        st.pyplot(fig)
        
        st.subheader("Hubungan Nilai Masuk dan Semester 1")
        fig, ax = plt.subplots()
        sns.scatterplot(data=dropout_data, x='Admission_grade', y='Curricular_units_1st_sem_grade', 
                        hue='Gender', palette={0: '#FF6B6B', 1: '#4E79A7'}, ax=ax)
        ax.set_xlabel("Nilai Masuk")
        ax.set_ylabel("Nilai Semester 1")
        ax.set_title("Korelasi Nilai Masuk dan Performa Awal")
        ax.legend(title='Gender', labels=['Perempuan', 'Laki-laki'])
        st.pyplot(fig)
    
    st.subheader("Trend Nilai per Status")
    fig, ax = plt.subplots(figsize=(10, 5))
    status_means = data.groupby('Status')[['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']].mean()
    status_means.plot(kind='line', marker='o', ax=ax, color=['#4ECDC4', '#FF6B6B'])
    ax.set_ylabel("Rata-rata Nilai")
    ax.set_xlabel("Status Mahasiswa")
    ax.set_title("Perbandingan Rata-rata Nilai per Status")
    ax.set_xticks(range(len(status_means.index)))
    ax.set_xticklabels(status_means.index)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

with tab2:
    st.header("Prediksi Status Mahasiswa & Solusi")
    st.markdown("Isi data utama berikut untuk prediksi status mahasiswa:")
    
    # Form input user
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Usia saat mendaftar", 16, 60, 20)
            admission = st.slider("Nilai masuk (0–200)", 0.0, 200.0, 120.0, step=0.1)
            grade1 = st.slider("Rata-rata nilai semester 1", 0.0, 20.0, 10.0, step=0.1)
        
        with col2:
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
            debtor = st.selectbox("Debtor (utang akademik)?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            fees = st.selectbox("Apakah bayar SPP tepat waktu?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            grade2 = st.slider("Rata-rata nilai semester 2", 0.0, 20.0, 10.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Prediksi Status", use_container_width=True)

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
        result_col = st.columns(3)
        
        with result_col[0]:
            st.metric("Status Prediksi", prediction)
        
        with result_col[1]:
            st.metric("Probabilitas", f"{max(proba)*100:.1f}%")
        
        with result_col[2]:
            if prediction == "Dropout":
                st.error("⚠️ Risiko Tinggi")
            elif prediction == "Enrolled":
                st.warning("ℹ️ Status Normal")
            else:
                st.success("✅ Status Optimal")

        # Visualisasi probabilitas
        st.subheader("Distribusi Probabilitas")
        labels = model.classes_
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF6B6B' if lbl == "Dropout" else '#4ECDC4' if lbl == "Enrolled" else '#1A535C' for lbl in labels]
        bars = ax.barh(labels, proba, color=colors)
        ax.set_xlim(0, 1)
        ax.set_title("Probabilitas Status Mahasiswa")
        ax.set_xlabel("Probabilitas")
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.03, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', 
                    ha='left', va='center')
        
        st.pyplot(fig)
        
        # Rekomendasi solusi
        st.subheader("🚀 Rekomendasi Solusi")
        
        if prediction == "Dropout":
            st.error("### Strategi Pencegahan Dropout")
            st.markdown("""
            **Intervensi yang Direkomendasikan:**
            - 🧠 **Program Bimbingan Akademik**: 
              - Konsultasi mata kuliah bermasalah 2x/minggu
              - Kelas tambahan untuk mata kuliah dengan nilai < 12
              
            - 💰 **Dukungan Finansial**:
              - Peninjauan ulang skema pembayaran
              - Beasiswa darurat untuk mahasiswa berprestasi
              - Program kerja paruh waktu kampus
              
            - ❤️ **Dukungan Psikologis**:
              - Konseling motivasi 1x/minggu
              - Program mentoring dengan alumni
              - Kelompok dukungan sebaya
              
            - 📚 **Penyesuaian Beban Studi**:
              - Opsi pengurangan SKS semester berikutnya
              - Fleksibilitas penjadwalan kuliah
            """)
            
            with st.expander("📈 Action Plan Detail"):
                st.markdown("""
                | Aktivitas | Frekuensi | Target |
                |----------|-----------|--------|
                | Konseling akademik | 2x/minggu | Tingkatkan nilai > 2 poin |
                | Workshop manajemen waktu | 1x/bulan | Tingkatkan kehadiran > 90% |
                | Program mentoring | 1x/minggu | Bangun jaringan dukungan |
                | Evaluasi beban studi | Per semester | Sesuaikan dengan kemampuan |
                """)
        
        elif prediction == "Enrolled":
            st.warning("### Strategi Optimalisasi Performa")
            st.markdown("""
            **Rekomendasi Pengembangan:**
            - 🎯 **Program Prestasi Akademik**:
              - Pelatihan teknik belajar efektif
              - Kompetisi akademik internal kampus
              - Akses ke sumber belajar premium
              
            - 🤝 **Pengembangan Jaringan Profesional**:
              - Magang industri selama liburan
              - Program mentoring dengan profesional
              - Keanggotaan organisasi mahasiswa
              
            - 🌐 **Pengembangan Keterampilan Tambahan**:
              - Pelatihan soft skills (komunikasi, kepemimpinan)
              - Sertifikasi kompetensi digital
              - Program bahasa asing intensif
            """)
            
            with st.expander("📈 Action Plan Detail"):
                st.markdown("""
                | Aktivitas | Frekuensi | Target |
                |----------|-----------|--------|
                | Pelatihan soft skills | 1x/bulan | Dapatkan 2 sertifikat |
                | Program magang | Minimal 1x | Dapatkan pengalaman industri |
                | Kompetisi akademik | 2x/semester | Raih minimal 1 penghargaan |
                """)
        
        else:
            st.success("### Strategi Percepatan Kelulusan")
            st.markdown("""
            **Rencana Persiapan Kelulusan:**
            - 🎓 **Akselerasi Studi**:
              - Opsi mengambil SKS tambahan
              - Program penelitian mandiri
              - Penyelesaian tugas akhir intensif
              
            - 🌟 **Pengembangan Karir**:
              - Workshop persiapan karir
              - Simulasi wawancara kerja
              - Kunjungan industri
              
            - 🔗 **Jaringan Alumni**:
              - Mentoring dengan alumni di industri
              - Forum alumni khusus jurusan
              - Program job matching kampus
              
            - 📝 **Persiapan Studi Lanjut**:
              - Konsultasi aplikasi pascasarjana
              - Persiapan tes masuk (GRE, GMAT, dll)
              - Bantuan aplikasi beasiswa
            """)
            
            with st.expander("📈 Action Plan Detail"):
                st.markdown("""
                | Aktivitas | Frekuensi | Target |
                |----------|-----------|--------|
                | Workshop karir | 1x/bulan | Terhubung dengan 5+ profesional |
                | Penyelesaian tugas akhir | - | Selesai 1 bulan sebelum deadline |
                | Aplikasi beasiswa | - | Apply ke 5+ program beasiswa |
                """)
        
        st.divider()
        st.markdown("💡 **Tips**: Untuk hasil lebih akurat, pastikan semua data yang dimasukkan valid dan terkini.")

# Footer
st.divider()
st.caption("© 2025 Dashboard Analisis Status Mahasiswa | Sistem Pendukung Keputusan Akademik")
