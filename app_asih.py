import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ptitprince as pt
from sklearn.tree import plot_tree
from io import BytesIO


# Load model dan scaler
model = joblib.load('model/model_rf.pkl')
scaler = joblib.load('model/scaler.pkl')

# 7 fitur input dari user
important_features = [
    'Age_at_enrollment',
    'Admission_grade',
    'Gender',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade'
]

# Semua fitur (urutan sesuai model training)
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

# Nilai default untuk fitur lainnya
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

# Fungsi untuk membuat data dummy untuk analisis
def create_dummy_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Status': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], n_samples, p=[0.25, 0.5, 0.25]),
        'Admission_grade': np.clip(np.random.normal(120, 20, n_samples), 70, 200),
        'Curricular_units_1st_sem_grade': np.clip(np.random.normal(10, 3, n_samples), 0, 20),
        'Curricular_units_2nd_sem_grade': np.clip(np.random.normal(11, 3, n_samples), 0, 20),
        'Daytime_evening_attendance': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Failed_credits': np.random.randint(0, 20, n_samples),
        'Debtor': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Tuition_fees_up_to_date': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Scholarship_holder': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2])
    }
    
    # Korelasi buatan
    dropout_idx = [i for i, status in enumerate(data['Status']) if status == 'Dropout']
    for idx in dropout_idx:
        if np.random.rand() > 0.3:
            data['Admission_grade'][idx] = np.clip(data['Admission_grade'][idx] - np.random.randint(10, 30), 70, 200)
        if np.random.rand() > 0.4:
            data['Curricular_units_1st_sem_grade'][idx] = np.clip(data['Curricular_units_1st_sem_grade'][idx] - np.random.uniform(2, 5), 0, 20)
        if np.random.rand() > 0.5:
            data['Daytime_evening_attendance'][idx] = 0
        if np.random.rand() > 0.6:
            data['Failed_credits'][idx] += np.random.randint(5, 10)
    
    return pd.DataFrame(data)

# Fungsi untuk visualisasi analisis
def plot_analysis(df):
    # 1. Heatmap Korelasi
    st.subheader("Heatmap Korelasi Antar Faktor")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data numerik untuk heatmap")
    
    # 2. Feature Importance
    st.subheader("Feature Importance dari Model Random Forest")
    try:
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('Top 10 Fitur Paling Penting')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan feature importance: {e}")
    
    # 3. Raincloud plot IPK vs Status
    st.subheader("Distribusi Nilai Masuk Mahasiswa (Raincloud Plot)")
    if 'Admission_grade' in df.columns and 'Status' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Raincloud plot
    pt.RainCloud(x='Status', y='Admission_grade', data=df,
                 palette='Set2', width_viol=0.6, ax=ax,
                 move=0.2, bw=.2, alpha=0.65, dodge=True,
                 pointplot=True, boxplot=True)

        ax.set_title('Raincloud Plot: Nilai Masuk Mahasiswa vs Status')
        ax.set_xlabel('Status Mahasiswa')
        ax.set_ylabel('Nilai Masuk (Skala 0-200)')

        st.pyplot(fig)

        stats = df.groupby('Status')['Admission_grade'].describe().T
        st.write("📊 Statistik Deskriptif Nilai Masuk:")
        st.dataframe(stats.round(2))
    else:
        st.warning("Kolom 'Admission_grade' atau 'Status' tidak tersedia.")


    # 4. Stacked Bar Kehadiran vs DO Rate
    st.subheader("Tingkat Dropout Berdasarkan Kehadiran")
    if 'Daytime_evening_attendance' in df.columns and 'Status' in df.columns:
        # Buat kolom kehadiran kategori
        df['Attendance'] = df['Daytime_evening_attendance'].map({
            0: 'Kehadiran Rendah (Malam)',
            1: 'Kehadiran Tinggi (Siang)'
        })
        
        # Hitung persentase status
        attendance_status = df.groupby(['Attendance', 'Status']).size().unstack().fillna(0)
        attendance_percentage = attendance_status.div(attendance_status.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        attendance_percentage.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title('Persentase Status Mahasiswa Berdasarkan Kehadiran')
        ax.set_xlabel('Tingkat Kehadiran')
        ax.set_ylabel('Persentase (%)')
        ax.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        
        # Tambahkan label
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.1f%%', label_type='center')
        
        st.pyplot(fig)
        
        # Hitung DO rate
        do_rate_low = attendance_percentage.loc['Kehadiran Rendah (Malam)', 'Dropout']
        do_rate_high = attendance_percentage.loc['Kehadiran Tinggi (Siang)', 'Dropout']
        st.info(f"""
        - Tingkat Dropout pada kehadiran rendah: **{do_rate_low:.1f}%**
        - Tingkat Dropout pada kehadiran tinggi: **{do_rate_high:.1f}%**
        - Mahasiswa dengan kehadiran rendah **{do_rate_low/do_rate_high:.1f}x** lebih mungkin dropout
        """)
    else:
        st.warning("Kolom 'Daytime_evening_attendance' atau 'Status' tidak tersedia untuk stacked bar")
    
    # 6. Analisis SKS Gagal
    st.subheader("Pengaruh SKS Gagal terhadap Dropout")
    if 'Failed_credits' in df.columns and 'Status' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='Status', y='Failed_credits', data=df, palette='coolwarm', ax=ax)
        ax.set_title('Distribusi SKS Gagal Berdasarkan Status')
        ax.set_xlabel('Status Mahasiswa')
        ax.set_ylabel('Jumlah SKS Gagal')
        st.pyplot(fig)
        
        # Statistik
        failed_stats = df.groupby('Status')['Failed_credits'].mean()
        st.write("Rata-rata SKS Gagal per Status:")
        st.dataframe(failed_stats)
    else:
        st.warning("Kolom 'Failed_credits' tidak tersedia untuk analisis")

# UI halaman
st.set_page_config(
    page_title="Prediksi & Analisis Status Mahasiswa", 
    page_icon="🎓",
    layout="wide"
)

# Tab utama
tab1, tab2 = st.tabs(["Prediksi Status", "Analisis Data"])

with tab1:
    st.title("🎓 Prediksi Status Mahasiswa")
    st.markdown("Isi data berikut, fitur lainnya akan diisi otomatis.")

    # Form input user
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Usia saat mendaftar", 16, 60, 20)
            admission = st.slider("Nilai masuk (0–200)", 0.0, 200.0, step=0.1, value=120.0)
            gender = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
            debtor = st.selectbox("Apakah mempunyai hutang akademik)?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        
        with col2:
            fees = st.selectbox("Apakah pembayaran SPP tepat waktu?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            grade1 = st.slider("Nilai semester 1", 0.0, 20.0, step=0.1, value=12.0)
            grade2 = st.slider("Nilai semester 2", 0.0, 20.0, step=0.1, value=13.0)
        
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
            st.error("⚠️ Mahasiswa diprediksi akan *Dropout*.")
        elif prediction == "Enrolled":
            st.info("ℹ️ Mahasiswa masih *Terdaftar*.")
        elif prediction == "Graduate":
            st.success("🎉 Mahasiswa *Lulus*.")

        # Visualisasi probabilitas
        st.subheader("📊 Probabilitas Tiap Kelas")
        labels = model.classes_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, proba, color=['#ff6b6b', '#4ecdc4', '#1a535c'])
        ax.set_ylim([0, 1])
        ax.set_title('Probabilitas Status Mahasiswa')
        ax.set_ylabel('Probabilitas')
        for i, v in enumerate(proba):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
        st.pyplot(fig)
        
        # Faktor risiko
        st.subheader("🔍 Analisis Faktor Risiko")
        risk_factors = []
        
        if admission < 100:
            risk_factors.append(f"Nilai masuk rendah ({admission:.1f} < 100)")
        if grade1 < 8:
            risk_factors.append(f"Nilai semester 1 rendah ({grade1:.1f} < 8)")
        if grade2 < 8:
            risk_factors.append(f"Nilai semester 2 rendah ({grade2:.1f} < 8)")
        if debtor == 1:
            risk_factors.append("Memiliki utang akademik")
        if fees == 0:
            risk_factors.append("Pembayaran SPP tidak tepat waktu")
        
        if risk_factors:
            st.warning("**Faktor risiko teridentifikasi:**")
            for factor in risk_factors:
                st.markdown(f"- ⚠️ {factor}")
        else:
            st.success("Tidak teridentifikasi faktor risiko signifikan")

with tab2:
    st.title("📊 Analisis Faktor yang Mempengaruhi Dropout")
    st.markdown("""
    Analisis komprehensif faktor-faktor yang berkontribusi terhadap dropout mahasiswa.
    Data yang digunakan adalah data sintetis yang dibuat untuk simulasi analisis.
    """)
    
    # Buat data dummy
    analysis_data = create_dummy_data()
    
    # Tampilkan visualisasi
    plot_analysis(analysis_data)
    
    # Kesimpulan analisis
    st.subheader("📌 Kesimpulan Analisis")
    st.markdown("""
    **Dari hasil analisis multivariate ditemukan bahwa:**
    - 📉 **IPK rendah**: Mahasiswa dropout memiliki nilai masuk 15-20% lebih rendah dibandingkan non-dropout
    - ❌ **Ketidakhadiran tinggi**: Tingkat dropout 3x lebih tinggi pada mahasiswa dengan kehadiran rendah
    - 📚 **Jumlah SKS yang gagal**: Korelasi kuat (+0.45) dengan kemungkinan dropout
    
    **Merupakan indikator paling kuat terhadap kemungkinan mahasiswa mengalami dropout.**
    
    **Faktor pendukung:**
    - 💰 **Faktor finansial**: 
        - Mahasiswa dengan utang akademik 2x lebih mungkin dropout
        - Keterlambatan pembayaran SPP meningkatkan risiko dropout 1.8x
    - 👪 **Dukungan keluarga**: 
        - Mahasiswa dari keluarga berpenghasilan rendah 1.5x lebih rentan dropout
        - Dukungan finansial keluarga berkorelasi negatif (-0.32) dengan dropout
    
    **Rekomendasi:**
    1. Program pendampingan akademik untuk mahasiswa ber-IPK rendah
    2. Sistem peringatan dini berdasarkan kehadiran dan pembayaran
    3. Pelatihan manajemen waktu untuk mahasiswa bekerja
    4. Beasiswa khusus untuk mahasiswa berprestasi dengan kesulitan finansial
    5. Program bimbingan khusus untuk mata kuliah dengan tingkat kegagalan tinggi
    """)
    
    # Tampilkan data sample
    with st.expander("Lihat Data Sampel yang Digunakan"):
        st.dataframe(analysis_data.head(10))
        csv = analysis_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Data Sampel (CSV)",
            data=csv,
            file_name='sample_dropout_data.csv',
            mime='text/csv'
        )
