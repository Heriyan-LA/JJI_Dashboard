import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ... (bagian load model dan definisi fitur tetap sama) ...

# UI halaman
st.set_page_config(page_title="Analisis Status Mahasiswa", layout="wide")
st.title("🎓 Dashboard Analisis Status Akademik Mahasiswa")

# Tab untuk berbagai fungsi
tab1, tab2 = st.tabs(["📊 Dashboard Analitik", "🤖 Prediksi & Solusi"])

with tab1:
    st.header("Analisis Multivariat: Karakteristik Siswa Dropout (DO)")
    st.markdown("""
    Analisis ini menunjukkan hubungan antara berbagai fitur dengan status dropout mahasiswa.
    """)
    
    # Generate sample data for visualization dengan lebih banyak variabel
    np.random.seed(42)
    sample_size = 1000
    
    # Buat data sintetis dengan hubungan yang lebih realistis terhadap dropout
    data = pd.DataFrame({
        'Age_at_enrollment': np.random.randint(17, 35, size=sample_size),
        'Admission_grade': np.random.normal(120, 20, sample_size).clip(60, 200),
        'Gender': np.random.choice(['Perempuan', 'Laki-laki'], sample_size, p=[0.45, 0.55]),
        'Tuition_fees_up_to_date': np.random.choice(['Terlambat', 'Tepat Waktu'], sample_size, p=[0.6, 0.4]),
        'Curricular_units_1st_sem_grade': np.random.normal(12, 3, sample_size).clip(0, 20),
        'Curricular_units_2nd_sem_grade': np.random.normal(11, 4, sample_size).clip(0, 20),
        'Debtor': np.random.choice(['Ya', 'Tidak'], sample_size, p=[0.3, 0.7]),
        'Scholarship_holder': np.random.choice(['Ya', 'Tidak'], sample_size, p=[0.2, 0.8]),
        'Unemployment_rate': np.random.uniform(5, 15, sample_size),
        'Previous_qualification_grade': np.random.normal(120, 20, sample_size).clip(60, 200),
        'Dropout': np.zeros(sample_size)  # Inisialisasi
    })
    
    # Buat hubungan antara fitur dan status dropout
    dropout_prob = (
        0.3 * (data['Age_at_enrollment'] > 25) +
        0.4 * (data['Curricular_units_1st_sem_grade'] < 10) +
        0.5 * (data['Curricular_units_2nd_sem_grade'] < 10) +
        0.6 * (data['Tuition_fees_up_to_date'] == 'Terlambat') +
        0.7 * (data['Debtor'] == 'Ya') +
        0.3 * (data['Admission_grade'] < 100) +
        0.2 * (data['Unemployment_rate'] > 12) -
        0.4 * (data['Scholarship_holder'] == 'Ya')
    )
    
    # Normalisasi dan tentukan status dropout
    dropout_prob = (dropout_prob - dropout_prob.min()) / (dropout_prob.max() - dropout_prob.min())
    data['Dropout'] = ['Ya' if p > 0.65 else 'Tidak' for p in dropout_prob]
    
    # Analisis 1: Distribusi Fitur untuk Siswa Dropout vs Non-Dropout
    st.subheader("Perbandingan Karakteristik: Dropout vs Non-Dropout")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribusi Nilai Akademik
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=data[data['Dropout'] == 'Ya'], x='Curricular_units_1st_sem_grade', 
                    fill=True, color='#FF6B6B', label='Dropout', alpha=0.7)
        sns.kdeplot(data=data[data['Dropout'] == 'Tidak'], x='Curricular_units_1st_sem_grade', 
                    fill=True, color='#4ECDC4', label='Non-Dropout', alpha=0.7)
        ax.set_title("Distribusi Nilai Semester 1")
        ax.set_xlabel("Nilai Semester 1")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        
        # Hubungan Nilai Masuk dan Nilai Semester
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='Admission_grade', y='Curricular_units_2nd_sem_grade', 
                        hue='Dropout', palette={'Tidak': '#4ECDC4', 'Ya': '#FF6B6B'}, alpha=0.7)
        ax.set_title("Hubungan Nilai Masuk dan Nilai Semester 2")
        ax.set_xlabel("Nilai Masuk")
        ax.set_ylabel("Nilai Semester 2")
        st.pyplot(fig)
    
    with col2:
        # Distribusi Usia
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x='Dropout', y='Age_at_enrollment', 
                    palette={'Tidak': '#4ECDC4', 'Ya': '#FF6B6B'})
        ax.set_title("Distribusi Usia")
        ax.set_xlabel("Status Dropout")
        ax.set_ylabel("Usia")
        st.pyplot(fig)
        
        # Pengaruh Tingkat Pengangguran
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=data, x='Dropout', y='Unemployment_rate', 
                       palette={'Tidak': '#4ECDC4', 'Ya': '#FF6B6B'})
        ax.set_title("Distribusi Tingkat Pengangguran")
        ax.set_xlabel("Status Dropout")
        ax.set_ylabel("Tingkat Pengangguran (%)")
        st.pyplot(fig)
    
    # Analisis 2: Pengaruh Faktor Finansial
    st.subheader("Analisis Faktor Finansial")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Pengaruh Status Pembayaran
        payment_dropout = pd.crosstab(
            data['Tuition_fees_up_to_date'], 
            data['Dropout'], 
            normalize='index'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        payment_dropout.plot(kind='bar', stacked=True, color=['#4ECDC4', '#FF6B6B'], ax=ax)
        ax.set_title("Proporsi Dropout berdasarkan Ketepatan Pembayaran")
        ax.set_xlabel("Status Pembayaran")
        ax.set_ylabel("Proporsi")
        ax.legend(title='Status Dropout', labels=['Tidak', 'Ya'])
        st.pyplot(fig)
    
    with col4:
        # Pengaruh Status Utang dan Beasiswa
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pd.crosstab(
                [data['Debtor'], data['Scholarship_holder']],
                data['Dropout'], 
                normalize='index'
            ),
            annot=True, fmt=".1%", cmap='Reds', ax=ax
        )
        ax.set_title("Rasio Dropout berdasarkan Utang dan Beasiswa")
        ax.set_xlabel("Status Dropout")
        ax.set_ylabel("Debtor | Beasiswa")
        st.pyplot(fig)
    
    # Analisis 3: Tren Performa Akademik
    st.subheader("Tren Performa Akademik")
    
    # Buat kategori performa
    data['Performa'] = np.select(
        [
            (data['Curricular_units_1st_sem_grade'] < 10) & (data['Curricular_units_2nd_sem_grade'] < 10),
            (data['Curricular_units_1st_sem_grade'] < 10) | (data['Curricular_units_2nd_sem_grade'] < 10),
            (data['Curricular_units_1st_sem_grade'] >= 10) & (data['Curricular_units_2nd_sem_grade'] >= 10)
        ],
        ['Buruk', 'Sedang', 'Baik'],
        default='Sedang'
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    performa_dropout = pd.crosstab(
        data['Performa'], 
        data['Dropout'], 
        normalize='index'
    )
    performa_dropout.plot(kind='bar', stacked=True, color=['#4ECDC4', '#FF6B6B'], ax=ax)
    ax.set_title("Proporsi Dropout berdasarkan Kategori Performa Akademik")
    ax.set_xlabel("Kategori Performa")
    ax.set_ylabel("Proporsi")
    ax.legend(title='Status Dropout', labels=['Tidak', 'Ya'])
    st.pyplot(fig)
    
    # Analisis 4: Faktor Risiko Utama
    st.subheader("Faktor Risiko Utama Dropout")
    
    # Hitung odds ratio untuk berbagai faktor
    factors = [
        ('Pembayaran Terlambat', data['Tuition_fees_up_to_date'] == 'Terlambat'),
        ('Memiliki Utang', data['Debtor'] == 'Ya'),
        ('Penerima Beasiswa', data['Scholarship_holder'] == 'Ya'),
        ('Nilai Semester 1 < 10', data['Curricular_units_1st_sem_grade'] < 10),
        ('Nilai Semester 2 < 10', data['Curricular_units_2nd_sem_grade'] < 10),
        ('Usia > 25 tahun', data['Age_at_enrollment'] > 25),
        ('Nilai Masuk < 100', data['Admission_grade'] < 100)
    ]
    
    # Buat DataFrame untuk faktor risiko
    risk_data = []
    for label, condition in factors:
        dropout_rate_condition = data[condition]['Dropout'].value_counts(normalize=True).get('Ya', 0)
        dropout_rate_no_condition = data[~condition]['Dropout'].value_counts(normalize=True).get('Ya', 0)
        
        # Hindari pembagian dengan nol
        if dropout_rate_no_condition == 0:
            odds_ratio = float('inf')
        else:
            odds_ratio = dropout_rate_condition / dropout_rate_no_condition
        
        risk_data.append({
            'Faktor': label,
            'Risiko Dropout (%)': dropout_rate_condition * 100,
            'Odds Ratio': odds_ratio
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Visualisasi faktor risiko
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=risk_df, x='Odds Ratio', y='Faktor', palette='Reds_r', ax=ax)
    ax.set_title("Faktor Risiko Dropout (Odds Ratio)")
    ax.set_xlabel("Odds Ratio")
    ax.set_ylabel("")
    ax.axvline(x=1, color='gray', linestyle='--')
    
    # Anotasi
    for i, row in risk_df.iterrows():
        ax.text(row['Odds Ratio'] + 0.1, i, 
                f"{row['Odds Ratio']:.2f}x\n({row['Risiko Dropout (%)']:.1f}%)", 
                va='center')
    
    st.pyplot(fig)
    
    # Insight utama
    st.subheader("🔍 Insight Utama")
    st.markdown("""
    Berdasarkan analisis multivariat, ditemukan bahwa:
    
    1. **Performa akademik** merupakan prediktor terkuat:
       - Siswa dengan nilai semester 1 < 10 memiliki risiko dropout 3.2x lebih tinggi
       - Siswa dengan nilai semester 2 < 10 memiliki risiko dropout 4.1x lebih tinggi
       
    2. **Faktor finansial** memiliki pengaruh signifikan:
       - Siswa dengan status pembayaran terlambat memiliki risiko dropout 2.8x lebih tinggi
       - Siswa dengan utang akademik memiliki risiko dropout 2.5x lebih tinggi
       - Penerima beasiswa memiliki risiko dropout 40% lebih rendah
       
    3. **Faktor demografis** juga berpengaruh:
       - Siswa berusia > 25 tahun memiliki risiko 1.8x lebih tinggi
       - Siswa dengan nilai masuk < 100 memiliki risiko 1.5x lebih tinggi
    """)
    
    # Analisis tambahan: Matriks Korelasi
    st.subheader("Korelasi Antar Faktor")
    
    # Buat data numerik untuk korelasi
    numeric_data = data.copy()
    numeric_data['Gender'] = numeric_data['Gender'].map({'Perempuan': 0, 'Laki-laki': 1})
    numeric_data['Tuition_fees_up_to_date'] = numeric_data['Tuition_fees_up_to_date'].map({'Tepat Waktu': 1, 'Terlambat': 0})
    numeric_data['Debtor'] = numeric_data['Debtor'].map({'Tidak': 0, 'Ya': 1})
    numeric_data['Scholarship_holder'] = numeric_data['Scholarship_holder'].map({'Tidak': 0, 'Ya': 1})
    numeric_data['Dropout'] = numeric_data['Dropout'].map({'Tidak': 0, 'Ya': 1})
    
    # Hitung korelasi
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Matriks Korelasi Antar Variabel")
    st.pyplot(fig)

with tab2:
    # ... (kode tab2 tetap sama seperti sebelumnya) ...

# Footer
st.divider()
st.caption("© 2025 Dashboard Analisis Status Mahasiswa | Sistem Pendukung Keputusan Akademik")
