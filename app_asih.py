%%writefile app_asih.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load model dan scaler
try:
    model = joblib.load('model/model_rf.pkl')
    scaler = joblib.load('model/scaler.pkl')
    # Load dataset untuk analisis
    df_insight = pd.read_csv('data/dropout_analysis.csv')
    
    # Cari kolom target
    target_col = None
    possible_target_names = ['Target', 'target', 'STATUS', 'status', 'Dropout', 'dropout', 'outcome']
    for name in possible_target_names:
        if name in df_insight.columns:
            target_col = name
            break
    
    if target_col is None:
        st.warning(f"Kolom target tidak ditemukan. Menggunakan kolom pertama '{df_insight.columns[0]}'")
        target_col = df_insight.columns[0]
    
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan! Pastikan file ada di lokasi yang benar: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
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
    'Course': 9500,
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
    layout="wide"
)

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("ℹ️ Panduan Penggunaan")
    st.markdown("""
    1. Pilih tab sesuai kebutuhan
    2. Gunakan tab **Prediksi** untuk prediksi individu
    3. Gunakan tab **Analisis** untuk melihat pola data
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

# Buat tab utama
tab1, tab2 = st.tabs(["🔮 Prediksi Individu", "📊 Analisis Data"])

with tab1:
    # Header
    st.title("🎓 Prediksi Status Akademik Mahasiswa")
    st.markdown("""
    **Isi data utama berikut untuk prediksi status mahasiswa.**
    Fitur lain akan diisi dengan nilai default secara otomatis.
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

with tab2:
    st.title("📊 Analisis Data Mahasiswa")
    st.markdown(f"""
    ## Analisis Faktor Dropout Berdasarkan Data Historis
    **Menggunakan kolom:** `{target_col}` sebagai status akademik
    """)
    
    # Analisis distribusi status
    st.subheader("Distribusi Status Akademik")
    
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        status_counts = df_insight[target_col].value_counts()
        
        # Cari label dropout untuk highlight
        dropout_label = None
        for label in status_counts.index:
            if 'dropout' in str(label).lower() or 'do' in str(label).lower():
                dropout_label = label
                break
        
        # Buat warna: merah untuk dropout, lainnya biru/hijau
        colors = []
        for label in status_counts.index:
            if label == dropout_label:
                colors.append('#FF4B4B')  # Merah untuk dropout
            else:
                colors.append('#1C83E1')  # Biru untuk status lain
        
        ax1.pie(status_counts, 
                labels=status_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        
        # Hitung persentase dropout
        if dropout_label:
            dropout_rate = status_counts[dropout_label] / len(df_insight) * 100
            st.markdown(f"""
            **Insight:**
            - Tingkat dropout: **{dropout_rate:.1f}%** ({status_counts[dropout_label]} mahasiswa)
            - Total sampel data: **{len(df_insight)}** mahasiswa
            """)
        else:
            st.warning("Label dropout tidak teridentifikasi dalam data")
    except Exception as e:
        st.error(f"Error visualisasi distribusi: {str(e)}")
    
    # Analisis faktor utama
    st.subheader("Faktor Utama yang Mempengaruhi Dropout")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pengaruh nilai akademik
        st.markdown("**Pengaruh Nilai Akademik**")
        try:
            if 'Curricular_units_1st_sem_grade' in df_insight.columns:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # Pisahkan data dropout dan non-dropout
                if dropout_label:
                    dropout_data = df_insight[df_insight[target_col] == dropout_label]
                    non_dropout_data = df_insight[df_insight[target_col] != dropout_label]
                    
                    sns.kdeplot(
                        data=non_dropout_data,
                        x='Curricular_units_1st_sem_grade',
                        label='Non-Dropout',
                        color='#1C83E1',
                        fill=True,
                        ax=ax2
                    )
                    
                    sns.kdeplot(
                        data=dropout_data,
                        x='Curricular_units_1st_sem_grade',
                        label='Dropout',
                        color='#FF4B4B',
                        fill=True,
                        ax=ax2
                    )
                
                ax2.set_title('Perbandingan Distribusi Nilai Semester 1')
                ax2.set_xlabel('Nilai Semester 1')
                ax2.set_ylabel('Densitas')
                ax2.legend()
                st.pyplot(fig2)
                
                # Hitung statistik
                if dropout_label:
                    mean_dropout = dropout_data['Curricular_units_1st_sem_grade'].mean()
                    mean_non_dropout = non_dropout_data['Curricular_units_1st_sem_grade'].mean()
                    
                    st.markdown(f"""
                    **Pola:**
                    - Rata-rata nilai dropout: **{mean_dropout:.1f}**
                    - Rata-rata nilai non-dropout: **{mean_non_dropout:.1f}**
                    - Selisih: **{mean_non_dropout - mean_dropout:.1f} poin**
                    """)
            else:
                st.warning("Kolom 'Curricular_units_1st_sem_grade' tidak ditemukan")
        except Exception as e:
            st.error(f"Error plot nilai: {str(e)}")
    
    with col2:
        # Pengaruh status keuangan
        st.markdown("**Pengaruh Faktor Keuangan**")
        try:
            if 'Debtor' in df_insight.columns and dropout_label:
                # Hitung persentase dropout dengan utang
                debtor_dropout_rate = df_insight[
                    (df_insight['Debtor'] == 1) & 
                    (df_insight[target_col] == dropout_label)
                ].shape[0] / df_insight[df_insight['Debtor'] == 1].shape[0] * 100
                
                # Hitung persentase dropout tanpa utang
                non_debtor_dropout_rate = df_insight[
                    (df_insight['Debtor'] == 0) & 
                    (df_insight[target_col] == dropout_label)
                ].shape[0] / df_insight[df_insight['Debtor'] == 0].shape[0] * 100
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                labels = ['Memiliki Utang', 'Tidak Memiliki Utang']
                rates = [debtor_dropout_rate, non_debtor_dropout_rate]
                
                bars = ax3.bar(labels, rates, color=['#FF4B4B', '#1C83E1'])
                ax3.set_title('Tingkat Dropout Berdasarkan Status Utang')
                ax3.set_ylabel('Persentase Dropout (%)')
                ax3.set_ylim(0, 100)
                
                # Tambah label nilai
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 2,
                            f'{height:.1f}%', ha='center')
                
                st.pyplot(fig3)
                
                st.markdown(f"""
                **Pola:**
                - Mahasiswa dengan utang: **{debtor_dropout_rate:.1f}%** dropout
                - Mahasiswa tanpa utang: **{non_debtor_dropout_rate:.1f}%** dropout
                """)
            else:
                st.warning("Kolom 'Debtor' tidak ditemukan atau label dropout tidak teridentifikasi")
        except Exception as e:
            st.error(f"Error plot keuangan: {str(e)}")
    
    # Analisis tambahan
    st.subheader("Faktor Pendukung Lainnya")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Pengaruh usia
        st.markdown("**Pengaruh Usia Saat Pendaftaran**")
        try:
            if 'Age_at_enrollment' in df_insight.columns and dropout_label:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                
                # Plot distribusi usia dropout
                dropout_ages = df_insight[df_insight[target_col] == dropout_label]['Age_at_enrollment']
                sns.histplot(
                    dropout_ages,
                    bins=20,
                    kde=True,
                    color='#FF4B4B',
                    ax=ax4
                )
                ax4.set_title('Distribusi Usia Mahasiswa Dropout')
                ax4.set_xlabel('Usia Saat Pendaftaran')
                ax4.set_ylabel('Jumlah Mahasiswa')
                
                # Hitung statistik
                mean_age = dropout_ages.mean()
                mode_age = dropout_ages.mode()[0] if not dropout_ages.empty else 'N/A'
                
                st.pyplot(fig4)
                st.markdown(f"""
                **Pola:**
                - Rata-rata usia dropout: **{mean_age:.1f} tahun**
                - Usia paling umum: **{mode_age} tahun**
                """)
            else:
                st.warning("Kolom 'Age_at_enrollment' tidak ditemukan")
        except Exception as e:
            st.error(f"Error plot usia: {str(e)}")
    
    with col4:
        # Pengaruh ekonomi makro
        st.markdown("**Pengaruh Kondisi Ekonomi**")
        try:
            if 'Unemployment_rate' in df_insight.columns and dropout_label:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Korelasi pengangguran dan dropout
                corr = df_insight['Unemployment_rate'].corr(
                    (df_insight[target_col] == dropout_label).astype(int)
                )
                
                sns.regplot(
                    data=df_insight,
                    x='Unemployment_rate',
                    y=(df_insight[target_col] == dropout_label).astype(int),
                    logistic=True,
                    color='#FF4B4B',
                    ax=ax5
                )
                ax5.set_title('Pengaruh Tingkat Pengangguran terhadap Risiko Dropout')
                ax5.set_xlabel('Tingkat Pengangguran (%)')
                ax5.set_ylabel('Probabilitas Dropout')
                ax5.set_yticks([0, 1])
                ax5.set_yticklabels(['Tidak Dropout', 'Dropout'])
                
                st.pyplot(fig5)
                
                st.markdown(f"""
                **Pola:**
                - Korelasi: **{corr:.2f}** (positif = meningkatkan risiko)
                - Setiap kenaikan 1% pengangguran meningkatkan risiko dropout
                """)
            else:
                st.warning("Kolom 'Unemployment_rate' tidak ditemukan")
        except Exception as e:
            st.error(f"Error plot ekonomi: {str(e)}")
    
    # Rekomendasi strategis
    st.subheader("Rekomendasi Strategis Berdasarkan Analisis Data")
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px">
    <h4>Strategi Pengurangan Tingkat Dropout:</h4>
    <ol>
        <li><b>Program Intervensi Akademik Dini</b>
            <ul>
                <li>Identifikasi mahasiswa dengan nilai semester 1 di bawah standar</li>
                <li>Beri bimbingan khusus dan mentoring akademik</li>
            </ul>
        </li>
        <li><b>Dukungan Keuangan Terarah</b>
            <ul>
                <li>Bantuan khusus untuk mahasiswa dengan masalah keuangan</li>
                <li>Program beasiswa tambahan berdasarkan kebutuhan</li>
            </ul>
        </li>
        <li><b>Program Pendampingan Mahasiswa Dewasa</b>
            <ul>
                <li>Kelas khusus dan fleksibilitas jadwal untuk mahasiswa >25 tahun</li>
                <li>Layanan konseling karir untuk mahasiswa pekerja</li>
            </ul>
        </li>
        <li><b>Kolaborasi dengan Pemerintah Daerah</b>
            <ul>
                <li>Program khusus untuk daerah dengan pengangguran tinggi</li>
                <li>Kerjasama industri untuk penempatan kerja magang</li>
            </ul>
        </li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("© 2025 Dashboard Prediksi Status Mahasiswa - Sistem Pendukung Keputusan Akademik")
