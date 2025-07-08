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
    # Load dataset
    df_insight = pd.read_csv('data.csv')  
    
    # Cari kolom target dengan berbagai kemungkinan nama
    target_col = None
    possible_target_names = ['Target', 'target', 'STATUS', 'status', 'Dropout', 'dropout', 'outcome']
    for name in possible_target_names:
        if name in df_insight.columns:
            target_col = name
            break
    
    # Jika tidak ditemukan, gunakan kolom pertama sebagai fallback
    if target_col is None:
        st.warning(f"Kolom target tidak ditemukan. Menggunakan kolom pertama '{df_insight.columns[0]}' sebagai status")
        target_col = df_insight.columns[0]
    
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan! Pastikan file ada di lokasi yang benar: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
    st.stop()

# [Kode lainnya tetap sama...]

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
                mode_age = dropout_ages.mode()[0]
                
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
                
                # Hitung korelasi
                corr = df_insight['Unemployment_rate'].corr(
                    (df_insight[target_col] == dropout_label).astype(int)
                
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
