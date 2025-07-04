import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data Pendidikan",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';', decimal=',')
    
    # Cek dan konversi kolom numerik yang mungkin terbaca sebagai string
    numeric_cols = ['Admission_grade', 'Previous_grade', 'Age']
    for col in numeric_cols:
        if col in data.columns and data[col].dtype == object:
            # Coba konversi ke float, ganti koma menjadi titik
            data[col] = data[col].str.replace(',', '.').astype(float)
            
    return data

# Fungsi preprocessing
def preprocess_data(df):
    # Mengatasi nilai yang hilang
    df = df.dropna()
    
    # Encode variabel kategorikal
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# Fungsi untuk visualisasi
def plot_distributions(df, chart_type):
    st.subheader("Distribusi Status Mahasiswa")
    
    # Visualisasi pertama: Distribusi status
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    if chart_type == "Bar Chart":
        sns.countplot(x='Status', data=df, ax=ax1, palette="viridis")
        ax1.set_title("Jumlah Mahasiswa per Status")
    elif chart_type == "Pie Chart":
        status_counts = df['Status'].value_counts()
        ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                colors=sns.color_palette('pastel'), startangle=90)
        ax1.set_title("Persentase Status Mahasiswa")
        ax1.axis('equal')  # Agar pie chart berbentuk lingkaran
    st.pyplot(fig1)
    
    # Visualisasi kedua: Hubungan IPK dengan Status
    st.subheader(f"Hubungan IPK dengan Status ({chart_type})")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Cek apakah kolom IPK ada dan numerik
    grade_col = None
    possible_grade_cols = ['Admission_grade', 'Admission grade', 'Grade', 'AdmissionGrade']
    
    for col in possible_grade_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            grade_col = col
            break
    
    if grade_col:
        if chart_type == "Bar Chart":
            # Bar chart: Rata-rata IPK per status
            avg_grades = df.groupby('Status')[grade_col].mean()
            sns.barplot(x=avg_grades.index, y=avg_grades.values, ax=ax2, palette="rocket")
            ax2.set_ylabel(f"Rata-rata {grade_col}")
            ax2.set_xlabel("Status")
            
            # Tambahkan nilai di atas bar
            for i, v in enumerate(avg_grades.values):
                ax2.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
                
        elif chart_type == "Pie Chart":
            # Pie chart: Persentase IPK per status
            status_grades = df.groupby('Status')[grade_col].sum()
            ax2.pie(status_grades, labels=status_grades.index, autopct='%1.1f%%', 
                    colors=sns.color_palette('Set2'), startangle=90)
            ax2.set_title(f"Distribusi Total {grade_col} per Status")
            ax2.axis('equal')
            
        elif chart_type == "Line Chart":
            # Line chart: Rata-rata IPK per status
            avg_grades = df.groupby('Status')[grade_col].mean()
            sns.lineplot(x=avg_grades.index, y=avg_grades.values, ax=ax2, 
                         marker='o', markersize=8, linewidth=2.5, color='purple')
            ax2.set_ylabel(f"Rata-rata {grade_col}")
            ax2.set_xlabel("Status")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Tambahkan nilai di titik
            for i, v in enumerate(avg_grades.values):
                ax2.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
        
        #st.pyplot(fig2)
    else:
        st.warning("Kolom IPK tidak ditemukan atau tidak numerik. Tidak dapat membuat visualisasi hubungan IPK-Status.")

# Fungsi utama
def main():
    st.title("🎓 Analisis Data Pendidikan Mahasiswa")
    st.markdown("""
    Aplikasi ini menganalisis data mahasiswa untuk memprediksi status kelulusan (Graduate, Dropout, atau Enrolled).
    """)
    
    # Sidebar
    st.sidebar.header("Pengaturan Model & Visualisasi")
    test_size = st.sidebar.slider("Ukuran Data Uji", 0.1, 0.5, 0.2)
    n_estimators = st.sidebar.slider("Jumlah Estimator", 10, 200, 100)
    
    # Tambahkan opsi pemilihan jenis grafik
    chart_type = st.sidebar.selectbox(
        "Jenis Visualisasi Data", 
        ["Bar Chart", "Pie Chart", "Line Chart"],
        index=0
    )
    
    # Memuat data
    data = load_data()
    
    # Tab untuk eksplorasi data
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualisasi", "Model", "Prediksi"])
    
    with tab1:
        st.header("Dataset Pendidikan Mahasiswa")
        st.write(f"Dataset berisi {data.shape[0]} baris dan {data.shape[1]} kolom")
        st.dataframe(data.head())
        
        # Tampilkan daftar kolom untuk debugging
        st.subheader("Daftar Kolom")
        st.write(data.columns.tolist())
        
        st.subheader("Statistik Deskriptif")
        st.write(data.describe(include='all'))
    
    with tab2:
        st.header("Visualisasi Data")
        plot_distributions(data, chart_type)
        
        st.subheader("Korelasi Fitur")
        # Hanya gunakan kolom numerik untuk heatmap
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(numeric_data.corr(), ax=ax, cmap='coolwarm', annot=True, fmt=".2f")
            st.pyplot(fig)
        else:
            st.warning("Tidak ada kolom numerik untuk ditampilkan dalam heatmap.")
    
    with tab3:
        st.header("Model Machine Learning")
        st.info("Menggunakan algoritma Random Forest untuk memprediksi status mahasiswa")
        
        # Preprocessing data
        df_processed = preprocess_data(data.copy())
        
        # Pastikan kolom Status ada
        if 'Status' not in df_processed.columns:
            st.error("Kolom 'Status' tidak ditemukan dalam dataset. Tidak dapat melanjutkan modeling.")
            return
        
        # Split data
        X = df_processed.drop('Status', axis=1)
        y = df_processed['Status']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluasi model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("Evaluasi Model")
        st.write(f"Akurasi: **{accuracy:.2f}**")
        
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with tab4:
        st.header("Prediksi Status Mahasiswa")
        st.warning("Masukkan data mahasiswa untuk memprediksi status")
        
        # Input form
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", min_value=17, max_value=60, value=20)
            marital_status = st.selectbox("Status Pernikahan", [1, 2, 3, 4, 5, 6])
            application_mode = st.number_input("Mode Aplikasi", min_value=1, max_value=60, value=1)
            
        with col2:
            admission_grade = st.number_input("IPK Masuk", min_value=0.0, max_value=200.0, value=120.0)
            previous_grade = st.number_input("IPK Sebelumnya", min_value=0.0, max_value=200.0, value=130.0)
            tuition_fees = st.selectbox("Pembayaran Uang Kuliah", [0, 1])
        
        # Membuat prediksi
        if st.button("Prediksi Status") and 'X_train' in locals():
            try:
                # Buat dictionary sesuai kolom di X_train
                input_dict = {col: 0 for col in X_train.columns}  # Inisialisasi dengan 0
                
                # Isi nilai untuk kolom tertentu
                # Cari nama kolom yang sesuai
                marital_col = next((col for col in X_train.columns if 'marital' in col.lower()), 'Marital status')
                app_mode_col = next((col for col in X_train.columns if 'application' in col.lower()), 'Application mode')
                grade_col = next((col for col in X_train.columns if 'admission' in col.lower() and 'grade' in col.lower()), 'Admission grade')
                prev_grade_col = next((col for col in X_train.columns if 'previous' in col.lower() and 'grade' in col.lower()), 'Previous grade')
                tuition_col = next((col for col in X_train.columns if 'tuition' in col.lower() or 'fee' in col.lower()), 'Tuition fees')
                age_col = next((col for col in X_train.columns if 'age' in col.lower()), 'Age')
                
                input_dict[marital_col] = marital_status
                input_dict[app_mode_col] = application_mode
                input_dict[grade_col] = admission_grade
                input_dict[prev_grade_col] = previous_grade
                input_dict[tuition_col] = tuition_fees
                input_dict[age_col] = age
                
                # Konversi ke DataFrame
                input_data = pd.DataFrame([input_dict], columns=X_train.columns)
                
                prediction = model.predict(input_data)[0]
                status_map = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
                st.success(f"Prediksi Status: **{status_map.get(prediction, 'Unknown')}**")
                
                # Feature importance
                st.subheader("Pengaruh Fitur dalam Prediksi")
                importance = pd.Series(model.feature_importances_, index=X.columns)
                top_features = importance.sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features.plot(kind='barh', ax=ax)
                ax.set_title("10 Fitur Terpenting")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat prediksi: {str(e)}")
        elif 'X_train' not in locals():
            st.warning("Silakan buka tab Model terlebih dahulu untuk melatih model sebelum melakukan prediksi.")

if __name__ == "__main__":
    main()
