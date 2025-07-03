# app.py
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
def plot_distributions(df):
    st.subheader("Distribusi Status Mahasiswa")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Status', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Hubungan IPK dengan Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Status', y='Admission_grade', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Fungsi utama
def main():
    st.title("🎓 Analisis Data Pendidikan Mahasiswa")
    st.markdown("""
    Aplikasi ini menganalisis data mahasiswa untuk memprediksi status kelulusan (Graduate, Dropout, atau Enrolled).
    """)
    
    # Sidebar
    st.sidebar.header("Pengaturan Model")
    test_size = st.sidebar.slider("Ukuran Data Uji", 0.1, 0.5, 0.2)
    n_estimators = st.sidebar.slider("Jumlah Estimator", 10, 200, 100)
    
    # Memuat data
    data = load_data()
    
    # Tab untuk eksplorasi data
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Visualisasi", "Model", "Prediksi"])
    
    with tab1:
        st.header("Dataset Pendidikan Mahasiswa")
        st.write(f"Dataset berisi {data.shape[0]} baris dan {data.shape[1]} kolom")
        st.dataframe(data.head())
        
        st.subheader("Statistik Deskriptif")
        st.write(data.describe())
    
    with tab2:
        st.header("Visualisasi Data")
        plot_distributions(data)
        
        st.subheader("Korelasi Fitur")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(data.corr(numeric_only=True), ax=ax, cmap='coolwarm')
        st.pyplot(fig)
    
    with tab3:
        st.header("Model Machine Learning")
        st.info("Menggunakan algoritma Random Forest untuk memprediksi status mahasiswa")
        
        # Preprocessing data
        df_processed = preprocess_data(data.copy())
        
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
    if st.button("Prediksi Status"):
        # Buat dictionary dengan nilai default untuk semua fitur
        input_dict = {col: 0 for col in X_train.columns}
        
        # Update dengan nilai yang diinput pengguna
        input_dict.update({
            'Marital status': marital_status,
            'Application mode': application_mode,
            'Previous qualification (grade)': previous_grade,
            'Admission grade': admission_grade,
            'Tuition fees up to date': tuition_fees,
            'Age': age
        })
        
        # Konversi ke DataFrame
        input_data = pd.DataFrame([input_dict])
        
        prediction = model.predict(input_data)[0]
        status_map = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}
        predicted_status = status_map.get(prediction, 'Unknown')
        
        # Tampilkan hasil dengan visual yang menarik
        status_colors = {
            'Dropout': '#FF9999',
            'Graduate': '#99FF99',
            'Enrolled': '#66B2FF'
        }
        
        st.success(f"Prediksi Status: **{predicted_status}**")
        
        # Progress bar indikator
        progress_value = {
            'Dropout': 0.3,
            'Enrolled': 0.6,
            'Graduate': 0.9
        }.get(predicted_status, 0.5)
        
        st.progress(progress_value, text=f"Indikator Kelulusan: {predicted_status}")
        
        # Feature importance
        st.subheader("Faktor Penentu Prediksi")
        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features.plot(
            kind='barh', 
            color=sns.color_palette('viridis', len(top_features)),
            edgecolor='black'
        )
        ax.set_title("10 Fitur Terpenting")
        ax.set_xlabel('Tingkat Kepentingan')
        st.pyplot(fig)
if __name__ == "__main__":
    main()
