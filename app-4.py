import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Judul Dashboard
st.title("🎓 Dashboard Prediksi Performa Siswa dengan Random Forest")

# Load data
uploaded_file = st.file_uploader("Unggah file data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Awal", df.head())

    # Pra-pemrosesan sederhana (menghapus nilai null)
    df.dropna(inplace=True)

    # Asumsi: kolom terakhir adalah target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter
    n_estimators = st.slider("Jumlah Trees dalam Random Forest", 10, 200, 100, step=10)

    # Model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.markdown("## 📈 Evaluasi Model")
    st.markdown(f"**Akurasi Model:** {acc:.2%}")

    # Confusion Matrix
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    st.pyplot(fig1)

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Visualisasi perbandingan
    st.markdown("## 🎯 Distribusi Hasil Prediksi vs Aktual")
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    comparison_count = comparison.value_counts().reset_index(name='count')

    fig2, ax2 = plt.subplots()
    sns.barplot(data=comparison_count, x='Actual', y='count', hue='Predicted', ax=ax2)
    st.pyplot(fig2)

    # Prediksi Berdasarkan Input User
    st.markdown("## 🔍 Prediksi Berdasarkan Input")
    input_data = st.text_input("Masukkan nilai fitur dipisahkan koma (sesuai urutan):")

    if input_data:
        try:
            input_list = [float(x.strip()) for x in input_data.split(',')]
            input_array = np.array(input_list).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            st.success(f"Hasil Prediksi: **{prediction}**")
        except Exception as e:
            st.error(f"Input tidak valid: {e}")

else:
    st.info("Silakan unggah file data untuk memulai analisis.")
