import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Risiko Dropout Mahasiswa",
    page_icon="⚠️",
    layout="wide"
)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';', decimal=',')
    
    # Cek dan konversi kolom numerik
    numeric_cols = ['Admission_grade', 'Previous_grade', 'Age']
    for col in numeric_cols:
        if col in data.columns and data[col].dtype == object:
            data[col] = data[col].str.replace(',', '.').astype(float)
            
    # Tambahkan kolom dropout
    if 'Status' in data.columns:
        data['is_dropout'] = data['Status'].apply(lambda x: 1 if x == 'Dropout' else 0)
    
    return data

# Fungsi preprocessing
def preprocess_data(df):
    df = df.dropna()
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

# Fungsi utama
def main():
    st.title("⚠️ Analisis Risiko Dropout Mahasiswa")
    st.markdown("""
    Aplikasi ini menganalisis faktor-faktor yang mempengaruhi risiko dropout mahasiswa menggunakan pendekatan multivariate.
    """)
    
    # Sidebar
    st.sidebar.header("Pengaturan Model & Analisis")
    test_size = st.sidebar.slider("Ukuran Data Uji", 0.1, 0.5, 0.2)
    n_estimators = st.sidebar.slider("Jumlah Estimator", 10, 200, 100)
    
    # Memuat data
    data = load_data()
    
    # Tab utama
    tab_model, tab_pred, tab_dropout = st.tabs(["Model Prediksi", "Prediksi Individu", "Analisis Faktor Dropout"])
    
    with tab_model:
        st.header("Model Prediksi Risiko Dropout")
        st.info("Menggunakan algoritma Random Forest untuk memprediksi risiko dropout mahasiswa")
        
        # Preprocessing data
        df_processed = preprocess_data(data.copy())
        
        if 'is_dropout' not in df_processed.columns:
            st.error("Kolom target 'is_dropout' tidak ditemukan.")
            return
        
        # Split data
        X = df_processed.drop('is_dropout', axis=1)
        y = df_processed['is_dropout']
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
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        st.subheader("Evaluasi Model")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1-Score", f"{f1:.2%}")
        
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        
        # Feature importance
        st.subheader("Faktor Paling Berpengaruh")
        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features.plot(kind='barh', ax=ax, color='darkred')
        ax.set_title("10 Faktor Risiko Terpenting")
        st.pyplot(fig)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', cmap='Reds', ax=ax,
                    annot_kws={"size": 14})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Non-Dropout', 'Dropout'])
        ax.set_yticklabels(['Non-Dropout', 'Dropout'])
        st.pyplot(fig)
    
    with tab_pred:
        st.header("Prediksi Risiko Dropout per Mahasiswa")
        st.warning("Masukkan data mahasiswa untuk memprediksi risiko dropout")
        
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
        
        # Analisis faktor risiko
        st.subheader("Analisis Faktor Risiko")
        risk_score = 0
        risk_factors = []
        
        if admission_grade < 100:
            risk_score += 1
            risk_factors.append(f"IPK Masuk rendah ({admission_grade})")
        if previous_grade < 100:
            risk_score += 1
            risk_factors.append(f"IPK Sebelumnya rendah ({previous_grade})")
        if tuition_fees == 0:
            risk_score += 1
            risk_factors.append("Status pembayaran tidak lunas")
        if age > 25:
            risk_score += 0.5
            risk_factors.append(f"Usia lebih tua ({age} tahun)")
        
        # Visualisasi radar chart untuk risiko
        risk_labels = ['IPK Masuk', 'IPK Sebelumnya', 'Pembayaran', 'Usia']
        risk_values = [
            1 if admission_grade < 100 else 0,
            1 if previous_grade < 100 else 0,
            1 if tuition_fees == 0 else 0,
            0.5 if age > 25 else 0
        ]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles := np.linspace(0, 2 * np.pi, len(risk_labels), risk_values, 'red', alpha=0.25)
        ax.plot(angles, risk_values, color='red', marker='o')
        ax.set_xticks(angles)
        ax.set_xticklabels(risk_labels)
        ax.set_title("Profil Risiko Mahasiswa", size=14, pad=20)
        st.pyplot(fig)
        
        # Tampilkan hasil analisis risiko
        risk_level = "Rendah"
        if risk_score >= 2.5:
            risk_level = "Sangat Tinggi"
        elif risk_score >= 1.5:
            risk_level = "Tinggi"
        elif risk_score >= 0.5:
            risk_level = "Sedang"
        
        st.info(f"**Tingkat Risiko:** {risk_level} ({risk_score}/3.5)")
        
        if risk_factors:
            st.warning("**Faktor Risiko Teridentifikasi:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        
        # Prediksi
        if st.button("Prediksi Risiko Dropout") and 'model' in locals():
            try:
                input_dict = {col: 0 for col in X_train.columns}
                
                # Mapping kolom
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
                
                input_data = pd.DataFrame([input_dict], columns=X_train.columns)
                proba = model.predict_proba(input_data)[0][1]
                
                # Visualisasi probabilitas
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Risiko Dropout'], [proba], color='darkred' if proba > 0.5 else 'green')
                ax.set_xlim(0, 1)
                ax.set_title(f"Probabilitas Dropout: {proba:.1%}", fontsize=14)
                st.pyplot(fig)
                
                if proba > 0.7:
                    st.error(f"**TINGGI** - Probabilitas dropout: {proba:.1%}")
                elif proba > 0.4:
                    st.warning(f"**SEDANG** - Probabilitas dropout: {proba:.1%}")
                else:
                    st.success(f"**RENDAH** - Probabilitas dropout: {proba:.1%}")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
        elif 'model' not in locals():
            st.warning("Silakan latih model di tab Model terlebih dahulu")
    
    with tab_dropout:
        st.header("🔍 Analisis Multivariat Faktor Dropout")
        st.markdown("""
        Analisis mendalam hubungan antara berbagai karakteristik mahasiswa dengan risiko dropout menggunakan pendekatan multivariate.
        """)
        
        if 'is_dropout' not in data.columns:
            st.warning("Data tidak memiliki informasi status dropout")
            st.stop()
        
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif Mahasiswa Dropout")
        dropout_stats = data[data['is_dropout'] == 1].describe().transpose()
        st.dataframe(dropout_stats.style.background_gradient(cmap='Reds'))
        
        # Pilih fitur untuk analisis
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_dropout' in numeric_features:
            numeric_features.remove('is_dropout')
        
        selected_features = st.multiselect(
            "Pilih Fitur untuk Analisis", 
            numeric_features,
            default=numeric_features[:4] if len(numeric_features) >= 4 else numeric_features
        )
        
        if not selected_features:
            st.warning("Pilih minimal 2 fitur untuk analisis multivariate")
            st.stop()
        
        # Analisis korelasi
        st.subheader("Matriks Korelasi Faktor Risiko")
        corr_matrix = data[selected_features + ['is_dropout']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax,
                   fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
        plt.title("Korelasi Antar Faktor Risiko", fontsize=14)
        st.pyplot(fig)
        
        # Analisis PCA
        st.subheader("Analisis Dimensi dengan PCA")
        st.markdown("Principal Component Analysis (PCA) membantu mengidentifikasi pola tersembunyi dalam data")
        
        # Standarisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[selected_features])
        
        # Lakukan PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        principal_df['is_dropout'] = data['is_dropout'].values
        
        # Visualisasi PCA
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=principal_df, hue='is_dropout',
                        palette={0: 'green', 1: 'red'}, alpha=0.6, s=80, ax=ax)
        plt.title('Visualisasi Data dengan PCA', fontsize=14)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        st.pyplot(fig)
        
        # Analisis cluster
        st.subheader("Segmentasi Mahasiswa Berdasarkan Risiko")
        st.markdown("Clustering membantu mengidentifikasi kelompok mahasiswa dengan karakteristik serupa")
        
        # Gunakan KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Visualisasi clustering
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], 
                            c=clusters, cmap='viridis', alpha=0.7, s=50)
        
        # Tandai dropout dengan simbol X
        dropout_indices = data[data['is_dropout'] == 1].index
        ax.scatter(principal_components[dropout_indices, 0], 
                  principal_components[dropout_indices, 1],
                  marker='x', s=100, c='red', label='Dropout')
        
        plt.title('Segmentasi Mahasiswa dengan K-Means Clustering', fontsize=14)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.legend(*scatter.legend_elements(), title='Cluster')
        plt.legend(loc='upper right')
        st.pyplot(fig)
        
        # Analisis fitur per cluster
        st.subheader("Karakteristik Setiap Cluster")
        data['Cluster'] = clusters
        cluster_summary = data.groupby('Cluster')[selected_features].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_summary.T, annot=True, cmap='YlOrBr', fmt=".1f", ax=ax)
        plt.title('Rata-rata Fitur per Cluster', fontsize=14)
        st.pyplot(fig)
        
        # Analisis interaksi fitur
        st.subheader("Interaksi Antar Faktor Risiko")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Fitur X", selected_features, index=0)
        with col2:
            y_feature = st.selectbox("Fitur Y", selected_features, index=1 if len(selected_features) > 1 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_feature, y=y_feature, hue='is_dropout', 
                        palette={0: 'green', 1: 'red'}, alpha=0.7, s=80, ax=ax)
        
        # Tambahkan garis regresi
        sns.regplot(data=data[data['is_dropout'] == 1], x=x_feature, y=y_feature, 
                   scatter=False, color='red', line_kws={"linewidth": 2})
        sns.regplot(data=data[data['is_dropout'] == 0], x=x_feature, y=y_feature, 
                   scatter=False, color='green', line_kws={"linewidth": 2})
        
        plt.title(f"Interaksi {x_feature} vs {y_feature}", fontsize=14)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
