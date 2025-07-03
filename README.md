# JJI_Dashboard
## Submission Menyelesaikan Permasalahan Institusi Pendidikan

## 🧠 Deskripsi Proyek
Aplikasi berbasis **Streamlit** untuk memprediksi status siswa (`Graduate`, `Dropout`, `Enrolled`) berdasarkan data numerik seperti kehadiran, nilai tugas, dan partisipasi kelas. Proyek ini dibuat untuk membantu **Jaya Jaya Institut** dalam memahami dan memantau performa siswa.


Model menggunakan pendekatan **Random Forest Classifier**


## 📂 Struktur Direktori
submission/
│
├── notebook.ipynb               # Notebook utama training & konversi
├── README.md                    # Deskripsi proyek
├── requirements.txt             # Daftar dependensi
├── data.csv                     # Dataset 
└── app.py                       # Aplikasi Dashboard
 

## 📊 Dataset
- **Nama Dataset:** data.csv
- **Kolom:** 37
- **Baris:** 4424



## 📌 Fitur Aplikasi

- Input manual data siswa melalui UI
- Prediksi status siswa menggunakan model machine learning (Random Forest)
- Aplikasi ringan dan siap dijalankan di Streamlit Cloud
- Visualisasi hasil sederhana (opsional)


## 🧪 Library yang Digunakan

- `streamlit` - UI web
- `pandas`, `numpy` - manipulasi data
- `matplotlib`, `seaborn` - visualisasi (opsional)
- `scikit-learn` - machine learning (RandomForestClassifier, evaluasi, preprocessing)


## 🚀 Cara Menjalankan
1. URL : https://jjidashboard-nshhuaq9kluboblfuhjuat.streamlit.app/
