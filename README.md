<div align="center">
  <img src="https://img.icons8.com/color/96/000000/air-quality.png" alt="EcoGuard Logo">
  <h1>🌍 EcoGuard Analytics</h1>
  <p><b>Sistem Cerdas Peringatan Dini Kualitas Udara (AQI) Berbasis Kondisi Meteorologi</b></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-orange.svg)](https://xgboost.readthedocs.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📖 Tentang Proyek
Polusi udara merupakan ancaman serius bagi kesehatan masyarakat perkotaan. Proyek **EcoGuard Analytics** dirancang sebagai sistem pendukung keputusan (*Decision Support System*) untuk memprediksi tingkat klasifikasi bahaya kualitas udara (Aman atau Tidak Sehat) menggunakan algoritma pemelajaran mesin **XGBoost**.

Tidak seperti sistem konvensional yang mengandalkan data polusi masa lalu, model ini menggunakan **prakiraan cuaca esok hari** (seperti kecepatan angin, curah hujan, dan suhu) sebagai fitur prediktor utama. Kondisi meteorologi terbukti memiliki pengaruh signifikan terhadap terperangkapnya atau tersebarnya polutan di lapisan atmosfer bawah.

Proyek ini disusun untuk memenuhi **Tugas Akhir Mata Kuliah Metodologi Data Science**.

---

## 👥 Tim Pengembang (Anak Data Nih Bosh)
- **Marchell Adi Pratama** (672023081)
- **Hendy Christian Nugroho** (672023161)
- **Maria Anne Pujara** (672023268)

---

## 🧠 Metodologi & Arsitektur Sistem

Proyek ini dibangun mengikuti kerangka kerja standar industri **CRISP-DM** (Cross-Industry Standard Process for Data Mining):
1. **Data Integration & Time-Shifting:** Menggabungkan dataset global observasi polusi dengan data ramalan cuaca (2024–2025).
2. **Feature Engineering:** Mengekstraksi 10 parameter kunci, termasuk dinamika angin (*windgusts*), faktor pencucian udara oleh presipitasi (*heavy rain flag*), dan tren musiman.
3. **Imbalance Handling:** Menggunakan algoritma **SMOTE** (Synthetic Minority Over-sampling Technique) untuk mengatasi ketimpangan kelas target bahaya.
4. **Modelling:** Melatih klasifikasi ansambel **XGBoost** berkinerja tinggi.
5. **Deployment:** Merancang arsitektur UI/UX analitik interaktif menggunakan kerangka kerja **Streamlit**.

---

## 📂 Struktur Direktori Repositori
Hanya direktori *production-ready* yang disertakan dalam *repository* ini:
```bash
├── Air_Quality_Weather_Project.py   # Skrip ETL, Pemodelan, & Evaluasi AI
├── app.py                           # Front-End Web Application (Streamlit)
├── Laporan_Tugas_Akhir.md           # Laporan Akademis Komprehensif (Metodologi Lengkap)
├── requirements.txt                 # Dependensi pustaka Python
├── results/                         # Ekspor Model (*.pkl) dan visualisasi evaluasi model
└── data/                            # Arsip dataset sumber sekunder
```

---

## 🚀 Panduan Instalasi Lokal

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/MarchellStudentGit/anak-data-nih-bosh.git
   cd anak-data-nih-bosh
   ```
2. **Instalasi Pustaka (Dependencies):**
   Disarankan menggunakan *virtual environment*.
   ```bash
   pip install -r requirements.txt
   ```
3. **Eksekusi Aplikasi Web:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis beroperasi pada `http://localhost:8501`.

---

## 🌐 Publikasi Jaringan (Cloud Deployment)
Untuk mempublikasikan aplikasi analitik ini agar dapat diakses publik, sistem telah dioptimalkan untuk di-deploy secara *seamless* di **Streamlit Community Cloud**:
1. Otorisasi akun GitHub Anda pada *dashboard* Streamlit Cloud.
2. Hubungkan *repository* ini.
3. Tetapkan `app.py` sebagai *entry point*.
4. Sistem akan mendeteksi `requirements.txt` dan mengkompilasi *environment* secara otomatis.