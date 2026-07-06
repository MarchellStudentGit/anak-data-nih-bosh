# Prediksi Kualitas Udara (AQI) Berbasis Data Meteorologi

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Indeks Kualitas Udara (AQI) Menggunakan Data Meteorologi (*Weather-based Air Quality Forecasting*)
**Kelompok:** Anak Data Nih Bosh

**Anggota Kelompok:**
1. Marchell Adi Pratama (672023081)
2. Hendy Christian Nugroho (672023161)
3. Maria Anne Pujara (672023268)

Repositori ini memuat *source code* dan dokumentasi untuk **Sistem Prediksi Kualitas Udara**. Sistem ini memanfaatkan model klasifikasi *Machine Learning* **XGBoost** untuk memprediksi probabilitas tingkat bahaya Kualitas Udara esok hari, dengan mengkorelasikan 10 parameter prakiraan cuaca sebagai variabel independen.

## 📂 Struktur Repositori
- `Air_Quality_Weather_Project.py` : Skrip utama yang mencakup pra-pemrosesan (*Data Integration, Time-Shifting, SMOTE*), pelatihan model XGBoost, dan visualisasi hasil evaluasi.
- `app.py` : Skrip *Front-end* aplikasi antarmuka menggunakan **Streamlit** untuk memfasilitasi prediksi interaktif oleh pengguna.
- `Laporan_Tugas_Akhir.md` : Dokumentasi lengkap yang mematuhi 6 fase metodologi CRISP-DM.
- `results/` : Direktori yang menyimpan artefak visual (*Confusion Matrix* & *Feature Importance*) serta *binary file* objek model (`.pkl`).
- `data/` : Direktori dataset mentah terkait data iklim dan observasi polusi (2024-2025).
- `Proyek_Air_Lama/` : Direktori arsip bagi pekerjaan eksplorasi data pada topik yang sebelumnya diajukan.

---

## 💻 Menjalankan Aplikasi secara Lokal
Untuk menjalankan purwarupa aplikasi ini di *local machine*, ikuti instruksi berikut:

1. **Instalasi Pustaka (Dependencies):**
   ```bash
   pip install -r requirements.txt
   ```
2. **Menjalankan Aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
3. Aplikasi akan beroperasi secara otomatis dan dapat diakses melalui web browser pada alamat (`http://localhost:8501`).

---

## 🚀 Panduan Deployment (Streamlit Community Cloud)
Aplikasi ini telah dirancang untuk mendukung *Continuous Deployment*.

1. Pastikan perubahan mutakhir telah di-*push* ke *branch* utama (`dev-marchell` atau `main`) pada repositori GitHub lokal Anda.
2. Akses platform [Streamlit Community Cloud](https://share.streamlit.io/) dan lakukan autentikasi dengan kredensial GitHub.
3. Klik tombol **"New app"**.
4. Pilih repositori `MarchellStudentGit/anak-data-nih-bosh`, pastikan pengaturan *branch* sudah tepat, lalu arahkan *Main file path* pada skrip `app.py`.
5. Klik **"Deploy!"**.
6. Sistem akan mengeksekusi kompilasi dependensi Python, dan aplikasi akan terpublikasi pada URL yang disediakan oleh Streamlit.