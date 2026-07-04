# 🌪️ EcoGuard: Air Quality Forecaster (Prediksi Kualitas Udara vs Cuaca)

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Indeks Kualitas Udara (AQI) Menggunakan Data Meteorologi (*Weather-based Air Quality Forecasting*)

Repositori ini memuat *source code* dan dokumentasi untuk **Sistem Peringatan Dini Polusi Udara**. Sistem ini menggunakan algoritma **XGBoost (AI)** untuk memprediksi tingkat bahaya Polusi Udara esok hari hanya berdasarkan kombinasi 10 parameter peramalan cuaca (suhu, curah hujan, angin, dll).

## 📂 Struktur Proyek
- `Air_Quality_Weather_Project.py` : Skrip Machine Learning utama (Data Integration, SMOTE, Training XGBoost, Visualisasi).
- `app.py` : Antarmuka Web App (*Front-end Dashboard*) cerdas menggunakan **Streamlit** dengan *2-Tab Analytics System*.
- `Laporan_Tugas_Akhir.md` : Laporan resmi tugas akhir yang sangat detail (memenuhi 6 aspek CRISP-DM lengkap dengan gambar *embedded*).
- `results/` : Menyimpan Plot AI (Confusion Matrix & Feature Importance) dan Model (*.pkl*).
- `data/` : Dataset mentah global (Cuaca & Kualitas Udara).
- `Proyek_Air_Lama/` : Folder arsip proyek Air sebelumnya.

---

## 💻 Menjalankan di Lokal (Local Development)
Untuk membuka **Dashboard Prediksi Udara (UI)** di komputer Anda:

1. **Pastikan library Python ter-install:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi Web Streamlit:**
   ```bash
   streamlit run app.py
   ```
3. Aplikasi secara otomatis terbuka di browser (`http://localhost:8501`).

---

## 🚀 Panduan Deployment (Cloud Gratis via Streamlit)
Aplikasi ini sudah dirancang untuk berjalan sebagai *Dashboard Cloud* yang *enterprise-grade*.

1. Tekan **Push** di repositori GitHub lokal Anda.
2. Buka situs [Streamlit Community Cloud](https://share.streamlit.io/) dan login dengan GitHub Anda.
3. Klik tombol **"New app"**.
4. Pilih repositori `MarchellStudentGit/anak-data-nih-bosh`, pastikan *branch* adalah `dev-marchell` atau `main`, dan set *Main file path* ke `app.py`.
5. Klik **"Deploy!"**.
6. Selesai! Aplikasi Anda akan online dengan *URL Publik* yang bisa Anda tunjukkan kepada dosen Anda saat presentasi! 

> 💡 **Fitur Dasbor Cerdas:** Anda bisa langsung mempresentasikan grafik akurasi (*Confusion Matrix*) tanpa perlu membuka laporan teks, cukup dengan membuka **Tab 2** pada situs Web App tersebut!