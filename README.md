# 💧 Sistem Prediksi Kelayakan dan Kualitas Air Minum
**Mata Kuliah:** Metodologi Data Science

Repositori ini memuat *source code* dan dokumentasi untuk memprediksi kelayakan air minum (apakah beracun atau aman untuk dikonsumsi) menggunakan Machine Learning (*Random Forest*).

## 📂 Struktur Proyek
- `Water_Quality_Project.py` : Skrip Machine Learning (EDA, Prep, SMOTE, Modeling).
- `app.py` : Antarmuka Web App (Front-end) yang dibangun menggunakan Streamlit.
- `Laporan_Tugas_Akhir.md` : Draf laporan akhir yang sangat lengkap (Business Understanding hingga Deployment).
- `results/` : Menyimpan plot evaluasi (Confusion Matrix) dan File Model Machine Learning (`rf_model.pkl` & `scaler.pkl`).
- `data/` : Dataset mentah yang digunakan untuk pembelajaran.

---

## 💻 Menjalankan di Lokal (Local Development)
Untuk menjalankan **Web Aplikasi Prediksi** (UI/Front-end) di komputer Anda sendiri:

1. **Pastikan library sudah ter-install:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi Web Streamlit:**
   ```bash
   streamlit run app.py
   ```
3. Aplikasi akan otomatis terbuka di *browser* Anda (biasanya di `http://localhost:8501`).

---

## 🚀 Panduan Deployment (Cloud Gratis)
Karena proyek ini dituntut untuk profesional dan gratis, aplikasi dapat dengan mudah di-*deploy* (*hosting*) agar bisa diakses oleh publik (seperti dosen penguji) via internet.

1. Pastikan Anda telah menekan **Push** agar semua kode terbaru ini masuk ke repositori GitHub Anda.
2. Buka situs [Streamlit Community Cloud](https://share.streamlit.io/) dan buat akun (Sign Up menggunakan akun GitHub Anda).
3. Setelah *login*, klik tombol biru **"New app"**.
4. Pilih repositori GitHub Anda (`MarchellStudentGit/anak-data-nih-bosh`), *branch* `dev-marchell` atau `main`, dan set *Main file path* ke `app.py`.
5. Klik **"Deploy!"**. 
6. Dalam hitungan menit, aplikasi Anda akan online secara otomatis dan Anda akan mendapatkan *link URL* publik untuk dipamerkan.

> **💡 Catatan Otomatis (CI/CD):** Jika ke depannya Anda memperbarui kode di GitHub, Streamlit Cloud akan secara otomatis mendeteksi pembaruan tersebut dan melakukan *deploy* ulang tanpa intervensi manual!