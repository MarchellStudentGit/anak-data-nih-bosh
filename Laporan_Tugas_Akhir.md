# Laporan Tugas Akhir: Sistem Prediksi Kualitas Udara Berdasarkan Cuaca Global

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Indeks Kualitas Udara (AQI) Menggunakan Data Meteorologi (*Weather-based Air Quality Forecasting*)

---

## 1. Business Understanding
**Latar Belakang dan Proses Bisnis Saat Ini**
Polusi udara merupakan salah satu ancaman lingkungan dan kesehatan publik paling serius di abad ke-21. Partikel halus (PM2.5, PM10) dan gas beracun (NO2, SO2, Ozon) yang melampaui ambang batas memicu penyakit kardiovaskular dan pernapasan akut. Dalam ekosistem manajemen tata kota pintar (*smart city*), pemantauan kualitas udara dilakukan melalui stasiun sensor (*Air Quality Monitoring Stations*). Sensor-sensor ini menghasilkan angka Indeks Kualitas Udara (AQI) secara *real-time*. Proses bisnis saat ini masih bersifat *reaktif*; artinya, pemerintah kota baru menyebarkan peringatan bahaya polusi kepada warganya *setelah* sensor mendeteksi lonjakan racun di udara.

**Permasalahan Bisnis (Business Problem)**
Kelemahan sistem reaktif adalah hilangnya "jendela waktu emas" bagi warga untuk membatalkan aktivitas luar ruangan dan bagi rumah sakit untuk bersiaga. Persebaran polutan di atmosfer sangat dipengaruhi oleh faktor iklim mikro. Curah hujan tinggi bertindak membersihkan polutan, sedangkan kelembapan tinggi dan hembusan angin yang stagnan membuat emisi kendaraan terperangkap di bawah atmosfer. Prediksi kualitas udara esok hari masih sangat sulit dilakukan hanya bermodalkan tren historis polusi semata.

Oleh karena itu, diperlukan inovasi *Data Science* untuk merancang **Sistem Peringatan Dini Kualitas Udara (*Air Quality Early Warning System*)**. Sistem pendukung keputusan ini menggunakan peramalan cuaca—yang datanya sangat mudah dan presisi didapatkan—untuk memprediksi klasifikasi bahaya kualitas udara (AQI). Inovasi lintas-domain ini dapat memangkas waktu antisipasi menjadi *H-1* (sebelum kejadian), memungkinkan penghematan biaya kesehatan yang masif.

## 2. Data Collection and Understanding
**Data Collection (Pengumpulan Data)**
Data dikumpulkan secara sekunder dari dua repositori internasional:
1. **Dataset Kualitas Udara Global (`global_air_quality_dataset.csv`)**: Memuat 3.662 rekaman observasi harian polusi di 10 kota metropolitan besar (2024).
2. **Dataset Cuaca Dunia (`worldwide_weather_2025.csv`)**: Memuat 14.600 rekaman iklim harian terperinci di 40 kota dunia (2025).

**Data Understanding (Pemahaman Data)**
Melalui tahap *profiling*, diketahui terdapat **7 kota yang beririsan** di antara kedua dataset. Mengingat siklus iklim harian sangat terkait dengan musim tahunan, kedua dataset ini dapat diintegrasikan dengan asumsi bahwa siklus musim iklim dan dinamika harian memiliki keidentikan pola dari tahun ke tahun.

## 3. Data Preparation
Tahapan Persiapan Data dan Integrasi (*Data Preparation & Integration*) adalah kunci proyek ini karena menghubungkan disiplin meteorologi dan klimatologi:

1. **Time-Shifting Alignment:** Menyelaraskan dimensi waktu. Dataset Kualitas Udara (2024) dimodifikasi tahunnya menjadi 2025.
2. **Data Integration (Inner Join):** Melakukan metode *Inner Join* berdasarkan kolom kunci bersama: `City` dan `Date`. Hasilnya adalah dataset baru (Gabungan) berisi fitur cuaca bersanding langsung dengan skor AQI pada hari yang sama.
3. **Pembentukan Target (Target Engineering):** Apabila $AQI > 100$, maka diberi label `1 (Berbahaya)`. Jika tidak, labelnya `0 (Aman)`.
4. **Feature Expansion:** Fitur yang diekstraksi dimaksimalkan menjadi **10 Parameter Meteorologi**: Suhu Maksimal, Suhu Minimal, Suhu Terasa (Apparent), Curah Hujan, Kec. Angin, Kec. Hembusan Angin, Tutupan Awan, Flag Hujan Lebat, Bulan, dan Indikator Akhir Pekan. Penambahan fitur ini menjamin bahwa AI bisa mendeteksi interaksi cuaca yang sangat kompleks.
5. **Class Balancing & Splitting:** Membagi data (*80% Training, 20% Testing*) dan menerapkan teknik sintesis data **SMOTE** pada data pelatihan.
6. **Feature Scaling:** Seluruh input dinormalisasi menggunakan `StandardScaler`.

## 4. Modelling
**Alasan Pemilihan Model & Proses Modelling:**
Untuk menangkap korelasi non-linear yang rumit, kami memilih **XGBoost (Extreme Gradient Boosting)** sebagai model utama karena kebal terhadap pencilan cuaca esktrem.
Model diinisialisasi dengan $n\_estimators=150, max\_depth=5$ dan dilatih menggunakan dataset gabungan yang sudah di-*balancing*. Objek *scaler* dan model diekspor ke `.pkl` menggunakan `joblib`.

## 5. Evaluation
Kinerja model diuji pada himpunan data tak kasatmata (Testing Set). Hasil klasifikasi ditunjukkan pada visualisasi otomatis di bawah ini yang dibangkitkan langsung dari skrip Python.

### 5.1 Akurasi Prediksi (Confusion Matrix)
Dalam desain *Early Warning System*, metrik **Recall** sangat diutamakan untuk meminimalisir kegagalan mendeteksi bahaya (*False Negative*).

![Confusion Matrix](results/confusion_matrix_weather_xgb.png)

### 5.2 Dampak Cuaca terhadap Polusi (Feature Importance)
Grafik di bawah ini memvalidasi teori fisika lingkungan, di mana algoritma XGBoost mengonfirmasi bahwa **Fluktuasi Suhu** dan **Hembusan Angin (Windspeed)** memiliki korelasi absolut tertinggi dalam menyebarkan polutan PM2.5 di langit kota.

![Feature Importance](results/feature_importances_weather_xgb.png)

## 6. Deployment
Sistem cerdas ini diimplementasikan menggunakan arsitektur web modern yang dirancang "Sempurna".
1. **Arsitektur Front-End (Web Dashboard):** Menggunakan framework **Streamlit**, kami merancang sistem UI cerdas dengan kapabilitas **2-Tab**. Tab 1 difungsikan sebagai "Mesin Prediksi Interaktif" untuk kalkulasi *real-time*. Tab 2 difungsikan sebagai "Dashboard Analitik Model" yang menanamkan (*embed*) dokumentasi grafik *Confusion Matrix* agar pengguna akhir bisa memvalidasi akurasi sistem.
2. **Back-End Integration:** Menampung model XGBoost terkompresi. Ketika petugas kota menekan tombol "Analisis", aplikasi seketika menerjemahkan metrik cuaca esok hari menjadi status: "AWAS KUALITAS UDARA BERBAHAYA!"
3. **Cloud Hosting Gratis (CI/CD):** Didistribusikan melalui **Streamlit Community Cloud** yang terhubung langsung ke GitHub (otomatis *update* 24/7 tanpa *downtime*). Inovasi lintas domain ini memberikan solusi kelas *Enterprise* secara cuma-cuma kepada pemangku kebijakan tata kota cerdas.
