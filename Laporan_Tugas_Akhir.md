# Laporan Tugas Akhir: Sistem Prediksi Kualitas Udara Berdasarkan Cuaca Global

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Indeks Kualitas Udara (AQI) Menggunakan Data Meteorologi (*Weather-based Air Quality Forecasting*)

---

## 1. Business Understanding
**Latar Belakang dan Proses Bisnis Saat Ini**
Polusi udara merupakan salah satu ancaman lingkungan dan kesehatan publik paling serius di abad ke-21. Partikel halus (PM2.5, PM10) dan gas beracun (NO2, SO2, Ozon) yang melampaui ambang batas memicu penyakit kardiovaskular dan pernapasan akut. Dalam ekosistem manajemen tata kota pintar (*smart city*), pemantauan kualitas udara dilakukan melalui stasiun sensor (*Air Quality Monitoring Stations*) yang tersebar di wilayah urban. Sensor-sensor ini menghasilkan angka Indeks Kualitas Udara (AQI) secara *real-time*. Proses bisnis saat ini masih bersifat *reaktif*; artinya, pemerintah kota baru menyebarkan peringatan bahaya polusi kepada warganya *setelah* sensor mendeteksi lonjakan racun di udara.

**Permasalahan Bisnis (Business Problem)**
Kelemahan sistem reaktif adalah hilangnya "jendela waktu emas" bagi warga untuk membatalkan aktivitas luar ruangan dan bagi rumah sakit untuk bersiaga. Persebaran polutan di atmosfer sangat dipengaruhi oleh faktor iklim mikro. Sebagai contoh, curah hujan tinggi bertindak membersihkan polutan, sedangkan kelembapan tinggi dan angin yang stagnan membuat emisi kendaraan terperangkap di bawah atmosfer. Prediksi kualitas udara esok hari masih sangat sulit dilakukan hanya bermodalkan tren historis polusi semata.

Oleh karena itu, diperlukan inovasi *Data Science* untuk merancang **Sistem Peringatan Dini Kualitas Udara (*Air Quality Early Warning System*)**. Sistem pendukung keputusan ini menggunakan peramalan cuaca (suhu, curah hujan, angin, dan tutupan awan)—yang data ramalannya sangat mudah dan presisi didapatkan secara publik—untuk memprediksi klasifikasi bahaya kualitas udara (AQI). Inovasi lintas-domain ini dapat memangkas waktu antisipasi dari hitungan jam (setelah kejadian) menjadi *H-1* (sebelum kejadian), memungkinkan penghematan biaya kesehatan masyarakat yang masif dan pencegahan krisis pernapasan.

## 2. Data Collection and Understanding
**Data Collection (Pengumpulan Data)**
Data dikumpulkan secara sekunder dari dua repositori internasional berformat *CSV*:
1. **Dataset Kualitas Udara Global (`global_air_quality_dataset.csv`)**: Memuat 3.662 rekaman observasi harian polusi (AQI, PM2.5, PM10, NO2, SO2, CO, O3) di 10 kota metropolitan besar (seperti Beijing, New York, Delhi) sepanjang tahun 2024.
2. **Dataset Cuaca Dunia (`worldwide_weather_2025.csv`)**: Memuat 14.600 rekaman iklim harian (*temperature_2m_max*, *precipitation_sum*, *windspeed_10m_max*, *cloudcover_mean*) di 40 kota dunia untuk tahun 2025.

**Data Understanding (Pemahaman Data)**
Melalui tahap *profiling*, diketahui bahwa terdapat **7 kota yang beririsan** di antara kedua dataset (contoh: London, Tokyo, Sydney). Dataset Cuaca tidak memiliki *missing value* yang bermakna, sementara Dataset Udara memiliki data target numerik kontinu (`AQI`). Masalah utamanya adalah perbedaan tahun pencatatan (2024 vs 2025). Mengingat siklus iklim harian sangat terkait dengan musim tahunan (Januari adalah musim dingin di belahan utara, dsb.), kedua dataset ini dapat diintegrasikan dengan asumsi bahwa siklus musim iklim dan dinamika harian pada tanggal yang sama memiliki keidentikan pola.

## 3. Data Preparation
Tahapan Persiapan Data dan Integrasi (*Data Preparation & Integration*) adalah kunci proyek ini karena menghubungkan dua disiplin (meteorologi dan klimatologi):

1. **Time-Shifting Alignment:** Menyelaraskan dimensi waktu. Dataset Kualitas Udara (2024) dimodifikasi tahunnya menjadi 2025 menggunakan pustaka *Pandas* (*DateTime manipulation*). Ini memungkinkan metrik udara untuk hari kalender tertentu bergabung mulus dengan cuaca hari tersebut.
2. **Data Integration (Inner Join):** Melakukan metode *Inner Join* berdasarkan kolom kunci bersama: nama `City` dan `Date`. Hasilnya adalah dataset baru (Gabungan) yang berisi kolom parameter cuaca bersanding langsung dengan skor AQI pada hari yang sama di kota yang sama.
3. **Pembentukan Target (Target Engineering):** AQI numerik diubah menjadi klasifikasi biner untuk tujuan *Early Warning System*. Apabila $AQI > 100$ (kategori internasional untuk udara Tidak Sehat bagi kelompok sensitif/berbahaya), maka labelnya adalah `1 (Berbahaya)`. Jika tidak, labelnya adalah `0 (Aman)`.
4. **Feature Selection:** Mengeliminasi variabel kebocoran (seperti konsentrasi PM2.5 dan CO) dan hanya mempertahankan fitur-fitur **cuaca murni** sebagai *predictors* (*Temperature, Precipitation, Windspeed, Cloudcover, Humidity*).
5. **Class Balancing & Splitting:** Membagi data (*80% Training, 20% Testing*) secara *stratified*. Karena jumlah hari "Berbahaya" cenderung lebih sedikit dari hari "Aman", kami menerapkan teknik sintesis data **SMOTE** pada data pelatihan untuk mencegah model bias ke kelas mayoritas.
6. **Feature Scaling:** Seluruh input meteorologi dinormalisasi menggunakan `StandardScaler` agar algoritma dapat memproses skala satuan yang berbeda (contoh: km/jam untuk angin vs derajat Celsius untuk suhu) secara adil.

## 4. Modelling
**Alasan Pemilihan Model:**
Untuk menangkap korelasi non-linear yang rumit antara kecepatan angin, tekanan, dan polusi, algoritma klasifikasi berbasis pohon ansambel (*tree-ensemble*) mutlak diperlukan. Kami mengimplementasikan:
- **Random Forest Classifier:** Sebagai model utama yang sangat kebal terhadap pencilan (*outliers*) cuaca ekstrem dan mampu memberikan matriks Kepentingan Fitur (*Feature Importance*).
- **XGBoost (Extreme Gradient Boosting):** Sebagai model *State-of-the-Art* yang sangat efisien dalam komputasi dan sering merajai kompetisi pemodelan iklim global.

**Proses Modelling:**
Model diinisialisasi dan dilatih menggunakan dataset gabungan (Data Cuaca) untuk menebak Target (Status AQI). Model dan parameter *Scaler* diekspor menggunakan pustaka `joblib` ke format biner `.pkl` untuk ditransfer ke ekosistem peluncuran (*Production*).

## 5. Evaluation
**Cara Evaluasi Model:**
Kinerja model diuji pada himpunan data Test (20% tak kasatmata) menggunakan metode *Confusion Matrix* dan metrik evaluasi *Accuracy, Precision, Recall*, serta *F1-Score*.

**Fokus Evaluasi (Business Justification):**
Dalam desain Sistem Peringatan Dini Bencana (*Early Warning System*), metrik **Recall** merupakan parameter absolut. Kesalahan *False Negative* (sistem meramalkan udara esok hari "Aman", padahal aslinya polusi pekat "Berbahaya") akan berujung pada hilangnya nyawa atau krisis astma massal. Oleh karena itu, pengoptimalan (*tuning* dan SMOTE) difokuskan untuk mendongkrak metrik Recall pada kelas "Berbahaya (1)", agar sistem lebih protektif (sedikit peringatan palsu / *False Positive* jauh lebih bisa ditoleransi daripada tidak ada peringatan sama sekali).

## 6. Deployment
Tahap akhir (*Deployment*) memastikan sistem *Decision Support* ini dapat dimanfaatkan secara instan, gratis, dan profesional oleh Walikota, Kementerian Lingkungan Hidup, hingga warga sipil tanpa harus memahami pemrograman.

**Rencana Deployment:**
1. **Infrastruktur Front-End:** Menggunakan pustaka *open-source* **Streamlit** (Python) untuk membangun antar-muka (*Web Interface*). Aplikasi menampilkan instrumen input *slider* numerik untuk memasukkan angka ramalan cuaca esok hari (Suhu, Angin, Curah Hujan).
2. **Back-End Integration:** File `app.py` menampung model `xgboost` terkompresi. Ketika petugas kota menekan tombol "Prediksi", aplikasi langsung mentransformasi *input*, mengujinya melalui *scaler*, dan menghitung probabilitas polusi dalam millidetik.
3. **Cloud Hosting Gratis (CI/CD):** Aplikasi ini akan langsung tayang ke publik melalui platform **Streamlit Community Cloud**. Platform ini terhubung langsung (*Continuous Deployment*) ke repositori GitHub. Setiap perubahan kode atau iterasi *retraining* model pada GitHub akan otomatis menyegarkan peluncuran aplikasi di *server cloud* tanpa henti (*zero-downtime*).
4. **Alur Pengguna (User Flow):** 
   - Warga mendengarkan siaran cuaca BMKG.
   - Mereka membuka tautan URL aplikasi Web kita.
   - Warga mengetik angka suhu (misal: 30°C) dan angin (15 m/s) ke dalam aplikasi.
   - Jika model menebak AQI > 100, aplikasi langsung menampilkan spanduk merah **"BAHAYA POLUSI: Wajib Gunakan Masker Esok Hari!"** disertai analisis fitur penyebabnya (misal: "Kecepatan angin yang stagnan memperburuk akumulasi partikel udara").
