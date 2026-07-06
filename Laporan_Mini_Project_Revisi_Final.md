# Laporan Mini Project Metodologi Data Science
## EcoGuard Analytics: Sistem Deteksi Dini Kualitas Udara Harian Berbasis Time-Series Classification

**Anggota Kelompok:**
1. Marchell Adi Pratama (672023081)
2. Hendy Christian Nugroho (672023161)
3. Maria Anne Pujara (672023268)

---

### 1. Business Understanding
Masalah polusi udara merupakan isu krusial di wilayah perkotaan, terutama pada area dengan kepadatan kendaraan, aktivitas industri, dan populasi yang tinggi. Kualitas udara yang buruk tidak selalu dapat dilihat secara kasatmata, namun dampaknya secara langsung dirasakan oleh masyarakat, khususnya kelompok rentan seperti anak-anak, lansia, pekerja lapangan, dan individu dengan gangguan pernapasan. Salah satu metrik standar yang digunakan untuk mengukur kondisi udara adalah *Air Quality Index* (AQI). Nilai AQI berfungsi untuk mengedukasi masyarakat mengenai apakah tingkat polusi di suatu wilayah tergolong aman atau berisiko.

Dalam praktiknya, pemantauan kualitas udara dilakukan menggunakan sensor pada stasiun pemantau lingkungan yang mengukur konsentrasi polutan seperti PM2.5, PM10, NO2, SO2, CO, dan O3. Akan tetapi, sistem peringatan yang ada saat ini mayoritas masih bersifat **reaktif**—peringatan baru disebarluaskan ketika nilai AQI sudah menembus ambang batas bahaya. Pendekatan ini memberikan waktu antisipasi yang sangat sempit bagi masyarakat maupun instansi kesehatan.

Permasalahan bisnis utama yang diselesaikan dalam proyek ini adalah keterlambatan deteksi risiko kualitas udara. Institusi seperti pemerintah kota, sekolah, rumah sakit, dan masyarakat membutuhkan informasi prediktif (h-1) agar dapat mengambil langkah preventif. Misalnya, sekolah dapat menjadwalkan ulang aktivitas luar ruangan, rumah sakit dapat mengantisipasi lonjakan pasien ISPA, dan masyarakat dapat membatasi kegiatan *outdoor*. 

Kualitas udara sangat dipengaruhi oleh parameter meteorologi. Hujan dapat melarutkan polutan di udara (*washing effect*), sementara hembusan angin dapat menyebarkan polutan. Sebaliknya, kondisi angin yang stagnan seringkali memerangkap polutan di wilayah tertentu. Suhu, tutupan awan, hingga aktivitas mingguan (hari kerja vs akhir pekan) juga memiliki korelasi dengan tingkat polusi. Oleh karena itu, data cuaca dapat dimanfaatkan secara optimal sebagai prediktor status kualitas udara.

Proyek ini dirancang sebagai pemodelan *time-series classification*, mengingat data yang diolah memiliki sekuens waktu harian. Target prediksi pada model bukanlah angka absolut AQI, melainkan klasifikasi status: **Aman** ($\leq 100$) atau **Berbahaya** ($> 100$). Hasil akhir dari proyek ini dideploy dalam bentuk purwarupa (*prototype*) aplikasi web **EcoGuard Analytics**, yang berperan sebagai sistem pendukung keputusan (*decision support system*) terintegrasi.

---

### 2. Data Collection and Understanding
Data yang digunakan dalam pemodelan ini bersumber dari dua himpunan data (*dataset*) sekunder: `global_air_quality_dataset.csv` dan `worldwide_weather_2025.csv`. 

1. **Global Air Quality Dataset (`global_air_quality_dataset.csv`)**
   Dataset ini terdiri dari 3.660 baris dan 13 kolom, memuat informasi historis kualitas udara harian di berbagai kota besar pada tahun 2024. Atribut yang relevan meliputi tanggal, kota, AQI, serta berbagai polutan (PM2.5, PM10, NO2, dll.). Atribut `AQI` digunakan sebagai dasar pembentukan kelas target klasifikasi (Aman/Berbahaya).
   
2. **Worldwide Weather Dataset (`worldwide_weather_2025.csv`)**
   Dataset ini memiliki 14.600 baris dan 33 kolom, mencakup data meteorologi harian global untuk tahun 2025. Beberapa variabel prediktor yang diekstraksi meliputi suhu maksimum/minimum (`temperature_2m_max`, `temperature_2m_min`), suhu terasa (`apparent_temperature_max`), total curah hujan (`precipitation_sum`), kecepatan dan hembusan angin (`windspeed_10m_max`, `windgusts_10m_max`), persentase tutupan awan (`cloudcover_mean`), serta fitur waktu (`month`, `is_weekend`, `heavy_rain_flag`).

Pada tahap *Data Understanding*, observasi difokuskan pada kunci komposit: `City` dan `Date`. Hasil *profiling* menemukan adanya irisan data pada 7 kota utama (Beijing, Cairo, Delhi, London, Paris, Sydney, dan Tokyo). Hal ini memfasilitasi proses integrasi data. 

*Catatan keterbatasan:* Dataset kualitas udara (2024) dan dataset cuaca (2025) memiliki perbedaan tahun. Untuk keperluan studi relasional antara pola cuaca dan polusi, dilakukan rekonsiliasi dengan menyesuaikan tahun pada dataset udara menjadi 2025. Setelah proses *inner join*, diperoleh dataset gabungan sebanyak **2.555 observasi** valid.

---

### 3. Data Preparation
Keberhasilan model *machine learning* sangat bergantung pada kualitas data, sehingga tahap pra-pemrosesan (*Data Preparation*) dilakukan secara komprehensif:

1. **Data Cleaning & Formatting:** Format tanggal pada kedua dataset diseragamkan ke tipe `datetime`. Pengecekan *missing values* menunjukkan bahwa dataset kualitas udara bersih dari data kosong. Beberapa atribut *lagging* pada data cuaca memiliki nilai kosong, namun karena fitur tersebut dieliminasi dalam pemodelan akhir, dampaknya dapat diabaikan.
2. **Time-Shifting Alignment:** Atribut tahun pada kolom tanggal data udara disesuaikan menjadi 2025. Data kemudian diurutkan berdasarkan `City` dan `Date` guna mempertahankan integritas urutan waktu (*temporal order*). Tanggal 29 Februari dibuang dari dataset karena 2025 bukan tahun kabisat.
3. **Data Integration:** Dilakukan *Inner Join* menggunakan atribut gabungan `City` dan `Date`, menghasilkan himpunan data terpadu yang memetakan cuaca dengan status AQI di hari yang sama.
4. **Target Engineering (Discretization):** Atribut numerik AQI didiskritisasi menjadi label biner (1 untuk Berbahaya jika AQI > 100, dan 0 untuk Aman jika AQI $\leq 100$).
5. **Feature Selection:** Dari 33 fitur, diekstraksi 10 prediktor paling relevan untuk menekan *curse of dimensionality* sekaligus mempertahankan fitur kausalitas secara meteorologi.

> **[MASUKKAN GAMBAR DI SINI: Cuplikan Tabel Data Hasil Penggabungan (Opsional, jika ada)]**

6. **Splitting & SMOTE:** Data dipartisi secara sekuensial (memperhatikan urutan waktu) menjadi 80% data latih (*training*) dan 20% data uji (*testing*). Mengingat kelas "Berbahaya" cenderung minoritas, teknik **SMOTE** (*Synthetic Minority Over-sampling Technique*) diimplementasikan secara eksklusif pada data latih.
7. **Feature Scaling:** Proses standardisasi (*Z-score Normalization*) menggunakan `StandardScaler` diterapkan agar rentang skala tiap variabel menjadi homogen, mencegah dominasi fitur ber-skala besar terhadap fitur berskala kecil.

> **[MASUKKAN KODE DI SINI: Potongan Script Data Splitting & Scaling (Opsional, dari skrip `Air_Quality_Weather_Project.py` baris 61-68)]**

---

### 4. Modelling
Pendekatan pemodelan yang diterapkan adalah *time-series classification*, di mana fitur independen mempertahankan karakteristik runtut waktu. Model klasifikasi yang dipilih adalah **XGBoost (Extreme Gradient Boosting)**.

XGBoost unggul dalam mengolah data tabular dan mengekstraksi korelasi non-linear yang rumit. Hubungan meteorologi tidak beroperasi secara linear sederhana; misalnya, hujan deras dapat menurunkan polusi secara drastis, namun efektivitasnya sangat dipengaruhi oleh kecepatan angin lokal dan fluktuasi suhu. XGBoost menggunakan metode *ensemble learning* dengan membangun pohon keputusan (*decision trees*) secara iteratif, di mana setiap pohon memperbaiki *error* dari pohon sebelumnya.

Konfigurasi *hyperparameter* yang digunakan adalah:
- `n_estimators = 150` (Membangun 150 iterasi pohon agar pembelajaran optimal tanpa *overfitting* berlebih).
- `max_depth = 5` (Membatasi kedalaman pohon untuk menjaga kemampuan generalisasi model).

Model dan objek *scaler* hasil pelatihan diekspor dalam bentuk `.pkl` menggunakan pustaka `joblib` agar dapat diintegrasikan langsung ke arsitektur *deployment* (*Web Server*).

---

### 5. Evaluation
Kinerja model dievaluasi secara ketat pada set data uji (*hold-out testing set*) yang merepresentasikan data pada periode masa depan relatif terhadap data latih. Hal ini mereplikasi kondisi operasional di dunia nyata.

#### 5.1 Evaluasi Matriks Kebingungan (Confusion Matrix)
Akurasi absolut (*accuracy*) model tercatat sebesar **59.45%** (berhasil memprediksi 305 dari 513 data uji secara tepat). Meskipun angka ini mengindikasikan bahwa model prototipe belum sempurna secara utuh, algoritma telah berhasil menangkap pola korelasi fundamental. 

> **[MASUKKAN GAMBAR DI SINI: Screenshot Plot Confusion Matrix dari hasil aplikasi (`results/confusion_matrix_weather_xgb.png`)]**

Untuk analisis bahaya, metrik **Recall (Sensitivitas)** pada kelas "Berbahaya" merupakan indikator yang jauh lebih krusial dibandingkan akurasi umum. Dari metrik evaluasi didapatkan **Recall sebesar 74.07%** dan **F1-Score 72.92%**. Tingginya nilai Recall menunjukkan keberhasilan model dalam menghindari *False Negative* (kondisi berbahaya yang gagal terdeteksi). Dalam ekosistem keselamatan publik, kegagalan membunyikan alarm saat udara beracun (*False Negative*) jauh lebih berisiko dibandingkan peringatan palsu saat udara bersih (*False Positive*).

#### 5.2 Analisis Signifikansi Fitur (Feature Importance)
Berdasarkan pembobotan internal XGBoost, didapatkan hierarki faktor meteorologi yang paling dominan dalam mengklasifikasikan tingkat polusi.

> **[MASUKKAN GAMBAR DI SINI: Screenshot Plot Feature Importance dari hasil aplikasi (`results/feature_importances_weather_xgb.png`)]**

Variabel yang paling berpengaruh secara berurutan meliputi fitur waktu mingguan (`is_weekend`), volume hujan (`precipitation_sum`), efek pencucian badai (`heavy_rain_flag`), hingga dinamika pergerakan angin (`windspeed_10m_max` dan `windgusts_10m_max`). Hal ini memberikan landasan empiris kuat yang selaras dengan literatur klimatologi lingkungan.

---

### 6. Deployment
Tahap akhir CRISP-DM diimplementasikan dengan mendeploy (*deployment*) model ke dalam antarmuka aplikasi web dinamis berbasis **Streamlit** (dinamai *EcoGuard Analytics*). Arsitektur ini dirancang sebagai *Dashboard Visual* agar proses interpretasi *machine learning* dapat diakses dengan mudah oleh pengguna non-teknis.

Tampilan aplikasi terdiri dari tiga segmen utama:
1. **Mesin Prediksi (AI Engine):** Formulir antarmuka interaktif yang menerima nilai masukan manual terkait prakiraan cuaca esok hari. Sistem di *back-end* kemudian memuat *scaler* dan model `.pkl` untuk memberikan vonis status "Aman" atau "Berbahaya" beserta probabilitas statistiknya secara *real-time*.
2. **Dashboard Analitik:** Panel transparan yang menyematkan grafik *Confusion Matrix* dan *Feature Importance*, memungkinkan pengguna memvalidasi akurasi sistem secara objektif.
3. **Kamus Glosarium:** Fitur edukatif untuk menjabarkan terminologi medis dan lingkungan kepada kelompok masyarakat awam.

Saat ini aplikasi beroperasi dalam versi purwarupa tertutup. Pada iterasi pengembangan di masa depan, sistem direncanakan untuk terintegrasi dengan *Weather API* secara langsung sehingga pengambilan parameter input dapat diotomatisasi, serta penerapan *hyperparameter tuning* lanjutan guna meningkatkan metrik akurasi melampaui 80%.

---

### Lampiran yang Disiapkan
1. **Dataset Raw** 
   - Global Air Quality: `https://drive.google.com/open?id=1nURyC3kPJDW0PpgngEO8LfkCOfXCUXRq&usp=drive_copy`
   - Worldwide Weather 2025: `https://drive.google.com/open?id=1tpa_4u69A6uIR-QtoRim5eDqdl29caJ1&usp=drive_copy`
2. **Repositori GitHub Publik**
   - Source Code: `https://github.com/MarchellStudentGit/anak-data-nih-bosh.git`
3. **Cloud Deployment URL**
   - Web App (*Live Streamlit*): `https://anak-data-nih-bosh.streamlit.app/`
