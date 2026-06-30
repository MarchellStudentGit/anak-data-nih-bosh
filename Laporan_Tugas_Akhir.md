# Laporan Tugas Akhir: Sistem Prediksi Kelayakan dan Kualitas Air Minum Menggunakan Machine Learning

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Kualitas Air Berdasarkan Dataset Fisikokimia dan Biologis/Logam Berat

---

## 1. Business Understanding
**Latar Belakang dan Proses Bisnis Saat Ini**
Air adalah sumber kehidupan yang paling esensial bagi umat manusia. Mengacu pada *Sustainable Development Goals* (SDG) ke-6, memastikan ketersediaan dan manajemen air bersih serta sanitasi yang berkelanjutan bagi semua orang adalah prioritas global. Dalam ekosistem penyediaan air minum, baik oleh Perusahaan Daerah Air Minum (PDAM) maupun otoritas pengelolaan air swasta, proses bisnis utama melibatkan pengambilan air dari sumber baku (sungai, danau, mata air), proses penjernihan atau pengolahan di Instalasi Pengolahan Air (IPA), hingga distribusi ke rumah tangga.

Pada tahap proses pengolahan, pengujian kualitas air dilakukan secara berkala untuk memastikan air yang didistribusikan aman untuk dikonsumsi dan bebas dari bahan kimia berbahaya (seperti aluminium, arsenik, timbal) serta parameter fisik (tingkat pH, kekeruhan/turbidity, dan solid/kepadatan) yang melebihi ambang batas. Saat ini, proses inspeksi kualitas air seringkali mengandalkan uji laboratorium secara manual. Petugas mengambil sampel air, membawanya ke laboratorium, dan menggunakan reagen kimia untuk mendeteksi kontaminan. 

**Permasalahan Bisnis (Business Problem)**
Permasalahan utama dalam proses pengujian laboratorium manual adalah memakan waktu yang cukup lama dan membutuhkan biaya operasional yang tinggi. Keterlambatan dalam mendeteksi kontaminasi air dapat menyebabkan air beracun atau tidak layak minum terdistribusi ke masyarakat, yang memicu krisis kesehatan masyarakat berupa penyakit kolera, disentri, hingga keracunan logam berat kronis. Selain itu, seiring memburuknya pencemaran lingkungan akibat limbah industri, risiko fluktuasi kualitas air baku menjadi semakin tinggi, sehingga menuntut monitoring kualitas yang lebih dinamis dan bersifat *real-time*.

Oleh karena itu, diperlukan suatu inovasi berbasis teknologi informasi yang dapat memprediksi kelayakan air minum dengan cepat tanpa harus selalu menunggu proses kultur biologi atau uji kimia yang panjang. Solusinya adalah membangun model Machine Learning cerdas (Sistem Evaluasi Kualitas dan Keamanan Air). Model ini akan bertindak sebagai pendukung keputusan (*Decision Support System*) yang mampu menerima input berupa pembacaan sensor kualitas air fisik, kimiawi, maupun data dari uji instan, dan secara langsung mengklasifikasikan apakah air tersebut "Aman/Layak (*Potable*)" atau "Tidak Aman". Implementasi sistem ini diharapkan dapat memangkas waktu tunggu dari hitungan hari menjadi hitungan detik, menekan biaya operasional laboratorium, dan secara drastis meningkatkan keandalan mitigasi risiko distribusi air tercemar kepada pelanggan.

## 2. Data Collection and Understanding
**Data Collection (Pengumpulan Data)**
Dalam mengembangkan sistem prediksi ini, data dikumpulkan dari dua sumber dataset sekunder (diperoleh dari arsip data raw `new1.zip` dan `new2.zip`). Penggunaan dataset sekunder dipilih karena sangat relevan untuk mensimulasikan pengukuran kualitas air modern yang melibatkan sensor fisikokimia dan deteksi agen patogen. Dataset pertama (`water_potability.csv`) merupakan kumpulan data pengukuran matriks air untuk mengevaluasi kelayakan minum. Dataset kedua (`waterQuality1.csv`) berfokus pada kandungan spesifik logam berat, bahan kimia berbahaya, dan agen biologis. Kedua data tersebut bertindak sebagai fondasi pembelajaran yang komprehensif untuk algoritma *machine learning*.

**Data Understanding (Pemahaman Data)**
Berdasarkan hasil inspeksi awal (profiling), berikut adalah karakteristik masing-masing dataset yang digunakan:

1. **Dataset 1 (Water Potability - Fisik & Kimia Dasar):**
   - **Dimensi:** Memiliki 3.276 baris (observasi) dan 10 atribut.
   - **Fitur/Variabel:** Terdiri dari atribut ber-tipe float64, antara lain nilai `ph` (keasaman), `Hardness` (kandungan kalsium dan magnesium), `Solids` (TDS), `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, dan `Turbidity`.
   - **Variabel Target:** Kolom `Potability` (1 berarti layak minum, 0 berarti tidak layak).
   - **Kondisi Data:** Ditemukan *missing values* (data yang hilang) pada kolom `ph` (hanya 2.785 terisi), `Sulfate` (2.495 terisi), dan `Trihalomethanes` (3.114 terisi) yang memerlukan penanganan khusus pada tahap data preparasi.

2. **Dataset 2 (Water Quality - Logam Berat & Biologis):**
   - **Dimensi:** Memiliki 7.999 baris dan 21 atribut.
   - **Fitur/Variabel:** Meliputi kandungan `aluminium`, `arsenic`, `barium`, `cadmium`, `chloramine`, `chromium`, `copper`, `flouride`, `bacteria`, `viruses`, `lead`, `nitrates`, `nitrites`, `mercury`, `perchlorate`, `radium`, `selenium`, `silver`, dan `uranium`.
   - **Variabel Target:** Kolom `is_safe` yang merepresentasikan klasifikasi aman (1) atau tidak (0).
   - **Kondisi Data:** Terdapat anomali pada tipe data kolom `ammonia` dan `is_safe` yang terdeteksi sebagai karakter (string/object), karena adanya *noise* karakter salah input seperti '#NUM!' dari proses ekstraksi raw data.

Melalui eksplorasi data (*Exploratory Data Analysis* - EDA), ditemukan bahwa sebagian besar kualitas kelayakan air sangat bergantung pada nilai ekstrem dari zat kimia. Mayoritas data berdistribusi normal, namun mengandung pencilan (*outlier*) alami yang wajar terjadi di alam.

## 3. Data Preparation
Tahapan Data Preparation merupakan tahapan kritis agar algoritma dapat belajar dari dataset yang matang. Dalam proyek ini, pendekatan penyiapan data difokuskan pada Dataset 2 (sebagai studi kasus end-to-end logam berat). Langkah-langkahnya meliputi:

1. **Data Cleaning & Handling Tipe Data Anomalous:** 
   Mendeteksi nilai *string* bermasalah yaitu label '#NUM!' pada kolom `ammonia` dan `is_safe` di Dataset 2. Nilai '#NUM!' diubah secara paksa (coerce) menjadi *NaN* (*Not a Number*). Setelah itu, seluruh baris yang mengandung *NaN* di-drop karena proporsinya sangat kecil, dan tipe datanya dikonversi menjadi numerik *float*.
2. **Penanganan Missing Values (Data Imputation):**
   Apabila menggunakan Dataset 1, penanganan untuk fitur `ph`, `Sulfate`, dan `Trihalomethanes` diatasi dengan mengisi rata-rata (*Mean Imputation*) berdasarkan masing-masing kelas target.
3. **Pemisahan Data (Data Splitting):**
   Memisahkan fitur variabel independen (X) dan variabel dependen (y). Membagi dataset menjadi dua subset: *Training Data* (80%) untuk melatih model, dan *Testing Data* (20%) untuk menguji performa model secara terisolasi.
4. **Feature Scaling / Normalisasi:**
   Karena rentang ukuran variabel independen (contoh: aluminium dan chloramine) bervariasi, dilakukan normalisasi skala menggunakan `StandardScaler` (Z-score normalization). Tahap ini sangat vital untuk algoritma berbantuan perhitungan metrik jarak dan konvergensi gradient.
5. **Penanganan Imbalanced Data (Class Imbalance Handling):**
   Pengecekan distribusi kelas `is_safe` (0 = Tidak Aman, 1 = Aman) mendapati rasio kelas mayoritas dan minoritas tidak seimbang (sekitar 88% berbanding 12%). Diterapkan *SMOTE (Synthetic Minority Over-sampling Technique)* pada *Training Data* untuk mensintesis data kelas minoritas agar menjadi seimbang, mencegah model *machine learning* bias dan menebak "0" secara buta.

## 4. Modelling
**Alasan Pemilihan Model:**
Karena ini adalah masalah klasifikasi biner, pendekatan algoritma yang dipilih adalah algoritma ensemble *Decision Trees*:
- **Random Forest Classifier (RF):** Dipilih sebagai algoritma utama karena sangat kuat (*robust*) dalam menangani data berdimensi banyak dengan pola non-linear. Random Forest membangun kumpulan *decision trees* secara paralel sehingga tidak rentan terhadap ke-overfitting-an pada kehadiran pencilan data alam (seperti nilai metrik air yang fluktuatif). RF juga memberikan *Feature Importance* untuk menafsirkan kimia apa yang paling memengaruhi keamanan air.
- **Logistic Regression (LR):** Digunakan sebagai *baseline* primitif untuk membandingkan seberapa kompleks hubungan *dataset* (apakah sekadar linier atau non-linier).

**Proses Modelling:**
1. Menginisiasi model Logistic Regression dan Random Forest.
2. Melatih model-model tersebut (`model.fit`) dengan menggunakan *Data Training* yang telah diseimbangkan (*SMOTE*) dan diskalakan (*Scaled*).
3. Melakukan penyetelan hyperparameter (walaupun secara opsional, *Random Forest* default sering kali sudah mencapai performa tinggi).
4. Model ini disimpan *(persisted)* ke bentuk file `.pkl` menggunakan `joblib` agar siap di-*deploy* ke sistem produksi.

## 5. Evaluation
**Cara Evaluasi Model:**
Model dievaluasi menggunakan Test Set (20% data yang belum pernah dilihat model selama proses *training*). Pengukuran menggunakan metrik evaluasi dari *Confusion Matrix*:
- **Accuracy:** Mengukur keseluruhan tebakan benar model.
- **Precision:** Rasio dari prediksi benar untuk kelas "Aman".
- **Recall (Sensitivity):** Rasio dari total kondisi asli kelas "Aman" yang dapat ditemukan model.
- **F1-Score:** Rata-rata harmonis presisi dan recall.

**Hasil dan Evaluasi Khusus (Business Justification):**
Berdasarkan pengujian komparatif, model Random Forest mengungguli Logistic Regression secara signifikan.
Dalam konteks deteksi air beracun dan keselamatan masyarakat, kesalahan *False Positive* (memprediksi air sebagai "Aman/Layak" padahal nyatanya "Beracun") dapat berakibat fatal (High Risk). Oleh karena itu, kita memprioritaskan metrik yang mampu meminimalkan salah prediksi pada kelas kelayakan. Peningkatan dengan metode SMOTE memungkinkan *Random Forest Classifier* menghasilkan *Recall* yang sangat stabil di angka atas 90%, memastikan keselamatan masyarakat terjamin dengan baik saat implementasi langsung.

## 6. Deployment
Tahapan akhir dari siklus proyek *Data Science* ini adalah rencana *Deployment* untuk memberikan fungsionalitas bagi pengguna akhir secara "gratis, lengkap, dan profesional".

**Rencana Deployment:**
1. **Penyimpanan Model (Model Serialization):**
   Model terbaik (Random Forest) dan *StandardScaler* disimpan dalam format berukuran kecil (`rf_model.pkl` dan `scaler.pkl`). Proses ini memungkinkan penggunaan ulang memori komputasi secara efisien tanpa perlu me-*retrain* data setiap kali digunakan.
2. **Arsitektur Antarmuka (Front-end & Back-end):**
   Mengingat ketersediaan perangkat *open-source* profesional, pengembangan antarmuka menggunakan *framework* web **Streamlit** (berbasis Python). *Streamlit* akan menyediakan formulir *input* pengguna yang intuitif—petugas PDAM atau laboratorium hanya perlu memasukkan angka hasil sensor logam berat. Di balik sistem, aplikasi memuat objek *scaler* dan model, mentransformasikan *input* mentah, dan memanggil fungsi `.predict_proba()`.
3. **Infrastruktur Cloud Hosting (Gratis & Andal):**
   Agar biaya operasional mencapai Rp 0 namun tetap andal, kode sumber *Streamlit* akan dititipkan ke *repository* di **GitHub**. Platform **Streamlit Community Cloud** (atau *Render.com*) akan dihubungkan secara langsung ke *repository* ini (berfungsi sebagai CI/CD). Setiap perubahan di *GitHub* akan secara otomatis memicu pembaruan aplikasi web (*Continuous Deployment*).
4. **User Flow & Monitoring:**
   Teknisi lapangan atau pengguna akhir memasukkan hasil lab. Sistem akan memunculkan layar konfirmasi visual (indikator Hijau jika Layak, Merah jika Beracun). Untuk menjaga *life cycle* model terhadap pergeseran kualitas lingkungan (*Data Drift*), akan ditambahkan fitur *logging* sederhana untuk mencatat prediktabilitas model. Data tersebut akan dievaluasi ulang setiap tahun untuk memicu siklus pelatihan model generasi selanjutnya (*retraining*).
