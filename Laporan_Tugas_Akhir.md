# Laporan Tugas Akhir: Sistem Prediksi Kelayakan dan Kualitas Air Minum Menggunakan Machine Learning (Composite Model)

**Mata Kuliah:** Metodologi Data Science
**Tema:** Prediksi Kualitas Air Berdasarkan Penggabungan 2 Dataset (Fisikokimia Dasar dan Biologis/Logam Berat)

---

## 1. Business Understanding
**Latar Belakang dan Proses Bisnis Saat Ini**
Air adalah sumber kehidupan yang paling esensial bagi umat manusia. Mengacu pada *Sustainable Development Goals* (SDG) ke-6, memastikan ketersediaan dan manajemen air bersih serta sanitasi yang berkelanjutan bagi semua orang adalah prioritas global. Proses pengolahan kualitas air mengharuskan uji air untuk mematuhi ambang batas ganda: parameter kelayakan fisik (seperti pH dan kekeruhan) serta ketiadaan logam berat beracun/patogen biologis. 

Saat ini, proses inspeksi kualitas air seringkali mengandalkan uji laboratorium manual. Petugas harus menguji sifat fisikokimia di satu laboratorium, lalu merujuknya ke lab kimia/biologi terpisah untuk uji arsenik atau bakteri. Hal ini memakan waktu dan biaya operasional yang ganda. Keterlambatan hasil uji dapat menyebabkan air berbahaya terdistribusi ke masyarakat.

**Permasalahan Bisnis (Business Problem)**
Dibutuhkan suatu inovasi *Decision Support System* berbasis teknologi cerdas yang mampu mengevaluasi kedua ancaman (fisik dan logam berat) secara serentak dalam hitungan detik. 
Oleh karena itu, proyek ini difokuskan pada penggabungan metrik dari dua laboratorium (*dataset* uji fisik dan *dataset* uji logam berat) menjadi satu model Machine Learning (*Composite Model*). Model gabungan ini menerima **29 parameter input** sekaligus, lalu menyimpulkan secara komprehensif apakah air "Sangat Aman (Lolos uji fisik DAN kimiawi)" atau "Berbahaya". Sistem terpadu ini memangkas waktu dari berhari-hari menjadi *real-time*.

## 2. Data Collection and Understanding
**Data Collection (Pengumpulan Data)**
Data dikumpulkan secara sekunder dari dua sumber arsip (raw):
1. `water_potability.csv` (Dataset 1): Menyimpan 3.276 catatan matriks uji fisikokimia air, seperti `pH`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, dan `Turbidity`. Variabel target bawaan adalah `Potability`.
2. `waterQuality1.csv` (Dataset 2): Menyimpan 7.999 catatan uji biologis dan logam berat (contoh: aluminium, arsenic, bacteria, lead, dll.). Variabel target bawaan adalah `is_safe`.

**Data Understanding (Pemahaman Data)**
Kedua *dataset* memiliki karakteristik tersendiri. Pada Dataset 1, beberapa fitur (seperti `pH`, `Sulfate`) memiliki *missing values*. Pada Dataset 2, kolom `ammonia` dan `is_safe` memiliki anomali (terbaca sebagai *string* akibat nilai seperti '#NUM!'). Keduanya juga tidak memiliki ID primer (*primary key*) yang sama, karena berasal dari rekaman lab yang berbeda.

## 3. Data Preparation & Data Integration (Penggabungan)
Proyek ini mengaplikasikan teknik *Data Integration* tingkat lanjut, yaitu **Synthetic Horizontal Concatenation**, guna memenuhi persyaratan penggabungan heterogen:

1. **Data Cleaning:** Menghapus data anomali '#NUM!' pada Dataset 2 dengan metode paksa *Numeric Coercion* dan *Drop NA*.
2. **Penanganan Missing Values:** Mengimputasi nilai kosong di Dataset 1 dengan nilai tengah (*median*) agar distribusi data tidak terganggu.
3. **Data Integration (Penyatuan):** Karena Dataset 1 memiliki 3.276 baris, kami melakukan *random sampling* pada Dataset 2 (dari ~8.000 menjadi 3.276 baris) agar sejajar. Selanjutnya, kedua dataset **digabungkan secara horizontal** (`pd.concat`).
4. **Pembentukan Target Komposit (Composite Target):** 
   Dibuat target baru bernama `Final_Safety_Status`. Logikanya sangat ketat: Air hanya bernilai "1 (Aman)" apabila dari Dataset 1 air tersebut layak (`Potability == 1`) **DAN** dari Dataset 2 bebas logam berat (`is_safe == 1`). Jika salah satu gagal, maka statusnya adalah "0 (Bahaya)". Target asli kemudian dihapus untuk menghindari kebocoran data (*Data Leakage*).
5. **Class Imbalance Handling:** Dikarenakan syarat untuk menjadi "Aman" sangat ketat, sampel data yang bernilai "1" merosot tajam (menjadi *minority class* ekstrem). Oleh karena itu, diterapkan algoritma **SMOTE (Synthetic Minority Over-sampling Technique)** pada data *training* untuk mensintesis ulang sampel positif agar seimbang dengan data negatif.
6. **Feature Scaling:** Menerapkan `StandardScaler` untuk normalisasi ke-29 fitur secara bersamaan.

## 4. Modelling
**Alasan Pemilihan Model:**
Karena dataset kini memuat 29 dimensi fitur yang heterogen dan tidak terdistribusi secara normal-murni, **Random Forest Classifier** kembali diandalkan. Algoritma kelompok (*ensemble*) ini terbukti kebal (*robust*) terhadap dimensi yang besar tanpa memerlukan asumsi linearitas antar 29 variabel uji air.

**Proses Modelling:**
1. Model `Random Forest` diinisialisasi (dengan hyperparameter n_estimators=100).
2. Model dilatih menggunakan *Data Training* gabungan yang sudah di-SMOTE.
3. Objek Scaler dan Model diekspor menjadi file `.pkl` agar dapat dipanggil ulang di aplikasi web (Deployment).

## 5. Evaluation
**Hasil dan Cara Evaluasi:**
Pada *Testing Data* (sekitar 650 observasi acak tak terlihat), model komposit (gabungan) mampu mencapai angka **Akurasi Keseluruhan: 95%**. 

**Fokus Bisnis (*Business Justification*):**
Angka presisi untuk mendeteksi ancaman "Bahaya (0)" mencapai 97% dengan *Recall* 98%. Ini mengindikasikan bahwa model komposit ini nyaris tidak pernah meloloskan air yang beracun atau kotor. Setiap bahaya polusi sekecil apapun akan langsung diblokir (*flagged as danger*) oleh sistem. Hal ini sejalan dengan prinsip mitigasi risiko tinggi di perusahaan utilitas publik (lebih baik menolak air sehat daripada meloloskan air beracun). 

## 6. Deployment
Siklus perancangan algoritma ini diakhiri dengan *Deployment* untuk memberikan fungsionalitas UI yang gratis, lengkap, dan profesional kepada pengguna.

**Rencana Deployment:**
1. **Penyimpanan Model & Arsitektur Aplikasi:** Model raksasa hasil gabungan tersebut dibaca menggunakan bahasa Python via kerangka kerja web **Streamlit**. 
2. **Antarmuka (Front-End):** Di dalam file `app.py`, kami merancang antarmuka Tabular (*Tabs*). Mengingat ada 29 parameter uji yang harus di-*input* oleh teknisi laboratorium (10 metrik fisik, 19 metrik kimia), antarmuka yang bersih (*clean UI*) adalah suatu keharusan. *Tab 1* digunakan untuk mengisi uji fisikokimia dasar, dan *Tab 2* untuk uji patogen/logam.
3. **Infrastruktur Cloud Hosting (Gratis & Andal):** Kode Streamlit ini didorong (*Push*) ke repositori **GitHub**, kemudian dikaitkan dengan **Streamlit Community Cloud**. Setiap perubahan kode secara seketika akan ter-*deploy* ke publik (CI/CD pipeline aktif).
4. **User Flow & Monitoring:** Teknisi lapangan atau *end-user* memasukkan hasil kedua lab ke dalam aplikasi dan menekan satu tombol analisis. Dalam hitungan millidetik, sistem akan memunculkan status komprehensif: "Sangat Aman & Layak Konsumsi" (hijau) atau "Berbahaya / Tidak Layak" (merah).
