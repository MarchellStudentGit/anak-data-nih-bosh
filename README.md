# 📊 Analisis Kesejahteraan Sosial & Pipeline Data Science 2026
> **Status Proyek:** `Branch: dev-marchell` — *Inisialisasi Pipeline & Sinkronisasi Database Selesai*

Repositori ini memuat seluruh rangkaian pengerjaan tugas kelompok Metodologi Data Science, mulai dari pembersihan data mentah (*Data Wrangling*), analisis karakteristik (*EDA*), rekayasa fitur, pemodelan mesin pencari pola (*Clustering*), hingga tahap deployment otomatis ke cloud database dan visualisasi interaktif.

---

## 🚀 Progres Pengerjaan (Update Terbaru)

Berikut adalah peta jalan (*roadmap*) arsitektur sistem yang telah selesai dibangun dan siap dikolaborasikan:

| Tahap | Berkas Eksekusi | Deskripsi Status | Status |
| :--- | :--- | :--- | :---: |
| **Langkah 1** | `data_wrangling.py` | Pembersihan data mentah BPS, penanganan *missing values*, & standardisasi format. |  ✅ *Done* |
| **Langkah 2** | `eda_kesejahteraan.py`| Eksplorasi karakteristik data sosial, visualisasi matriks korelasi, & distribusi fitur. | ✅ *Done* |
| **Langkah 3** | `feature_engineering.py`| Seleksi fitur sensitif, reduksi dimensi, dan transformasi skala variabel numerik. | ✅ *Done* |
| **Langkah 4** | `clustering_kmeans.py` | Implementasi algoritma K-Means untuk pengelompokan tingkat kesejahteraan. | ✅ *Done* |
| **Langkah 5** | `deploy_supabase.py` | Automasi migrasi data hasil olahan dari lokal menuju Cloud Database (Supabase). | ✅ *Done* |
| **Langkah 6** | `app.py` | Pembuatan visualisasi dan antarmuka *dashboard* interaktif menggunakan Streamlit. | ✅ *Done* |

---

## 🛠️ Spesifikasi Environment & Struktur Folder

Untuk memastikan kode berjalan dengan lancar tanpa kendala *package mismatch*, proyek ini menggunakan Virtual Environment (`env/`) terisolasi yang mendasarkan instalasinya pada pustaka berikut:

### Komponen Utama `requirements.txt`
* **Data Processing & Analytics:** `pandas`, `numpy`
* **Visualizations:** `matplotlib`, `seaborn`
* **Machine Learning Model:** `scikit-learn`
* **Cloud Infrastructure:** `supabase`, `python-dotenv`
* **Web Application:** `streamlit`

### Struktur Repositori
```text
anak-data-nih-bosh/
├── data/
│   ├── raw/                 # Data mentah hasil unduhan BPS
│   └── clean/               # Berkas CSV hasil eksekusi data_wrangling.py
├── plots/                   # Output visualisasi distribusi dan korelasi (.png)
├── .env                     # [LOCAL ONLY] Kredensial & URL API Supabase (Hidden)
├── .gitignore               # Proteksi internal Git agar folder env/ & .env tidak bocor
├── requirements.txt         # Daftar dependencies library Python
├── app.py                   # Berkas utama aplikasi web Streamlit
└── README.md                # Dokumentasi proyek