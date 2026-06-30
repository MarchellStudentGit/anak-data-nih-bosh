import pandas as pd
import numpy as np
import os

print("=== LANGKAH 2 (SELEPAS ENV): REKAYASA FITUR EKONOMI 2025 ===")

# 1. Menentukan path input dan output sesuai struktur folder baru
file_input = "data/clean/analisis_sektoral_2025.csv"
file_output = "data/clean/dataset_fitur_sektoral_2025.csv"

if not os.path.exists(file_input):
    print(f"[ERROR] File {file_input} tidak ditemukan! Jalankan data_wrangling.py dulu.")
    exit()

# 2. Memuat data hasil wrangling
df = pd.read_csv(file_input)
print(f"-> Sukses memuat {len(df)} baris data bersih.")

# 3. FITUR A: Menghitung Rasio Daya Beli (Upah dibagi GKM Makanan)
# Menunjukkan berapa kali lipat upah pekerja dibanding biaya makan minimum per kapita
df['rasio_daya_beli'] = (df['rata_rata_upah'] / df['gkm_makanan']).round(2)

# 4. FITUR B: Menghitung Gap Kesejahteraan (Selisih dalam Rupiah riil)
df['gap_kesejahteraan'] = df['rata_rata_upah'] - df['gkm_makanan']

# 5. FITUR C: Klasifikasi Status Kerentanan Ekonomi (Rule-Based Labeling)
# KRITIS : Gaji pekerja sangat mepet, <= 3 kali lipat dari batas makan minimum daerahnya
# RENTAN : Gaji pekerja berada di rentang menengah, antara 3 hingga 5 kali lipat GKM
# AMAN   : Gaji pekerja sudah sejahtera, > 5 kali lipat dari batas makan minimum
kondisi = [
    (df['rasio_daya_beli'] <= 3.0),
    (df['rasio_daya_beli'] > 3.0) & (df['rasio_daya_beli'] <= 5.0),
    (df['rasio_daya_beli'] > 5.0)
]
pilihan_label = ['KRITIS', 'RENTAN', 'AMAN']

df['status_kerentanan'] = np.select(kondisi, pilihan_label, default='TIDAK TERDEFINISI')

# 6. Menyimpan hasil rekayasa fitur ke folder data/clean/
df.to_csv(file_output, index=False)

print("\n[PROSES FEATURE ENGINEERING BERHASIL COY!]")
print(f"File kaya fitur disimpan di: {file_output}")
print(f"Total data siap olah: {len(df)} baris.")

print("\nSampel hasil rekayasa fitur (3 baris teratas):")
print(df[['provinsi', 'sektor', 'rata_rata_upah', 'gkm_makanan', 'rasio_daya_beli', 'status_kerentanan']].head(3))