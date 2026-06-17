import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=== LANGKAH 3 (SELEPAS ENV): EXPLORATORY DATA ANALYSIS (EDA) ===")

# 1. Tentukan path file input dan folder output grafik
file_input = "data/clean/dataset_fitur_sektoral_2025.csv"
folder_plots = "plots"
os.makedirs(folder_plots, exist_ok=True)

if not os.path.exists(file_input):
    print(f"[ERROR] File {file_input} tidak ditemukan! Jalankan feature_engineering.py dulu.")
    exit()

# 2. Muat data hasil rekayasa fitur
df = pd.read_csv(file_input)
print(f"-> Sukses memuat {len(df)} baris data berfitur untuk dianalisis.")

# ==========================================
# ANALISIS 1: SEKTOR LAPANGAN KERJA PALING KRITIS
# ==========================================
print("\n[ANALISIS 1] Mencari Sektor Lapangan Kerja Paling Rentan/Kritis...")
rerata_sektor = df.groupby('sektor')['rasio_daya_beli'].mean().sort_values().reset_index()

print("\n--- 3 Sektor dengan Rata-Rata Rasio Daya Beli TERENDAH (Paling Rentan) ---")
print(rerata_sektor.head(3))

# ==========================================
# ANALISIS 2: PROVINSI DENGAN KETIMPANGAN UPAH TERTINGGI
# ==========================================
print("\n[ANALISIS 2] Mencari Provinsi dengan Variasi/Ketimpangan Upah Antarsektor Tertinggi...")
# Menggunakan Standar Deviasi (std) untuk melihat jarak ketimpangan antar sektor di tiap provinsi
ketimpangan_prov = df.groupby('provinsi')['rata_rata_upah'].std().sort_values(ascending=False).reset_index()
ketimpangan_prov.columns = ['provinsi', 'ketimpangan_upah_std']

print("\n--- 3 Provinsi dengan Ketimpangan Upah Sektoral PALING TINGGI ---")
print(ketimpangan_prov.head(3))

# ==========================================
# VISUALISASI 1: DISTRIBUSI STATUS KERENTANAN NASIONAL
# ==========================================
print("\n[VISUALISASI 1] Membuat Grafik Distribusi Status Kerentanan Pekerja...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='status_kerentanan', order=['KRITIS', 'RENTAN', 'AMAN'], palette='Set2')
plt.title('Distribusi Status Kerentanan Kesejahteraan Sektoral Indonesia (2025)')
plt.xlabel('Status Kerentanan')
plt.ylabel('Jumlah Record (Sektor-Provinsi)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Simpan Grafik ke folder plots/
path_plot1 = os.path.join(folder_plots, "distribusi_kerentanan.png")
plt.savefig(path_plot1, bbox_inches='tight', dpi=150)
plt.close()
print(f" -> Grafik distribusi berhasil disimpan di: {path_plot1}")

# ==========================================
# VISUALISASI 2: BOXPLOT RATIO DAYA BELI PER SEKTOR
# ==========================================
print("\n[VISUALISASI 2] Membuat Grafik Boxplot Rasio Daya Beli per Sektor...")
plt.figure(figsize=(12, 8))
# Mengurutkan berdasarkan median terendah agar terlihat sektor yang menderita
order_sektor = df.groupby('sektor')['rasio_daya_beli'].median().sort_values().index

sns.boxplot(data=df, y='sektor', x='rasio_daya_beli', order=order_sektor, palette='coolwarm')
plt.axvline(x=3.0, color='red', linestyle='--', label='Batas Kritis (Rasio <= 3.0)')
plt.title('Perbandingan Rasio Daya Beli Sektoral Terhadap Garis Kemiskinan Makanan (2025)')
plt.xlabel('Rasio Daya Beli (Upah / GKM Makanan)')
plt.ylabel('Sektor Lapangan Pekerjaan Utama')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Simpan Grafik Sektor
path_plot2 = os.path.join(folder_plots, "boxplot_rasio_sektoral.png")
plt.savefig(path_plot2, bbox_inches='tight', dpi=150)
plt.close()
print(f" -> Grafik komparasi sektoral berhasil disimpan di: {path_plot2}")

print("\n[PROSES EDA SELESAI COY!] Folder 'plots/' kamu sekarang sudah terisi grafik analitik.")