import os
import pandas as pd
import numpy as np  
from dotenv import load_dotenv
from supabase import create_client, Client

print("=== LANGKAH 5 (SELEPAS ENV): INTEGRASI DATABASE SUPABASE ===")

# 1. Memuat variabel dari berkas .env untuk keamanan kredensial
load_dotenv()

url_supabase = os.getenv("SUPABASE_URL")
kunci_supabase = os.getenv("SUPABASE_KEY")

if not url_supabase or not kunci_supabase:
    print("[ERROR] Kredensial Supabase di berkas .env belum disetel dengan benar!")
    exit()

# 2. Inisialisasi Klien Supabase
supabase: Client = create_client(url_supabase, kunci_supabase)
print(" -> Berhasil terhubung ke server awan Supabase.")

# 3. Memuat file hasil klasterisasi K-Means
file_input = "data/clean/hasil_cluster_2025.csv"

if not os.path.exists(file_input):
    print(f"[ERROR] Berkas {file_input} tidak ditemukan! Jalankan clustering_kmeans.py dulu.")
    exit()

df_hasil = pd.read_csv(file_input)
print(f" -> Memuat {len(df_hasil)} baris data hasil klasterisasi untuk diunggah.")

# 4. Mengubah tipe data NaN / Kosong menjadi None agar kompatibel dengan PostgreSQL Supabase
df_hasil = df_hasil.replace({np.nan: None})

# 5. Mengonversi DataFrame menjadi format JSON/List of Dictionaries untuk di-insert
catatan_unggahan = df_hasil.to_dict(orient="records")

print("\n=== MEMULAI PROSES UNGGAH DATA KE CLOUD ===")
try:
    # Mengunggah data ke tabel bernama 'kesejahteraan_provinsi_2025'
    # Catatan: Pastikan kamu sudah membuat tabel ini di dashboard Supabase milikmu
    data_respons = supabase.table("kesejahteraan_provinsi_2025").insert(catatan_unggahan).execute()
    
    print("[PROSES DEPLOYMENT BERHASIL COY!]")
    print(" -> Data klasterisasi ekonomi 2025 sudah nangkring di database awan Supabase.")
    print(" -> Data science pipeline dari raw data hingga cloud database selesai 100%.")

except Exception as e:
    print("\n[PERINGATAN] Gagal mengunggah otomatis ke tabel Supabase.")
    print(f"Detail kendala: {e}")
    print("\n[SOLUSI ALTERNATIF JIKA BELUM SET UP TABEL DI SQL SUPABASE]:")
    print("Kamu bisa masuk ke Dashboard Supabase -> Table Editor -> Import data via CSV,")
    print("lalu unggah langsung file lokal kamu: 'data/clean/hasil_cluster_2025.csv'")