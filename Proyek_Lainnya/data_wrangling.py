import pandas as pd
import numpy as np
import os

# Memastikan folder untuk hasil pembersihan 'data/clean' otomatis tersedia
os.makedirs('data/clean', exist_ok=True)

print("=== LANGKAH 1: MEMBEDAH DATA SEKTORAL BPS 2025 ===")

# Menyesuaikan path ke folder data/raw/ sesuai direktori kamu
file_upah_mentah = "data/raw/Rata-Rata Upah_Gaji Bersih Sebulan Buruh_Karyawan_Pegawai menurut Provinsi dan Lapangan Pekerjaan Utama di 17 Sektor, 2025.csv"

if not os.path.exists(file_upah_mentah):
    print(f"[ERROR] File mentah tidak ditemukan di: {file_upah_mentah}")
    print("Pastikan nama file di dalam folder 'data/raw/' sudah benar.")
    exit()

df_raw_upah = pd.read_csv(file_upah_mentah, header=None)

# Nama sektor ada di baris indeks 2, Nama periode bulan ada di baris indeks 4
daftar_sektor = df_raw_upah.iloc[2].tolist()
daftar_periode = df_raw_upah.iloc[4].tolist()

# Mengisi kekosongan nama sektor akibat merge cells pada file CSV
sektor_sekarang = None
for i in range(len(daftar_sektor)):
    if pd.notna(daftar_sektor[i]) and daftar_sektor[i].strip() != "":
        sektor_sekarang = daftar_sektor[i].strip()
    daftar_sektor[i] = sektor_sekarang

list_data_upah = []
# Data provinsi riil dimulai dari baris indeks ke-5 ke bawah
for idx in range(5, len(df_raw_upah)):
    row = df_raw_upah.iloc[idx]
    nama_provinsi = row.iloc[0]
    
    if pd.isna(nama_provinsi) or 'indonesia' in str(nama_provinsi).lower() or str(nama_provinsi).strip() == "":
        continue
        
    nama_provinsi = str(nama_provinsi).strip().upper()
    
    for col_idx in range(1, len(row)):
        sektor = daftar_sektor[col_idx]
        periode = str(daftar_periode[col_idx]).strip().upper() if pd.notna(daftar_periode[col_idx]) else "TAHUNAN"
        nilai_upah = row.iloc[col_idx]
        
        if pd.isna(nilai_upah) or str(nilai_upah).strip() in ['-', '', '.']:
            continue
            
        try:
            upah_angka = float(str(nilai_upah).replace(',', '').strip())
            list_data_upah.append({
                'provinsi': nama_provinsi,
                'tahun': 2025,
                'sektor': sektor,
                'periode_upah': periode,
                'rata_rata_upah': upah_angka
            })
        except ValueError:
            continue

df_upah_clean = pd.DataFrame(list_data_upah)
print(f"[OK] Berhasil merapikan upah sektoral menjadi format LONG: {len(df_upah_clean)} baris.")


print("\n=== LANGKAH 2: MEMBEDAH DATA GARIS KEMISKINAN MAKANAN 2025 ===")

# Menyesuaikan path ke folder data/raw/ untuk garis kemiskinan
file_gkm_mentah = "data/raw/Garis Kemiskinan Makanan (Rupiah_Kapita_Bulan) Menurut Provinsi dan Daerah, 2025.csv"
df_raw_gkm = pd.read_csv(file_gkm_mentah, header=None)

list_data_gkm = []
for idx in range(5, len(df_raw_gkm)):
    row = df_raw_gkm.iloc[idx]
    nama_provinsi = row.iloc[0]
    
    if pd.isna(nama_provinsi) or 'indonesia' in str(nama_provinsi).lower() or str(nama_provinsi).strip() == "":
        continue
        
    nama_provinsi = str(nama_provinsi).strip().upper()
    
    # Kolom indeks 7 = Semester 1 (Maret), Kolom indeks 8 = Semester 2 (September)
    gkm_maret = row.iloc[7]
    gkm_september = row.iloc[8]
    
    for p_nama, gkm_val in [('MARET', gkm_maret), ('SEPTEMBER', gkm_september)]:
        if pd.isna(gkm_val) or str(gkm_val).strip() in ['-', '', '.']:
            continue
        try:
            gkm_angka = float(str(gkm_val).replace(',', '').strip())
            list_data_gkm.append({
                'provinsi': nama_provinsi,
                'tahun': 2025,
                'periode_kemiskinan': p_nama,
                'gkm_makanan': gkm_angka
            })
        except ValueError:
            continue

df_gkm_clean = pd.DataFrame(list_data_gkm)
print(f"[OK] Berhasil mengekstrak data kemiskinan: {len(df_gkm_clean)} baris.")


print("\n=== LANGKAH 3: INTEGRASI DAN PENYELARASAN PERIODE SURVEI ===")

# Menyelaraskan Sakernas Februari -> Susenas Maret
df_maret = df_upah_clean[df_upah_clean['periode_upah'] == 'FEBRUARI'].copy()
df_maret['periode_kemiskinan'] = 'MARET'

# Sakernas Agustus -> Susenas September
df_september = df_upah_clean[df_upah_clean['periode_upah'] == 'AGUSTUS'].copy()
df_september['periode_kemiskinan'] = 'SEPTEMBER'

df_upah_mapped = pd.concat([df_maret, df_september], ignore_index=True)

# Merge data berdasarkan provinsi dan periode
df_final = pd.merge(df_upah_mapped, df_gkm_clean, on=['provinsi', 'tahun', 'periode_kemiskinan'], how='inner')

# Kita simpan hasil bersihnya ke dalam folder baru: data/clean/
target_simpan = "data/clean/analisis_sektoral_2025.csv"
df_final.to_csv(target_simpan, index=False)

print("\n[PROSES DATA WRANGLING SELESAI & SUKSES COY!]")
print(f"Hasil gabungan bersih disimpan di: {target_simpan}")
print(f"Total baris terintegrasi riil: {len(df_final)} baris.")
print("\nSampel data terintegrasi (3 baris teratas):")
print(df_final.head(3))