import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("=== LANGKAH 4 (SELEPAS ENV): MODELLING K-MEANS CLUSTERING ===")

# 1. Tentukan path file input dan output sesuai direktori clean
file_input = "data/clean/dataset_fitur_sektoral_2025.csv"
file_output = "data/clean/hasil_cluster_2025.csv"
folder_plots = "plots"

if not os.path.exists(file_input):
    print(f"[ERROR] File {file_input} tidak ditemukan! Jalankan eda_kesejahteraan.py dulu.")
    exit()

# 2. Muat dataset
df = pd.read_csv(file_input)

# Kita lakukan aggregasi per provinsi untuk melihat profil makro wilayah
df_provinsi = df.groupby('provinsi').agg({
    'rata_rata_upah': 'mean',
    'gkm_makanan': 'mean',
    'rasio_daya_beli': 'mean',
    'gap_kesejahteraan': 'mean'
}).reset_index()

print(f"-> Sukses merangkum profil makro untuk {len(df_provinsi)} Provinsi.")

# 3. Ekstraksi Fitur untuk Clustering & Standardisasi (Scaling)
# Kita gunakan Rasio Daya Beli dan Gap Kesejahteraan sebagai basis segmentasi
X = df_provinsi[['rasio_daya_beli', 'gap_kesejahteraan']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. METODE ELBOW: Mencari K Optimal (Penjumlahan Kuadrat Jarak Intra-Klaster / WCSS)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Visualisasi Grafik Metode Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
plt.xlabel('Jumlah Klaster (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)

path_elbow = os.path.join(folder_plots, "metode_elbow.png")
plt.savefig(path_elbow, bbox_inches='tight', dpi=150)
plt.close()
print(f" -> Grafik Metode Elbow disimpan di: {path_elbow}")

# 5. EKSEKUSI K-MEANS CLUSTERING (Berdasarkan proposal 1.txt, dipilih k=3: Sejahtera, Rentan, Kritis)
k_optimal = 3
model_kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
df_provinsi['cluster_label'] = model_kmeans.fit_transform(X_scaled).argmax(axis=1) # Menentukan index klaster

# 6. EVALUASI MODEL
sil_score = silhouette_score(X_scaled, df_provinsi['cluster_label'])
db_index = davies_bouldin_score(X_scaled, df_provinsi['cluster_label'])

print("\n====== METRIK EVALUASI KLASTERISASI ======")
print(f"-> Silhouette Score      : {sil_score:.4f} (Mendekati 1 = Klaster Terpisah dengan Baik)")
print(f"-> Davies-Bouldin Index : {db_index:.4f} (Mendekati 0 = Batas Klaster Sangat Bagus)")

# 7. PROFILING KLASTER (Menganalisis karakteristik ekonomi tiap klaster)
profil_klaster = df_provinsi.groupby('cluster_label').agg({
    'provinsi': 'count',
    'rata_rata_upah': 'mean',
    'rasio_daya_beli': 'mean',
    'gap_kesejahteraan': 'mean'
}).rename(columns={'provinsi': 'jumlah_provinsi'}).reset_index()

print("\n====== PROFIL KARAKTERISTIK KLASTER ======")
print(profil_klaster)

# Simpan hasil pemetaan provinsi dan klasternya ke folder clean
df_provinsi.to_csv(file_output, index=False)
print(f"\n[PROSES MODELLING SELESAI!] Hasil pemetaan klaster disimpan di: {file_output}")