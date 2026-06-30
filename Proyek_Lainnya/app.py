import streamlit as pd
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Konfigurasi Awal Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Kesejahteraan Sektoral 2025",
    page_icon="📊",
    layout="wide"
)

# 2. Memuat berkas .env untuk koneksi database
load_dotenv()
url_supabase = os.getenv("SUPABASE_URL")
kunci_supabase = os.getenv("SUPABASE_KEY")

@st.cache_resource
def inisialisasi_supabase():
    """Fungsi di-cache agar Streamlit tidak membuat koneksi ulang setiap ada interaksi"""
    return create_client(url_supabase, kunci_supabase)

try:
    supabase = inisialisasi_supabase()
except Exception as e:
    st.error(f"Gagal terhubung ke Supabase. Periksa file .env Anda. Detail: {e}")
    st.stop()

# 3. Mengambil Data Langsung dari Supabase Cloud
@st.cache_data
def ambil_data_klaster():
    # Mengambil data dari tabel yang baru saja kamu unggah
    respons = supabase.table("kesejahteraan_provinsi_2025").select("*").execute()
    return pd.DataFrame(respons.data)

with st.spinner("Mengunduh data klasterisasi dari Supabase Cloud..."):
    df_prov = ambil_data_klaster()

# 4. Header Aplikasi UI
st.title("📊 Dashboard Analisis Kesejahteraan & Kerentanan Sektoral 2025")
st.markdown("""
Aplikasi ini menampilkan hasil analisis tingkat lanjut menggunakan siklus **CRISP-DM** terhadap data upah sektoral BPS dan Garis Kemiskinan Makanan tahun 2025. Data ditarik secara 
*live* dari basis data cloud **Supabase**.
""")
st.divider()

# 5. Bagian Ringkasan Utama (Metrics)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Provinsi Teranalisis", f"{len(df_prov)} Wilayah")
with col2:
    rerata_upah_nasional = df_prov['rata_rata_upah'].mean()
    st.metric("Rata-Rata Upah Makro Nasional", f"Rp {rerata_upah_nasional:,.0f}")
with col3:
    rerata_ratio = df_prov['rasio_daya_beli'].mean()
    st.metric("Rata-Rata Rasio Daya Beli", f"{rerata_ratio:.2f} x GKM")

st.divider()

# 6. Pembuatan Layout Dashboard (Kiri: Grafik, Kanan: Data Tabel)
Kolom_Kiri, Kolom_Kanan = st.columns([3, 2])

with Kolom_Kiri:
    st.subheader("🎯 Sebaran Hasil Klasterisasi K-Means")
    
    # Membuat visualisasi scatter plot dari data Supabase
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_prov, 
        x="rasio_daya_beli", 
        y="gap_kesejahteraan", 
        hue="cluster_label", 
        palette="bright", 
        s=100, 
        ax=ax
    )
    ax.set_title("Segmentasi Provinsi Berdasarkan Daya Beli & Gap Kesejahteraan")
    ax.set_xlabel("Rasio Daya Beli (Upah / GKM Makanan)")
    ax.set_ylabel("Gap Kesejahteraan (Rupiah)")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    st.pyplot(fig)

with Kolom_Kanan:
    st.subheader("🔍 Telusuri Data Provinsi")
    
    # Opsi Filter Interaktif untuk User Dashboard
    pilihan_cluster = st.multiselect(
        "Filter Berdasarkan Klaster:",
        options=sorted(df_prov['cluster_label'].unique()),
        default=sorted(df_prov['cluster_label'].unique())
    )
    
    # Filter DataFrame berdasarkan input user
    df_filtered = df_prov[df_prov['cluster_label'].isin(pilihan_cluster)]
    
    # Menampilkan tabel interaktif yang bisa di-sort oleh user
    st.dataframe(
        df_filtered[['provinsi', 'rata_rata_upah', 'rasio_daya_beli', 'gap_kesejahteraan', 'cluster_label']],
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.caption("📊 Sistem Analisis Kesejahteraan & Kerentanan Sektoral - Tugas Metodologi Data Science | 2026")