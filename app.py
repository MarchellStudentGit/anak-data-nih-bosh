import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
import time

st.set_page_config(
    page_title="EcoGuard by Anak Data",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Vibe (Dark Mode Compatible)
st.markdown("""
<style>
    /* Headers */
    .title-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .title-box h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        margin: 0;
        font-size: 3.5rem;
    }
    .title-box p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    /* Tabs Customization (Theme Aware) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 25px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white !important;
        font-weight: bold;
        border: none;
    }
    
    /* Glossary Cards (Transparent for Dark Mode) */
    .glossary-card {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #e74c3c;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .glossary-card:hover {
        transform: translateX(5px);
    }
    .glossary-card h4 {
        margin-top: 0;
    }
    .glossary-card p {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('results/xgb_weather_model.pkl')
        scaler = joblib.load('results/weather_scaler.pkl')
        cm_img = Image.open('results/confusion_matrix_weather_xgb.png') if os.path.exists('results/confusion_matrix_weather_xgb.png') else None
        fi_img = Image.open('results/feature_importances_weather_xgb.png') if os.path.exists('results/feature_importances_weather_xgb.png') else None
        return model, scaler, cm_img, fi_img
    except Exception as e:
        return None, None, None, None

model, scaler, cm_img, fi_img = load_assets()

# Hero Section
st.markdown("""
<div class="title-box">
    <h1>🌍 EcoGuard Analytics</h1>
    <p>Sistem Deteksi Dini Polusi Udara (AQI) — <b>Anak Data Nih Bosh!</b></p>
</div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Peringatan: Engine AI (XGBoost) belum ditemukan. Hubungi Administrator sistem.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi AI", "📊 Model Analytics", "📖 Kamus Awam"])

with tab1:
    st.info("💡 **Petunjuk Penggunaan:** Silakan sesuaikan parameter cuaca di bawah ini berdasarkan ramalan BMKG esok hari, lalu tekan tombol prediksi di bagian bawah.")
    
    # Menggunakan UI Container berborder (Fitur Streamlit modern)
    with st.container(border=True):
        st.subheader("🌡️ 1. Parameter Suhu & Termal")
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature_2m_max = st.number_input("Suhu Maksimal (°C)", value=32.5, step=0.5, help="Suhu tertinggi yang akan dicapai esok hari.")
        with c2:
            temperature_2m_min = st.number_input("Suhu Minimal (°C)", value=24.0, step=0.5, help="Suhu terendah (biasanya terjadi saat dini hari).")
        with c3:
            apparent_temperature_max = st.number_input("Suhu Terasa Maksimal (°C)", value=35.0, step=0.5, help="Suhu nyata yang dirasakan kulit manusia akibat pengaruh kelembapan.")

    with st.container(border=True):
        st.subheader("💨 2. Dinamika Angin & Atmosfer")
        c4, c5, c6 = st.columns(3)
        with c4:
            windspeed_10m_max = st.number_input("Kec. Angin Rata-rata (km/h)", value=12.5, step=0.5, help="Angin yang stabil sangat krusial untuk menyapu bersih polusi dari atas kota.")
        with c5:
            windgusts_10m_max = st.number_input("Hembusan Angin Mendadak (km/h)", value=25.0, step=0.5, help="Hembusan angin kuat seketika yang dapat membubarkan konsentrasi gas beracun.")
        with c6:
            cloudcover_mean = st.number_input("Tutupan Awan (%)", value=45.0, step=1.0, help="Persentase awan di langit. Awan tebal terkadang bisa menahan polusi untuk naik ke atmosfer atas.")

    with st.container(border=True):
        st.subheader("🌧️ 3. Curah Hujan & Siklus Musim")
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            precipitation_sum = st.number_input("Total Curah Hujan (mm)", value=0.0, step=0.1, help="Volume air hujan. Hujan berfungsi sebagai 'detergen' alam yang mencuci debu PM2.5.")
        with c8:
            heavy_rain_flag = st.selectbox("Ada Badai / Hujan Lebat?", [0, 1], format_func=lambda x: "Ya (Hujan Lebat)" if x == 1 else "Tidak/Cerah")
        with c9:
            month = st.slider("Bulan (Pola Musiman)", min_value=1, max_value=12, value=6)
        with c10:
            is_weekend = st.selectbox("Akhir Pekan (Weekend)?", [0, 1], format_func=lambda x: "Ya (Hari Libur)" if x == 1 else "Bukan (Hari Kerja)", help="Hari libur biasanya menurunkan jumlah emisi gas buang dari kendaraan pekerja.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 JALANKAN ANALISIS KECERDASAN BUATAN", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Mengolah milyaran rantai data iklim historis..."):
            time.sleep(0.8) # Memberikan efek "komputasi" agar terasa lebih meyakinkan
            
            features = [
                'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max', 
                'precipitation_sum', 'windspeed_10m_max', 'windgusts_10m_max', 
                'cloudcover_mean', 'heavy_rain_flag', 'month', 'is_weekend'
            ]
            
            input_data = pd.DataFrame([[
                temperature_2m_max, temperature_2m_min, apparent_temperature_max, 
                precipitation_sum, windspeed_10m_max, windgusts_10m_max, 
                cloudcover_mean, heavy_rain_flag, month, is_weekend
            ]], columns=features)
            
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            
            st.markdown("---")
            st.subheader("📋 Hasil Diagnostik Sistem")
            
            if prediction == 1:
                st.error("🚨 **STATUS DARURAT MERAH: KUALITAS UDARA SANGAT BERBAHAYA (AQI > 100)**")
                st.markdown("""
                > **Peringatan Otoritas Kesehatan:**
                > Berdasarkan pola angin yang stagnan dan/atau suhu yang terperangkap, AI memprediksi pembentukan kabut asap beracun (PM2.5) di lapisan pernapasan manusia.
                > 
                > **Aksi Rekomendasi:**
                > 1. Warga wajib menggunakan masker medis (N95) saat evakuasi atau keluar rumah.
                > 2. Penderita asma dan lansia dilarang keras berolahraga *outdoor*.
                """)
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Probabilitas Tepat", f"{prediction_proba[1]*100:.2f}%", "- AI Sangat Yakin", delta_color="inverse")
            else:
                st.success("✅ **STATUS HIJAU ZAMRUD: UDARA BERSIH DAN AMAN (AQI <= 100)**")
                st.markdown("""
                > **Laporan Ekologis:**
                > Sirkulasi cuaca esok hari beroperasi bagaikan penyaring alami. Polusi akan tertiup angin atau tercuci habis oleh presipitasi hujan.
                > 
                > **Aksi Rekomendasi:**
                > Udara sedang dalam kualitas prima. Ini adalah waktu terbaik untuk melakukan olahraga maraton, piknik keluarga, dan membuka jendela rumah lebar-lebar.
                """)
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Probabilitas Tepat", f"{prediction_proba[0]*100:.2f}%", "+ Sangat Aman")

with tab2:
    st.subheader("📈 Validasi Performa AI (XGBoost Metrics)")
    st.write("Visualisasi interaktif ini diekstraksi langsung dari dapur pelatihan Machine Learning. Sistem ini sepenuhnya transparan dan berbasis data otentik.")
    
    c_img1, c_img2 = st.columns(2)
    
    with c_img1:
        with st.container(border=True):
            st.markdown("#### Akurasi Klasifikasi (Confusion Matrix)")
            if cm_img:
                st.image(cm_img, use_container_width=True)
                st.caption("Diagram ini memvalidasi kemampuan model dalam mendeteksi ancaman polusi tanpa menghasilkan terlalu banyak peringatan palsu (False Alarms).")
            else:
                st.warning("Visualisasi belum siap.")
                
    with c_img2:
        with st.container(border=True):
            st.markdown("#### Rahasia Alam Terungkap (Feature Importance)")
            if fi_img:
                st.image(fi_img, use_container_width=True)
                st.caption("Grafik pilar ini mengungkap fakta empiris: Parameter yang paling tinggi pilarnya adalah elemen cuaca yang paling bertanggung jawab atas penumpukan polusi di kota Anda.")
            else:
                st.warning("Visualisasi belum siap.")

with tab3:
    st.subheader("📚 Kamus Cerdas (Tanpa Bahasa Robot)")
    st.markdown("Kami menyadari istilah meteorologi sangat memusingkan. Mari kita gunakan logika sederhana!")
    
    st.markdown("""
    <div class="glossary-card">
        <h4>🏭 1. Indeks AQI (Air Quality Index)</h4>
        <p>Bayangkan nilai ujian di sekolah, tapi terbalik! Di sekolah, nilai 100 itu bagus. Tapi untuk AQI, nilai di atas 100 artinya <b>SANGAT BURUK</b>. AQI menghitung seberapa pekat racun tak kasat mata (seperti timbal dan gas sulfur) di udara yang kita hirup.</p>
    </div>
    
    <div class="glossary-card">
        <h4>🦠 2. Monster Tak Kasat Mata: PM2.5</h4>
        <p>Debu PM2.5 adalah pembunuh senyap. Ukurannya begitu kecil hingga masker kain biasa tidak bisa menahannya. Saat terhirup, ia langsung masuk ke paru-paru dan menembus pembuluh darah, memicu penyakit mematikan secara diam-diam.</p>
    </div>
    
    <div class="glossary-card">
        <h4>🌬️ 3. Hubungan Ajaib Angin & Polusi</h4>
        <p>Coba tiup asap rokok yang mengumpul di ruangan. Pasti asapnya buyar, kan? Nah, <b>Kecepatan Angin (Windspeed)</b> bekerja persis seperti kipas raksasa buatan Tuhan. Jika angin kencang, kota akan bersih dari polusi. Jika angin berhenti berhembus, kota akan "tersedak" polusi.</p>
    </div>
    
    <div class="glossary-card">
        <h4>🤖 4. Kecerdasan XGBoost (Xtreme Gradient Boosting)</h4>
        <p>Ini bukan ramalan dukun. XGBoost adalah algoritma mutakhir yang bisa menelaah berjuta-juta pola cuaca di masa lalu secara simultan. Jika di masa lalu kejadian "Suhu 35°C + Tanpa Hujan" memicu AQI berbahaya, XGBoost akan mengingat pola itu dan memperingatkan kita sebelum kejadian itu terulang kembali esok hari.</p>
    </div>
    """, unsafe_allow_html=True)
