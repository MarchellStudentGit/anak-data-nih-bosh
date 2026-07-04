import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

st.set_page_config(
    page_title="Sistem Peringatan Dini AQI",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    .glossary-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
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

st.markdown('<div class="main-header">🌪️ EcoGuard: Air Quality Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Platform Cerdas Peringatan Dini Polusi Udara Berbasis Ramalan Cuaca</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model AI belum diinisiasi. Harap melatih model terlebih dahulu melalui script Python.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Mesin Prediksi", "📊 Dashboard Analitik", "📖 Kamus Istilah (Untuk Awam)"])

with tab1:
    st.markdown("### 🌤️ Parameter Ramalan Cuaca Esok Hari")
    st.info("💡 **Tips:** Anda bisa mendapatkan parameter ini dari prakiraan cuaca di HP Anda (seperti aplikasi Weather bawaan Apple/Android) atau website BMKG lokal.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature_2m_max = st.number_input("Suhu Maksimal (°C)", value=32.5, step=0.5, help="Suhu udara paling panas yang diramalkan terjadi esok hari.")
        temperature_2m_min = st.number_input("Suhu Minimal (°C)", value=24.0, step=0.5, help="Suhu udara paling dingin (biasanya saat subuh/malam).")
        apparent_temperature_max = st.number_input("Suhu Terasa (Apparent) (°C)", value=35.0, step=0.5, help="Suhu yang benar-benar dirasakan oleh kulit manusia, dipengaruhi oleh kelembapan (seperti indeks panas/gerah).")
        
    with col2:
        precipitation_sum = st.number_input("Total Curah Hujan (mm)", value=0.0, step=0.1, help="Perkiraan total air hujan yang akan turun sepanjang hari. Jika 0, berarti cerah/mendung tanpa hujan.")
        heavy_rain_flag = st.selectbox("Indikator Hujan Lebat?", [0, 1], format_func=lambda x: "Ya (Hujan Deras)" if x == 1 else "Tidak", help="Pilih 'Ya' jika ada peringatan badai atau hujan sangat deras yang membasuh udara.")
        
    with col3:
        windspeed_10m_max = st.number_input("Kec. Angin Maksimal (km/h)", value=12.5, step=0.5, help="Seberapa kencang angin rata-rata yang bertiup. Angin kencang membantu menyapu polusi/asap dari kota.")
        windgusts_10m_max = st.number_input("Hembusan Angin Tiba-tiba (km/h)", value=25.0, step=0.5, help="Hembusan angin kencang (Windgusts) yang muncul secara tiba-tiba/seketika.")
        
    with col4:
        cloudcover_mean = st.number_input("Rata-rata Tutupan Awan (%)", value=45.0, step=1.0, help="0% berarti langit sangat cerah tanpa awan. 100% berarti langit mendung total.")
        month = st.slider("Bulan (Siklus Musim)", min_value=1, max_value=12, value=6, help="Bulan kejadian untuk membantu sistem mengenali tren polusi musiman (contoh: polusi lebih parah di musim kemarau).")
        is_weekend = st.selectbox("Akhir Pekan?", [0, 1], format_func=lambda x: "Ya (Sabtu/Minggu)" if x == 1 else "Bukan", help="Akhir pekan biasanya mengurangi emisi kendaraan kantoran, yang mempengaruhi tingkat polusi.")
    
    st.markdown("---")
    predict_button = st.button("🚀 ANALISIS TINGKAT BAHAYA UDARA", use_container_width=True)
    
    if predict_button:
        with st.spinner("AI sedang mengkalkulasi dispersi polutan..."):
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
            
            st.markdown("### 📋 Hasil Diagnosis AI")
            
            if prediction == 1:
                st.error("🚨 **STATUS MERAH: KUALITAS UDARA BERBAHAYA (AQI > 100)**")
                st.markdown("""
                Berdasarkan kombinasi cuaca di atas, sirkulasi atmosfer diperkirakan memburuk dan akan menjebak asap kotor, emisi kendaraan (PM2.5), dan gas beracun di dekat permukaan tanah (zona napas manusia).
                - **Saran Medis:** Sangat disarankan memakai masker (N95) saat berada di luar ruangan. Penderita asma harap berhati-hati!
                """)
                st.metric("Probabilitas Keakuratan", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.success("✅ **STATUS HIJAU: UDARA AMAN TERKENDALI (AQI <= 100)**")
                st.markdown("""
                Kondisi cuaca besok (seperti curah hujan atau tiupan angin) mendukung proses "pencucian" atau penyebaran polutan, sehingga udara perkotaan menjadi jauh lebih bersih.
                - **Saran Medis:** Udara segar! Sangat aman untuk berolahraga pagi atau beraktivitas santai di luar rumah tanpa masker.
                """)
                st.metric("Probabilitas Keakuratan", f"{prediction_proba[0]*100:.1f}%")

with tab2:
    st.markdown("### 🧠 Dokumentasi Intelegensi Buatan (XGBoost)")
    st.write("Tab ini ditujukan bagi teknisi, analis data, maupun dosen penguji untuk memverifikasi kecerdasan sistem prediksi yang berjalan di balik layar.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 1. Confusion Matrix (Matriks Kebingungan)")
        st.write("Menunjukkan rapor algoritma saat diuji coba (seperti ujian kelulusan). Semakin tebal warna merah di kotak persimpangan yang sama (Aman ditebak Aman, Bahaya ditebak Bahaya), artinya model AI sangat pintar.")
        if cm_img:
            st.image(cm_img, use_column_width=True)
        else:
            st.warning("Gambar belum di-generate.")
            
    with col_b:
        st.markdown("#### 2. Dampak Cuaca vs Polusi (Feature Importance)")
        st.write("Berdasarkan ratusan juta perbandingan komputasi, grafik ini merangkum *kunci rahasia* dari alam: elemen cuaca manakah yang paling kuat memicu penumpukan debu/polusi beracun di langit.")
        if fi_img:
            st.image(fi_img, use_column_width=True)
        else:
            st.warning("Gambar belum di-generate.")

with tab3:
    st.markdown("### 📖 Kamus Istilah Cepat (Untuk Masyarakat Umum)")
    st.markdown("""
    Jika Anda merasa bingung dengan istilah-istilah di atas, Anda berada di tempat yang tepat! Berikut adalah penjelasannya dalam bahasa sehari-hari:
    
    <div class="glossary-box">
        <h4>1. Apa itu AQI?</h4>
        <p><b>AQI (Air Quality Index)</b> adalah "rapor" untuk mengukur seberapa kotor atau beracun udara hari ini. Mirip seperti nilai ulangan, tapi bedanya, makin tinggi nilainya, makin <b>buruk</b> udaranya! Nilai AQI di atas 100 berarti udara sudah sangat tidak sehat dan bisa membuat Anda batuk atau sesak napas.</p>
    </div>
    
    <div class="glossary-box">
        <h4>2. Apa itu PM2.5? Kenapa berbahaya?</h4>
        <p>Ini adalah debu halus polusi (dari asap kendaraan atau pabrik) yang ukurannya sangat amat kecil—bahkan lebih kecil dari rambut Anda yang dibelah 30! Saking kecilnya, dia bisa menembus masker kain biasa, masuk menembus paru-paru, bahkan sampai ke aliran darah dan memicu penyakit jantung/kanker.</p>
    </div>
    
    <div class="glossary-box">
        <h4>3. Kok cuaca bisa pengaruhi polusi udara?</h4>
        <p>Bayangkan Anda menyalakan api unggun di dalam toples tertutup. Asapnya pasti diam mengumpul (AQI tinggi). Tapi jika toples itu disemprot air (Hujan) atau ditiup kipas angin kuat (Angin Kencang), asapnya akan bersih atau menyebar menjauh. Cuaca bertindak layaknya penyedot debu atau kipas raksasa bagi udara kota kita!</p>
    </div>
    
    <div class="glossary-box">
        <h4>4. Apa itu XGBoost (AI) yang disebut-sebut dari tadi?</h4>
        <p>XGBoost adalah nama otak robot/komputer yang sangat pintar. Kita menyuapinya dengan jutaan baris data ramalan cuaca dan tingkat polusi tahun-tahun sebelumnya. Robot ini belajar sendiri mencari tahu: "Oh, kalau suhunya sekian dan anginnya tenang, besok pasti udaranya beracun".</p>
    </div>
    """, unsafe_allow_html=True)
