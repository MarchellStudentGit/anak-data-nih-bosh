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

tab1, tab2 = st.tabs(["🔮 Mesin Prediksi Interaktif", "📊 Dashboard Analitik Model"])

with tab1:
    st.markdown("### 🌤️ Parameter Ramalan Cuaca Esok Hari")
    st.info("💡 **Tips:** Anda bisa mendapatkan parameter ini dari prakiraan BMKG atau aplikasi cuaca global.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature_2m_max = st.number_input("Suhu Maksimal (°C)", value=32.5, step=0.5)
        temperature_2m_min = st.number_input("Suhu Minimal (°C)", value=24.0, step=0.5)
        apparent_temperature_max = st.number_input("Suhu Terasa (Apparent) (°C)", value=35.0, step=0.5)
        
    with col2:
        precipitation_sum = st.number_input("Total Curah Hujan (mm)", value=0.0, step=0.1)
        heavy_rain_flag = st.selectbox("Indikator Hujan Lebat?", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
        
    with col3:
        windspeed_10m_max = st.number_input("Kec. Angin Maksimal (km/h)", value=12.5, step=0.5)
        windgusts_10m_max = st.number_input("Hembusan Angin Tiba-tiba (km/h)", value=25.0, step=0.5)
        
    with col4:
        cloudcover_mean = st.number_input("Rata-rata Tutupan Awan (%)", value=45.0, step=1.0)
        month = st.slider("Bulan (Siklus Musim)", min_value=1, max_value=12, value=6)
        is_weekend = st.selectbox("Akhir Pekan?", [0, 1], format_func=lambda x: "Ya (Sabtu/Minggu)" if x == 1 else "Bukan")
    
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
                Berdasarkan kombinasi cuaca di atas, sirkulasi atmosfer diperkirakan memburuk dan akan menjebak emisi beracun (seperti PM2.5 dan Karbon Monoksida) di lapisan bawah.
                - **Saran Tindakan:** Otoritas wajib mengeluarkan peringatan penggunaan masker.
                """)
                st.metric("Tingkat Keyakinan (Confidence)", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.success("✅ **STATUS HIJAU: UDARA AMAN TERKENDALI (AQI <= 100)**")
                st.markdown("""
                Kondisi iklim mendukung pencucian polutan (*washing effect* oleh curah hujan) atau penyebaran polutan oleh sirkulasi angin yang sehat.
                - **Saran Tindakan:** Sangat aman untuk berolahraga dan beraktivitas luar ruangan.
                """)
                st.metric("Tingkat Keyakinan (Confidence)", f"{prediction_proba[0]*100:.1f}%")

with tab2:
    st.markdown("### 🧠 Dokumentasi Intelegensi Buatan (XGBoost)")
    st.write("Di tab ini, Anda dapat memverifikasi kapabilitas teknis dan bukti ilmiah dari sistem Machine Learning yang bekerja di balik layar.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 1. Confusion Matrix (Akurasi Prediksi)")
        st.write("Matriks ini menunjukkan keberhasilan model pada data pengujian yang tak kasatmata (20% *Hold-out set*). Semakin gelap warna pada diagonal utama, semakin baik model mendeteksi hari berbahaya (Recall).")
        if cm_img:
            st.image(cm_img, use_column_width=True)
        else:
            st.warning("Gambar belum di-generate.")
            
    with col_b:
        st.markdown("#### 2. Kausalitas Iklim vs Polusi (Feature Importance)")
        st.write("Berdasarkan miliaran rantai keputusan pohon (*trees*), algoritma kami menyimpulkan parameter cuaca mana yang paling ekstrem mempengaruhi jebakan polusi udara.")
        if fi_img:
            st.image(fi_img, use_column_width=True)
        else:
            st.warning("Gambar belum di-generate.")
