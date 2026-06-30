import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sistem Peringatan Dini AQI",
    page_icon="🌪️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #D32F2F;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('results/xgb_weather_model.pkl')
        scaler = joblib.load('results/weather_scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model_and_scaler()

st.markdown('<div class="main-header">🌪️ Sistem Prediksi Bahaya Polusi Udara (AQI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Menebak tingkat bahaya kualitas udara esok hari hanya dari ramalan cuaca</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model XGBoost belum tersedia! Harap jalankan kode `Air_Quality_Weather_Project.py` terlebih dahulu.")
    st.stop()

st.markdown("### 🌤️ Masukkan Ramalan Cuaca Kota Anda")
st.info("Data ini bisa Anda peroleh dari stasiun Meteorologi lokal atau aplikasi cuaca di HP Anda.")

col1, col2, col3 = st.columns(3)

with col1:
    temperature_2m_max = st.number_input("Suhu Maksimal Harian (°C)", value=32.5, step=0.5)
    temperature_2m_mean = st.number_input("Suhu Rata-rata Harian (°C)", value=28.0, step=0.5)

with col2:
    apparent_temperature_max = st.number_input("Suhu Terasa Maksimal (°C)", value=35.0, step=0.5)
    precipitation_sum = st.number_input("Curah Hujan (mm)", value=0.0, step=0.1)

with col3:
    windspeed_10m_max = st.number_input("Kecepatan Angin Maksimal (km/h)", value=12.5, step=0.5)
    cloudcover_mean = st.number_input("Rata-rata Tutupan Awan (%)", value=45.0, step=1.0)

st.markdown("---")
predict_button = st.button("🔮 Ramal Kualitas Udara Besok", use_container_width=True)

if predict_button:
    # Orde kolom harus sama dengan training (X = ['temperature_2m_max', 'temperature_2m_mean', 'apparent_temperature_max', 'precipitation_sum', 'windspeed_10m_max', 'cloudcover_mean'])
    input_data = pd.DataFrame([{
        'temperature_2m_max': temperature_2m_max,
        'temperature_2m_mean': temperature_2m_mean,
        'apparent_temperature_max': apparent_temperature_max,
        'precipitation_sum': precipitation_sum,
        'windspeed_10m_max': windspeed_10m_max,
        'cloudcover_mean': cloudcover_mean
    }])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]
    
    st.markdown("### 📊 Hasil Prediksi")
    
    if prediction == 1:
        st.error("🚨 **PERINGATAN DINI: KUALITAS UDARA BERBAHAYA (AQI > 100)**")
        st.markdown("Berdasarkan ramalan cuaca tersebut, polusi (PM2.5 / Gas beracun) berpotensi terjebak di permukaan tanah. **Warga dihimbau menggunakan masker jika keluar rumah!**")
        st.metric("Probabilitas Berbahaya", f"{prediction_proba[1]*100:.1f}%")
    else:
        st.success("✅ **STATUS: UDARA AMAN (AQI <= 100)**")
        st.markdown("Kondisi meteorologi diperkirakan sangat baik untuk membersihkan / meniup polutan menjauhi permukaan. Aman untuk beraktivitas luar ruangan.")
        st.metric("Probabilitas Aman", f"{prediction_proba[0]*100:.1f}%")
