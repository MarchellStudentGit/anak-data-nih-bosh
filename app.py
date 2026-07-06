import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
import time

st.set_page_config(
    page_title="Prediksi AQI - Anak Data Nih Bosh",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Headers */
    .title-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .title-box h1 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    .title-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    .author-box {
        font-size: 1rem;
        margin-top: 15px;
        font-style: italic;
    }
    
    /* Tabs Customization (Theme Aware) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white !important;
        font-weight: bold;
        border: none;
    }
    
    /* Glossary Cards */
    .glossary-card {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        margin-bottom: 15px;
    }
    .glossary-card h4 {
        margin-top: 0;
        font-size: 1.1rem;
    }
    .glossary-card p {
        margin-bottom: 0;
        font-size: 0.95rem;
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
    <h1>Prediksi Indeks Kualitas Udara (AQI) Berbasis Cuaca</h1>
    <p>Tugas Akhir Metodologi Data Science — <b>Kelompok: Anak Data Nih Bosh</b></p>
    <div class="author-box">
        Disusun oleh:<br>
        Marchell Adi Pratama (672023081) | Hendy Christian Nugroho (672023161) | Maria Anne Pujara (672023268)
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("Model XGBoost belum tersedia. Pastikan script pelatihan model telah dijalankan.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Form Prediksi AQI", "Evaluasi Model", "Glosarium"])

with tab1:
    st.info("Masukkan parameter prakiraan cuaca harian untuk memprediksi probabilitas tingkat bahaya kualitas udara (AQI).")
    
    with st.container(border=True):
        st.subheader("1. Parameter Suhu")
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature_2m_max = st.number_input("Suhu Maksimal (°C)", value=32.5, step=0.5, help="Prakiraan suhu tertinggi harian.")
        with c2:
            temperature_2m_min = st.number_input("Suhu Minimal (°C)", value=24.0, step=0.5, help="Prakiraan suhu terendah harian.")
        with c3:
            apparent_temperature_max = st.number_input("Suhu Terasa Maksimal (°C)", value=35.0, step=0.5, help="Suhu yang dirasakan dengan memperhitungkan faktor kelembapan.")

    with st.container(border=True):
        st.subheader("2. Parameter Angin & Awan")
        c4, c5, c6 = st.columns(3)
        with c4:
            windspeed_10m_max = st.number_input("Kecepatan Angin (km/h)", value=12.5, step=0.5, help="Kecepatan rata-rata hembusan angin harian.")
        with c5:
            windgusts_10m_max = st.number_input("Hembusan Angin Maksimal (km/h)", value=25.0, step=0.5, help="Kecepatan hembusan angin paling tinggi.")
        with c6:
            cloudcover_mean = st.number_input("Rata-rata Tutupan Awan (%)", value=45.0, step=1.0, help="Persentase tutupan awan harian.")

    with st.container(border=True):
        st.subheader("3. Parameter Presipitasi & Waktu")
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            precipitation_sum = st.number_input("Curah Hujan (mm)", value=0.0, step=0.1, help="Total curah hujan harian.")
        with c8:
            heavy_rain_flag = st.selectbox("Indikator Hujan Lebat", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
        with c9:
            month = st.slider("Bulan", min_value=1, max_value=12, value=6)
        with c10:
            is_weekend = st.selectbox("Akhir Pekan", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)", help="Menandakan hari libur (Sabtu/Minggu).")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Jalankan Prediksi", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Memproses data melalui model XGBoost..."):
            time.sleep(0.5) 
            
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
            st.subheader("Hasil Klasifikasi Model")
            
            if prediction == 1:
                st.error("Klasifikasi: Kualitas Udara Tidak Sehat (AQI > 100)")
                st.markdown("""
                Kondisi meteorologi yang diinputkan berpotensi menyebabkan akumulasi polutan di udara mencapai tingkat yang tidak sehat bagi masyarakat. 
                Disarankan untuk mengurangi aktivitas luar ruangan.
                """)
                st.metric("Probabilitas Prediksi", f"{prediction_proba[1]*100:.2f}%")
            else:
                st.success("Klasifikasi: Kualitas Udara Wajar/Baik (AQI <= 100)")
                st.markdown("""
                Kondisi meteorologi diperkirakan cukup mendukung dispersi polutan, sehingga kualitas udara berada dalam ambang batas yang wajar untuk beraktivitas.
                """)
                st.metric("Probabilitas Prediksi", f"{prediction_proba[0]*100:.2f}%")

with tab2:
    st.subheader("Evaluasi Kinerja Model")
    st.write("Visualisasi ini menyajikan hasil evaluasi dari model XGBoost yang telah dilatih menggunakan data historis.")
    
    c_img1, c_img2 = st.columns(2)
    
    with c_img1:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")
            if cm_img:
                st.image(cm_img, use_container_width=True)
                st.caption("Matriks kebingungan menunjukkan distribusi prediksi benar dan salah pada data uji (20%).")
            else:
                st.warning("Visualisasi belum tersedia.")
                
    with c_img2:
        with st.container(border=True):
            st.markdown("**Feature Importance**")
            if fi_img:
                st.image(fi_img, use_container_width=True)
                st.caption("Grafik yang menunjukkan tingkat signifikansi tiap parameter cuaca dalam mempengaruhi hasil prediksi polusi udara.")
            else:
                st.warning("Visualisasi belum tersedia.")

with tab3:
    st.subheader("Glosarium Istilah")
    st.markdown("Penjelasan singkat mengenai beberapa istilah teknis yang digunakan dalam aplikasi ini:")
    
    st.markdown("""
    <div class="glossary-card">
        <h4>1. AQI (Air Quality Index)</h4>
        <p>Indeks Kualitas Udara adalah parameter standar yang digunakan untuk mengkomunikasikan tingkat polusi udara kepada publik. Nilai AQI di atas 100 menunjukkan bahwa udara sudah memasuki kategori tidak sehat untuk kelompok tertentu maupun masyarakat umum.</p>
    </div>
    
    <div class="glossary-card">
        <h4>2. PM2.5 & PM10</h4>
        <p>Particulate Matter (PM) merujuk pada partikel padat atau cair yang melayang di udara. Angka 2.5 dan 10 merujuk pada diameter partikel dalam satuan mikrometer. Partikel ini sangat kecil dan dapat masuk ke saluran pernapasan sehingga berbahaya bagi kesehatan.</p>
    </div>
    
    <div class="glossary-card">
        <h4>3. Pengaruh Cuaca terhadap Polusi</h4>
        <p>Variabel meteorologi seperti kecepatan angin dapat menyebarkan polutan, sedangkan presipitasi (hujan) memiliki efek pencucian yang dapat menurunkan konsentrasi partikel di udara. Sebaliknya, kondisi angin yang stagnan seringkali menyebabkan akumulasi polusi.</p>
    </div>
    
    <div class="glossary-card">
        <h4>4. XGBoost Classifier</h4>
        <p>Extreme Gradient Boosting adalah algoritma <i>Machine Learning</i> berbasis <i>decision trees</i> yang banyak digunakan dalam pemodelan data tabular karena efisiensinya yang tinggi dalam mengenali pola data yang kompleks dan kemampuannya menangani <i>imbalanced data</i>.</p>
    </div>
    """, unsafe_allow_html=True)
