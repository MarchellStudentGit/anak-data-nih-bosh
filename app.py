import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman (Harus di awal)
st.set_page_config(
    page_title="Prediksi Kualitas Air Minum",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk memperindah tampilan
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache model agar tidak dimuat berulang kali
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('results/rf_model.pkl')
        scaler = joblib.load('results/scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model_and_scaler()

# Judul Utama
st.markdown('<div class="main-header">💧 Sistem Prediksi Kelayakan Air Minum</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mendeteksi potensi bahaya beracun dari kandungan logam berat & biologis (Machine Learning)</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model atau Scaler tidak ditemukan! Harap jalankan `Water_Quality_Project.py` terlebih dahulu untuk menghasilkan model.")
    st.stop()

# Sidebar untuk Input
st.sidebar.header("🧪 Masukkan Hasil Uji Laboratorium")
st.sidebar.markdown("Silakan isi parameter air berdasarkan pengujian:")

# Membagi input ke dalam beberapa kategori
st.sidebar.subheader("Polutan Biologis & Umum")
ammonia = st.sidebar.number_input("Ammonia", min_value=0.0, max_value=30.0, value=9.08, step=0.1)
bacteria = st.sidebar.number_input("Bacteria", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
viruses = st.sidebar.number_input("Viruses", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
chloramine = st.sidebar.number_input("Chloramine", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

st.sidebar.subheader("Logam Berat Berbahaya")
aluminium = st.sidebar.number_input("Aluminium", min_value=0.0, max_value=10.0, value=1.65, step=0.01)
arsenic = st.sidebar.number_input("Arsenic", min_value=0.0, max_value=2.0, value=0.04, step=0.01)
barium = st.sidebar.number_input("Barium", min_value=0.0, max_value=5.0, value=2.85, step=0.01)
cadmium = st.sidebar.number_input("Cadmium", min_value=0.0, max_value=0.5, value=0.007, step=0.001)
chromium = st.sidebar.number_input("Chromium", min_value=0.0, max_value=1.0, value=0.83, step=0.01)
copper = st.sidebar.number_input("Copper", min_value=0.0, max_value=3.0, value=0.17, step=0.01)
lead = st.sidebar.number_input("Lead", min_value=0.0, max_value=0.5, value=0.054, step=0.001)
mercury = st.sidebar.number_input("Mercury", min_value=0.0, max_value=0.05, value=0.007, step=0.001)

st.sidebar.subheader("Zat Kimia Lainnya")
flouride = st.sidebar.number_input("Flouride", min_value=0.0, max_value=5.0, value=0.05, step=0.01)
nitrates = st.sidebar.number_input("Nitrates", min_value=0.0, max_value=30.0, value=16.08, step=0.1)
nitrites = st.sidebar.number_input("Nitrites", min_value=0.0, max_value=5.0, value=1.13, step=0.01)
perchlorate = st.sidebar.number_input("Perchlorate", min_value=0.0, max_value=60.0, value=37.75, step=0.1)
radium = st.sidebar.number_input("Radium", min_value=0.0, max_value=10.0, value=6.78, step=0.1)
selenium = st.sidebar.number_input("Selenium", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
silver = st.sidebar.number_input("Silver", min_value=0.0, max_value=1.0, value=0.34, step=0.01)
uranium = st.sidebar.number_input("Uranium", min_value=0.0, max_value=0.1, value=0.02, step=0.01)

# Tombol Prediksi
predict_button = st.sidebar.button("🔍 Cek Kualitas Air")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Interpretasi Parameter")
    st.info("""
    Aplikasi ini menggunakan model **Random Forest Classifier** yang telah dilatih pada ribuan sampel air.
    Masukkan data pada panel di sebelah kiri dan klik **'Cek Kualitas Air'** untuk memprediksi apakah air tersebut memenuhi standar konsumsi atau berpotensi meracuni.
    """)
    
    if predict_button:
        # Mengumpulkan input dalam format yang sama dengan data training
        input_data = pd.DataFrame([{
            'aluminium': aluminium, 'ammonia': ammonia, 'arsenic': arsenic, 
            'barium': barium, 'cadmium': cadmium, 'chloramine': chloramine, 
            'chromium': chromium, 'copper': copper, 'flouride': flouride, 
            'bacteria': bacteria, 'viruses': viruses, 'lead': lead, 
            'nitrates': nitrates, 'nitrites': nitrites, 'mercury': mercury, 
            'perchlorate': perchlorate, 'radium': radium, 'selenium': selenium, 
            'silver': silver, 'uranium': uranium
        }])
        
        # Scaling Input
        scaled_input = scaler.transform(input_data)
        
        # Prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")
        
        if prediction == 1:
            st.success("✅ **STATUS: AIR LAYAK MINUM (AMAN)**")
            st.markdown("Berdasarkan parameter yang dimasukkan, kualitas air ini diprediksi **AMAN** dari racun biologis dan logam berat. Dapat didistribusikan untuk konsumsi masyarakat.")
            st.metric("Tingkat Keyakinan Model", f"{prediction_proba[1]*100:.1f}%")
        else:
            st.error("🚨 **STATUS: BAHAYA! AIR TIDAK LAYAK (BERACUN)**")
            st.markdown("Kandungan polutan atau logam berat dalam sampel ini berpotensi membahayakan kesehatan manusia. Diperlukan tahap penyaringan/purifikasi lebih lanjut.")
            st.metric("Tingkat Kepastian Bahaya", f"{prediction_proba[0]*100:.1f}%")

with col2:
    st.image("https://images.unsplash.com/photo-1548842100-34989069d254?q=80&w=1543&auto=format&fit=crop", caption="Water Quality Check")
