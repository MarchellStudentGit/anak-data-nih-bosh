import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Prediksi Kualitas Air (Composite)",
    page_icon="💧",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #616161;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('results/rf_model.pkl')
        scaler = joblib.load('results/scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_model_and_scaler()

st.markdown('<div class="main-header">💧 Sistem Prediksi Kelayakan & Keamanan Air (Composite)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mendeteksi kelayakan uji fisik dan potensi bahaya logam berat secara bersamaan</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model belum tersedia! Harap jalankan kode Python untuk melatih model gabungan terlebih dahulu.")
    st.stop()

st.info("Silakan masukkan parameter hasil laboratorium pada kedua tab di bawah ini, lalu klik tombol **'Cek Status Kualitas Air'** di bagian paling bawah.")

# Membuat sistem Tabs untuk input
tab1, tab2 = st.tabs(["🧪 Parameter Fisik (Dataset 1)", "☣️ Parameter Logam Berat & Biologis (Dataset 2)"])

with tab1:
    st.markdown("### 1. Karakteristik Fisik & Kimia Dasar Air")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        ph = st.number_input("pH (Keasaman)", value=7.0, step=0.1)
        hardness = st.number_input("Hardness (Kekerasan)", value=196.0, step=1.0)
        solids = st.number_input("Solids (TDS)", value=22014.0, step=100.0)
    with col_p2:
        chloramines_phys = st.number_input("Chloramines (Fisik)", value=7.1, step=0.1)
        sulfate = st.number_input("Sulfate", value=333.0, step=1.0)
        conductivity = st.number_input("Conductivity", value=426.0, step=1.0)
    with col_p3:
        organic_carbon = st.number_input("Organic Carbon", value=14.2, step=0.1)
        trihalomethanes = st.number_input("Trihalomethanes", value=66.3, step=0.1)
        turbidity = st.number_input("Turbidity (Kekeruhan)", value=3.9, step=0.1)

with tab2:
    st.markdown("### 2. Kadar Logam Berat, Patogen, & Polutan")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        aluminium = st.number_input("Aluminium", value=1.65, step=0.01)
        ammonia = st.number_input("Ammonia", value=9.08, step=0.1)
        arsenic = st.number_input("Arsenic", value=0.04, step=0.01)
        barium = st.number_input("Barium", value=2.85, step=0.01)
        cadmium = st.number_input("Cadmium", value=0.007, step=0.001)
    with col_c2:
        chloramine = st.number_input("Chloramine (Kimia)", value=2.0, step=0.1)
        chromium = st.number_input("Chromium", value=0.83, step=0.01)
        copper = st.number_input("Copper", value=0.17, step=0.01)
        flouride = st.number_input("Flouride", value=0.05, step=0.01)
        bacteria = st.number_input("Bacteria", value=0.2, step=0.01)
    with col_c3:
        viruses = st.number_input("Viruses", value=0.0, step=0.01)
        lead = st.number_input("Lead (Timbal)", value=0.054, step=0.001)
        nitrates = st.number_input("Nitrates", value=16.08, step=0.1)
        nitrites = st.number_input("Nitrites", value=1.13, step=0.01)
        mercury = st.number_input("Mercury", value=0.007, step=0.001)
    with col_c4:
        perchlorate = st.number_input("Perchlorate", value=37.75, step=0.1)
        radium = st.number_input("Radium", value=6.78, step=0.1)
        selenium = st.number_input("Selenium", value=0.08, step=0.01)
        silver = st.number_input("Silver", value=0.34, step=0.01)
        uranium = st.number_input("Uranium", value=0.02, step=0.01)

st.markdown("---")
predict_button = st.button("🔍 CEK STATUS KUALITAS AIR", use_container_width=True)

if predict_button:
    # Urutan fitur harus persis sama dengan DataFrame saat training (df_phys lalu df_chem)
    # Order: ph, Hardness, Solids, Chloramines_phys, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    # aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, flouride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, uranium
    
    input_data = pd.DataFrame([{
        'ph': ph, 'Hardness': hardness, 'Solids': solids, 'Chloramines_phys': chloramines_phys,
        'Sulfate': sulfate, 'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity,
        'aluminium': aluminium, 'ammonia': ammonia, 'arsenic': arsenic, 
        'barium': barium, 'cadmium': cadmium, 'chloramine': chloramine, 
        'chromium': chromium, 'copper': copper, 'flouride': flouride, 
        'bacteria': bacteria, 'viruses': viruses, 'lead': lead, 
        'nitrates': nitrates, 'nitrites': nitrites, 'mercury': mercury, 
        'perchlorate': perchlorate, 'radium': radium, 'selenium': selenium, 
        'silver': silver, 'uranium': uranium
    }])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]
    
    if prediction == 1:
        st.success("✅ **STATUS: SANGAT AMAN & LAYAK KONSUMSI**")
        st.markdown("Berdasarkan uji komposit (Fisik dan Kimiawi), air ini diprediksi aman dan memenuhi seluruh standar potabilitas maupun ambang batas logam berat.")
        st.metric("Tingkat Keyakinan (Confidence)", f"{prediction_proba[1]*100:.1f}%")
    else:
        st.error("🚨 **STATUS: BERBAHAYA / TIDAK LAYAK**")
        st.markdown("Air gagal memenuhi satu atau kedua kriteria (uji kelayakan fisik ATAU bebas polutan/logam berat). Dilarang untuk dikonsumsi secara langsung.")
        st.metric("Tingkat Kepastian Bahaya", f"{prediction_proba[0]*100:.1f}%")
