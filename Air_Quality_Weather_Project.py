# %% [markdown]
# # Sistem Prediksi Kualitas Udara Berdasarkan Kondisi Cuaca
# Proyek Data Science menggunakan Machine Learning untuk Klasifikasi Bahaya Polusi Udara.
# Sumber data: `global_air_quality_dataset.csv` & `worldwide_weather_2025.csv`

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# %% [markdown]
# ## 2. Data Collection
# Memuat data dari DUA dataset berbeda (Kualitas Udara 2024 dan Cuaca 2025).

# %%
print("==== 2. DATA COLLECTION ====")
df_air = pd.read_csv('data/raw/global_air_quality_dataset.csv')
df_weather = pd.read_csv('data/raw/worldwide_weather_2025.csv')

print(f"Dimensi Dataset Air Quality: {df_air.shape[0]} baris.")
print(f"Dimensi Dataset Weather: {df_weather.shape[0]} baris.")

# %% [markdown]
# ## 3. Data Integration & Data Preparation
# Melakukan Time-Shifting (Penyelarasan Waktu) dan Inner Join.

# %%
print("\n==== 3. DATA PREPARATION & INTEGRATION ====")

# 3.1 Konversi ke tipe Datetime
df_air['Date'] = pd.to_datetime(df_air['Date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

# 3.2 Time-Shifting Alignment (Menyelaraskan tahun 2024 ke 2025)
# Asumsi: Siklus cuaca tahunan (bulan dan hari) memiliki pola berulang yang identik
df_air['Date'] = df_air['Date'] + pd.DateOffset(years=1)

# 3.3 INNER JOIN (Penggabungan berdasarkan Kota dan Tanggal)
df_merged = pd.merge(df_air, df_weather, left_on=['Date', 'City'], right_on=['date', 'city'], how='inner')
print(f"Berhasil menggabungkan {df_merged.shape[0]} observasi hari yang sama di kota yang saling beririsan.")

# 3.4 Pembentukan Variabel Target (AQI > 100 = Berbahaya)
df_merged['is_hazardous'] = (df_merged['AQI'] > 100).astype(int)
print("Distribusi Target (0: Aman, 1: Berbahaya):")
print(df_merged['is_hazardous'].value_counts())

# 3.5 Pemilihan Fitur Cuaca (Feature Selection)
features = ['temperature_2m_max', 'temperature_2m_mean', 'apparent_temperature_max', 
            'precipitation_sum', 'windspeed_10m_max', 'cloudcover_mean']
X = df_merged[features]
y = df_merged['is_hazardous']

# Penanganan missing values jika ada pada data cuaca
X = X.fillna(X.median())

# %% [markdown]
# ## 4. Data Splitting & Handling Imbalance

# %%
print("\n==== 4. SPLITTING & SCALING ====")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE (Menyeimbangkan kelas target "Berbahaya")
print("Menerapkan SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'results/weather_scaler.pkl')
print("Scaler cuaca disimpan sebagai 'results/weather_scaler.pkl'.")

# %% [markdown]
# ## 5. Modelling (XGBoost)

# %%
print("\n==== 5. MODELLING ====")
print("Melatih model XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train_smote)

joblib.dump(xgb_model, 'results/xgb_weather_model.pkl')
print("Model diekspor ke 'results/xgb_weather_model.pkl'.")

# %% [markdown]
# ## 6. Evaluation

# %%
print("\n==== 6. EVALUATION ====")
y_pred = xgb_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Aman (0)', 'Berbahaya (1)'], yticklabels=['Aman (0)', 'Berbahaya (1)'])
plt.title('Confusion Matrix - XGBoost (Air Quality)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('results/confusion_matrix_weather_xgb.png')
plt.close()

# Visualisasi Feature Importance
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Weather Impact on AQI)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('results/feature_importances_weather_xgb.png')
plt.close()

print("Eksekusi Selesai. Hasil plot telah disimpan di direktori results/.")
