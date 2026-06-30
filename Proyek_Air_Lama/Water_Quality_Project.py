# %% [markdown]
# # Sistem Prediksi Kelayakan dan Kualitas Air Minum (Composite 2 Dataset)
# Proyek Data Science menggunakan Machine Learning untuk Klasifikasi Kualitas Air.
# Sumber data: `water_potability.csv` (Dataset Fisikokimia) & `waterQuality1.csv` (Dataset Logam Berat)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# %% [markdown]
# ## 2. Data Collection
# Memuat data dari DUA dataset yang berbeda.

# %%
print("==== 2. DATA COLLECTION ====")
df_phys = pd.read_csv('data/raw/new1/water_potability.csv')
df_chem = pd.read_csv('data/raw/new2/waterQuality1.csv')

print(f"Dimensi Dataset 1 (Fisik): {df_phys.shape[0]} baris, {df_phys.shape[1]} kolom.")
print(f"Dimensi Dataset 2 (Kimia/Biologis): {df_chem.shape[0]} baris, {df_chem.shape[1]} kolom.")

# %% [markdown]
# ## 3. Data Integration & Data Preparation
# Melakukan Synthetic Horizontal Concatenation karena kedua dataset tidak memiliki ID unik.

# %%
print("\n==== 3. DATA PREPARATION & INTEGRATION ====")

# 3.1 Cleaning Dataset 2 (Kimia)
df_chem['ammonia'] = pd.to_numeric(df_chem['ammonia'], errors='coerce')
df_chem['is_safe'] = pd.to_numeric(df_chem['is_safe'], errors='coerce')
df_chem = df_chem.dropna().reset_index(drop=True)

# 3.2 Menyamakan jumlah baris dengan Dataset 1
# Mengambil 3.276 baris acak dari Dataset 2 agar jumlah observasi sama
min_len = min(len(df_phys), len(df_chem))
df_chem_sampled = df_chem.sample(n=min_len, random_state=42).reset_index(drop=True)
df_phys_sampled = df_phys.iloc[:min_len].reset_index(drop=True)

# 3.3 Penanganan Missing Values di Dataset 1
# Mengisi NaN dengan median kolom
for col in df_phys_sampled.columns:
    if df_phys_sampled[col].isnull().sum() > 0:
        df_phys_sampled[col] = df_phys_sampled[col].fillna(df_phys_sampled[col].median())

# Ganti nama agar tidak tabrakan (Keduanya punya semacam kloramin)
df_phys_sampled = df_phys_sampled.rename(columns={'Chloramines': 'Chloramines_phys'})

# 3.4 PENGGABUNGAN (CONCATENATION)
df_combined = pd.concat([df_phys_sampled, df_chem_sampled], axis=1)

# Membuat Variabel Target Komposit: 
# Air aman HANYA JIKA lolos uji kelayakan (Potability == 1) DAN bebas polutan (is_safe == 1)
df_combined['Final_Safety_Status'] = ((df_combined['Potability'] == 1) & (df_combined['is_safe'] == 1)).astype(int)

# Hapus target asli agar model tidak "bocor" (Data Leakage)
df_combined = df_combined.drop(['Potability', 'is_safe'], axis=1)

print(f"Dimensi Dataset Gabungan (Composite): {df_combined.shape[0]} baris, {df_combined.shape[1]} kolom.")
print("Distribusi Kelas Final_Safety_Status (0: Bahaya, 1: Sangat Aman):")
print(df_combined['Final_Safety_Status'].value_counts())

# %% [markdown]
# ## 4. Data Splitting & Handling Imbalance

# %%
print("\n==== 4. SPLITTING & SCALING ====")
X = df_combined.drop('Final_Safety_Status', axis=1)
y = df_combined['Final_Safety_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE (Menyeimbangkan kelas target)
print("Menerapkan SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'results/scaler.pkl')

# %% [markdown]
# ## 5. Modelling

# %%
print("\n==== 5. MODELLING ====")
print("Melatih model Random Forest Composite...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)

joblib.dump(rf_model, 'results/rf_model.pkl')
print("Model diekspor ke 'results/rf_model.pkl'.")

# %% [markdown]
# ## 6. Evaluation

# %%
print("\n==== 6. EVALUATION ====")
y_pred_rf = rf_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bahaya (0)', 'Sangat Aman (1)'], yticklabels=['Bahaya (0)', 'Sangat Aman (1)'])
plt.title('Confusion Matrix - Composite Random Forest')
plt.savefig('results/confusion_matrix_rf.png')
plt.close()

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Composite Dataset)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('results/feature_importances_rf.png')
plt.close()

print("Eksekusi Selesai. Hasil visualisasi disimpan di folder results/.")
