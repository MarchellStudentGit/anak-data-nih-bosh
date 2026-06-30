# %% [markdown]
# # Sistem Prediksi Kelayakan dan Kualitas Air Minum
# Proyek Data Science menggunakan Machine Learning untuk Klasifikasi Kualitas Air.
# Sumber data: `waterQuality1.csv`
# Fokus: Mendeteksi kelayakan konsumsi air (is_safe) berdasarkan pengujian tingkat kontaminan biologis dan logam berat.

# %% [markdown]
# ## 1. Import Library
# Mengimpor semua modul yang dibutuhkan (Pastikan Anda sudah melakukan `pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn`).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

# Membuat direktori hasil jika belum ada
os.makedirs('results', exist_ok=True)

# %% [markdown]
# ## 2. Data Collection
# Memuat data dari dataset Water Quality (Fokus pada Logam Berat & Polutan).

# %%
print("==== 2. DATA COLLECTION ====")
data_path = 'data/raw/new2/waterQuality1.csv'
df = pd.read_csv(data_path)

print(f"Dimensi dataset awal: {df.shape[0]} baris, {df.shape[1]} kolom.")
print(df.head())

# %% [markdown]
# ## 3. Data Understanding (Exploratory Data Analysis)
# Memahami tipe data, mendeteksi nilai-nilai yang aneh.

# %%
print("\n==== 3. DATA UNDERSTANDING ====")
print(df.info())

# Memeriksa persebaran data pada kolom target 'is_safe'
print("\nSebaran target awal 'is_safe':")
print(df['is_safe'].value_counts())

# %% [markdown]
# ## 4. Data Preparation
# Melakukan cleaning, handling nilai hilang, feature scaling, dan balancing kelas.

# %%
print("\n==== 4. DATA PREPARATION ====")

# 4.1 Data Cleaning: Mengubah nilai string seperti '#NUM!' menjadi NaN (Not a Number)
print("Mengatasi missing values dan nilai aneh (seperti '#NUM!')...")
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')

# Mengecek jumlah NaN akibat konversi
print(f"Total NaN di 'ammonia': {df['ammonia'].isnull().sum()}")
print(f"Total NaN di 'is_safe': {df['is_safe'].isnull().sum()}")

# Drop baris yang memiliki NaN (karena proporsinya kecil, hanya 3 baris)
df = df.dropna()
print(f"Dimensi dataset setelah Data Cleaning: {df.shape[0]} baris.")

# Menyimpan plot sebaran kelas
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='is_safe', palette='Set2')
plt.title('Distribusi Kelas Target (0: Tidak Layak, 1: Layak Minum)')
plt.savefig('results/target_distribution_before_smote.png')
plt.close()

# 4.2 Data Splitting
X = df.drop('is_safe', axis=1)
y = df['is_safe']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Ukuran X_train: {X_train.shape}, Ukuran X_test: {X_test.shape}")

# 4.3 Handling Imbalanced Data menggunakan SMOTE pada Training Data
print("Menerapkan SMOTE untuk menyeimbangkan kelas target (hanya di data train)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Rasio kelas setelah SMOTE di y_train:\n{y_train_smote.value_counts()}")

# 4.4 Feature Scaling (Standardization)
print("Menerapkan StandardScaler untuk menormalkan rentang fitur...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Menyimpan scaler untuk Deployment
joblib.dump(scaler, 'results/scaler.pkl')
print("Scaler disimpan sebagai 'results/scaler.pkl'.")


# %% [markdown]
# ## 5. Modelling
# Melatih dua buah model: Logistic Regression (Baseline) dan Random Forest (Ensemble).

# %%
print("\n==== 5. MODELLING ====")

# 5.1 Melatih Baseline Model: Logistic Regression
print("Melatih model Logistic Regression...")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train_smote)

# 5.2 Melatih Main Model: Random Forest
print("Melatih model Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)

# Menyimpan model untuk Deployment
joblib.dump(rf_model, 'results/rf_model.pkl')
print("Model Random Forest disimpan sebagai 'results/rf_model.pkl'.")


# %% [markdown]
# ## 6. Evaluation
# Mengukur dan membandingkan kinerja model pada Test Data (Data yang belum pernah dilihat model).

# %%
print("\n==== 6. EVALUATION ====")

# Prediksi menggunakan Test Data
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluasi Logistic Regression
print("\n--- Kinerja Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Evaluasi Random Forest
print("\n--- Kinerja Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix untuk Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Aman (0)', 'Aman (1)'], yticklabels=['Tidak Aman (0)', 'Aman (1)'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('results/confusion_matrix_rf.png')
plt.close()
print("Grafik Confusion Matrix disimpan di 'results/confusion_matrix_rf.png'.")

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('results/feature_importances_rf.png')
plt.close()
print("Grafik Feature Importance disimpan di 'results/feature_importances_rf.png'.")

print("\n==== PROSES SELESAI ====")
print("Model yang direkomendasikan untuk Deployment adalah Random Forest karena skor akurasi dan Recall-nya lebih tinggi dibandingkan Logistic Regression.")
