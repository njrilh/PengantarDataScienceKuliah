import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate data pasien (contoh)
df = pd.read_csv("titanic.csv")
print("Dataset Titanic berhasil dimuat.")

print(f'data missing {df.isnull().sum()}')

# Ubah sex menjadi angka
df['sex'] = df['sex'].map({'male': 1, 'female': 2})
print(df['sex'].unique())

# Bagi data menjadi fitur (X) dan target (y)
X = df[['pclass', 'fare']]  # eksplisit & aman
y = df['survived']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skala fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bangun model k-Nearest Neighbors
k = 5  # Jumlah tetangga terdekat
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test_scaled)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
print("\nMatriks Konfusi:")
print(confusion_matrix(y_test, y_pred))

# # Contoh penggunaan model untuk prediksi pasien baru
# new_patient = pd.DataFrame({
#     'usia': [50],
#     'tekanan_darah': [130],
#     'kolesterol': [200]
# })

# # Pastikan urutan kolom sama dengan X_train
# new_patient = new_patient[X.columns]  # amankan urutan

# # Skala data baru menggunakan scaler yang sudah dilatih
# new_patient_scaled = scaler.transform(new_patient)

# # Prediksi
# prediction = model.predict(new_patient_scaled)

# if prediction[0] == 1:
#     print("\nPasien baru memiliki risiko penyakit jantung.")
# else:
#     print("\nPasien baru tidak memiliki risiko penyakit jantung.")