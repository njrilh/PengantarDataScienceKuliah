import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate data pasien (contoh)
df = pd.read_csv("loan_data.csv")
print("Dataset berhasil dimuat.")

# print(f'data missing {df.isnull().sum()}')
df = df.dropna()
df = df.drop_duplicates()

# ubah ke int
df['Employment_Status'] = df['Employment_Status'].map({'unemployed': 0, 'employed': 1 })
# print(df['Employment_Status'].unique())
# ubah ke int
df['Approval'] = df['Approval'].map({'Rejected': 0, 'Approved': 1})
# print(df['Approval'].unique())

# Bagi data menjadi fitur (X) dan target (y)
X = df[['Income','Credit_Score','Loan_Amount','DTI_Ratio','Employment_Status']]  # eksplisit & aman
y = df['Approval']

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

# Contoh penggunaan model untuk prediksi pasien baru
new = pd.DataFrame({
    'Income' : [26556],
    'Credit_Score' : [389],
    'Loan_Amount' : [34118],
    'DTI_Ratio' : [10.22],
    'Employment_Status' : [1]
})

# Pastikan urutan kolom sama dengan X_train
new = new[X.columns]  # amankan urutan

# Skala data baru menggunakan scaler yang sudah dilatih
new_scaled = scaler.transform(new)

# Prediksi
prediction = model.predict(new_scaled)

print(f'Hasil Approved : {prediction}')
