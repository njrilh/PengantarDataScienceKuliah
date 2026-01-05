import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate data acak
np.random.seed(42)
pengalaman = np.random.randint(0, 20, 100)
gaji = 3000 + 2500 * pengalaman + np.random.normal(0, 5000, 100)  # gaji = 3jt + 2.5jt*p

df = pd.DataFrame({'pengalaman': pengalaman, 'gaji': gaji})

# Siapkan data
X = df[['pengalaman']]
y = df['gaji']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# data baru diinput
lama_kerja = int(input("masukkan lama bekerja: "))
pengalaman_baru = pd.DataFrame({'pengalaman': [lama_kerja]})
prediksi_gaji_baru = model.predict(pengalaman_baru)

print()
print(f"Prediksi pengalaman {lama_kerja} tahun maka gaji = {prediksi_gaji_baru[0]:.0f}")

# Visualisasi
plt.scatter(X_test, y_test, color='blue', label='Data Asli')
plt.plot(X_test, y_pred, color='red', label='Garis Regresi')
plt.xlabel('Pengalaman (tahun)')
plt.ylabel('Gaji (ribu rupiah)')
plt.legend()
plt.title('Regresi Linear Sederhana')
plt.show()