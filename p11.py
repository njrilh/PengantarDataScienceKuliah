import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
advertising_spend = np.random.randint(1000, 5000, size=20)
sales = 2 * advertising_spend + np.random.normal(0, 500, size=20) 

df = pd.DataFrame({'pengeluaran_iklan': advertising_spend, 'penjualan': sales})

plt.figure(figsize=(10, 6))
plt.scatter(df['pengeluaran_iklan'], df['penjualan'])
plt.xlabel('Pengeluaran Iklan')
plt.ylabel('Penjualan')
plt.title('Hubungan antara Pengeluaran Iklan dan Penjualan')
plt.grid(True)
plt.show()

X = df[['pengeluaran_iklan']]
y = df['penjualan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r_squared = model.score(X_test, y_test)
print("R-squared:", r_squared)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Data Uji')
plt.plot(X_test, y_pred, color='red', label='Prediksi')
plt.xlabel('Pengeluaran Iklan')
plt.ylabel('Penjualan')
plt.title('Prediksi Penjualan berdasarkan Pengeluaran Iklan')
plt.legend()
plt.grid(True)
plt.show()

new_advertising_spend = [[3000]]
predicted_sales = model.predict(new_advertising_spend)
print("Prediksi penjualan untuk pengeluaran iklan sebesar 3000:", predicted_sales[0])
