import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate data pasien (contoh)
df = pd.read_csv("digital_habits_vs_mental_health.csv")
print("Dataset berhasil dimuat.")

x = df[['screen_time_hours','hours_on_TikTok','sleep_hours','stress_level']]
y = df['mood_score']

for col in x:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], y, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('Mood Score')
    plt.title(f'{col} vs Mood Score')
    plt.grid(True)
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

r_squared = model.score(X_test, y_test)
print("R-squared:", r_squared)

y_pred = model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Mood Score Aktual')
plt.ylabel('Mood Score Prediksi')
plt.title('Perbandingan Nilai Aktual dan Prediksi Mood Score')
plt.grid(True)
plt.show()

new_data = pd.DataFrame({
    'screen_time_hours': [13],
    'hours_on_TikTok': [10],
    'sleep_hours': [1],
    'stress_level': [10]
})

prediction = model.predict(new_data)
print("Prediksi Mood Score:", prediction[0])

print('Mood Jelek' if prediction < 5 else 
      'Mood Oke' if prediction < 8 else 
      'Mood Baik')