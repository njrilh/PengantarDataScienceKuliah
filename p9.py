import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('titanic.csv')
print(f"1. rata-rata age: {df['age'].mean():.2f} tahun")
print(f"rata-rata fare: {df['fare'].mean():.2f}")
print(f"rata-rata survived: {df['survived'].mean():.2f} ({df['survived'].mean()*100:.1f}% selamat)")
print(f"rata-rata pclass: {df['pclass'].mean():.2f}")

print(f"\n2. age - maks: {df['age'].max():.1f} tahun, min: {df['age'].min():.1f} tahun")
print(f"fare - maks: {df['fare'].max():.2f}, min: {df['fare'].min():.2f}")

threshold = 3
z_scores = np.abs(stats.zscore(df['fare']))
outliers = df[z_scores > threshold]
print(f"\n3. threshold Z-Score: {threshold}")
print(f"jumlah outlier: {len(outliers)}")
print(f"outliers:\n{outliers}\n")

plt.figure(figsize=(8, 5))
plt.boxplot(df['fare'], vert=False)
plt.title('boxplot fare dengan outliers - Dataset Titanic')
plt.xlabel('fare')
plt.show()
