# Project Clustering Pertama: Segmentasi Customer
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# 2. Pillh 2 fitur untuk visualisasi
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Cari optimal k dengan Elbow Method
WCSS = [] # Within-Cluster Sum of Square
for i in range(1, 11):
    kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto')
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

# Plot Elbow
plt.plot(range(1, 11), WCSS)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 4. Apply K-Means dengan ke5 (biasanya optimal untuk dataset ini)
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X)

# 5. Visualisasi hasil clustering
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# 6. Analisis cluster (versi eksplisit: hanya kolom numerik)
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
cluster_analysis = df.groupby('Cluster')[numeric_cols].mean()

print("\nAnalisis per Cluster:")
print(cluster_analysis)