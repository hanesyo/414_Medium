import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(42)
categories = ['FPS', 'MOBA', 'RPG', 'IRL', 'Strategy']

fps_fans = np.random.normal(loc=[150, 15, 10, 8, 5], scale=15, size=(30, 5))
moba_fans = np.random.normal(loc=[12, 160, 15, 5, 10], scale=12, size=(30, 5))
irl_fans = np.random.normal(loc=[20, 20, 15, 140, 10], scale=18, size=(30, 5))
strategy_fans = np.random.normal(loc=[8, 10, 12, 5, 145], scale=12, size=(30, 5))
variety_fans = np.random.normal(loc=[50, 50, 40, 45, 35], scale=20, size=(30, 5))

data = np.vstack([fps_fans, moba_fans, irl_fans, strategy_fans, variety_fans])
data = np.maximum(data, 0)
df = pd.DataFrame(data, columns=categories)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df)

print("CLUSTER PROFILES (Average Minutes per Week)")
print("="*60)
cluster_profiles = df.groupby('Cluster')[categories].mean()
print(cluster_profiles.round(1))
print()

print("SILHOUETTE SCORES BY K")
print("="*60)
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(df[categories])
    score = silhouette_score(df[categories], labels)
    print(f"K={k:2d}  |  Silhouette Score = {score:.4f}")
