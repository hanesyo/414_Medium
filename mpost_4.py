import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Using simulated Twitch viewer data (150 users, 5 categories)\n")

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

print(f"Dataset: {len(df)} users across {len(categories)} content categories\n")

print("Testing cluster configurations (K=2 to K=10)...")
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)
    sil_score = silhouette_score(df, labels)
    silhouette_scores.append(sil_score)
    print(f"K={k:2d}: Silhouette Score = {sil_score:.4f}")

optimal_k = k_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
print(f"\nOptimal: K={optimal_k} (Score: {best_score:.4f})\n")

print(f"Clustering with K={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df)
print("Done.\n")

print("Cluster Profiles:")
cluster_profiles = df.groupby('Cluster')[categories].mean()
print(cluster_profiles.round(1))
print(f"\nDistribution: {dict(df['Cluster'].value_counts().sort_index())}\n")

print("Viewer Personas:\n")
for cluster_id in range(optimal_k):
    profile = cluster_profiles.loc[cluster_id]
    top_category = profile.idxmax()
    top_value = profile.max()
    size = (df['Cluster'] == cluster_id).sum()
    examples = list(df[df['Cluster'] == cluster_id].head(2).index)
    
    print(f"Cluster {cluster_id}: {top_category} Enthusiasts")
    print(f"  {size} users | {top_value:.0f} min/week in {top_category}")
    print(f"  Examples: {examples[0]}, {examples[1]}\n")

print(f"Silhouette Score: {silhouette_score(df[categories], df['Cluster']):.4f}\n")

df.to_csv('twitch_clusters.csv', index=False)
print("Saved: twitch_clusters.csv")
