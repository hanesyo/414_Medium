import pandas as pd
import networkx as nx
from collections import Counter

df = pd.read_csv('edges.csv')  

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Unique heroes: {df['hero'].nunique()}")
print(f"Unique comics: {df['comic'].nunique()}")
print(f"\nFirst 10 rows:")
print(df.head(10))

print("\n" + "=" * 60)
print("BUILDING HERO NETWORK")
print("=" * 60)

comics_grouped = df.groupby('comic')['hero'].apply(list).reset_index()

edges = []
edge_weights = Counter()

for idx, row in comics_grouped.iterrows():
    heroes = row['hero']
    for i in range(len(heroes)):
        for j in range(i+1, len(heroes)):
            hero1, hero2 = sorted([heroes[i], heroes[j]])  
            edge_weights[(hero1, hero2)] += 1

edges = [(h1, h2, w) for (h1, h2), w in edge_weights.items()]

print(f"Total hero-hero edges: {len(edges)}")
print(f"Total appearances (sum of weights): {sum(w for _, _, w in edges)}")

G = nx.Graph()
G.add_weighted_edges_from(edges)

print(f"Nodes in graph: {G.number_of_nodes()}")
print(f"Edges in graph: {G.number_of_edges()}")

print("\n" + "=" * 60)
print("COMPUTING CENTRALITY MEASURES")
print("=" * 60)

degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G, weight='weight')

print("\nTOP 10 BY DEGREE CENTRALITY:")
top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (hero, score) in enumerate(top_degree, 1):
    print(f"{i}. {hero}: {score:.4f}")

print("\nTOP 10 BY BETWEENNESS CENTRALITY:")
top_between = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (hero, score) in enumerate(top_between, 1):
    print(f"{i}. {hero}: {score:.4f}")

print("\nTOP 10 BY PAGERANK:")
top_page = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (hero, score) in enumerate(top_page, 1):
    print(f"{i}. {hero}: {score:.4f}")

print("\n" + "=" * 60)
print("ADDITIONAL STATS")
print("=" * 60)
degrees = dict(G.degree())
top_connections = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTOP 10 BY RAW NUMBER OF CONNECTIONS:")
for i, (hero, deg) in enumerate(top_connections, 1):
    print(f"{i}. {hero}: {deg} connections")

results_df = pd.DataFrame({
    'hero': list(degree_cent.keys()),
    'degree_centrality': list(degree_cent.values()),
    'betweenness_centrality': [betweenness_cent[h] for h in degree_cent.keys()],
    'pagerank': [pagerank[h] for h in degree_cent.keys()],
    'num_connections': [degrees[h] for h in degree_cent.keys()]
})
results_df = results_df.sort_values('degree_centrality', ascending=False)
results_df.to_csv('marvel_centrality_results.csv', index=False)
print("\n✓ Saved full results to 'marvel_centrality_results.csv'")
