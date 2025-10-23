import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pyvis.network as net
from collections import Counter

df = pd.read_csv('marvel_centrality_results.csv')

edges_df = pd.read_csv('edges.csv')  
comics_grouped = edges_df.groupby('comic')['hero'].apply(list)

edge_weights = Counter()
for heroes in comics_grouped:
    for i in range(len(heroes)):
        for j in range(i+1, len(heroes)):
            hero1, hero2 = sorted([heroes[i], heroes[j]])
            edge_weights[(hero1, hero2)] += 1

G = nx.Graph()
G.add_weighted_edges_from([
    (h1, h2, w) for (h1, h2), w in edge_weights.items()
])

fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

top_25_heroes = set(df.nlargest(25, 'degree_centrality')['hero'].values)
G_subgraph = G.subgraph(top_25_heroes).copy()

pos = nx.spring_layout(G_subgraph, k=0.5, iterations=50, seed=42)

nx.draw_networkx_edges(
    G_subgraph, pos,
    ax=ax,
    alpha=0.15,
    width=0.5,
    edge_color='#888888'
)

node_sizes = []
node_colors = []
top_10 = set(df.nlargest(10, 'degree_centrality')['hero'].values)

for node in G_subgraph.nodes():
    centrality = df[df['hero'] == node]['degree_centrality'].values[0]
    node_sizes.append(centrality * 3000)
    
    if node in top_10:
        node_colors.append('#E63946')  
    else:
        node_colors.append('#457B9D')  

nx.draw_networkx_nodes(
    G_subgraph, pos,
    node_size=node_sizes,
    node_color=node_colors,
    ax=ax,
    alpha=0.8,
    edgecolors='white',
    linewidths=1.5
)

top_20 = df.nlargest(20, 'degree_centrality')['hero'].values
labels = {node: node if node in top_20 else '' for node in G_subgraph.nodes()}

nx.draw_networkx_labels(
    G_subgraph, pos,
    labels=labels,
    ax=ax,
    font_size=8,
    font_weight='bold',
    font_color='black'
)

ax.set_title('Marvel Universe Hero Network\n(Top 25 heroes by connections)', fontsize=18, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig('marvel_top_25.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved static network image to: marvel_network_static.png")
plt.close()
