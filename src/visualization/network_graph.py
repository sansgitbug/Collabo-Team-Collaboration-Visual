# src/visualization/network_graph.py

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
METRICS_DIR = os.path.join(PROCESSED_DIR, "metrics")
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
metrics = pd.read_csv(os.path.join(METRICS_DIR, "node_metrics.csv"))
edges = pd.read_csv(os.path.join(METRICS_DIR, "edge_metrics.csv"))
members = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_members.csv"))

# Build graph
G = nx.DiGraph()

# Add nodes with attributes
for _, m in members.iterrows():
    G.add_node(m["member_id"], role=m["role"])

# Add ALL edges (no filtering)
for _, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# -------------------------
# VISUAL TWEAKS FOR READABILITY
# -------------------------

# Node size (reduced)
node_sizes = [
    300 + (metrics.loc[metrics.member_id == n, "activity_score"].values[0] * 4)
    for n in G.nodes()
]

# Node color by role
role_colors = {
    "leader": "#ff6b6b",
    "active": "#feca57",
    "regular": "#48dbfb",
    "passive": "#1dd1a1",
    "isolated": "#c8d6e5",
}

node_colors = [
    role_colors.get(G.nodes[n]["role"], "#8395a7")
    for n in G.nodes()
]

# Edge thickness (very thin)
edge_widths = [
    min( max(row["weight"] * 0.01, 0.2 ), 2 )
    for _, row in edges.iterrows()
]

# Edge transparency
edge_alpha = 0.08    # *massively* improves readability

# Layout: more spacing
pos = nx.spring_layout(G, seed=42, k=2.0, iterations=200)

# -------------------------
# DRAW
# -------------------------
plt.figure(figsize=(13, 9))

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=1,
    edgecolors="black"
)

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    alpha=edge_alpha,
    edge_color="black",
    arrows=False
)

nx.draw_networkx_labels(
    G, pos,
    font_size=10,
    font_color="black"
)

plt.title("Collaboration Network (All Interactions Shown)", fontsize=16)
plt.axis("off")

out_file = os.path.join(FIG_DIR, "network_graph_clean.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved improved network graph to {out_file}")
