# src/analysis/compute_metrics.py
import os
import json
import pandas as pd
import networkx as nx

PROCESSED_DIR = "data/processed"
METRICS_DIR = os.path.join(PROCESSED_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
clean_interactions = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_interactions.csv"))
adj_list = pd.read_csv(os.path.join(PROCESSED_DIR, "adj_list.csv"))
members = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_members.csv"))

# ------------------------------
# Prepare base metrics table
# ------------------------------
metrics = pd.DataFrame()
metrics["member_id"] = members["member_id"]

# ------------------------------
# TOTAL SENT
# ------------------------------
sent = (
    clean_interactions.groupby("source")
    .size()
    .reset_index()
    .rename(columns={"source": "member_id", 0: "total_sent"})
)
metrics = metrics.merge(sent, on="member_id", how="left")
metrics["total_sent"] = metrics["total_sent"].fillna(0).astype(int)

# ------------------------------
# TOTAL RECEIVED
# ------------------------------
received = (
    clean_interactions.dropna(subset=["target"])
    .groupby("target")
    .size()
    .reset_index()
    .rename(columns={"target": "member_id", 0: "total_received"})
)
metrics = metrics.merge(received, on="member_id", how="left")
metrics["total_received"] = metrics["total_received"].fillna(0).astype(int)

# ------------------------------
# WEIGHTED SENT
# ------------------------------
weighted_sent = (
    adj_list.groupby("source")["weight"]
    .sum()
    .reset_index()
    .rename(columns={"source": "member_id", "weight": "weighted_sent"})
)
metrics = metrics.merge(weighted_sent, on="member_id", how="left")
metrics["weighted_sent"] = metrics["weighted_sent"].fillna(0)

# ------------------------------
# WEIGHTED RECEIVED
# ------------------------------
weighted_received = (
    adj_list.groupby("target")["weight"]
    .sum()
    .reset_index()
    .rename(columns={"target": "member_id", "weight": "weighted_received"})
)
metrics = metrics.merge(weighted_received, on="member_id", how="left")
metrics["weighted_received"] = metrics["weighted_received"].fillna(0)

# ------------------------------
# BUILD GRAPH FOR CENTRALITY
# ------------------------------
G = nx.DiGraph()

# add nodes
for m in members["member_id"]:
    G.add_node(m)

# add edges (weighted)
for _, row in adj_list.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# ------------------------------
# CENTRALITY MEASURES
# ------------------------------
deg_cent = nx.degree_centrality(G)
in_deg_cent = nx.in_degree_centrality(G)
out_deg_cent = nx.out_degree_centrality(G)
close_cent = nx.closeness_centrality(G)
between_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)

metrics["degree_centrality"] = metrics["member_id"].map(deg_cent)
metrics["in_degree_centrality"] = metrics["member_id"].map(in_deg_cent)
metrics["out_degree_centrality"] = metrics["member_id"].map(out_deg_cent)
metrics["closeness_centrality"] = metrics["member_id"].map(close_cent)
metrics["betweenness_centrality"] = metrics["member_id"].map(between_cent)

# ------------------------------
# ACTIVITY SCORE
# ------------------------------
metrics["activity_score"] = (
    metrics["total_sent"] * 0.4
    + metrics["total_received"] * 0.2
    + metrics["weighted_sent"] * 0.4
)

# ------------------------------
# INFLUENCE SCORE
# ------------------------------
metrics["influence_score"] = (
    metrics["weighted_sent"] * 0.6
    + metrics["degree_centrality"] * 0.3
    + metrics["betweenness_centrality"] * 0.1
)

# ------------------------------
# SAVE NODE METRICS
# ------------------------------
metrics.to_csv(os.path.join(METRICS_DIR, "node_metrics.csv"), index=False)

# ------------------------------
# EDGE METRICS
# ------------------------------
edge_metrics = adj_list.copy()
edge_metrics["norm_weight"] = edge_metrics["weight"] / edge_metrics["weight"].max()
edge_metrics.to_csv(os.path.join(METRICS_DIR, "edge_metrics.csv"), index=False)

# ------------------------------
# TEAM-LEVEL METRICS
# ------------------------------
team_metrics = {
    "density": nx.density(G),
    "reciprocity": nx.reciprocity(G),
    "num_nodes": G.number_of_nodes(),
    "num_edges": G.number_of_edges(),
    "average_clustering": nx.average_clustering(G.to_undirected()),
}

with open(os.path.join(METRICS_DIR, "team_metrics.json"), "w") as f:
    json.dump(team_metrics, f, indent=4)

print("SUCCESS: Metrics computed.")
print(f"Saved in {METRICS_DIR}")
