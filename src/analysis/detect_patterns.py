# src/analysis/detect_patterns.py

import os
import json
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

PROCESSED_DIR = "data/processed"
METRICS_DIR = os.path.join(PROCESSED_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

# --------------------------
# LOAD METRICS
# --------------------------
node_metrics = pd.read_csv(os.path.join(METRICS_DIR, "node_metrics.csv"))
edge_metrics = pd.read_csv(os.path.join(METRICS_DIR, "edge_metrics.csv"))
members = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_members.csv"))

patterns = {}

# --------------------------
# REBUILD GRAPH
# --------------------------
G = nx.DiGraph()

for m in members["member_id"]:
    G.add_node(m, role=members[members.member_id == m]["role"].iloc[0])

for _, row in edge_metrics.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# --------------------------
# 1. ISOLATED MEMBERS
# (No sent + no received)
# --------------------------
isolated = node_metrics[
    (node_metrics.total_sent == 0) &
    (node_metrics.total_received == 0)
]["member_id"].tolist()

patterns["isolated_members"] = isolated

# --------------------------
# 2. PASSIVE MEMBERS
# (Low activity score)
# --------------------------
avg_activity = node_metrics["activity_score"].mean()
passive = node_metrics[node_metrics.activity_score < avg_activity * 0.4]["member_id"].tolist()

patterns["passive_members"] = passive

# --------------------------
# 3. DOMINANT MEMBERS
# (Extremely high activity score)
# --------------------------
dominant = node_metrics[node_metrics.activity_score > avg_activity * 1.8]["member_id"].tolist()

patterns["dominant_members"] = dominant

# --------------------------
# 4. STRONG COLLABORATION PAIRS
# --------------------------
strong_threshold = edge_metrics["weight"].mean() * 1.5

strong_pairs = edge_metrics[edge_metrics.weight >= strong_threshold][["source", "target", "weight"]]
patterns["strong_pairs"] = strong_pairs.to_dict("records")

# --------------------------
# 5. WEAK COLLABORATION PAIRS
# --------------------------
weak_threshold = edge_metrics["weight"].mean() * 0.5

weak_pairs = edge_metrics[edge_metrics.weight <= weak_threshold][["source", "target", "weight"]]
patterns["weak_pairs"] = weak_pairs.to_dict("records")

# --------------------------
# 6. SUBGROUPS (COMMUNITIES)
# --------------------------
UG = G.to_undirected()

if UG.number_of_edges() > 0:
    communities = greedy_modularity_communities(UG, weight="weight")
    patterns["subgroups"] = [list(c) for c in communities]
else:
    patterns["subgroups"] = []

# --------------------------
# 7. ROLE MISMATCH (LEADER SHOULD NOT BE WEAK)
# --------------------------
leader_row = members[members.role == "leader"]

role_mismatch_reasons = []

if not leader_row.empty:
    leader = leader_row.member_id.iloc[0]
    leader_stats = node_metrics[node_metrics.member_id == leader].iloc[0]

    # leader must have high centrality
    if leader_stats["degree_centrality"] < node_metrics["degree_centrality"].mean():
        role_mismatch_reasons.append("Leader has low degree centrality (not well-connected).")

    # leader must not be passive
    if leader_stats["activity_score"] < avg_activity * 0.6:
        role_mismatch_reasons.append("Leader is unusually inactive.")

    # leader should receive replies
    if leader_stats["total_received"] < node_metrics["total_received"].mean() * 0.5:
        role_mismatch_reasons.append("Leader receives unusually low communication.")

patterns["role_mismatch"] = role_mismatch_reasons

# --------------------------
# SAVE OUTPUT
# --------------------------
with open(os.path.join(METRICS_DIR, "patterns.json"), "w") as f:
    json.dump(patterns, f, indent=4)

print("Pattern detection completed!")
print("Saved to:", os.path.join(METRICS_DIR, "patterns.json"))
