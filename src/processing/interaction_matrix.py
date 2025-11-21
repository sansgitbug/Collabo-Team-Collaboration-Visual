# src/processing/build_interaction_matrix.py
"""
Builds:
- Raw interaction matrix (counts)
- Weighted interaction matrix
- Adjacency list

Outputs:
- interaction_matrix.csv
- weighted_matrix.csv
- adj_list.csv
"""

import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
OUT_DIR = PROCESSED_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Load cleaned interactions
df = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_interactions.csv"))

# Only consider interactions with a valid target
df_valid = df.dropna(subset=["target"]).copy()

members = sorted(set(df_valid["source"].unique()) | set(df_valid["target"].unique()))

# ---------------------------
# Build interaction matrix
# ---------------------------
matrix = pd.DataFrame(0, index=members, columns=members, dtype=int)
weighted = pd.DataFrame(0.0, index=members, columns=members, dtype=float)

for _, row in df_valid.iterrows():
    src = row["source"]
    tgt = row["target"]
    w = row["weight"]

    matrix.loc[src, tgt] += 1
    weighted.loc[src, tgt] += w

# ---------------------------
# Build adjacency list
# ---------------------------
adj = []

for src in members:
    for tgt in members:
        if matrix.loc[src, tgt] > 0:
            adj.append({
                "source": src,
                "target": tgt,
                "count": matrix.loc[src, tgt],
                "weight": round(weighted.loc[src, tgt], 3)
            })

adj_df = pd.DataFrame(adj)

# ---------------------------
# Save outputs
# ---------------------------
matrix.to_csv(os.path.join(OUT_DIR, "interaction_matrix.csv"))
weighted.to_csv(os.path.join(OUT_DIR, "weighted_matrix.csv"))
adj_df.to_csv(os.path.join(OUT_DIR, "adj_list.csv"), index=False)

print("Interaction matrices generated!")
print(f"Saved to {OUT_DIR}")
