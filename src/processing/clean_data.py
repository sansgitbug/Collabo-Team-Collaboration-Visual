# src/processing/clean_data.py
"""
Cleans the raw datasets:
- interactions.csv
- members.csv
- tasks.csv

Outputs cleaned versions to data/processed/
"""

import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Load raw files
# ----------------------------
interactions = pd.read_csv(os.path.join(RAW_DIR, "interactions.csv"))
members = pd.read_csv(os.path.join(RAW_DIR, "members.csv"))
tasks = pd.read_csv(os.path.join(RAW_DIR, "tasks.csv"))

# ----------------------------
# Clean interactions
# ----------------------------
# Convert timestamps
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")

# Drop rows with invalid timestamps
interactions = interactions.dropna(subset=["timestamp"])

# Replace blank targets with None
interactions["target"] = interactions["target"].replace("", None)

# Sort chronologically
interactions = interactions.sort_values("timestamp").reset_index(drop=True)

# Ensure types
interactions["source"] = interactions["source"].astype(str)
interactions["interaction_type"] = interactions["interaction_type"].astype(str)
interactions["platform"] = interactions["platform"].astype(str)
interactions["weight"] = interactions["weight"].astype(float)

# Merge member roles into interactions
interactions = interactions.merge(
    members[["member_id", "role"]],
    how="left",
    left_on="source",
    right_on="member_id"
).drop(columns=["member_id"]).rename(columns={"role": "source_role"})

# ----------------------------
# Clean members
# ----------------------------
members["joined_at"] = pd.to_datetime(members["joined_at"], errors="coerce")

# ----------------------------
# Clean tasks
# ----------------------------
# Convert timestamps
for col in ["assigned_at", "due_date", "completed_at"]:
    if col in tasks.columns:
        tasks[col] = pd.to_datetime(tasks[col], errors="coerce")

# ----------------------------
# Save cleaned datasets
# ----------------------------
interactions.to_csv(os.path.join(OUT_DIR, "clean_interactions.csv"), index=False)
members.to_csv(os.path.join(OUT_DIR, "clean_members.csv"), index=False)
tasks.to_csv(os.path.join(OUT_DIR, "clean_tasks.csv"), index=False)

print("Cleaning complete!")
print(f"Saved cleaned files to {OUT_DIR}")
