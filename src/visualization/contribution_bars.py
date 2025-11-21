# src/visualization/contribution_bars.py

import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = "data/processed/metrics"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

metrics = pd.read_csv(os.path.join(METRICS_DIR, "node_metrics.csv"))

# Sort by activity score
metrics_sorted = metrics.sort_values("activity_score", ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(metrics_sorted["member_id"], metrics_sorted["activity_score"])

plt.title("Member Contribution (Activity Score)")
plt.xlabel("Member")
plt.ylabel("Activity Score")
plt.grid(axis="y")

out_file = os.path.join(FIG_DIR, "contribution_bars.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved contribution bar chart to {out_file}")
