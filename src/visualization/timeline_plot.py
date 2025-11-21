# src/visualization/timeline_plot.py

import os
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load interactions
interactions = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_interactions.csv"))
interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

# Count interactions per day
daily_counts = interactions.resample("D", on="timestamp").size()

plt.figure(figsize=(10, 5))
daily_counts.plot(kind="line")

plt.title("Daily Collaboration Activity")
plt.xlabel("Date")
plt.ylabel("Number of Interactions")
plt.grid(True)

out_file = os.path.join(FIG_DIR, "timeline_plot.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved timeline plot to {out_file}")
