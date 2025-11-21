# src/data_generation/generate_synthetic_data_single_team.py
"""
Single-team realistic synthetic collaboration data generator.

Features:
- Single team only.
- Team size: 5-6 members (randomly chosen).
- Roles: leader, active, regular, passive, optional isolated.
- Interaction types: message, reply, task_assign, task_complete, comment, review.
- Bursty days, quiet days, daily rhythm, pairwise affinity (subgroups).
- Generates ~TARGET_INTERACTIONS interactions (configurable; default 7000).
- Outputs CSVs to data/raw/: interactions.csv, tasks.csv, members.csv

Run:
    python src/data_generation/generate_synthetic_data_single_team.py
"""
import os
import random
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEAM_ID = "T01"
MIN_MEMBERS = 5
MAX_MEMBERS = 6

DAYS = 30                       # simulate last N days
START_DATE = datetime.now() - timedelta(days=DAYS)
TARGET_INTERACTIONS = 7000      # user requested ~7000 interactions

PLATFORMS = ["slack", "whatsapp", "github", "trello", "googledocs"]
INTERACTION_TYPES = [
    "message", "reply", "task_assign", "task_complete", "comment", "review"
]
ROLE_POOL = ["leader", "active", "regular", "passive", "isolated"]

# -----------------------
# HELPERS
# -----------------------
def rand_name():
    first = ["Alex","Sam","Chris","Taylor","Jordan","Casey","Riley","Morgan","Avery","Jamie","Kai","Lee"]
    last = ["Patel","Singh","Kumar","Sharma","Das","Iyer","Gupta","Rao","Verma","Nair","Fernandes"]
    return f"{random.choice(first)} {random.choice(last)}"

def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(round(x))))

# -----------------------
# BUILD TEAM & MEMBERS
# -----------------------
team_size = random.randint(MIN_MEMBERS, MAX_MEMBERS)
members = []
for i in range(team_size):
    mid = f"M{i+1:03d}"
    members.append({
        "member_id": mid,
        "team_id": TEAM_ID,
        "name": rand_name(),
        "role": "regular",
        "joined_at": (START_DATE - timedelta(days=random.randint(0,60))).isoformat()
    })

# assign roles: 1 leader, 1-2 active, maybe 1 passive, maybe 1 isolated
member_ids = [m["member_id"] for m in members]
leader = random.choice(member_ids)
for m in members:
    if m["member_id"] == leader:
        m["role"] = "leader"

remaining = [m for m in member_ids if m != leader]
n_actives = min(2, max(1, len(remaining)//3))
actives = random.sample(remaining, n_actives)
for a in actives:
    for m in members:
        if m["member_id"] == a:
            m["role"] = "active"

# passive
if len(member_ids) >= 5:
    possible = [m for m in member_ids if m not in [leader] + actives]
    if possible:
        passive = random.choice(possible)
        for m in members:
            if m["member_id"] == passive:
                m["role"] = "passive"

# isolated (optional)
if random.random() < 0.5:
    possible = [m for m in member_ids if m not in [leader] + actives]
    if possible:
        iso = random.choice(possible)
        for m in members:
            if m["member_id"] == iso:
                m["role"] = "isolated"

members_df = pd.DataFrame(members)

# -----------------------
# PAIR AFFINITY (subgroups / stronger ties)
# -----------------------
pair_affinity = {}
# create 1 small clique (if possible) to simulate close collaborators
clique = []
if len(member_ids) >= 4:
    clique_size = max(2, len(member_ids)//2)  # half the team typically
    clique = random.sample(member_ids, clique_size)

for a in member_ids:
    for b in member_ids:
        if a == b:
            continue
        base = 1.0
        if a in clique and b in clique:
            base *= random.uniform(2.0, 3.5)
        # role effects
        role_a = members_df.loc[members_df.member_id == a, "role"].values[0]
        role_b = members_df.loc[members_df.member_id == b, "role"].values[0]
        if role_a == "leader":
            base *= 1.6
        if role_b == "isolated":
            base *= 0.3
        pair_affinity[(a, b)] = base

# -----------------------
# SIMULATE INTERACTIONS
# -----------------------
interactions = []
tasks = []
interaction_count = 0

date_list = [START_DATE + timedelta(days=d) for d in range(DAYS)]
# burst days for team
team_bursts = set(random.sample(range(DAYS), k=max(1, DAYS//8)))  # ~4 burst days if 30 days

while interaction_count < TARGET_INTERACTIONS:
    for day_idx, day in enumerate(date_list):
        # per-day base activity
        base_lambda = max(4, len(member_ids) * 10)  # average events per day
        if day_idx in team_bursts:
            base_lambda = int(base_lambda * random.uniform(2.0, 3.0))
        # quiet chance
        if random.random() < 0.06:
            base_lambda = int(base_lambda * random.uniform(0.1, 0.6))
        n_events = np.random.poisson(base_lambda)
        for _ in range(n_events):
            # event timestamp within day with daily rhythm (peak afternoon)
            hour = int(np.clip(np.random.normal(loc=15, scale=4), 8, 23))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = datetime(day.year, day.month, day.day, hour, minute, second)

            # choose source based on role weights
            role_weights = []
            for mid in member_ids:
                role = members_df.loc[members_df.member_id == mid, "role"].values[0]
                if role == "leader":
                    w = 2.2
                elif role == "active":
                    w = 1.8
                elif role == "regular":
                    w = 1.0
                elif role == "passive":
                    w = 0.5
                elif role == "isolated":
                    w = 0.25
                else:
                    w = 1.0
                role_weights.append(w)
            source = random.choices(member_ids, weights=role_weights, k=1)[0]

            # interaction type probabilities (task events rarer)
            itype = random.choices(
                INTERACTION_TYPES,
                weights=[0.48, 0.18, 0.07, 0.05, 0.16, 0.06],
                k=1
            )[0]

            # target selection using pair affinity; small chance of channel message
            possible_targets = [m for m in member_ids if m != source]
            affinities = [pair_affinity[(source, t)] for t in possible_targets]
            if random.random() < 0.09:
                target = None  # broadcast / channel message
            else:
                target = random.choices(possible_targets, weights=affinities, k=1)[0]

            platform = random.choices(PLATFORMS, weights=[0.4,0.25,0.12,0.13,0.1], k=1)[0]

            base_weight = {
                "message": 1.0,
                "reply": 1.2,
                "task_assign": 2.1,
                "task_complete": 2.6,
                "comment": 1.3,
                "review": 1.8
            }.get(itype, 1.0)

            affinity_scale = 1.0 if target is None else pair_affinity[(source, target)]
            weight = base_weight * affinity_scale * random.uniform(0.85, 1.25)
            weight = round(float(weight), 3)

            # generate simple placeholder content length to simulate message sizes
            text_len = int(np.clip(np.random.normal(loc=60 if itype in ["message","reply"] else 140, scale=25), 5, 600))
            content = f"<{itype}> " + ("x" * max(4, text_len//2))

            interactions.append({
                "timestamp": ts.isoformat(),
                "team_id": TEAM_ID,
                "source": source,
                "target": target if target is not None else "",
                "interaction_type": itype,
                "platform": platform,
                "weight": weight,
                "content": content
            })
            interaction_count += 1

            # tasks logic
            if itype == "task_assign":
                task_id = f"TASK-{uuid.uuid4().hex[:8]}"
                due_days = random.randint(1, 10)
                tasks.append({
                    "task_id": task_id,
                    "team_id": TEAM_ID,
                    "assigned_by": source,
                    "assigned_to": target if target is not None else source,
                    "assigned_at": ts.isoformat(),
                    "due_date": (ts + timedelta(days=due_days)).isoformat(),
                    "status": "assigned"
                })
            elif itype == "task_complete":
                team_tasks = [t for t in tasks if t["team_id"] == TEAM_ID and t["status"] == "assigned"]
                if team_tasks:
                    task_to_complete = random.choice(team_tasks)
                    task_to_complete["status"] = "completed"
                    task_to_complete["completed_by"] = source
                    task_to_complete["completed_at"] = ts.isoformat()

            if interaction_count >= TARGET_INTERACTIONS:
                break
        if interaction_count >= TARGET_INTERACTIONS:
            break
    if interaction_count >= TARGET_INTERACTIONS:
        break

# -----------------------
# SAVE CSVs
# -----------------------
interactions_df = pd.DataFrame(interactions).sort_values("timestamp").reset_index(drop=True)
tasks_df = pd.DataFrame(tasks)
members_df = members_df.copy()

# compute simple contribution estimate
contrib = interactions_df.groupby("source")["weight"].sum().reset_index().rename(columns={"weight":"sum_weight"})
task_completions = tasks_df[tasks_df.status == "completed"].groupby("completed_by").size().reset_index().rename(columns={0:"tasks_completed"})
members_df = members_df.merge(contrib, how="left", left_on="member_id", right_on="source").drop(columns=["source"], errors="ignore")
members_df = members_df.merge(task_completions, how="left", left_on="member_id", right_on="completed_by").drop(columns=["completed_by"], errors="ignore")
members_df["sum_weight"] = members_df["sum_weight"].fillna(0.0)
members_df["tasks_completed"] = members_df["tasks_completed"].fillna(0).astype(int)
members_df["estimated_contribution"] = (members_df["sum_weight"] * 0.7) + (members_df["tasks_completed"] * 1.6)

# file paths
interactions_csv = os.path.join(OUTPUT_DIR, "interactions.csv")
tasks_csv = os.path.join(OUTPUT_DIR, "tasks.csv")
members_csv = os.path.join(OUTPUT_DIR, "members.csv")

interactions_df.to_csv(interactions_csv, index=False)
tasks_df.to_csv(tasks_csv, index=False)
members_df.to_csv(members_csv, index=False)

print(f"Generated {len(interactions_df)} interactions, {len(tasks_df)} tasks, {len(members_df)} members.")
print(f"Wrote: {interactions_csv}")
print(f"Wrote: {tasks_csv}")
print(f"Wrote: {members_csv}")
