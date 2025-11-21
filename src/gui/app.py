import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
import json
import community.community_louvain as community_louvain

# --------------------------------------------
# LOAD DATA
# --------------------------------------------
PROCESSED_DIR = "data/processed"
METRICS_DIR = os.path.join(PROCESSED_DIR, "metrics")

clean_interactions = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_interactions.csv"))
members = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_members.csv"))
node_metrics = pd.read_csv(os.path.join(METRICS_DIR, "node_metrics.csv"))
edge_metrics = pd.read_csv(os.path.join(METRICS_DIR, "edge_metrics.csv"))

with open(os.path.join(METRICS_DIR, "team_metrics.json")) as f:
    team_metrics = json.load(f)

with open(os.path.join(METRICS_DIR, "patterns.json")) as f:
    patterns = json.load(f)

clean_interactions["timestamp"] = pd.to_datetime(clean_interactions["timestamp"])


# --------------------------------------------
# HELPER – INTERACTIVE NETWORK GRAPH (TIME FILTERED)
# --------------------------------------------
def generate_pyvis_graph_filtered(start_date, end_date, member_filter, type_filter, platform_filter, max_edges):
    """
    Generate interactive network graph with community detection.
    Communities are detected using the Louvain algorithm and visualized with distinct colors.
    """
    
    # Filter by time
    filtered = clean_interactions[
        (clean_interactions["timestamp"] >= start_date) &
        (clean_interactions["timestamp"] <= end_date)
    ]

    # Filter by member
    if member_filter != "ALL":
        filtered = filtered[
            (filtered["source"] == member_filter) |
            (filtered["target"] == member_filter)
        ]

    # Filter by interaction type
    if type_filter != "ALL":
        filtered = filtered[filtered["interaction_type"] == type_filter]

    # Filter by platform
    if platform_filter != "ALL":
        filtered = filtered[filtered["platform"] == platform_filter]

    # Limit edges to prevent lag
    if len(filtered) > max_edges:
        filtered = filtered.sample(max_edges, random_state=42)

    # Build graph for community detection (undirected for better community detection)
    G_undirected = nx.Graph()
    
    # Add weighted edges based on interaction frequency
    edge_weights = filtered.groupby(['source', 'target']).size().reset_index(name='weight')
    
    for _, row in edge_weights.iterrows():
        src = str(row['source'])
        tgt = str(row['target'])
        weight = row['weight']
        
        if G_undirected.has_edge(src, tgt):
            G_undirected[src][tgt]['weight'] += weight
        else:
            G_undirected.add_edge(src, tgt, weight=weight)
    
    # Detect communities using Louvain algorithm
    communities = community_louvain.best_partition(G_undirected, weight='weight')
    
    # Identify isolated nodes (not in the main graph)
    all_members = set(str(row["member_id"]) for _, row in members.iterrows())
    connected_members = set(G_undirected.nodes())
    isolated_members = all_members - connected_members
    
    # Assign isolated members to community -1
    for isolated in isolated_members:
        communities[isolated] = -1
    
    # Color palette for communities (up to 20 distinct colors)
    community_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8',
        '#52BE80', '#F1948A', '#85929E', '#F39C12', '#8E44AD',
        '#16A085', '#E74C3C', '#3498DB', '#2ECC71', '#E67E22'
    ]
    
    # Special color for isolated nodes
    isolated_color = '#CCCCCC'  # Gray for isolated members
    
    # Build directed graph for visualization
    G_directed = nx.MultiDiGraph()
    
    # Add nodes with community colors
    for _, row in members.iterrows():
        member_id = str(row["member_id"])
        role = row["role"]
        
        # Get community assignment
        comm_id = communities.get(member_id, 0)
        color = community_colors[comm_id % len(community_colors)]
        
        # Calculate node size based on degree in undirected graph
        degree = G_undirected.degree(member_id) if member_id in G_undirected else 0
        node_size = 20 + (degree * 3)
        
        # Create clean, readable tooltip
        tooltip = f"{member_id}\nRole: {role}\nCommunity: {comm_id}\nConnections: {degree}"
        
        G_directed.add_node(
            member_id,
            label=member_id,
            title=tooltip,
            color=color,
            size=node_size
        )
    
    # Aggregate edges to prevent multiple edge chaos
    edge_aggregation = filtered.groupby(['source', 'target']).agg({
        'interaction_type': lambda x: ', '.join(x.unique()),
        'platform': lambda x: ', '.join(x.unique()),
        'timestamp': 'count'
    }).reset_index()
    edge_aggregation.columns = ['source', 'target', 'types', 'platforms', 'count']
    
    # Add aggregated edges to directed graph
    for _, row in edge_aggregation.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        count = row["count"]
        
        # Create clean tooltip
        tooltip = f"{src} → {tgt}\nInteractions: {count}\nTypes: {row['types']}\nPlatforms: {row['platforms']}"
        
        G_directed.add_edge(
            src,
            tgt,
            title=tooltip,
            color="rgba(0,0,0,0.15)",
            value=count,  # thickness based on interaction count
            width=1 + (count * 0.5)  # visual thickness
        )
    
    # Create PyVis network
    net = Network(height="650px", width="100%", directed=True, bgcolor="#FFFFFF")
    net.from_nx(G_directed)
    
    # Enhanced visualization options
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "font": {
          "size": 18,
          "face": "Tahoma",
          "color": "#000000"
        },
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.2)",
          "size": 10,
          "x": 3,
          "y": 3
        }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic",
          "roundness": 0.5
        },
        "color": {
          "color": "rgba(100,100,100,0.2)",
          "highlight": "rgba(0,0,0,0.5)"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.4
          }
        },
        "width": 1.5
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.25,
          "springLength": 200,
          "springConstant": 0.03,
          "damping": 0.12,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": true,
        "hideEdgesOnZoom": false
      }
    }
    """)
    
    net.save_graph("network_graph_filtered.html")
    
    # Return both the file path and community info
    num_communities = len(set(communities.values()))
    community_sizes = {}
    for node, comm in communities.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    return "network_graph_filtered.html", num_communities, community_sizes, communities


# --------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------
st.sidebar.title("Team Collaboration Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "👤 Members",
        "🔗 Pairwise Collaboration",
        "⏱️ Timeline",
        "🕸️ Network Graph",
        "🧠 Patterns & Insights",
        "💡 Recommendations",
        "⭐ Real time implementation!"
    ]
)


# --------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------
if page == "🏠 Dashboard":
    st.title("📊 Team Collaboration Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Interactions", len(clean_interactions))
    col2.metric("Team Density", round(team_metrics["density"], 3))
    col3.metric("Edges (Relationships)", team_metrics["num_edges"])

    st.subheader("Activity Summary")
    top_member = node_metrics.sort_values("activity_score", ascending=False).iloc[0]

    st.success(
        f"**Most Active Member:** {top_member['member_id']} "
        f"(Score: {round(top_member['activity_score'],2)})"
    )

    st.subheader("Team Health Indicators")
    st.json(team_metrics)


# --------------------------------------------
# PAGE 2: MEMBERS
# --------------------------------------------
elif page == "👤 Members":
    st.title("👤 Member Analytics")

    member_list = sorted(members["member_id"].unique())
    selected = st.selectbox("Select a Member", member_list)

    data = node_metrics[node_metrics["member_id"] == selected].iloc[0]

    st.subheader(f"Metrics for {selected}")
    st.json(data.to_dict())

    st.subheader("Contribution Activity")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(["Sent", "Received"], [data["total_sent"], data["total_received"]])
    st.pyplot(fig)


# --------------------------------------------
# PAGE 3: PAIRWISE COLLABORATION
# --------------------------------------------
# --------------------------------------------
# PAGE 3: PAIRWISE COLLABORATION
# --------------------------------------------
elif page == "🔗 Pairwise Collaboration":
    st.title("🔗 Pairwise Collaboration Analysis")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔥 Heatmap", "📋 Interaction Table", "🎯 Top Pairs"])
    
    with tab1:
        st.header("Collaboration Overview")
        
        # Load matrix
        matrix = pd.read_csv(os.path.join(PROCESSED_DIR, "interaction_matrix.csv"), index_col=0)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        total_interactions = clean_interactions.shape[0]
        unique_pairs = len(clean_interactions.groupby(['source', 'target']).size())
        avg_per_pair = total_interactions / unique_pairs if unique_pairs > 0 else 0
        
        col1.metric("Total Interactions", f"{total_interactions:,}")
        col2.metric("Unique Pairs", unique_pairs)
        col3.metric("Avg per Pair", f"{avg_per_pair:.1f}")
        
        # Member-by-member breakdown
        st.subheader("Member Interaction Breakdown")
        
        member_stats = []
        for member in matrix.index:
            sent = matrix.loc[member].sum()  # Row sum = messages sent
            received = matrix[member].sum()  # Column sum = messages received
            total = sent + received
            member_stats.append({
                "Member": member,
                "Sent": int(sent),
                "Received": int(received),
                "Total": int(total)
            })
        
        stats_df = pd.DataFrame(member_stats).sort_values("Total", ascending=False)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.subheader("Sent vs Received Comparison")
        chart_data = stats_df.set_index("Member")[["Sent", "Received"]]
        st.bar_chart(chart_data)
    
    with tab2:
        st.header("Interaction Heatmap")
        
        # Load matrix
        matrix = pd.read_csv(os.path.join(PROCESSED_DIR, "interaction_matrix.csv"), index_col=0)
        
        st.write("**How to read:** Rows = Source (who sent), Columns = Target (who received)")
        
        # Create styled heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd', 
                    linewidths=0.5, cbar_kws={'label': 'Interaction Count'},
                    ax=ax)
        ax.set_title("Pairwise Interaction Heatmap", fontsize=16, fontweight='bold')
        ax.set_xlabel("Target (Recipient)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Source (Sender)", fontsize=12, fontweight='bold')
        
        st.pyplot(fig)
        
        # Show raw matrix
        st.subheader("Raw Matrix Data")
        st.dataframe(matrix, use_container_width=True)
    
    with tab3:
        st.header("Detailed Interaction Table")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            source_filter = st.selectbox(
                "Filter by Source",
                ["All"] + list(clean_interactions["source"].unique())
            )
        
        with col2:
            target_filter = st.selectbox(
                "Filter by Target",
                ["All"] + list(clean_interactions["target"].dropna().unique())
            )
        
        # Apply filters
        filtered_data = clean_interactions.copy()
        if source_filter != "All":
            filtered_data = filtered_data[filtered_data["source"] == source_filter]
        if target_filter != "All":
            filtered_data = filtered_data[filtered_data["target"] == target_filter]
        
        st.write(f"Showing {len(filtered_data):,} of {len(clean_interactions):,} interactions")
        
        # Show data
        st.dataframe(
            filtered_data[["timestamp", "source", "target", "interaction_type", "platform"]],
            height=400,
            use_container_width=True
        )
        
        # Download option
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_interactions.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.header("Top Collaboration Pairs")
        
        # Calculate pair interactions
        pair_counts = clean_interactions[clean_interactions["target"].notna()].groupby(
            ["source", "target"]
        ).size().reset_index(name="count")
        
        pair_counts = pair_counts.sort_values("count", ascending=False)
        
        # Top 10 pairs
        st.subheader("🔥 Top 10 Most Active Pairs")
        top_10 = pair_counts.head(10).copy()
        top_10["Pair"] = top_10["source"] + " → " + top_10["target"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_10)), top_10["count"], color='skyblue')
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10["Pair"])
        ax.set_xlabel("Number of Interactions", fontweight='bold')
        ax.set_title("Top 10 Collaboration Pairs", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for i, v in enumerate(top_10["count"]):
            ax.text(v + 10, i, str(v), va='center')
        
        st.pyplot(fig)
        
        # Bottom 10 pairs
        st.subheader("❄️ Bottom 10 Least Active Pairs")
        bottom_10 = pair_counts.tail(10).copy()
        bottom_10["Pair"] = bottom_10["source"] + " → " + bottom_10["target"]
        st.dataframe(
            bottom_10[["Pair", "count"]].rename(columns={"count": "Interactions"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Reciprocity analysis
        st.subheader("🔄 Reciprocity Analysis")
        
        reciprocal_pairs = []
        for _, row in pair_counts.iterrows():
            source, target, count = row["source"], row["target"], row["count"]
            
            # Check if reverse pair exists
            reverse = pair_counts[
                (pair_counts["source"] == target) & 
                (pair_counts["target"] == source)
            ]
            
            if not reverse.empty:
                reverse_count = reverse.iloc[0]["count"]
                balance = abs(count - reverse_count)
                reciprocal_pairs.append({
                    "Pair": f"{source} ↔ {target}",
                    f"{source} → {target}": count,
                    f"{target} → {source}": reverse_count,
                    "Balance": balance,
                    "Status": "Balanced" if balance < 50 else "Imbalanced"
                })
        
        if reciprocal_pairs:
            recip_df = pd.DataFrame(reciprocal_pairs).sort_values("Balance")
            
            balanced = recip_df[recip_df["Status"] == "Balanced"]
            imbalanced = recip_df[recip_df["Status"] == "Imbalanced"]
            
            col1, col2 = st.columns(2)
            col1.metric("Balanced Pairs", len(balanced))
            col2.metric("Imbalanced Pairs", len(imbalanced))
            
            st.dataframe(recip_df, use_container_width=True, hide_index=True)
        else:
            st.info("No reciprocal pairs found in the data.")

# --------------------------------------------
# PAGE 4: TIMELINE
# --------------------------------------------
elif page == "⏱️ Timeline":
    st.title("⏱️ Collaboration Timeline")

    daily = clean_interactions.resample("D", on="timestamp").size()
    st.line_chart(daily)


# --------------------------------------------
# PAGE 5: NETWORK GRAPH (TIME-FILTERED RAW INTERACTIONS)
# --------------------------------------------
elif page == "🕸️ Network Graph":
    st.title("🕸️ Interactive Collaboration Network with Community Detection")
    
    st.info("""
    **Community Detection:** Nodes are colored by detected collaboration communities using the Louvain algorithm.
    Members in the same community tend to work together more frequently.
    """)

    # TIME SLIDER
    min_date = clean_interactions["timestamp"].min().to_pydatetime()
    max_date = clean_interactions["timestamp"].max().to_pydatetime()

    start, end = st.slider(
        "Select time range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # MEMBER FILTER
    member_list = ["ALL"] + list(members["member_id"])
    member_filter = st.selectbox("Filter by member:", member_list)

    # INTERACTION TYPE FILTER
    type_list = ["ALL"] + sorted(clean_interactions["interaction_type"].unique())
    type_filter = st.selectbox("Filter by interaction type:", type_list)

    # PLATFORM FILTER
    platform_list = ["ALL"] + sorted(clean_interactions["platform"].unique())
    platform_filter = st.selectbox("Filter by platform:", platform_list)

    # MAX EDGES FILTER
    max_edges = st.slider("Max edges to display:", 50, 1000, 300)

    st.info(f"Showing up to **{max_edges} interactions** with current filters.")

    # Generate graph with community detection
    html_file, num_communities, community_sizes, communities = generate_pyvis_graph_filtered(
        start, end, member_filter, type_filter, platform_filter, max_edges
    )
    
    # Display community statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Communities", num_communities)
    with col2:
        avg_size = sum(community_sizes.values()) / len(community_sizes) if community_sizes else 0
        st.metric("Avg Community Size", f"{avg_size:.1f}")
    
    # Show community breakdown
    with st.expander("📊 View Community Details"):
        st.write("**Members by Community:**")
        
        # Organize by community
        communities_dict = {}
        for member, comm_id in communities.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(member)
        
        for comm_id in sorted(communities_dict.keys()):
            st.write(f"**Community {comm_id}:** {', '.join(sorted(communities_dict[comm_id]))}")

    # Display graph
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=700, scrolling=True)
    
    st.caption("""
    💡 **Tip:** Larger nodes indicate members with more connections. 
    Hover over nodes and edges to see details. Drag nodes to rearrange the layout.
    """)

# --------------------------------------------
# PAGE: RECOMMENDATIONS
# --------------------------------------------
elif page == "💡 Recommendations":
    st.title("💡 Actionable Recommendations")
    
    # Load recommendations
    try:
        import os
        REC_DIR = "data/recommendations"
        
        with open(os.path.join(REC_DIR, "detailed_recommendations.json")) as f:
            detailed_recs = json.load(f)
        
        with open(os.path.join(REC_DIR, "action_plan.json")) as f:
            action_plan = json.load(f)
        
        with open(os.path.join(REC_DIR, "communication_protocol.json")) as f:
            comm_protocol = json.load(f)
        
        # Summary metrics
        st.header("📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Recommendations", detailed_recs["total_recommendations"])
        col2.metric("Critical Issues", action_plan["summary"]["critical"])
        col3.metric("High Priority", action_plan["summary"]["high"])
        col4.metric("Medium Priority", action_plan["summary"]["medium"])
        
        # Tab view
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚨 Action Plan", 
            "👥 Member-Specific", 
            "💬 Communication Protocol",
            "📄 Full Report"
        ])
        
        # TAB 1: ACTION PLAN
        with tab1:
            st.subheader("🚨 Prioritized Action Plan")
            
            if action_plan["immediate_actions_this_week"]:
                st.markdown("### ⚠️ THIS WEEK (Critical)")
                for i, action in enumerate(action_plan["immediate_actions_this_week"], 1):
                    with st.expander(f"#{i}: {action['issue']}", expanded=True):
                        st.markdown(action['top_action'])
            
            if action_plan["short_term_1_2_weeks"]:
                st.markdown("### 📅 NEXT 1-2 WEEKS (High Priority)")
                for i, action in enumerate(action_plan["short_term_1_2_weeks"], 1):
                    with st.expander(f"#{i}: {action['issue']}"):
                        st.markdown(action['top_action'])
            
            if action_plan["medium_term_3_4_weeks"]:
                st.markdown("### 📆 NEXT 3-4 WEEKS (Medium Priority)")
                for i, action in enumerate(action_plan["medium_term_3_4_weeks"], 1):
                    with st.expander(f"#{i}: {action['issue']}"):
                        st.markdown(action['top_action'])
            
            st.markdown("### ♻️ ONGOING PRACTICES")
            for practice in action_plan["ongoing_practices"]:
                st.markdown(f"- {practice}")
        
        # TAB 2: MEMBER-SPECIFIC
        with tab2:
            st.subheader("👥 Member-Specific Recommendations")
            
            for rec in detailed_recs["recommendations"]:
                if "member_id" in rec:
                    severity_emoji = {
                        "CRITICAL": "🚨",
                        "HIGH": "⚠️",
                        "MEDIUM": "ℹ️",
                        "INFO": "💡",
                        "POSITIVE": "✅"
                    }.get(rec.get("severity", "INFO"), "•")
                    
                    with st.expander(
                        f"{severity_emoji} {rec['member_name']} ({rec['member_id']}) - {rec['issue']}", 
                        expanded=(rec.get("severity") in ["CRITICAL", "HIGH"])
                    ):
                        if "current_metrics" in rec:
                            st.markdown("**Current Metrics:**")
                            st.json(rec["current_metrics"])
                        
                        if "positive_notes" in rec:
                            st.success("**Positive Notes:**")
                            for note in rec["positive_notes"]:
                                st.markdown(f"✓ {note}")
                        
                        st.markdown("**Recommendations:**")
                        for i, recommendation in enumerate(rec.get("recommendations", []), 1):
                            st.markdown(f"{i}. {recommendation}")
                        
                        if "suggested_actions" in rec:
                            st.markdown("**Action Timeline:**")
                            for timeframe, actions in rec["suggested_actions"].items():
                                st.markdown(f"*{timeframe.replace('_', ' ').title()}:*")
                                for action in actions:
                                    st.markdown(f"  - {action}")
                        
                        if "expected_outcomes" in rec:
                            st.info("**Expected Outcomes:**")
                            for outcome in rec["expected_outcomes"]:
                                st.markdown(f"→ {outcome}")
        
        # TAB 3: COMMUNICATION PROTOCOL
        with tab3:
            st.subheader("💬 Recommended Communication Protocol")
            
            st.markdown("### 📋 Core Practices")
            for practice, details in comm_protocol.get("core_practices", {}).items():
                with st.expander(practice.replace("_", " ").title()):
                    if isinstance(details, dict):
                        for key, value in details.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(details)
            
            st.markdown("### 🛠️ Platform Guidelines")
            platform_usage = comm_protocol.get("platform_usage_guidelines", {})
            for platform, guidelines in platform_usage.items():
                with st.expander(platform.replace("_", " ").title()):
                    for key, value in guidelines.items():
                        if isinstance(value, list):
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                st.markdown(f"  - {item}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.markdown("### ⬆️ Escalation Protocol")
            escalation = comm_protocol.get("escalation_protocol", {})
            for level, action in escalation.items():
                st.markdown(f"**{level.replace('_', ' ').title()}:** {action}")
            
            st.markdown("### 🤝 Inclusion Practices")
            inclusion = comm_protocol.get("inclusion_practices", {})
            for practice, description in inclusion.items():
                st.markdown(f"**{practice.replace('_', ' ').title()}:** {description}")
        
        # TAB 4: FULL REPORT
        with tab4:
            st.subheader("📄 Complete Recommendations Report")
            st.json(detailed_recs)
            
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json.dumps(detailed_recs, indent=4),
                file_name="team_recommendations.json",
                mime="application/json"
            )
    
    except FileNotFoundError:
        st.error("⚠️ Recommendations not generated yet!")
        st.info("Run the following command to generate recommendations:")
        st.code("python src/analysis/generate_recommendations.py")


#--------------------------------------------
#real life data (whatsapp + dc)
#---------------------------------------------
#--------------------------------------------
#real life data (whatsapp + dc)
#---------------------------------------------
elif page == "⭐ Real time implementation!":
    st.title("📱 Real-World Collaboration Analysis (WhatsApp + Discord)")

    # Load combined real dataset
    real = pd.read_csv("data/real/combined_real_interactions.csv")
    real["timestamp"] = pd.to_datetime(real["timestamp"])

    sub = st.radio("Choose a view:", [
        "📊 Team Performance Metrics",
        "🔥 Contribution Analysis",
        "🕸️ Real Network Graph",
        "🔗 Strongest & Weakest Ties",
        "🧠 Real Pattern Detection",
        "💡 Detailed Recommendations",
        "⚖️ Compare Synthetic vs Real",
        "📄 Raw Real-Life Data"
    ])

    # ------------------ 1. TEAM PERFORMANCE METRICS ------------------
    if sub == "📊 Team Performance Metrics":
        st.header("📊 Team Performance Metrics")
        
        # Basic counts
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Interactions", len(real))
        col2.metric("Team Members", real["source"].nunique())
        col3.metric("Mentions", (real["interaction_type"] == "mention").sum())
        
        # Activity distribution
        st.subheader("Activity Distribution")
        counts = real["source"].value_counts()
        total_messages = counts.sum()
        
        activity_df = pd.DataFrame({
            "Member": counts.index,
            "Messages": counts.values,
            "Percentage": (counts.values / total_messages * 100).round(1)
        })
        st.dataframe(activity_df, use_container_width=True)
        
        # Communication balance
        st.subheader("Communication Balance")
        st.bar_chart(counts)
        
        # Mention analysis
        if (real["interaction_type"] == "mention").sum() > 0:
            st.subheader("Mention Patterns")
            mentions = real[real["interaction_type"] == "mention"]
            mention_counts = mentions.groupby(["source", "target"]).size().reset_index(name="count")
            mention_counts = mention_counts.sort_values("count", ascending=False)
            st.dataframe(mention_counts.head(10), use_container_width=True)
        
        # Response patterns
        st.subheader("Response Patterns")
        response_data = []
        for member in real["source"].unique():
            sent = len(real[real["source"] == member])
            received = len(real[real["target"] == member])
            response_data.append({
                "Member": member,
                "Sent": sent,
                "Received": received,
                "Ratio": f"{sent}:{received}" if received > 0 else f"{sent}:0"
            })
        response_df = pd.DataFrame(response_data)
        st.dataframe(response_df, use_container_width=True)

    # ------------------ 2. CONTRIBUTION ANALYSIS ------------------
    elif sub == "🔥 Contribution Analysis":
        st.header("🔥 Real Contribution Levels")

        counts = real["source"].value_counts()

        st.bar_chart(counts)

        st.write("### Lowest Contributor")
        st.error(counts.sort_values().idxmin())

        st.write("### Highest Contributor")
        st.success(counts.idxmax())

    # ------------------ 3. NETWORK GRAPH ------------------
 # ------------------ 3. NETWORK GRAPH ------------------
    elif sub == "🕸️ Real Network Graph":
        st.header("🕸️ Real Interaction Network")

        # Build graph
        G = nx.DiGraph()

        # Calculate statistics
        activity = real["source"].value_counts()
        received = real[real["target"].notna()].groupby("target").size()

        # Add nodes
        for name in activity.index:
            sent = activity.get(name, 0)
            recv = received.get(name, 0)
            
            tooltip = f"{name}\nSent: {sent}\nReceived: {recv}"
            
            G.add_node(
                name,
                size=20 + sent * 1.5,
                color="#3498db",
                title=tooltip,
                label=name
            )

        # Add edges with detailed info
        edge_freq = real[real["target"].notna()].groupby(["source", "target"])
        
        for (src, tgt), group in edge_freq:
            count = len(group)
            
            # Count interaction types
            types = group["interaction_type"].value_counts()
            type_str = ", ".join([f"{t}: {c}" for t, c in types.items()])
            
            # Build tooltip
            edge_tooltip = f"{src} → {tgt}\nTotal: {count} messages\n{type_str}"
            
            G.add_edge(
                src, 
                tgt, 
                value=1,
                width=1,
                title=edge_tooltip,
                color="rgba(100,100,100,0.4)"
            )

        # Create network
        net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff")
        net.from_nx(G)

        # Static layout with visible arrows
        net.set_options("""
        {
            "physics": {
                "enabled": false
            },
            "nodes": {
                "shape": "dot",
                "font": {"size": 14}
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.6
                    }
                },
                "width": 1,
                "smooth": {
                    "enabled": true,
                    "type": "curvedCW",
                    "roundness": 0.2
                }
            }
        }
        """)

        net.save_graph("real_network.html")
        
        with open("real_network.html", "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=750, scrolling=True)

    # ------------------ 4. STRONGEST / WEAKEST TIES ------------------
    elif sub == "🔗 Strongest & Weakest Ties":
        st.header("🔗 Strongest & Weakest Collaboration")

        grouped = real[real["target"] != ""].groupby(["source", "target"]).size()

        st.subheader("Strongest Pairs")
        st.table(grouped.sort_values(ascending=False).head(5))

        st.subheader("Weakest Pairs")
        st.table(grouped.sort_values().head(5))

    # ------------------ 5. PATTERN DETECTION ------------------
    elif sub == "🧠 Real Pattern Detection":
        st.header("🧠 Real Collaboration Patterns")

        counts = real["source"].value_counts()

        isolated = counts[counts == 0]
        passive = counts[counts < counts.mean()]
        dominant = counts[counts > counts.mean() + counts.std()]

        st.subheader("Dominant Members")
        st.success(list(dominant.index))

        st.subheader("Passive Members")
        st.warning(list(passive.index))

        st.subheader("Isolated Members (no outgoing)")
        st.error(list(isolated.index) if len(isolated) else "None")

    # ------------------ 6. DETAILED RECOMMENDATIONS ------------------
    elif sub == "💡 Detailed Recommendations":
        st.header("💡 Detailed Situational Analysis")
        
        # Calculate metrics
        counts = real["source"].value_counts()
        total_messages = counts.sum()
        mentions = real[real["interaction_type"] == "mention"]
        
        # Per-member analysis
        for member in counts.index:
            with st.expander(f"📊 {member} - Detailed Analysis"):
                member_messages = counts[member]
                member_percentage = (member_messages / total_messages * 100)
                
                st.markdown(f"### Current Situation")
                
                # Communication burden
                st.markdown(f"**Communication Load:** You are carrying {member_percentage:.1f}% of team communication burden.")
                
                # Mention analysis
                member_mentions = mentions[mentions["source"] == member]
                if len(member_mentions) > 0:
                    mention_targets = member_mentions["target"].value_counts()
                    
                    for target, mention_count in mention_targets.items():
                        # Count responses from that target
                        responses = len(real[(real["source"] == target) & (real["target"] == member)])
                        
                        st.markdown(f"**Interaction with {target}:** You mentioned {target} {mention_count} times with {responses} responses - {'indicating good responsiveness' if responses >= mention_count * 0.5 else 'indicating a responsiveness issue'}.")
                
                # Activity level
                messages_sent = counts[member]
                messages_received = len(real[real["target"] == member])
                
                st.markdown(f"**Activity Balance:** {messages_sent} messages sent vs {messages_received} received.")
                
                if messages_sent > messages_received * 2:
                    st.markdown("**Pattern:** You initiate significantly more conversations than you receive - showing strong leadership but possible communication imbalance.")
                elif messages_received > messages_sent * 2:
                    st.markdown("**Pattern:** You respond more than you initiate - showing reactive rather than proactive engagement.")
                else:
                    st.markdown("**Pattern:** Balanced give-and-take communication style.")
                
                # Relative contribution
                mean_contribution = counts.mean()
                if member_messages > mean_contribution * 1.5:
                    st.markdown("**Team Impact:** Significantly above average contribution - you are a key driver of team communication.")
                elif member_messages < mean_contribution * 0.5:
                    st.markdown("**Team Impact:** Below average contribution - there may be opportunities to engage more with the team.")
                else:
                    st.markdown("**Team Impact:** Contributing at a healthy, sustainable level relative to team average.")

    # ------------------ 7. SYNTHETIC VS REAL ------------------
    elif sub == "⚖️ Compare Synthetic vs Real":
        st.header("⚖️ Synthetic vs Real Team Comparison")

        real_counts = real["source"].value_counts()
        syn_counts = clean_interactions["source"].value_counts()

        compare = pd.DataFrame({
            "Real": real_counts,
            "Synthetic": syn_counts
        }).fillna(0)

        st.bar_chart(compare)

        st.write("### Observations:")
        st.write("- Real data has fewer total interactions but stronger patterns.")
        st.write("- Synthetic data is denser and more balanced.")

    # ------------------ 8. RAW DATA ------------------
    elif sub == "📄 Raw Real-Life Data":
        st.header("📄 Complete Real Interaction Logs")
        st.dataframe(real, height=500)