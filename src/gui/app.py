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
elif page == "🔗 Pairwise Collaboration":
    st.title("🔗 Pairwise Collaboration")

    st.write("### Interaction Table (ALL 7000 events)")
    st.dataframe(clean_interactions, height=400)

    st.write("### Collaboration Heatmap (Counts)")
    matrix = pd.read_csv(os.path.join(PROCESSED_DIR, "interaction_matrix.csv"))
    st.dataframe(matrix)


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
elif page == "⭐ Real time implementation!":
    st.title("📱 Real-World Collaboration Analysis (WhatsApp + Discord)")

    # Load combined real dataset
    real = pd.read_csv("data/real/combined_real_interactions.csv")
    real["timestamp"] = pd.to_datetime(real["timestamp"])

    sub = st.radio("Choose a view:", [
        "📊 Overview Dashboard",
        "🔥 Contribution Analysis",
        "🕸️ Real Network Graph",
        "🔗 Strongest & Weakest Ties",
        "🧠 Real Pattern Detection",
        "💡 Professional Team Recommendations",
        "⚖️ Compare Synthetic vs Real",
        "📄 Raw Real-Life Data"
    ])

    # ------------------ 1. OVERVIEW ------------------
    if sub == "📊 Overview Dashboard":
        st.header("📊 Real Data Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Real Interactions", len(real))
        col2.metric("Unique Contributors", real["source"].nunique())
        col3.metric("Mentions", (real["interaction_type"] == "mention").sum())

    # ------------------ 2. CONTRIBUTION ------------------
    elif sub == "🔥 Contribution Analysis":
        st.header("🔥 Real Contribution Levels")

        counts = real["source"].value_counts()

        st.bar_chart(counts)

        st.write("### Lowest Contributor")
        st.error(counts.sort_values().idxmin())

        st.write("### Highest Contributor")
        st.success(counts.idxmax())

    # ------------------ 3. NETWORK GRAPH ------------------
    elif sub == "🕸️ Real Network Graph":
        st.header("🕸️ Real Interaction Network (Raw Interactions Like Synthetic)")

        # Use ALL real edges (raw data)
        real_edges = real.copy()

        # Build graph
        G = nx.DiGraph()

        # Node activity = # messages sent
        activity = real_edges["source"].value_counts()

        # Add nodes with good scaling
        for name in activity.index:
            G.add_node(
                name,
                size=20 + activity[name] * 1.5,      # gentle scaling
                color="#90caf9",                     # soft blue
                title=f"{name}<br>Total Sent: {activity[name]}"
            )

        # Edge weighting = frequency of interactions between pairs
        edge_freq = real_edges.groupby(["source", "target"]).size()

        for (src, tgt), w in edge_freq.items():
            if pd.isna(tgt) or tgt == "":
                continue

            # build tooltip showing details
            info_rows = real_edges[(real_edges["source"] == src) & (real_edges["target"] == tgt)]
            tooltip = "<br>".join([
                f"{row['interaction_type']} on {row['platform']} — {row['timestamp']}"
                for _, row in info_rows.iterrows()
            ])

            G.add_edge(
                src,
                tgt,
                value=w,  # thickness
                title=tooltip,
                color="rgba(80,80,80,0.5)"
            )

        # Create PyVis Network
        net = Network(
            height="650px",
            width="100%",
            directed=True,
            bgcolor="#FFFFFF",
            font_color="#000000"
        )

        net.from_nx(G)

        # Physics like synthetic graph (stable & nice)
        net.set_options("""
        {
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -2500,
            "centralGravity": 0.2,
            "springLength": 150,
            "springConstant": 0.04,
            "damping": 0.09
            },
            "minVelocity": 0.75
        },
        "edges": {
            "smooth": {"enabled": true, "type": "dynamic"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "nodes": {
            "shape": "dot",
            "font": {"size": 16},
            "borderWidth": 1
        }
        }
        """)

        net.save_graph("real_network.html")

        with open("real_network.html", "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=700, scrolling=True)




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

    # ------------------ 6. PROFESSIONAL WORK RECOMMENDATIONS ------------------
    elif sub == "💡 Professional Team Recommendations":
        st.header("💡 Professional Work Project Recommendations")
        st.subheader("Team: Sanjana (Lead), Tvisha (Active), Vikram (Unresponsive)")
        
        try:
            import os
            import json
            
            REC_DIR = "data/recommendations/real"
            
            with open(os.path.join(REC_DIR, "professional_recommendations.json")) as f:
                prof_recs = json.load(f)
            
            # Summary
            st.subheader("📊 Analysis Summary")
            summary = prof_recs["analysis_summary"]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", summary['total_interactions'])
            col2.metric("Duration", f"{summary['duration_minutes']} min")
            col3.metric("Issues Found", summary['critical_issues_found'])
            col4.metric("Severity", summary['severity'])
            
            # Key Metrics
            st.markdown("---")
            st.subheader("👥 Team Performance Metrics")
            
            metrics = prof_recs["key_metrics"]
            
            for name, data in metrics.items():
                status_emoji = "🚨" if "CRITICAL" in data['status'] else "⚠️" if "Overburdened" in data['status'] else "✅"
                
                with st.expander(f"{status_emoji} {name} - {data['role']}", expanded=("CRITICAL" in data['status'])):
                    st.markdown(f"**Status:** {data['status']}")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Messages Sent", data['messages_sent'])
                    col2.metric("Workload %", f"{data['workload_percentage']}%")
                    
                    if 'response_ratio' in data:
                        st.error(f"⚠️ Response Ratio: {data['response_ratio']} (Healthy: >0.7)")
            
            # Detailed Recommendations
            st.markdown("---")
            st.subheader("💡 Detailed Recommendations")
            
            recs = prof_recs["recommendations"]
            
            # Sanjana (Team Lead)
            if "Sanjana_Team_Lead" in recs:
                sanjana_rec = recs["Sanjana_Team_Lead"]
                
                with st.expander("🔥 FOR SANJANA (TEAM LEAD) - CRITICAL ACTIONS", expanded=True):
                    st.warning(f"**Current Situation:** {sanjana_rec['current_situation']}")
                    
                    st.markdown("### Immediate Actions Required:")
                    for action in sanjana_rec['immediate_actions']:
                        priority_color = "🚨" if action['priority'] == "CRITICAL" else "⚠️" if action['priority'] == "HIGH" else "ℹ️"
                        st.markdown(f"**{priority_color} {action['priority']}: {action['action']}**")
                        st.markdown(f"*Rationale:* {action['rationale']}")
                        st.markdown("*Steps:*")
                        for step in action['specific_steps']:
                            st.markdown(f"  - {step}")
                        st.markdown("")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ❌ Stop Doing:")
                        for item in sanjana_rec['what_to_stop_doing']:
                            st.markdown(f"- {item}")
                    
                    with col2:
                        st.markdown("### ✅ Start Doing:")
                        for item in sanjana_rec['what_to_start_doing']:
                            st.markdown(f"- {item}")
                    
                    st.info(f"**Leadership Feedback:** {sanjana_rec['leadership_feedback']}")
            
            # Tvisha
            if "Tvisha_Active_Collaborator" in recs:
                tvisha_rec = recs["Tvisha_Active_Collaborator"]
                
                with st.expander("✅ FOR TVISHA (ACTIVE MEMBER)"):
                    st.success(f"**Current Situation:** {tvisha_rec['current_situation']}")
                    
                    st.markdown("### Strengths:")
                    for strength in tvisha_rec['strengths']:
                        st.markdown(f"- {strength}")
                    
                    st.markdown("### Recommendations:")
                    for rec in tvisha_rec['recommendations']:
                        st.markdown(f"**{rec['priority']} Priority: {rec['action']}**")
                        st.markdown(f"*{rec['rationale']}*")
                        for step in rec['specific_steps']:
                            st.markdown(f"  - {step}")
                    
                    st.info(f"**Peer Feedback:** {tvisha_rec['peer_feedback']}")
            
            # Vikram
            if "Vikram_Unresponsive_Member" in recs:
                vikram_rec = recs["Vikram_Unresponsive_Member"]
                
                with st.expander("🚨 FOR VIKRAM (PROBLEMATIC MEMBER) - CRITICAL", expanded=True):
                    st.error(f"**{vikram_rec['severity']}**")
                    st.error(f"**Current Situation:** {vikram_rec['current_situation']}")
                    
                    st.markdown("### Issues Identified:")
                    for issue in vikram_rec['issues_identified']:
                        st.markdown(f"- {issue}")
                    
                    st.markdown("### Immediate Requirements:")
                    for req in vikram_rec['immediate_requirements']:
                        st.markdown(f"**Requirement:** {req['requirement']}")
                        st.markdown(f"**Consequence:** {req['consequence']}")
                        st.markdown(f"**Action:** {req['action']}")
                        st.markdown("")
                    
                    st.markdown("### Performance Improvement Plan:")
                    pip = vikram_rec['performance_improvement_plan']
                    st.markdown(f"**Week 1:** {', '.join(pip['week_1'])}")
                    st.markdown(f"**Week 2:** {', '.join(pip['week_2'])}")
                    st.markdown(f"**Success Criteria:** {pip['success_criteria']}")
                    st.markdown(f"**Failure Outcome:** {pip['failure_outcome']}")
                    
                    st.error(f"**Direct Feedback:** {vikram_rec['direct_feedback']}")
            
            # Management Actions
            st.markdown("---")
            st.subheader("🎓 For Supervisor/Professor")
            
            if "management_actions" in prof_recs:
                mgmt = prof_recs["management_actions"]["for_supervisor_professor"]
                
                st.info(f"**Situation:** {mgmt['situation']}")
                
                st.markdown("### Immediate Actions (48-72 hours):")
                for action in mgmt['immediate_actions']:
                    with st.expander(f"{action['action']} - {action['timeline']}"):
                        st.markdown(f"**Purpose:** {action['purpose']}")
                        if 'questions_to_ask' in action:
                            st.markdown("**Questions to Ask:**")
                            for q in action['questions_to_ask']:
                                st.markdown(f"  - {q}")
                        if 'talking_points' in action:
                            st.markdown("**Talking Points:**")
                            for tp in action['talking_points']:
                                st.markdown(f"  - {tp}")
                        if 'requirements' in action:
                            st.markdown("**Requirements:**")
                            for req in action['requirements']:
                                st.markdown(f"  - {req}")
            
            # Download
            st.markdown("---")
            st.download_button(
                label="📥 Download Professional Analysis Report",
                data=json.dumps(prof_recs, indent=4),
                file_name="professional_work_analysis.json",
                mime="application/json"
            )
        
        except FileNotFoundError:
            st.error("⚠️ Professional recommendations not generated yet!")
            st.info("Run the following command to generate:")
            st.code("python src/analysis/generate_recommendations_real_professional.py")
        except Exception as e:
            st.error(f"Error loading recommendations: {str(e)}")

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
        st.write("- Real data clearly shows Sanjana > Tvisha > Vikram.")

    # ------------------ 7. RAW DATA ------------------
    elif sub == "📄 Raw Real-Life Data":
        st.header("📄 Complete Real Interaction Logs")
        st.dataframe(real, height=500)


# --------------------------------------------
# PAGE 6: PATTERNS + INSIGHTS
# --------------------------------------------
elif page == "🧠 Patterns & Insights":
    st.title("🧠 Pattern Detection & Insights")

    st.subheader("Isolated Members")
    st.write(patterns["isolated_members"])

    st.subheader("Passive Members")
    st.write(patterns["passive_members"])

    st.subheader("Dominant Members")
    st.write(patterns["dominant_members"])

    st.subheader("Strong Collaboration Pairs")
    st.write(patterns["strong_pairs"])

    st.subheader("Weak Collaboration Pairs")
    st.write(patterns["weak_pairs"])

    st.subheader("Subgroups")
    st.write(patterns["subgroups"])

    st.subheader("Role Mismatch")
    st.error(patterns["role_mismatch"])


# --------------------------------------------
# PAGE 7: RECOMMENDATIONS & ACTION PLAN
# --------------------------------------------
elif page == "💡 Recommendations & Action Plan":
    st.title("💡 AI-Powered Recommendations & Action Plan")
    
    st.markdown("""
    This page provides **actionable, data-driven recommendations** to improve team collaboration 
    based on detected patterns. Use this to make informed interventions and track progress.
    """)
    
    # Check if recommendations exist
    rec_file = "data/recommendations/detailed_recommendations.json"
    action_file = "data/recommendations/action_plan.json"
    protocol_file = "data/recommendations/communication_protocol.json"
    
    if not os.path.exists(rec_file):
        st.warning("⚠️ Recommendations not yet generated!")
        st.info("**To generate recommendations, run:** `python src/analysis/generate_recommendations.py`")
        
        if st.button("🚀 Generate Recommendations Now"):
            with st.spinner("Analyzing patterns and generating recommendations..."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/analysis/generate_recommendations.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("✅ Recommendations generated successfully!")
                    st.rerun()
                else:
                    st.error(f"Error: {result.stderr}")
    else:
        # Load recommendations
        with open(rec_file) as f:
            recommendations_data = json.load(f)
        
        with open(action_file) as f:
            action_plan_data = json.load(f)
        
        with open(protocol_file) as f:
            protocol_data = json.load(f)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Action Plan",
            "💡 Detailed Recommendations", 
            "📞 Communication Protocol",
            "📊 Summary Stats"
        ])
        
        # TAB 1: ACTION PLAN
        with tab1:
            st.header("📋 Prioritized Action Plan")
            st.caption(f"Generated: {action_plan_data.get('generated_date', 'N/A')}")
            
            # Summary metrics
            summary = action_plan_data.get("summary", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Issues", summary.get("total_issues", 0))
            col2.metric("🔴 Critical", summary.get("critical", 0))
            col3.metric("🟠 High", summary.get("high", 0))
            col4.metric("🟡 Medium", summary.get("medium", 0))
            
            st.divider()
            
            # Immediate actions
            st.subheader("🚨 Immediate Actions (This Week)")
            immediate = action_plan_data.get("immediate_actions_this_week", [])
            if immediate:
                for idx, action in enumerate(immediate, 1):
                    with st.expander(f"{idx}. {action['issue']}", expanded=True):
                        st.markdown(action['top_action'])
            else:
                st.success("✅ No critical issues requiring immediate action!")
            
            # Short term
            st.subheader("📅 Short-Term Actions (1-2 Weeks)")
            short_term = action_plan_data.get("short_term_1_2_weeks", [])
            if short_term:
                for idx, action in enumerate(short_term, 1):
                    with st.expander(f"{idx}. {action['issue']}"):
                        st.markdown(action['top_action'])
            else:
                st.info("No high-priority actions scheduled.")
            
            # Medium term
            st.subheader("📆 Medium-Term Actions (3-4 Weeks)")
            medium_term = action_plan_data.get("medium_term_3_4_weeks", [])
            if medium_term:
                for idx, action in enumerate(medium_term, 1):
                    with st.expander(f"{idx}. {action['issue']}"):
                        st.markdown(action['top_action'])
            else:
                st.info("No medium-term actions scheduled.")
            
            # Ongoing practices
            st.subheader("🔄 Ongoing Practices")
            ongoing = action_plan_data.get("ongoing_practices", [])
            for practice in ongoing:
                st.markdown(f"- {practice}")