# src/analysis/generate_recommendations.py
"""
Recommendation Engine for Team Collaboration Improvement

This module analyzes detected patterns and generates actionable,
specific recommendations for educators and project managers.

Features:
- Member-specific interventions
- Team-level strategies
- Priority scoring
- Communication protocol suggestions
- Timeline-based action plans
"""

import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

PROCESSED_DIR = "data/processed"
METRICS_DIR = os.path.join(PROCESSED_DIR, "metrics")
OUTPUT_DIR = "data/recommendations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# LOAD DATA
# --------------------------
node_metrics = pd.read_csv(os.path.join(METRICS_DIR, "node_metrics.csv"))
edge_metrics = pd.read_csv(os.path.join(METRICS_DIR, "edge_metrics.csv"))
members = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_members.csv"))

with open(os.path.join(METRICS_DIR, "patterns.json")) as f:
    patterns = json.load(f)

with open(os.path.join(METRICS_DIR, "team_metrics.json")) as f:
    team_metrics = json.load(f)

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def get_member_role(member_id):
    """Get role for a member"""
    role_row = members[members.member_id == member_id]
    if not role_row.empty:
        return role_row.iloc[0]["role"]
    return "unknown"

def get_member_name(member_id):
    """Get name for a member"""
    name_row = members[members.member_id == member_id]
    if not name_row.empty:
        return name_row.iloc[0]["name"]
    return member_id

def get_member_metrics(member_id):
    """Get all metrics for a member"""
    metrics_row = node_metrics[node_metrics.member_id == member_id]
    if not metrics_row.empty:
        return metrics_row.iloc[0].to_dict()
    return {}

# --------------------------
# RECOMMENDATION GENERATORS
# --------------------------

def recommend_for_isolated_members(isolated):
    """Generate recommendations for isolated team members"""
    recommendations = []
    
    for member_id in isolated:
        role = get_member_role(member_id)
        name = get_member_name(member_id)
        
        rec = {
            "member_id": member_id,
            "member_name": name,
            "issue": "Isolated - No Interactions",
            "severity": "CRITICAL",
            "priority": 1,
            "recommendations": [
                f"**Immediate Action:** Schedule 1-on-1 check-in with {name} to understand barriers to participation",
                f"**Pair Assignment:** Assign {name} as a collaborator on a task with the most active team member",
                f"**Onboarding Review:** Verify {name} has access to all communication platforms (Slack, GitHub, etc.)",
                "**Buddy System:** Designate a team mentor to actively engage with this member",
                "**Task Allocation:** Assign a small, achievable task to build confidence and encourage interaction"
            ],
            "expected_outcomes": [
                "First interaction within 2-3 days",
                "Regular participation within 1 week",
                "Integration into team workflow within 2 weeks"
            ],
            "metrics_to_track": [
                "Number of messages sent per day",
                "Task completion rate",
                "Response time to team communications"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_passive_members(passive):
    """Generate recommendations for passive team members"""
    recommendations = []
    
    for member_id in passive:
        role = get_member_role(member_id)
        name = get_member_name(member_id)
        metrics = get_member_metrics(member_id)
        
        # Determine specific issue
        issues = []
        if metrics.get("total_sent", 0) < metrics.get("total_received", 0) * 0.3:
            issues.append("Low outbound communication")
        if metrics.get("betweenness_centrality", 0) < 0.1:
            issues.append("Not involved in key discussions")
        if metrics.get("weighted_sent", 0) < node_metrics["weighted_sent"].mean() * 0.4:
            issues.append("Minimal contribution to important tasks")
        
        rec = {
            "member_id": member_id,
            "member_name": name,
            "role": role,
            "issue": f"Passive Participation - {', '.join(issues)}",
            "severity": "HIGH",
            "priority": 2,
            "current_metrics": {
                "messages_sent": int(metrics.get("total_sent", 0)),
                "messages_received": int(metrics.get("total_received", 0)),
                "activity_score": round(metrics.get("activity_score", 0), 2)
            },
            "recommendations": [
                f"**Engagement Strategy:** Directly tag {name} in discussions requiring their expertise",
                f"**Task Leadership:** Assign {name} as lead on a small sub-task to increase ownership",
                "**Check-in Protocol:** Weekly progress meetings to identify blockers",
                f"**Peer Pairing:** Partner {name} with a dominant contributor for knowledge transfer",
                "**Recognition:** Publicly acknowledge any contributions in team channels to build confidence",
                "**Barrier Analysis:** Survey to identify what's preventing active participation"
            ],
            "suggested_actions": {
                "this_week": [
                    f"Assign 1 specific task to {name} with clear deliverables",
                    f"Tag {name} in 2-3 relevant discussions",
                    "Send encouraging direct message acknowledging recent work"
                ],
                "next_2_weeks": [
                    "Increase task complexity gradually",
                    "Include in decision-making discussions",
                    "Monitor for improved interaction patterns"
                ]
            },
            "expected_outcomes": [
                "30% increase in outbound messages within 1 week",
                "Complete assigned tasks on time",
                "Begin initiating conversations within 2 weeks"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_dominant_members(dominant):
    """Generate recommendations for dominant team members"""
    recommendations = []
    
    for member_id in dominant:
        role = get_member_role(member_id)
        name = get_member_name(member_id)
        metrics = get_member_metrics(member_id)
        
        rec = {
            "member_id": member_id,
            "member_name": name,
            "role": role,
            "issue": "Over-Dominant - Potential Bottleneck",
            "severity": "MEDIUM",
            "priority": 3,
            "current_metrics": {
                "messages_sent": int(metrics.get("total_sent", 0)),
                "activity_score": round(metrics.get("activity_score", 0), 2),
                "centrality": round(metrics.get("degree_centrality", 0), 3)
            },
            "positive_notes": [
                f"{name} is highly engaged and contributes significantly",
                "Strong network centrality indicates key role in team coordination"
            ],
            "recommendations": [
                f"**Risk Mitigation:** {name} may be a single point of failure - distribute critical knowledge",
                f"**Delegation Training:** Coach {name} to delegate more tasks to passive members",
                "**Burnout Prevention:** Monitor workload to prevent exhaustion",
                f"**Mentorship Role:** Formalize {name} as mentor to passive team members",
                "**Documentation:** Ensure {name}'s knowledge is documented for team resilience",
                "**Balanced Participation:** Encourage pauses in discussions to let others contribute"
            ],
            "suggested_actions": {
                "immediate": [
                    f"Thank {name} for exceptional contributions",
                    "Assess current workload and identify tasks to redistribute",
                    "Schedule discussion about sustainability"
                ],
                "ongoing": [
                    "Pair with passive members for knowledge transfer",
                    "Create documentation of processes owned by this member",
                    "Monitor for signs of burnout"
                ]
            },
            "expected_outcomes": [
                "More balanced team participation",
                "Reduced dependency on single member",
                "Improved team resilience"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_weak_pairs(weak_pairs):
    """Generate recommendations for weak collaboration pairs"""
    recommendations = []
    
    # Group by source for clearer recommendations
    weak_by_source = defaultdict(list)
    for pair in weak_pairs:
        weak_by_source[pair["source"]].append(pair)
    
    for source, targets in weak_by_source.items():
        source_name = get_member_name(source)
        target_names = [get_member_name(t["target"]) for t in targets]
        
        rec = {
            "issue_type": "Weak Collaboration Links",
            "severity": "MEDIUM",
            "priority": 4,
            "pattern": f"{source_name} has weak connections with {len(targets)} team member(s)",
            "affected_pairs": [
                {
                    "from": source_name,
                    "to": get_member_name(t["target"]),
                    "weight": round(t["weight"], 2)
                }
                for t in targets
            ],
            "recommendations": [
                f"**Structured Collaboration:** Assign joint tasks requiring {source_name} to work with: {', '.join(target_names)}",
                f"**Ice Breaker Activities:** Facilitate informal team bonding sessions",
                "**Cross-Functional Projects:** Create opportunities for these members to collaborate",
                "**Communication Channels:** Ensure all members are active in shared communication spaces",
                "**Conflict Check:** Verify no interpersonal issues are causing avoidance"
            ],
            "suggested_activities": [
                "Pair programming sessions (if technical team)",
                "Joint presentation preparation",
                "Peer review assignments",
                "Shared responsibility for deliverable sections"
            ],
            "expected_outcomes": [
                "Increased interaction frequency between weak-link pairs",
                "More balanced communication network",
                "Improved team cohesion"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_strong_pairs(strong_pairs):
    """Generate insights on strong collaboration pairs"""
    recommendations = []
    
    if len(strong_pairs) > 0:
        rec = {
            "issue_type": "Strong Collaboration Pairs Identified",
            "severity": "INFO",
            "priority": 6,
            "pattern": f"{len(strong_pairs)} strong collaboration pair(s) detected",
            "strong_pairs": [
                {
                    "from": get_member_name(p["source"]),
                    "to": get_member_name(p["target"]),
                    "weight": round(p["weight"], 2)
                }
                for p in strong_pairs
            ],
            "positive_notes": [
                "Strong pairs indicate effective working relationships",
                "These partnerships can be leveraged for team success"
            ],
            "recommendations": [
                "**Best Practice Sharing:** Document what makes these collaborations successful",
                "**Mentorship Pairing:** Use strong pairs as models for weaker collaborators",
                "**Risk Awareness:** Ensure these pairs aren't becoming siloed from the rest of the team",
                "**Knowledge Distribution:** Rotate these members into different pairings occasionally",
                "**Celebration:** Recognize and celebrate effective collaboration publicly"
            ],
            "cautions": [
                "Monitor for potential clique formation excluding other members",
                "Ensure dependency doesn't create bottlenecks if one member is unavailable"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_subgroups(subgroups):
    """Generate recommendations for detected subgroups"""
    recommendations = []
    
    if len(subgroups) > 1:
        subgroup_details = []
        for idx, group in enumerate(subgroups, 1):
            names = [get_member_name(m) for m in group]
            subgroup_details.append({
                "subgroup_id": idx,
                "members": names,
                "size": len(names)
            })
        
        rec = {
            "issue_type": "Multiple Subgroups Detected",
            "severity": "MEDIUM",
            "priority": 4,
            "pattern": f"Team has fragmented into {len(subgroups)} subgroups",
            "subgroups": subgroup_details,
            "implications": [
                "Subgroups can lead to information silos",
                "Reduced cross-pollination of ideas",
                "Potential for misalignment on project goals"
            ],
            "recommendations": [
                "**Cross-Group Tasks:** Assign projects requiring collaboration across subgroups",
                "**Rotation Strategy:** Periodically rotate members between subgroups",
                "**All-Hands Meetings:** Regular full-team sync meetings to ensure alignment",
                "**Shared Goals:** Emphasize team-level objectives over subgroup achievements",
                "**Bridge Builders:** Identify and empower members who connect subgroups",
                "**Communication Audit:** Review if certain platforms favor specific subgroups"
            ],
            "suggested_activities": [
                "Team-building activities involving cross-subgroup mixing",
                "Shared deliverables requiring input from all subgroups",
                "Round-robin task assignments to break patterns"
            ],
            "expected_outcomes": [
                "Increased inter-subgroup communication",
                "More unified team identity",
                "Better information flow across the entire team"
            ]
        }
        
        recommendations.append(rec)
    
    elif len(subgroups) == 1:
        rec = {
            "issue_type": "Single Cohesive Group",
            "severity": "POSITIVE",
            "priority": 7,
            "pattern": "Team operates as one unified group",
            "positive_notes": [
                "Good team cohesion detected",
                "No concerning fragmentation",
                "Healthy communication flow"
            ],
            "recommendations": [
                "**Maintain Momentum:** Continue current collaboration practices",
                "**Scale Carefully:** Monitor cohesion if team size increases",
                "**Document Success:** Record what practices are working well"
            ]
        }
        
        recommendations.append(rec)
    
    return recommendations


def recommend_for_role_mismatch(role_mismatch, members_df):
    """Generate recommendations for role mismatches"""
    recommendations = []
    
    if len(role_mismatch) > 0:
        leader_row = members_df[members_df.role == "leader"]
        if not leader_row.empty:
            leader_id = leader_row.iloc[0]["member_id"]
            leader_name = leader_row.iloc[0]["name"]
            
            rec = {
                "issue_type": "Leader Performance Issues",
                "severity": "HIGH",
                "priority": 2,
                "pattern": f"Leader ({leader_name}) showing concerning patterns",
                "specific_issues": role_mismatch,
                "recommendations": [
                    f"**Leadership Support:** Provide additional resources or training for {leader_name}",
                    "**Expectation Alignment:** Clarify leadership responsibilities and expectations",
                    "**Co-Leadership:** Consider adding a deputy or co-leader to share responsibilities",
                    "**Skills Assessment:** Identify if leader needs specific skill development",
                    "**Team Feedback:** Gather anonymous feedback on leadership effectiveness",
                    "**Coaching:** Provide executive coaching or mentorship for the leader"
                ],
                "immediate_actions": [
                    "1-on-1 discussion with leader about challenges",
                    "Assess if leader is overloaded with technical vs. leadership tasks",
                    "Review team dynamics and potential conflicts"
                ],
                "expected_outcomes": [
                    "Improved leader visibility and engagement",
                    "Better team coordination",
                    "Clearer direction and decision-making"
                ]
            }
            
            recommendations.append(rec)
    
    return recommendations


def generate_team_level_recommendations(team_metrics, node_metrics_df):
    """Generate overall team-level recommendations"""
    recommendations = []
    
    density = team_metrics.get("density", 0)
    reciprocity = team_metrics.get("reciprocity", 0)
    avg_clustering = team_metrics.get("average_clustering", 0)
    
    # Density analysis
    if density < 0.5:
        rec = {
            "issue_type": "Low Team Density",
            "severity": "MEDIUM",
            "priority": 5,
            "current_value": round(density, 3),
            "interpretation": "Team has sparse interactions - many potential connections are unused",
            "recommendations": [
                "**Increase Touchpoints:** Schedule more frequent team check-ins",
                "**Collaboration Tools:** Ensure team is using shared platforms effectively",
                "**Task Interdependence:** Design tasks that require cross-member collaboration",
                "**Team Events:** Regular team-building activities to increase connections"
            ],
            "target": "Increase density to > 0.6 within 2-3 weeks"
        }
        recommendations.append(rec)
    
    # Reciprocity analysis
    if reciprocity < 0.4:
        rec = {
            "issue_type": "Low Reciprocity",
            "severity": "MEDIUM",
            "priority": 5,
            "current_value": round(reciprocity, 3),
            "interpretation": "Communication is mostly one-way - team members aren't responding to each other",
            "recommendations": [
                "**Response Culture:** Establish norm that messages should be acknowledged",
                "**Discussion Threads:** Encourage threaded conversations rather than broadcasts",
                "**Question-Asking:** Train team to ask clarifying questions",
                "**Feedback Loops:** Create structured opportunities for back-and-forth dialogue"
            ],
            "target": "Increase reciprocity to > 0.5 within 2 weeks"
        }
        recommendations.append(rec)
    
    # Activity distribution analysis
    activity_std = node_metrics_df["activity_score"].std()
    activity_mean = node_metrics_df["activity_score"].mean()
    cv = activity_std / activity_mean if activity_mean > 0 else 0
    
    if cv > 0.6:  # High coefficient of variation
        rec = {
            "issue_type": "Unbalanced Participation",
            "severity": "HIGH",
            "priority": 3,
            "interpretation": "Large disparity in contribution levels across team members",
            "statistics": {
                "mean_activity": round(activity_mean, 2),
                "std_deviation": round(activity_std, 2),
                "coefficient_variation": round(cv, 2)
            },
            "recommendations": [
                "**Load Balancing:** Redistribute tasks from high to low contributors",
                "**Skill Development:** Train less active members to build confidence",
                "**Participation Equity:** Set minimum participation expectations",
                "**Rotating Roles:** Implement rotating responsibilities to spread engagement"
            ],
            "expected_outcomes": [
                "More balanced activity scores across team",
                "Reduced coefficient of variation to < 0.5",
                "Improved team satisfaction"
            ]
        }
        recommendations.append(rec)
    
    return recommendations


def generate_communication_protocol():
    """Generate communication protocol template"""
    protocol = {
        "title": "Recommended Team Communication Protocol",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "daily_practices": {
            "morning_standup": {
                "frequency": "Daily",
                "duration": "15 minutes",
                "format": "Each member shares: (1) Yesterday's progress, (2) Today's plan, (3) Blockers",
                "platform": "Video call or Slack"
            },
            "active_hours": {
                "recommended": "9 AM - 6 PM team timezone",
                "expectation": "Respond to urgent messages within 2 hours during active hours"
            },
            "status_updates": {
                "frequency": "End of day",
                "format": "Brief message in team channel summarizing accomplishments"
            }
        },
        "weekly_practices": {
            "team_sync": {
                "frequency": "Weekly",
                "duration": "30-45 minutes",
                "agenda": [
                    "Review progress toward milestones",
                    "Discuss blockers and solutions",
                    "Plan next week's priorities",
                    "Celebrate wins"
                ]
            },
            "retrospective": {
                "frequency": "Every 2 weeks",
                "format": "What went well, What didn't, What to improve"
            }
        },
        "platform_usage_guidelines": {
            "slack_teams": {
                "use_for": ["Quick questions", "Informal discussion", "Daily updates"],
                "response_time": "2 hours during work hours"
            },
            "github": {
                "use_for": ["Code review", "Technical discussions", "Bug reports"],
                "response_time": "24 hours"
            },
            "trello_asana": {
                "use_for": ["Task tracking", "Milestone planning", "Assignment clarity"],
                "update_frequency": "Daily"
            },
            "email": {
                "use_for": ["Formal communications", "Stakeholder updates", "Documentation"],
                "response_time": "48 hours"
            }
        },
        "escalation_protocol": {
            "level_1": "Direct message to relevant team member",
            "level_2": "Tag team lead in team channel",
            "level_3": "Escalate to project manager/educator"
        },
        "inclusion_practices": {
            "tagging": "Tag specific members when their input is needed",
            "time_zones": "Be mindful of time zones when scheduling",
            "accessibility": "Provide written summaries of video meetings",
            "quiet_voices": "Actively solicit input from quieter team members"
        }
    }
    
    return protocol


def generate_action_plan(all_recommendations):
    """Generate prioritized action plan with timeline"""
    
    # Sort by priority
    sorted_recs = sorted(
        [r for r in all_recommendations if "priority" in r],
        key=lambda x: x.get("priority", 99)
    )
    
    action_plan = {
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_issues": len(sorted_recs),
            "critical": len([r for r in sorted_recs if r.get("severity") == "CRITICAL"]),
            "high": len([r for r in sorted_recs if r.get("severity") == "HIGH"]),
            "medium": len([r for r in sorted_recs if r.get("severity") == "MEDIUM"]),
            "info": len([r for r in sorted_recs if r.get("severity") in ["INFO", "POSITIVE"]])
        },
        "immediate_actions_this_week": [],
        "short_term_1_2_weeks": [],
        "medium_term_3_4_weeks": [],
        "ongoing_practices": []
    }
    
    # Categorize by urgency
    for rec in sorted_recs:
        severity = rec.get("severity", "MEDIUM")
        issue_type = rec.get("issue_type", rec.get("issue", "Unknown"))
        
        if severity == "CRITICAL":
            action_plan["immediate_actions_this_week"].append({
                "issue": issue_type,
                "top_action": rec.get("recommendations", ["Review recommendations"])[0] if rec.get("recommendations") else "Take action"
            })
        elif severity == "HIGH":
            action_plan["short_term_1_2_weeks"].append({
                "issue": issue_type,
                "top_action": rec.get("recommendations", ["Review recommendations"])[0] if rec.get("recommendations") else "Take action"
            })
        elif severity == "MEDIUM":
            action_plan["medium_term_3_4_weeks"].append({
                "issue": issue_type,
                "top_action": rec.get("recommendations", ["Review recommendations"])[0] if rec.get("recommendations") else "Take action"
            })
    
    # Add ongoing practices
    action_plan["ongoing_practices"] = [
        "Monitor team metrics weekly using the dashboard",
        "Conduct bi-weekly retrospectives",
        "Track individual activity scores",
        "Review network graphs for emerging patterns",
        "Gather team feedback monthly"
    ]
    
    return action_plan


# --------------------------
# MAIN EXECUTION
# --------------------------

def generate_all_recommendations():
    """Generate comprehensive recommendations"""
    
    all_recommendations = []
    
    print("=" * 60)
    print("GENERATING COMPREHENSIVE RECOMMENDATIONS")
    print("=" * 60)
    
    # 1. Isolated members
    if patterns["isolated_members"]:
        print(f"\n✓ Processing {len(patterns['isolated_members'])} isolated member(s)...")
        recs = recommend_for_isolated_members(patterns["isolated_members"])
        all_recommendations.extend(recs)
    
    # 2. Passive members
    if patterns["passive_members"]:
        print(f"✓ Processing {len(patterns['passive_members'])} passive member(s)...")
        recs = recommend_for_passive_members(patterns["passive_members"])
        all_recommendations.extend(recs)
    
    # 3. Dominant members
    if patterns["dominant_members"]:
        print(f"✓ Processing {len(patterns['dominant_members'])} dominant member(s)...")
        recs = recommend_for_dominant_members(patterns["dominant_members"])
        all_recommendations.extend(recs)
    
    # 4. Weak pairs
    if patterns["weak_pairs"]:
        print(f"✓ Processing {len(patterns['weak_pairs'])} weak collaboration pair(s)...")
        recs = recommend_for_weak_pairs(patterns["weak_pairs"])
        all_recommendations.extend(recs)
    
    # 5. Strong pairs
    if patterns["strong_pairs"]:
        print(f"✓ Processing {len(patterns['strong_pairs'])} strong collaboration pair(s)...")
        recs = recommend_for_strong_pairs(patterns["strong_pairs"])
        all_recommendations.extend(recs)
    
    # 6. Subgroups
    print(f"✓ Processing {len(patterns['subgroups'])} subgroup(s)...")
    recs = recommend_for_subgroups(patterns["subgroups"])
    all_recommendations.extend(recs)
    
    # 7. Role mismatch
    if patterns["role_mismatch"]:
        print(f"✓ Processing role mismatch issues...")
        recs = recommend_for_role_mismatch(patterns["role_mismatch"], members)
        all_recommendations.extend(recs)
    
    # 8. Team-level
    print("✓ Processing team-level metrics...")
    recs = generate_team_level_recommendations(team_metrics, node_metrics)
    all_recommendations.extend(recs)
    
    # --------------------------
    # SAVE OUTPUTS
    # --------------------------
    
    # Save detailed recommendations
    output_file = os.path.join(OUTPUT_DIR, "detailed_recommendations.json")
    with open(output_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "team_id": "T01",
            "total_recommendations": len(all_recommendations),
            "recommendations": all_recommendations
        }, f, indent=4)
    print(f"\n✓ Saved detailed recommendations: {output_file}")
    
    # Generate communication protocol
    protocol = generate_communication_protocol()
    protocol_file = os.path.join(OUTPUT_DIR, "communication_protocol.json")
    with open(protocol_file, "w") as f:
        json.dump(protocol, f, indent=4)
    print(f"✓ Saved communication protocol: {protocol_file}")
    
    # Generate action plan
    action_plan = generate_action_plan(all_recommendations)
    action_file = os.path.join(OUTPUT_DIR, "action_plan.json")
    with open(action_file, "w") as f:
        json.dump(action_plan, f, indent=4)
    print(f"✓ Saved action plan: {action_file}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal recommendations generated: {len(all_recommendations)}")
    print(f"\nOutput location: {OUTPUT_DIR}/")
    print("\nFiles created:")
    print("  1. detailed_recommendations.json - Full recommendations with context")
    print("  2. communication_protocol.json - Team communication guidelines")
    print("  3. action_plan.json - Prioritized timeline of actions")
    print("\n✅ Ready to integrate with Streamlit dashboard!")
    
    return all_recommendations, protocol, action_plan


if __name__ == "__main__":
    generate_all_recommendations()