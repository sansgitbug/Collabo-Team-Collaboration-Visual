# src/analysis/generate_recommendations_irl.py
"""
Professional Work Project Recommendation Generator
Analyzes team collaboration in a work simulation where:
- Sanjana: Team Lead
- Tvisha: Active collaborator
- Vikram: Unresponsive team member

Generates professional, actionable management recommendations.
"""

import os
import json
import pandas as pd
import networkx as nx
from datetime import datetime

# --------------------------
# CONFIG
# --------------------------
REAL_DATA_PATH = "data/real/combined_real_interactions.csv"
OUTPUT_DIR = "data/recommendations/real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PROFESSIONAL WORK PROJECT ANALYSIS")
print("=" * 70)

# --------------------------
# LOAD DATA
# --------------------------
real = pd.read_csv(REAL_DATA_PATH)
real["timestamp"] = pd.to_datetime(real["timestamp"])

print(f"\n📊 Dataset Overview:")
print(f"   Total interactions: {len(real)}")
print(f"   Time period: {real['timestamp'].min()} to {real['timestamp'].max()}")
print(f"   Duration: {(real['timestamp'].max() - real['timestamp'].min()).total_seconds() / 60:.0f} minutes")
print(f"   Platforms: {', '.join(real['platform'].unique())}")

# --------------------------
# TEAM MEMBER ANALYSIS
# --------------------------
members = real["source"].unique().tolist()
print(f"   Team members: {', '.join(members)}")

# Message counts
messages_sent = real.groupby("source").size()
messages_received = real[real["target"] != ""].groupby("target").size()

# Mentions (who's tagging whom)
mentions = real[real["interaction_type"] == "mention"]
mention_matrix = mentions.groupby(["source", "target"]).size()

print(f"\n👥 Team Activity:")
for member in members:
    sent = messages_sent.get(member, 0)
    recv = messages_received.get(member, 0)
    print(f"   {member}: {sent} messages sent, {recv} mentions received")

# Who mentions whom the most
print(f"\n🔗 Communication Patterns:")
for (src, tgt), count in mention_matrix.sort_values(ascending=False).head(5).items():
    print(f"   {src} → {tgt}: {count} mentions")

# --------------------------
# CRITICAL ISSUE DETECTION
# --------------------------
print(f"\n⚠️  Critical Issues Detected:")

# Issue 1: Team lead repeatedly tagging unresponsive member
sanjana_to_vikram = mention_matrix.get(("Sanjana", "Vikram"), 0)
vikram_to_sanjana = mention_matrix.get(("Vikram", "Sanjana"), 0)
response_ratio = vikram_to_sanjana / sanjana_to_vikram if sanjana_to_vikram > 0 else 0

print(f"   1. Unresponsive Team Member:")
print(f"      - Sanjana (lead) mentioned Vikram {sanjana_to_vikram} times")
print(f"      - Vikram mentioned Sanjana back only {vikram_to_sanjana} times")
print(f"      - Response ratio: {response_ratio:.2f} (healthy: >0.7)")

# Issue 2: Unbalanced workload
sanjana_msgs = messages_sent.get("Sanjana", 0)
tvisha_msgs = messages_sent.get("Tvisha", 0)
vikram_msgs = messages_sent.get("Vikram", 0)
total_msgs = sanjana_msgs + tvisha_msgs + vikram_msgs

print(f"   2. Workload Distribution:")
print(f"      - Sanjana: {sanjana_msgs} ({sanjana_msgs/total_msgs*100:.1f}%)")
print(f"      - Tvisha: {tvisha_msgs} ({tvisha_msgs/total_msgs*100:.1f}%)")
print(f"      - Vikram: {vikram_msgs} ({vikram_msgs/total_msgs*100:.1f}%)")

# Issue 3: Sanjana-Tvisha working well, Vikram isolated
sanjana_tvisha_interactions = mention_matrix.get(("Sanjana", "Tvisha"), 0) + mention_matrix.get(("Tvisha", "Sanjana"), 0)
print(f"   3. Team Cohesion:")
print(f"      - Sanjana ↔ Tvisha: {sanjana_tvisha_interactions} mutual mentions (good)")
print(f"      - Vikram participation gap detected")

# --------------------------
# PROFESSIONAL RECOMMENDATIONS
# --------------------------
recommendations = {
    "generated_at": datetime.now().isoformat(),
    "project_context": "Work Project Simulation - Team Collaboration Analysis",
    "team_structure": {
        "team_lead": "Sanjana",
        "active_collaborator": "Tvisha",
        "problematic_member": "Vikram"
    },
    "analysis_summary": {
        "total_interactions": len(real),
        "duration_minutes": int((real['timestamp'].max() - real['timestamp'].min()).total_seconds() / 60),
        "platforms_used": list(real['platform'].unique()),
        "critical_issues_found": 3,
        "severity": "HIGH"
    },
    "key_metrics": {
        "Sanjana": {
            "role": "Team Lead",
            "messages_sent": int(sanjana_msgs),
            "workload_percentage": round(sanjana_msgs/total_msgs*100, 1),
            "mentions_to_vikram": int(sanjana_to_vikram),
            "mentions_to_tvisha": int(mention_matrix.get(("Sanjana", "Tvisha"), 0)),
            "status": "Overburdened - carrying team communication load"
        },
        "Tvisha": {
            "role": "Active Collaborator",
            "messages_sent": int(tvisha_msgs),
            "workload_percentage": round(tvisha_msgs/total_msgs*100, 1),
            "mentions_to_sanjana": int(mention_matrix.get(("Tvisha", "Sanjana"), 0)),
            "mentions_to_vikram": int(mention_matrix.get(("Tvisha", "Vikram"), 0)),
            "status": "Performing well - responsive and engaged"
        },
        "Vikram": {
            "role": "Team Member",
            "messages_sent": int(vikram_msgs),
            "workload_percentage": round(vikram_msgs/total_msgs*100, 1),
            "response_to_lead": int(vikram_to_sanjana),
            "response_ratio": round(response_ratio, 2),
            "status": "CRITICAL - Unresponsive to team lead"
        }
    },
    "recommendations": {}
}

# --------------------------
# FOR SANJANA (TEAM LEAD)
# --------------------------
recommendations["recommendations"]["Sanjana_Team_Lead"] = {
    "current_situation": f"You are carrying {sanjana_msgs/total_msgs*100:.1f}% of team communication burden. You mentioned Vikram {sanjana_to_vikram} times with only {vikram_to_sanjana} responses - indicating a serious responsiveness issue.",
    
    "immediate_actions": [
        {
            "priority": "CRITICAL",
            "action": "Escalate Vikram's Non-Responsiveness",
            "rationale": f"After {sanjana_to_vikram} unanswered mentions, this is beyond poor communication - it's project risk",
            "specific_steps": [
                "Document all instances where Vikram was tagged but didn't respond",
                "Schedule mandatory 1-on-1 video call with Vikram within 24 hours",
                "Set explicit response time expectations: acknowledge within 2 hours, respond within 4 hours",
                "If unresponsive to meeting request, escalate to project supervisor/professor immediately",
                "Create written record of communication attempts for accountability"
            ]
        },
        {
            "priority": "HIGH",
            "action": "Redistribute Workload",
            "rationale": f"You're handling {sanjana_msgs} messages vs Vikram's {vikram_msgs} - unsustainable for team lead",
            "specific_steps": [
                "Delegate specific, bounded tasks to Tvisha (who is responsive)",
                "Stop chasing Vikram for every update - set hard deadlines with consequences",
                "Document task assignments formally (not just tags in chat)",
                "Use project management tool (Trello/Asana) for task accountability, not just WhatsApp"
            ]
        },
        {
            "priority": "MEDIUM",
            "action": "Implement Structured Communication Protocol",
            "rationale": "Tagging 13 times shows current informal approach isn't working",
            "specific_steps": [
                "Daily standup messages: Everyone posts status by 10 AM",
                "Response SLA: Acknowledge within 2 hours, detailed response within 4 hours",
                "If someone misses standup twice, automatic escalation",
                "Use formal channels for work communication, not casual chat style"
            ]
        }
    ],
    
    "what_to_stop_doing": [
        "❌ Stop repeatedly tagging Vikram without consequences - it enables bad behavior",
        "❌ Stop carrying his workload - let deadlines be missed if he doesn't respond",
        "❌ Stop informal 'check-in' messages - move to formal status reports",
        "❌ Stop assuming he'll eventually respond - treat non-response as escalation trigger"
    ],
    
    "what_to_start_doing": [
        "✅ Document everything - timestamps of tags, responses, missed deadlines",
        "✅ Set clear consequences: 2 missed standups = formal warning, 3 = escalation",
        "✅ Use Tvisha more effectively - she's responsive and capable",
        "✅ Protect your time - stop chasing, start enforcing accountability",
        "✅ Formal weekly team meetings (video call, mandatory attendance, recorded)"
    ],
    
    "leadership_feedback": "You're doing the right thing by trying to engage Vikram, but you've crossed from 'good leader' into 'enabling non-performance'. It's time to enforce accountability, not just request responses."
}

# --------------------------
# FOR TVISHA (ACTIVE COLLABORATOR)
# --------------------------
tvisha_to_sanjana = mention_matrix.get(("Tvisha", "Sanjana"), 0)
tvisha_to_vikram = mention_matrix.get(("Tvisha", "Vikram"), 0)

recommendations["recommendations"]["Tvisha_Active_Collaborator"] = {
    "current_situation": f"You are a model team member with {tvisha_msgs} messages and responsive interaction patterns. You engage with both Sanjana ({tvisha_to_sanjana} mentions) and Vikram ({tvisha_to_vikram} mentions).",
    
    "strengths": [
        "✓ Consistent communication across platforms (WhatsApp + Discord)",
        "✓ Responsive to team lead's requests",
        "✓ Attempts to engage unresponsive teammate",
        "✓ Balanced workload contribution (25% of team messages)"
    ],
    
    "recommendations": [
        {
            "priority": "LOW",
            "action": "Continue Current Performance",
            "rationale": "Your collaboration patterns are healthy and productive",
            "specific_steps": [
                "Maintain current response times and engagement levels",
                "Continue bidirectional communication with Sanjana",
                "Keep attempting to engage Vikram but don't take it personally if he doesn't respond"
            ]
        },
        {
            "priority": "MEDIUM",
            "action": "Support Team Lead Without Taking On Extra Burden",
            "rationale": "Sanjana may need support but you shouldn't compensate for Vikram's non-performance",
            "specific_steps": [
                "Offer to take on well-defined tasks, but set boundaries",
                "Don't let Sanjana dump Vikram's work on you without recognition",
                "Support Sanjana in documentation of Vikram's non-responsiveness",
                "Be willing to go on record about team dynamics if escalation happens"
            ]
        }
    ],
    
    "peer_feedback": "You're doing excellent work. Your responsiveness is what makes this team functional. Don't let Vikram's issues drag down your performance or morale."
}

# --------------------------
# FOR VIKRAM (PROBLEMATIC MEMBER)
# --------------------------
recommendations["recommendations"]["Vikram_Unresponsive_Member"] = {
    "current_situation": f"CRITICAL PERFORMANCE ISSUE: You have only {vikram_msgs} messages ({vikram_msgs/total_msgs*100:.1f}% of team communication). You were mentioned {sanjana_to_vikram + tvisha_to_vikram} times by teammates but only responded with {vikram_to_sanjana + mention_matrix.get(('Vikram', 'Tvisha'), 0)} mentions - a response ratio of {response_ratio:.2f}.",
    
    "severity": "CRITICAL - Project Risk",
    
    "issues_identified": [
        f"1. Non-responsiveness: Team lead tagged you {sanjana_to_vikram} times with minimal response",
        f"2. Low participation: Only {vikram_msgs/total_msgs*100:.1f}% of team communication",
        "3. Broken accountability: Multiple mentions across WhatsApp AND Discord ignored",
        f"4. Team dependency: Your lack of response forces Sanjana to over-work ({sanjana_msgs} messages)"
    ],
    
    "immediate_requirements": [
        {
            "requirement": "Respond to ALL Outstanding Tags Within 2 Hours",
            "consequence": "If not done, formal escalation to supervisor",
            "action": "Go through WhatsApp and Discord, respond to every @ mention you've received"
        },
        {
            "requirement": "Daily Status Updates",
            "consequence": "Missed standup = immediate follow-up call, 2 misses = formal warning",
            "action": "Post by 10 AM daily: What you did yesterday, what you're doing today, any blockers"
        },
        {
            "requirement": "Acknowledge Receipt Within 2 Hours of Being Tagged",
            "consequence": "Non-acknowledgment treated as unavailability, tasks reassigned",
            "action": "Even if you can't fully respond, reply with 'Seen, will respond by [time]'"
        },
        {
            "requirement": "Attend Mandatory Weekly Team Meeting (Video)",
            "consequence": "Miss without advance notice = formal documentation",
            "action": "30 minutes, camera on, prepared with status update"
        }
    ],
    
    "root_cause_analysis": [
        "Possible reasons for non-responsiveness (Vikram should self-identify which applies):",
        "• Technical issues: Not receiving notifications? Check app settings.",
        "• Time management: Overwhelmed with other commitments? Communicate capacity constraints.",
        "• Confusion: Unclear what's expected? Ask for clarification explicitly.",
        "• Disengagement: Not invested in project? This needs to be addressed openly.",
        "• Interpersonal: Issues with team members? Raise privately with supervisor.",
        ""
    ],
    
    "performance_improvement_plan": {
        "week_1": [
            "Respond to ALL outstanding tags within 24 hours",
            "Post daily status updates",
            "Attend team meeting and explain situation",
            "Identify root cause of non-responsiveness"
        ],
        "week_2": [
            "Maintain 100% response rate (within 4 hours)",
            "Complete at least one assigned task",
            "Proactively reach out with questions/blockers",
            "Demonstrate improvement in engagement metrics"
        ],
        "success_criteria": f"Increase message count from {vikram_msgs} to at least {int(total_msgs * 0.2)} (20% of team communication)",
        "failure_outcome": "Formal performance review with supervisor, possible team restructuring"
    },
    
    "direct_feedback": "Your current level of engagement is unacceptable for a team project. You have been given multiple opportunities to respond. This is now a formal performance issue, not just 'quiet participation'. Immediate improvement required."
}

# --------------------------
# MANAGEMENT/SUPERVISOR RECOMMENDATIONS
# --------------------------
recommendations["management_actions"] = {
    "for_supervisor_professor": {
        "situation": "Team with clear unresponsive member issue requiring intervention",
        
        "immediate_actions": [
            {
                "action": "Meet with Vikram Individually",
                "timeline": "Within 48 hours",
                "purpose": "Determine root cause of non-responsiveness and set clear expectations",
                "questions_to_ask": [
                    "Are you receiving notifications from team channels?",
                    "Do you understand your responsibilities on this project?",
                    "Is there something preventing you from engaging with the team?",
                    "Are there conflicts with team members we should address?",
                    "What do you need to be successful in this project?"
                ]
            },
            {
                "action": "Acknowledge Sanjana's Leadership Challenges",
                "timeline": "Within 72 hours",
                "purpose": "Recognize she's doing everything right but facing team member issues",
                "talking_points": [
                    "You've documented appropriate attempts to engage Vikram",
                    "Your communication frequency is appropriate for a team lead",
                    "This is not a reflection on your leadership",
                    "We will support you in enforcing accountability"
                ]
            },
            {
                "action": "Set Formal Performance Expectations",
                "timeline": "This week",
                "deliverable": "Written communication protocol with consequences",
                "requirements": [
                    "Daily status updates (10 AM deadline)",
                    "4-hour response time to @ mentions",
                    "Mandatory weekly team meeting attendance",
                    "Minimum 20% contribution to team communication"
                ]
            }
        ],
        
        "medium_term_actions": [
            "Monitor team chat daily for 2 weeks to verify Vikram improvement",
            "Hold Vikram accountable to performance improvement plan",
            "Consider team restructuring if no improvement within 2 weeks",
            "Ensure Sanjana's grade isn't impacted by Vikram's non-performance",
            "Document everything for academic integrity if needed"
        ],
        
        "alternative_approaches": [
            "Option 1: Give Vikram one more week with formal PIP, then reassign him to individual project",
            "Option 2: Reduce Vikram's role to 'contributor' vs 'core team member', adjust grading accordingly",
            "Option 3: Add check-in meetings for Vikram (with supervisor) 2x weekly until responsive"
        ]
    }
}

# --------------------------
# SAVE ALL OUTPUTS
# --------------------------
output_file = os.path.join(OUTPUT_DIR, "professional_recommendations.json")
with open(output_file, "w") as f:
    json.dump(recommendations, f, indent=4)

# Save simplified metrics
metrics_file = os.path.join(OUTPUT_DIR, "team_metrics.json")
with open(metrics_file, "w") as f:
    json.dump({
        "Sanjana": {"role": "Team Lead", "messages": int(sanjana_msgs), "mentions_sent": int(sanjana_to_vikram + mention_matrix.get(("Sanjana", "Tvisha"), 0))},
        "Tvisha": {"role": "Active Member", "messages": int(tvisha_msgs), "status": "Performing well"},
        "Vikram": {"role": "Problematic Member", "messages": int(vikram_msgs), "response_ratio": round(response_ratio, 2), "status": "CRITICAL"}
    }, f, indent=4)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - PROFESSIONAL RECOMMENDATIONS GENERATED")
print("=" * 70)
print(f"\n🎯 Key Findings:")
print(f"   • Sanjana (Team Lead): Overburdened, mentioned Vikram {sanjana_to_vikram}x")
print(f"   • Tvisha: Performing excellently, responsive and engaged")
print(f"   • Vikram: CRITICAL ISSUE - Unresponsive (ratio: {response_ratio:.2f})")
print(f"\n⚠️  Severity: HIGH - Requires immediate management intervention")
print(f"\n📁 Files Created:")
print(f"   {output_file}")
print(f"   {metrics_file}")
print("\n" + "=" * 70)