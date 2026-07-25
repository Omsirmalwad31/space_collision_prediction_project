"""
mission_assistant.py — AI Mission Control chat assistant

A chat panel scoped ONLY to the current run's data (objects, events, scores).
Answers questions like "which pair is highest risk?" by querying the actual
session data — grounded in real numbers, not free-form generation.

Supports two modes:
  1. LLM-powered (Anthropic Claude) — structured context injection
  2. Template/rule-based fallback — pattern-matched responses
"""

from __future__ import annotations
import os
import re
import warnings
import pandas as pd
from typing import Dict, Any, List, Optional


def _build_context_summary(session_data: Dict[str, Any]) -> str:
    """Build a concise text summary of the current session for LLM context."""
    events_df = session_data.get("events", pd.DataFrame())
    records = session_data.get("records", [])
    regime_map = session_data.get("regime_map", {})

    lines = [
        f"Current session: {len(records)} objects tracked.",
        f"Conjunction events detected: {len(events_df)}.",
    ]

    if len(events_df) > 0:
        critical = events_df[events_df["risk_category"] == "Critical"]
        high = events_df[events_df["risk_category"] == "High"]
        lines.append(f"Critical events: {len(critical)}, High events: {len(high)}.")

        # Top 5 events
        top = events_df.head(5)
        lines.append("\nTop 5 highest-risk events:")
        for _, row in top.iterrows():
            lines.append(
                f"  • {row.get('object_a', '?')} ↔ {row.get('object_b', '?')}: "
                f"risk={row.get('risk_score', 0):.1f}/100 [{row.get('risk_category', '?')}], "
                f"miss={row.get('miss_distance_km', 0):.3f} km, "
                f"TCA={row.get('tca_hours_from_now', 0):.1f}h"
            )

        # Regime distribution
        if regime_map:
            regimes = {}
            for nid, regime in regime_map.items():
                regimes[regime] = regimes.get(regime, 0) + 1
            lines.append(f"\nOrbital regimes: {regimes}")

    return "\n".join(lines)


def _rule_based_answer(
    question: str,
    session_data: Dict[str, Any],
) -> str:
    """Pattern-matched answers for common questions (no LLM needed)."""
    q = question.lower().strip()
    events_df = session_data.get("events", pd.DataFrame())
    records = session_data.get("records", [])

    if len(events_df) == 0:
        return ("No conjunction events have been detected in this run. "
                "Try adjusting the conjunction threshold or prediction window.")

    # ── Highest risk ──
    if any(kw in q for kw in ["highest risk", "most dangerous", "worst", "top risk"]):
        top = events_df.iloc[0]
        return (
            f"The highest-risk conjunction is between {top['object_a']} and "
            f"{top['object_b']} with a risk score of {top['risk_score']:.1f}/100 "
            f"[{top['risk_category']}]. Miss distance: {top['miss_distance_km']:.3f} km, "
            f"relative velocity: {top['relative_velocity_kms']:.2f} km/s, "
            f"TCA: {top['tca_hours_from_now']:.1f} hours from now."
        )

    # ── Next TCA ──
    if any(kw in q for kw in ["next tca", "soonest", "earliest", "next 24", "next conjunction"]):
        next_ev = events_df.sort_values("tca_hours_from_now").iloc[0]
        return (
            f"The soonest conjunction is {next_ev['object_a']} ↔ "
            f"{next_ev['object_b']} at TCA in {next_ev['tca_hours_from_now']:.1f} hours. "
            f"Risk score: {next_ev['risk_score']:.1f}/100 [{next_ev['risk_category']}]."
        )

    # ── Count / summary ──
    if any(kw in q for kw in ["how many", "count", "total", "summary"]):
        n_crit = len(events_df[events_df["risk_category"] == "Critical"])
        n_high = len(events_df[events_df["risk_category"] == "High"])
        n_med = len(events_df[events_df["risk_category"] == "Medium"])
        n_low = len(events_df[events_df["risk_category"] == "Low"])
        return (
            f"Session summary: {len(records)} objects tracked, "
            f"{len(events_df)} conjunction events detected.\n"
            f"• Critical: {n_crit}\n• High: {n_high}\n"
            f"• Medium: {n_med}\n• Low: {n_low}"
        )

    # ── Specific object ──
    for _, row in events_df.iterrows():
        if (row.get("object_a", "").lower() in q or
                row.get("object_b", "").lower() in q):
            return (
                f"Found event involving the queried object: "
                f"{row['object_a']} ↔ {row['object_b']}, "
                f"risk={row['risk_score']:.1f}/100 [{row['risk_category']}], "
                f"miss={row['miss_distance_km']:.3f} km, "
                f"TCA={row['tca_hours_from_now']:.1f}h."
            )

    # ── Starlink / constellation ──
    if "starlink" in q or "constellation" in q:
        starlink_events = events_df[
            events_df["object_a"].str.contains("STARLINK", case=False, na=False) |
            events_df["object_b"].str.contains("STARLINK", case=False, na=False)
        ]
        if len(starlink_events) > 0:
            return (
                f"Found {len(starlink_events)} conjunction events involving Starlink objects. "
                f"Highest risk: {starlink_events.iloc[0]['risk_score']:.1f}/100."
            )
        return "No Starlink conjunction events in this run."

    # ── Default ──
    return (
        f"I can answer questions about the current run's data: "
        f"{len(records)} objects tracked, {len(events_df)} events detected. "
        f"Try asking about 'highest risk pair', 'next TCA', 'how many critical events', "
        f"or a specific object name."
    )


def answer_question(
    question: str,
    session_data: Dict[str, Any],
    force_template: bool = False,
) -> str:
    """
    Answer a question about the current session data.

    Parameters
    ----------
    question : str
        User's natural-language question.
    session_data : dict
        Current st.session_state.results.
    force_template : bool
        Skip LLM even if API key is set.

    Returns
    -------
    str — Answer text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key or force_template:
        return _rule_based_answer(question, session_data)

    # ── LLM-powered answer ──
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        context = _build_context_summary(session_data)

        prompt = f"""You are an AI assistant for a Space Situational Awareness (SSA) Mission Control system.
You ONLY answer questions using the data provided below — do not make up data or events.
If the data doesn't contain the answer, say so honestly.

CURRENT SESSION DATA:
{context}

USER QUESTION: {question}

Answer concisely (2-4 sentences), grounding every claim in the data above.
Always include specific numbers (risk scores, distances, times).
End with a note: "Based on current session data — advisory only."
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as exc:
        warnings.warn(f"LLM assistant failed: {exc}")
        return _rule_based_answer(question, session_data)


# ─── Smoke test ──

if __name__ == "__main__":
    mock_data = {
        "records": [{"name": f"SAT-{i}"} for i in range(30)],
        "events": pd.DataFrame([
            {"object_a": "STARLINK-1007", "object_b": "COSMOS 2251 DEB [A]",
             "miss_distance_km": 2.5, "relative_velocity_kms": 7.8,
             "tca_hours_from_now": 36.5, "risk_score": 72.3, "risk_category": "High"},
            {"object_a": "ISS (ZARYA)", "object_b": "FENGYUN 1C DEB [A]",
             "miss_distance_km": 15.0, "relative_velocity_kms": 3.2,
             "tca_hours_from_now": 120.0, "risk_score": 35.0, "risk_category": "Medium"},
        ]),
        "regime_map": {},
    }

    questions = [
        "Which pair is highest risk?",
        "How many events total?",
        "What about Starlink?",
        "When is the next TCA?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q, mock_data, force_template=True)}")
