"""
ai_report.py — Generative AI risk advisory reports (FR-11, FR-12)

Generates natural-language risk advisories for scored conjunction events.
Two paths:
  1. LLM (Anthropic Claude) — when ANTHROPIC_API_KEY is set
  2. Template fallback — deterministic, same output format, zero network

Every report is labeled "ADVISORY ONLY — NOT AN OPERATIONAL DIRECTIVE"
per 04_UI_UX_DESIGN_BRIEF.md principle #4 (trust through transparency).

Data contract in  (Scored Event Record dict):
  {object_a, object_b, miss_distance_km, relative_velocity_kms,
   tca_hours_from_now, risk_score, risk_category, ...}

Data contract out (05_BACKEND_SCHEMA.md §7 — plain text, fixed structure):
  RISK ADVISORY
  Object: <A> | Object: <B>
  ...
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, Any, Optional


def _template_report(event: Dict[str, Any]) -> str:
    """
    Deterministic template-based report.  Same structure as LLM output.
    Used when no API key is set or LLM call fails.
    """
    name_a = event.get("object_a", "Unknown-A")
    name_b = event.get("object_b", "Unknown-B")
    miss = event.get("miss_distance_km", 0)
    vel = event.get("relative_velocity_kms", 0)
    tca = event.get("tca_hours_from_now", 0)
    score = event.get("risk_score", 0)
    category = event.get("risk_category", "Unknown")
    pc = event.get("pc", None)
    delta_v = event.get("delta_v_ms", None)

    # Summary generation based on risk level
    if category == "Critical":
        summary = (
            f"CRITICAL conjunction detected between {name_a} and {name_b}. "
            f"Predicted miss distance of {miss:.3f} km with closing velocity "
            f"of {vel:.2f} km/s. Time to closest approach: {tca:.1f} hours. "
            f"This event requires immediate operator attention and potential "
            f"avoidance maneuver assessment."
        )
        action = (
            "IMMEDIATE ACTION RECOMMENDED: Assess avoidance maneuver feasibility. "
            "Verify latest tracking data. Coordinate with space surveillance network "
            "for updated conjunction data message (CDM). "
            "Prepare contingency plan if maneuver is not executed."
        )
    elif category == "High":
        summary = (
            f"High-risk conjunction identified between {name_a} and {name_b}. "
            f"Predicted miss distance: {miss:.3f} km at relative velocity "
            f"{vel:.2f} km/s. TCA in {tca:.1f} hours. "
            f"Event warrants close monitoring and maneuver readiness."
        )
        action = (
            "MONITOR CLOSELY: Request updated tracking data. Prepare preliminary "
            "avoidance maneuver profile. Re-evaluate risk at next CDM update. "
            "Escalate if miss distance decreases or uncertainty grows."
        )
    elif category == "Medium":
        summary = (
            f"Medium-risk conjunction between {name_a} and {name_b}. "
            f"Miss distance: {miss:.3f} km, relative velocity: {vel:.2f} km/s, "
            f"TCA: {tca:.1f} hours. Event is within monitoring threshold but "
            f"does not currently require intervention."
        )
        action = (
            "ROUTINE MONITORING: Continue tracking. No immediate action required. "
            "Re-evaluate at next tracking update or if risk score trends upward."
        )
    else:
        summary = (
            f"Low-risk conjunction between {name_a} and {name_b}. "
            f"Miss distance: {miss:.3f} km, relative velocity: {vel:.2f} km/s, "
            f"TCA: {tca:.1f} hours. Conjunction is within normal operational "
            f"bounds for the current orbital regime."
        )
        action = "NO ACTION REQUIRED: Event logged for record-keeping."

    # Build report
    lines = [
        "+==============================================================+",
        "|                      RISK ADVISORY                           |",
        "|         [!] ADVISORY ONLY -- NOT AN OPERATIONAL DIRECTIVE    |",
        "+==============================================================+",
        "",
        f"Object: {name_a}  |  Object: {name_b}",
        f"Time to TCA: {tca:.1f} hours",
        f"Risk Score: {score:.1f}/100  [{category}]",
    ]

    if pc is not None:
        lines.append(f"Analytical Pc: {pc:.2e} (Chan 2D method)")

    lines += [
        "",
        "SUMMARY:",
        summary,
        "",
        "RECOMMENDED ACTION:",
        action,
        "",
        f"PRIORITY: {category.upper()}",
    ]

    if delta_v is not None and delta_v > 0:
        lines += [
            "",
            f"ESTIMATED AVOIDANCE Delta-v: {delta_v:.4f} m/s",
            "(Simplified impulse approximation -- see assumptions in Threat Assessment tab)",
        ]

    lines += [
        "",
        "--- Method & Uncertainty ---",
        f"Risk score: ML model (GBR, 4 features). Observed MAE ~25 pts on ESA CDM test data.",
        f"This score is a statistical estimate, not a certainty.",
        "All figures are observed on sample/test data -- not operational guarantees.",
        "",
        f"Generated by: template engine (offline mode)",
    ]

    return "\n".join(lines)


def _llm_report(event: Dict[str, Any]) -> str:
    """
    Generate a report via Anthropic Claude API.
    Falls back to template on any error.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _template_report(event)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a space situational awareness (SSA) analyst assistant.
Generate a structured risk advisory for this conjunction event.

Event data:
- Object A: {event.get('object_a', 'Unknown')}
- Object B: {event.get('object_b', 'Unknown')}
- Miss distance: {event.get('miss_distance_km', 0):.3f} km
- Relative velocity: {event.get('relative_velocity_kms', 0):.2f} km/s
- Time to TCA: {event.get('tca_hours_from_now', 0):.1f} hours
- ML Risk score: {event.get('risk_score', 0):.1f}/100 [{event.get('risk_category', 'Unknown')}]
- Analytical Pc: {event.get('pc', 'N/A')}

Format your response EXACTLY as:

RISK ADVISORY
⚠️ ADVISORY ONLY — NOT AN OPERATIONAL DIRECTIVE

Object: [A] | Object: [B]
Time to TCA: [hours] hours
Risk Score: [score]/100 [category]

SUMMARY:
[2-3 sentence analysis]

RECOMMENDED ACTION:
[Specific, actionable recommendation]

PRIORITY: [LOW|MEDIUM|HIGH|CRITICAL]

IMPORTANT: State that all figures are statistical estimates observed on sample data, not guarantees.
End with: "Generated by: Claude LLM (advisory use only)"
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as exc:
        warnings.warn(f"LLM report generation failed: {exc} — using template.")
        return _template_report(event)


def generate_report(
    event: Dict[str, Any],
    force_template: bool = False,
) -> str:
    """
    Generate a risk advisory report for a scored conjunction event.

    Parameters
    ----------
    event : dict
        Scored Event Record.
    force_template : bool
        If True, skip LLM even if API key is available.

    Returns
    -------
    str
        Formatted risk advisory text.
    """
    if force_template:
        return _template_report(event)
    return _llm_report(event)


def get_report_mode() -> str:
    """Return 'llm' if API key is set, else 'template'."""
    return "llm" if os.environ.get("ANTHROPIC_API_KEY") else "template"


# ─── Smoke test ──

if __name__ == "__main__":
    test_event = {
        "object_a": "STARLINK-1007",
        "object_b": "COSMOS 2251 DEB [A]",
        "miss_distance_km": 2.5,
        "relative_velocity_kms": 7.8,
        "tca_hours_from_now": 36.5,
        "risk_score": 72.3,
        "risk_category": "High",
        "pc": 1.2e-6,
        "delta_v_ms": 0.019,
    }

    print(f"Report mode: {get_report_mode()}")
    print()
    print(generate_report(test_event, force_template=True))
