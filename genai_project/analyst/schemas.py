"""JSON schemas that pin down every agent's output.

Each agent returns structured data, never prose the next agent has to
re-parse. That is what makes the hand-offs reliable: the Risk Agent receives
typed findings, not paragraphs, and the report renderer never has to guess.

The schemas are written for OpenAI *strict* structured outputs, which requires
`additionalProperties: false` and every property listed in `required`, and
which does **not** accept validation keywords like `minimum`/`maxItems`.
Ranges are therefore stated in the field descriptions and clamped in Python.
"""

from __future__ import annotations

from typing import Any


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    """Strict object schema: no extra keys, every key required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def arr(items: dict[str, Any], description: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"type": "array", "items": items}
    if description:
        out["description"] = description
    return out


def s(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def i(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def enum(values: list[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": description}


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------
AGENT_NAMES = ["finance", "sales", "customer", "market", "operations"]

PLANNER_SCHEMA = obj(
    {
        "restated_question": s("The user's question restated as a decision to be made."),
        "decision_type": enum(
            ["launch", "expansion", "pricing", "cost_reduction", "investment",
             "portfolio", "diagnostic", "other"],
            "The kind of business decision this question represents.",
        ),
        "sub_questions": arr(
            s("A specific analytical question that must be answered."),
            "3-6 sub-questions the analysis must resolve.",
        ),
        "assignments": arr(
            obj(
                {
                    "agent": enum(AGENT_NAMES, "Which specialist agent takes this task."),
                    "task": s("The concrete analytical task for that agent."),
                    "why": s("Why this task matters to the decision."),
                    "priority": i("1 = critical to the decision, 2 = supporting, 3 = nice to have."),
                }
            ),
            "One entry per specialist that should run. Omit agents whose data "
            "cannot inform this question.",
        ),
        "success_criteria": arr(
            s("What a good answer must demonstrate."), "2-4 criteria."
        ),
        "assumptions": arr(s("An assumption the plan rests on."), "0-4 assumptions."),
    }
)

# ---------------------------------------------------------------------------
# 2-6. Domain specialists (one shared shape, so hand-offs stay uniform)
# ---------------------------------------------------------------------------
SPECIALIST_SCHEMA = obj(
    {
        "headline": s("One sentence a CFO could read alone and still learn something."),
        "findings": arr(
            obj(
                {
                    "title": s("Short label for the finding, e.g. 'Operating margin compression'."),
                    "detail": s(
                        "2-4 sentences. State the number, the period it covers, and the "
                        "most likely driver. Cite figures from the evidence only."
                    ),
                    "metric": s(
                        "The headline figure, formatted for a report "
                        "(e.g. '+18.4% YoY', '-5.2 pts', '$1.4M'). Empty string if none."
                    ),
                    "direction": enum(
                        ["up", "down", "flat", "mixed"], "Direction of travel of the metric."
                    ),
                    "impact": enum(
                        ["high", "medium", "low"], "How much this should weigh on the decision."
                    ),
                    "evidence": s("Which file or figures in the evidence support this."),
                }
            ),
            "3-6 findings, ordered most important first.",
        ),
        "risks": arr(s("A risk this domain raises for the decision."), "1-4 risks."),
        "opportunities": arr(s("An opportunity this domain surfaces."), "1-4 items."),
        "recommendation": s("What this agent would advise, within its own remit only."),
        "data_gaps": arr(
            s("Something the agent needed but the evidence does not contain."), "0-3 gaps."
        ),
        "confidence": i("0-100. How much the evidence supports these conclusions."),
    }
)

# ---------------------------------------------------------------------------
# 7. Risk
# ---------------------------------------------------------------------------
RISK_SCHEMA = obj(
    {
        "summary": s("2-3 sentences on the overall risk posture of this decision."),
        "risks": arr(
            obj(
                {
                    "category": enum(
                        ["financial", "operational", "market", "reputational",
                         "regulatory", "execution"],
                        "Risk category.",
                    ),
                    "title": s("Short name for the risk."),
                    "description": s("2-3 sentences: the mechanism by which this hurts."),
                    "likelihood": i("1-5, where 5 is near-certain."),
                    "impact": i("1-5, where 5 is severe."),
                    "evidence": arr(
                        s("An agent finding or figure that supports this risk."), "1-3 items."
                    ),
                    "mitigation": s("A concrete, ownable action that reduces this risk."),
                    "early_warning_indicator": s("The metric that would show this risk landing."),
                }
            ),
            "4-8 risks, most severe first.",
        ),
        "category_scores": arr(
            obj(
                {
                    "category": enum(
                        ["financial", "operational", "market", "reputational",
                         "regulatory", "execution"],
                        "Risk category.",
                    ),
                    "score": i("1-5 overall severity for this category (5 = worst)."),
                    "rationale": s("One sentence justifying the score."),
                }
            ),
            "One entry per category that is materially in play.",
        ),
        "overall_risk": i("1-5 overall risk of proceeding as proposed."),
        "top_concern": s("The single risk that most deserves executive attention."),
        "deal_breakers": arr(
            s("A condition that, if true, should stop the decision outright."), "0-3 items."
        ),
    }
)

# ---------------------------------------------------------------------------
# 8. Strategy
# ---------------------------------------------------------------------------
STRATEGY_SCHEMA = obj(
    {
        "recommendation": s("The recommendation in one imperative sentence, e.g. 'Launch the Pulse Pro in India in Q1 FY27.'"),
        "decision": enum(
            ["go", "conditional_go", "hold", "no_go"], "The verdict."
        ),
        "conditions": arr(
            s("A condition that must hold for a conditional_go. Empty for a clean go/no_go."),
            "0-4 conditions.",
        ),
        "reasons": arr(
            obj(
                {
                    "reason": s("One reason supporting the recommendation."),
                    "evidence": s("The agent finding or figure behind it."),
                    "weight": enum(["primary", "supporting"], "How much this reason carries."),
                }
            ),
            "3-6 reasons.",
        ),
        "counter_arguments": arr(
            obj(
                {
                    "argument": s("The strongest case against the recommendation."),
                    "response": s("Why it does not change the recommendation — or why it nearly does."),
                }
            ),
            "1-3 counter-arguments. Steelman them.",
        ),
        "action_plan": arr(
            obj(
                {
                    "step": s("A concrete action."),
                    "owner": s("The function that owns it (e.g. 'Supply Chain', 'Finance')."),
                    "timeline": s("When, e.g. 'next 30 days', 'Q1 FY27'."),
                    "success_metric": s("How we will know it worked."),
                }
            ),
            "3-6 steps, in execution order.",
        ),
        "kpis_to_watch": arr(s("A metric to track after the decision."), "3-5 KPIs."),
        "confidence": i("0-100 confidence in the recommendation."),
        "confidence_rationale": s("Why that confidence level and not higher or lower."),
        "executive_summary": s(
            "5-8 sentences for a board audience: the decision, the two or three "
            "numbers that drive it, the main risk, and the first action. No bullet "
            "markers, no headings — plain prose."
        ),
    }
)

# ---------------------------------------------------------------------------
# 9. Evaluator (LLM-as-judge)
# ---------------------------------------------------------------------------
EVAL_CRITERIA = [
    "groundedness",
    "relevance",
    "completeness",
    "actionability",
    "internal_consistency",
]

EVAL_SCHEMA = obj(
    {
        "scores": arr(
            obj(
                {
                    "criterion": enum(EVAL_CRITERIA, "Which criterion this scores."),
                    "score": i("0-100."),
                    "justification": s("2-3 sentences citing what drove the score."),
                }
            ),
            "One entry per criterion, all five, in the order listed.",
        ),
        "unsupported_claims": arr(
            obj(
                {
                    "claim": s("The claim as written in the report, quoted."),
                    "problem": s(
                        "Why the evidence does not support it — figure absent, "
                        "figure contradicted, or inference beyond what the data shows."
                    ),
                    "severity": enum(
                        ["high", "medium", "low"],
                        "high if it would change the decision, low if cosmetic.",
                    ),
                }
            ),
            "0-6 claims. Empty if every figure checks out.",
        ),
        "strengths": arr(s("Something the report does well."), "1-3 items."),
        "weaknesses": arr(s("Something that materially weakens it."), "1-4 items."),
        "overall_score": i("0-100, weighing groundedness most heavily."),
        "verdict": enum(
            ["publish", "revise", "reject"],
            "publish = board-ready; revise = fixable gaps; reject = unsound.",
        ),
        "summary": s("2-3 sentences an editor could act on."),
    }
)