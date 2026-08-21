"""Rendering the finished analysis as an executive report.

Everything here is presentation only — no judgement is added at this layer.
The renderer must survive a partial run: any agent may be missing or failed,
and the report still has to say something useful about what did come back
rather than raising on a `None`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ROLE_LABELS

if TYPE_CHECKING:  # avoids a cycle — graph imports reporting, not the reverse
    from .agents import AgentResult
    from .graph import AnalysisResult


# ---------------------------------------------------------------------------
# small formatters
# ---------------------------------------------------------------------------
def stars(score: int, out_of: int = 5) -> str:
    """1-5 severity as filled/empty stars, the format the brief asks for."""
    try:
        n = max(0, min(out_of, int(score)))
    except (TypeError, ValueError):
        n = 0
    return "★" * n + "☆" * (out_of - n)


def confidence_bar(pct: int, width: int = 20) -> str:
    try:
        p = max(0, min(100, int(pct)))
    except (TypeError, ValueError):
        p = 0
    filled = round(width * p / 100)
    return "█" * filled + "░" * (width - filled) + f"  {p}%"


DECISION_LABELS = {
    "go": "GO",
    "conditional_go": "CONDITIONAL GO",
    "hold": "HOLD",
    "no_go": "NO GO",
}

_DIRECTION_ARROWS = {"up": "▲", "down": "▼", "flat": "▬", "mixed": "◆"}


def _lst(value: Any) -> list:
    return value if isinstance(value, list) else []


def _bullets(items: Any, empty: str = "_None recorded._") -> str:
    items = [str(i) for i in _lst(items) if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else empty


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def _header(result: "AnalysisResult") -> str:
    lines = ["# Executive Analysis", "", f"**Question:** {result.question}", ""]
    plan = result.plan.data if result.plan.ok else {}
    if plan.get("restated_question"):
        lines += [f"**Decision:** {plan['restated_question']}", ""]
    if plan.get("decision_type"):
        lines += [f"**Decision type:** `{plan['decision_type']}`", ""]
    if result.aborted:
        lines += [f"> **Run stopped early:** {result.aborted}", ""]
    return "\n".join(lines)


def _summary_section(result: "AnalysisResult") -> str:
    if not (result.strategy and result.strategy.ok):
        return (
            "## Executive summary\n\n"
            "_Not available — the Strategy Agent did not complete"
            + (f" ({result.strategy.error})" if result.strategy else "")
            + "._\n"
        )
    d = result.strategy.data
    decision = DECISION_LABELS.get(d.get("decision", ""), str(d.get("decision", "?")).upper())
    out = [
        "## Executive summary",
        "",
        f"### {decision} — {d.get('recommendation', '')}",
        "",
        d.get("executive_summary", ""),
        "",
        "**Confidence**",
        "",
        "```",
        confidence_bar(d.get("confidence", 0)),
        "```",
        "",
        f"_{d.get('confidence_rationale', '')}_",
        "",
    ]
    conditions = _lst(d.get("conditions"))
    if conditions:
        out += ["**Conditions attached to this recommendation**", "", _bullets(conditions), ""]
    return "\n".join(out)


def _reasoning_section(result: "AnalysisResult") -> str:
    if not (result.strategy and result.strategy.ok):
        return ""
    d = result.strategy.data
    out = ["## Why", ""]

    primary = [r for r in _lst(d.get("reasons")) if r.get("weight") == "primary"]
    supporting = [r for r in _lst(d.get("reasons")) if r.get("weight") != "primary"]
    for title, group in (("Primary", primary), ("Supporting", supporting)):
        if not group:
            continue
        out.append(f"**{title}**")
        out.append("")
        for r in group:
            out.append(f"- **{r.get('reason', '')}**  \n  _Evidence: {r.get('evidence', '')}_")
        out.append("")

    counters = _lst(d.get("counter_arguments"))
    if counters:
        out += ["### The case against", ""]
        for c in counters:
            out.append(f"- **{c.get('argument', '')}**  \n  _Response: {c.get('response', '')}_")
        out.append("")
    return "\n".join(out)


def _risk_section(result: "AnalysisResult") -> str:
    if not (result.risk and result.risk.ok):
        return (
            "## Risk assessment\n\n_Not available"
            + (f" ({result.risk.error})" if result.risk else "")
            + "._\n"
        )
    d = result.risk.data
    out = [
        "## Risk assessment",
        "",
        f"**Overall risk of proceeding:** {stars(d.get('overall_risk', 3))} "
        f"({d.get('overall_risk', '?')}/5)",
        "",
        d.get("summary", ""),
        "",
    ]

    cats = _lst(d.get("category_scores"))
    if cats:
        out += [
            "| Category | Severity | Why |",
            "|---|---|---|",
        ]
        for c in sorted(cats, key=lambda x: -int(x.get("score", 0) or 0)):
            out.append(
                f"| {str(c.get('category', '')).title()} | {stars(c.get('score', 0))} "
                f"| {c.get('rationale', '')} |"
            )
        out.append("")

    risks = _lst(d.get("risks"))
    if risks:
        out += ["### Individual risks", ""]
        # Severity is likelihood x impact, so a 5x1 and a 1x5 sort apart from
        # a 3x3 — which is the point of scoring them separately.
        for r in sorted(
            risks,
            key=lambda x: -(int(x.get("likelihood", 0) or 0) * int(x.get("impact", 0) or 0)),
        ):
            lk, im = int(r.get("likelihood", 0) or 0), int(r.get("impact", 0) or 0)
            out += [
                f"**{r.get('title', '')}**  `{str(r.get('category', '')).lower()}`  "
                f"— likelihood {lk}/5 × impact {im}/5 = **{lk * im}**",
                "",
                r.get("description", ""),
                "",
                f"- _Mitigation:_ {r.get('mitigation', '')}",
                f"- _Early warning:_ {r.get('early_warning_indicator', '')}",
                f"- _Based on:_ {'; '.join(str(e) for e in _lst(r.get('evidence'))) or 'unstated'}",
                "",
            ]

    if d.get("top_concern"):
        out += [f"> **Top concern:** {d['top_concern']}", ""]
    breakers = _lst(d.get("deal_breakers"))
    if breakers:
        out += ["**Deal breakers**", "", _bullets(breakers), ""]
    return "\n".join(out)


def _action_section(result: "AnalysisResult") -> str:
    if not (result.strategy and result.strategy.ok):
        return ""
    d = result.strategy.data
    steps = _lst(d.get("action_plan"))
    out = []
    if steps:
        out += [
            "## Action plan",
            "",
            "| # | Action | Owner | Timeline | Success metric |",
            "|---|---|---|---|---|",
        ]
        for n, s in enumerate(steps, 1):
            out.append(
                f"| {n} | {s.get('step', '')} | {s.get('owner', '')} "
                f"| {s.get('timeline', '')} | {s.get('success_metric', '')} |"
            )
        out.append("")
    kpis = _lst(d.get("kpis_to_watch"))
    if kpis:
        out += ["**KPIs to watch**", "", _bullets(kpis), ""]
    return "\n".join(out)


def _specialist_section(results: list["AgentResult"]) -> str:
    if not results:
        return ""
    out = ["## Specialist findings", ""]
    for r in results:
        out.append(f"### {r.label}")
        out.append("")
        if not r.ok:
            out += [f"_Agent failed: {r.error}_", ""]
            continue
        d = r.data
        out += [
            f"**{d.get('headline', '')}**",
            "",
            f"_Confidence: {d.get('confidence', '?')}/100_",
            "",
        ]
        findings = _lst(d.get("findings"))
        if findings:
            out += ["| Finding | Metric | Impact | Detail |", "|---|---|---|---|"]
            for f in findings:
                arrow = _DIRECTION_ARROWS.get(f.get("direction", ""), "")
                metric = f"{arrow} {f.get('metric', '')}".strip() or "—"
                detail = str(f.get("detail", "")).replace("\n", " ").replace("|", "\\|")
                out.append(
                    f"| **{f.get('title', '')}** | {metric} "
                    f"| {f.get('impact', '')} | {detail} |"
                )
            out.append("")
        for label, key in (
            ("Risks raised", "risks"),
            ("Opportunities", "opportunities"),
            ("Data gaps", "data_gaps"),
        ):
            items = _lst(d.get(key))
            if items:
                out += [f"**{label}**", "", _bullets(items), ""]
        if d.get("recommendation"):
            out += [f"_Within its own remit, this agent advises:_ {d['recommendation']}", ""]
        if r.citations:
            unique = list(dict.fromkeys(r.citations))
            out += ["_Sources retrieved:_ " + ", ".join(f"`{c}`" for c in unique), ""]
    return "\n".join(out)


def _plan_section(result: "AnalysisResult") -> str:
    if not result.plan.ok:
        return ""
    d = result.plan.data
    out = ["## How the question was broken down", ""]
    sub = _lst(d.get("sub_questions"))
    if sub:
        out += ["**Sub-questions**", "", _bullets(sub), ""]
    assignments = _lst(d.get("assignments"))
    if assignments:
        out += ["| Agent | Task | Priority |", "|---|---|---|"]
        for a in assignments:
            out.append(
                f"| {ROLE_LABELS.get(a.get('agent', ''), a.get('agent', ''))} "
                f"| {a.get('task', '')} | {a.get('priority', '')} |"
            )
        out.append("")
    assumptions = _lst(d.get("assumptions"))
    if assumptions:
        out += ["**Assumptions this plan rests on**", "", _bullets(assumptions), ""]
    return "\n".join(out)


def _footer(result: "AnalysisResult") -> str:
    u = result.usage or {}
    parts = [
        "---",
        "",
        "### Run detail",
        "",
        f"- Backend: `{result.settings.backend}` / `{result.settings.model}`",
        f"- Wall clock: {result.seconds:.1f}s",
        f"- Model calls: {u.get('calls', 0)} ({u.get('cache_hits', 0)} served from cache)",
        f"- Tokens: {u.get('input_tokens', 0):,} in / {u.get('output_tokens', 0):,} out",
        f"- Estimated cost: ${u.get('cost_usd', 0):.4f}",
        "",
        "_Figures are computed from the company's files by code; the agents "
        "interpret them but never calculate them._",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def render_report(result: "AnalysisResult") -> str:
    """The full executive report as markdown."""
    sections = [
        _header(result),
        _summary_section(result),
        _reasoning_section(result),
        _risk_section(result),
        _action_section(result),
        _specialist_section(result.specialists),
        _plan_section(result),
        _footer(result),
    ]
    return "\n".join(s for s in sections if s.strip())


def render_evaluation(evaluation: "AgentResult | None") -> str:
    """The judge's grading, rendered separately from the report it grades."""
    if evaluation is None:
        return "_Evaluation was not run._"
    if not evaluation.ok:
        return f"_Evaluation failed: {evaluation.error}_"

    d = evaluation.data
    verdict = str(d.get("verdict", "?")).upper()
    out = [
        "## Independent evaluation",
        "",
        f"**{d.get('overall_score', '?')}/100 — {verdict}**",
        "",
        d.get("summary", ""),
        "",
        "| Criterion | Score | Justification |",
        "|---|---|---|",
    ]
    for row in _lst(d.get("scores")):
        name = str(row.get("criterion", "")).replace("_", " ").title()
        just = str(row.get("justification", "")).replace("\n", " ").replace("|", "\\|")
        out.append(f"| {name} | {row.get('score', '?')}/100 | {just} |")
    out.append("")

    claims = _lst(d.get("unsupported_claims"))
    if claims:
        out += ["### Claims the evidence does not support", ""]
        for c in claims:
            out += [
                f"- **[{c.get('severity', '?')}]** \"{c.get('claim', '')}\"  \n"
                f"  _{c.get('problem', '')}_"
            ]
        out.append("")
    else:
        out += ["_The judge found no unsupported figures._", ""]

    for label, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses")):
        items = _lst(d.get(key))
        if items:
            out += [f"**{label}**", "", _bullets(items), ""]
    return "\n".join(out)