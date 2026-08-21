"""The agents: one function per role, each returning validated structured data.

Every agent follows the same three steps — assemble evidence, call the model
under a JSON schema, then clamp and normalise what comes back. That last step
matters more than it looks: strict structured outputs guarantee the *shape* of
the response but cannot express numeric bounds (see the note at the top of
`schemas.py`), so a model is free to return `confidence: 250`. Clamping here
means every downstream consumer — the risk agent, the report renderer, the
star ratings — can trust its inputs without re-checking them.

An agent that fails returns an `AgentResult` with `ok=False` rather than
raising, so one dead specialist degrades the report instead of sinking the run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import analytics, schemas
from .config import (
    ROLE_EVALUATOR,
    ROLE_LABELS,
    ROLE_PLANNER,
    ROLE_RISK,
    ROLE_STRATEGY,
    Settings,
)
from .datasets import DataRoom
from .llm import AgentCallError, BudgetExceeded, LLMBackend, extract_json
from .prompts import system_for
from .retrieval import HybridRetriever, format_hits


# ---------------------------------------------------------------------------
# result envelope
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    role: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0
    cached: bool = False
    evidence_chars: int = 0
    # Which retrieved passages this agent saw, for the provenance panel.
    citations: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role.title())

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "seconds": round(self.seconds, 2),
            "cached": self.cached,
            "evidence_chars": self.evidence_chars,
            "citations": list(self.citations),
        }


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _normalise_specialist(data: dict) -> dict:
    data["confidence"] = _clamp(data.get("confidence"), 0, 100, 50)
    for finding in _as_list(data.get("findings")):
        if isinstance(finding, dict):
            finding.setdefault("metric", "")
            finding.setdefault("direction", "flat")
            finding.setdefault("impact", "medium")
    return data


def _normalise_risk(data: dict) -> dict:
    data["overall_risk"] = _clamp(data.get("overall_risk"), 1, 5, 3)
    for risk in _as_list(data.get("risks")):
        if isinstance(risk, dict):
            risk["likelihood"] = _clamp(risk.get("likelihood"), 1, 5, 3)
            risk["impact"] = _clamp(risk.get("impact"), 1, 5, 3)
    for cat in _as_list(data.get("category_scores")):
        if isinstance(cat, dict):
            cat["score"] = _clamp(cat.get("score"), 1, 5, 3)
    return data


def _normalise_strategy(data: dict) -> dict:
    data["confidence"] = _clamp(data.get("confidence"), 0, 100, 50)
    return data


def _normalise_eval(data: dict) -> dict:
    data["overall_score"] = _clamp(data.get("overall_score"), 0, 100, 50)
    for row in _as_list(data.get("scores")):
        if isinstance(row, dict):
            row["score"] = _clamp(row.get("score"), 0, 100, 50)
    return data


# ---------------------------------------------------------------------------
# the one call every agent makes
# ---------------------------------------------------------------------------
def _call(
    backend: LLMBackend,
    role: str,
    prompt: str,
    schema: dict,
    normalise=None,
    citations: list[str] | None = None,
) -> AgentResult:
    started = time.perf_counter()
    try:
        completion = backend.complete(
            role=role, prompt=prompt, system=system_for(role), schema=schema
        )
        data = extract_json(completion.text)
        if not isinstance(data, dict):
            raise AgentCallError(f"expected a JSON object, got {type(data).__name__}")
        if normalise:
            data = normalise(data)
        return AgentResult(
            role=role,
            ok=True,
            data=data,
            seconds=time.perf_counter() - started,
            cached=completion.cached,
            evidence_chars=len(prompt),
            citations=list(citations or []),
        )
    except BudgetExceeded:
        # Not this agent's failure — the whole run is out of budget. Swallowing
        # it here would let the graph keep dispatching agents against a ceiling
        # that has already been hit, so it propagates.
        raise
    except Exception as exc:  # noqa: BLE001 — one agent must not sink the run
        return AgentResult(
            role=role,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - started,
            evidence_chars=len(prompt),
        )


# ---------------------------------------------------------------------------
# evidence assembly
# ---------------------------------------------------------------------------
# The market agent's sources are entirely prose, so retrieval replaces the
# digest there. Everywhere else retrieval only adds background context on top
# of figures that `analytics.py` has already computed.
_RETRIEVAL_DOMAINS = {
    "market": ["market", "profile"],
    "finance": ["profile"],
    "sales": ["profile", "market"],
    "customer": ["profile"],
    "operations": ["profile"],
}


def build_evidence(
    role: str,
    question: str,
    room: DataRoom,
    settings: Settings,
    retriever: HybridRetriever | None,
) -> tuple[str, list[str]]:
    """Return the evidence block for one specialist and the passages it cites."""
    sections: list[str] = []
    citations: list[str] = []

    if role != "market":
        sections.append(
            "## Computed figures\n"
            "Every number below was calculated directly from the company's "
            "files by code, not by a model.\n\n" + analytics.build_digest(role, room, settings)
        )
    else:
        # The market digest is a raw dump; retrieval below is the real input.
        sections.append(
            "## Market source overview\n"
            + analytics.build_digest("market", room, settings)[:2000]
        )

    if retriever is not None and settings.use_retrieval:
        domains = _RETRIEVAL_DOMAINS.get(role, ["profile"])
        hits = retriever.search(question, top_k=settings.retrieval_top_k)
        hits = [h for h in hits if _domain_of(h.passage.source) in domains]
        if hits:
            citations = [h.passage.citation for h in hits]
            sections.append(
                "## Retrieved passages\n"
                "Selected by hybrid search against the question being asked. "
                "Cite these by their [P#] label.\n\n" + format_hits(hits)
            )

    return "\n\n".join(sections), citations


def _domain_of(source: str) -> str:
    return source.split("/")[0] if "/" in source else "profile"


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------
def run_planner(
    backend: LLMBackend, settings: Settings, question: str, room: DataRoom
) -> AgentResult:
    enabled = ", ".join(settings.enabled_specialists)
    prompt = f"""\
# Executive question
{question}

# Company
{analytics.company_profile(room)[:1500]}

# Data room inventory
This is everything the team can see. Do not assign work that this data cannot
support.

{room.inventory_report()}

# Specialists available
{enabled}

Produce the analysis plan.
"""
    return _call(backend, ROLE_PLANNER, prompt, schemas.PLANNER_SCHEMA)


# ---------------------------------------------------------------------------
# 2-6. Specialists
# ---------------------------------------------------------------------------
def run_specialist(
    backend: LLMBackend,
    settings: Settings,
    role: str,
    question: str,
    task: str,
    why: str,
    evidence: str,
    citations: list[str],
) -> AgentResult:
    prompt = f"""\
# Executive question being decided
{question}

# Your assignment
{task}

Why this matters to the decision: {why}

# Evidence
{evidence}

Report your findings on this assignment, using only the evidence above.
"""
    return _call(
        backend,
        role,
        prompt,
        schemas.SPECIALIST_SCHEMA,
        _normalise_specialist,
        citations,
    )


# ---------------------------------------------------------------------------
# 7. Risk
# ---------------------------------------------------------------------------
def _findings_brief(results: list[AgentResult]) -> str:
    """Compact every specialist's output into one block for the later agents."""
    blocks = []
    for r in results:
        if not r.ok:
            blocks.append(f"## {r.label}\n(this agent failed: {r.error})")
            continue
        d = r.data
        lines = [f"## {r.label}", f"Headline: {d.get('headline', '')}", ""]
        for f in _as_list(d.get("findings")):
            metric = f.get("metric") or "n/a"
            lines.append(
                f"- [{f.get('impact', '?')} impact] {f.get('title', '')} "
                f"({metric}, {f.get('direction', '?')})\n"
                f"  {f.get('detail', '')}\n"
                f"  evidence: {f.get('evidence', 'unstated')}"
            )
        for label, key in (("Risks raised", "risks"), ("Opportunities", "opportunities")):
            items = _as_list(d.get(key))
            if items:
                lines.append(f"{label}: " + "; ".join(str(x) for x in items))
        gaps = _as_list(d.get("data_gaps"))
        if gaps:
            lines.append("Data gaps: " + "; ".join(str(g) for g in gaps))
        lines.append(f"Recommendation (own remit): {d.get('recommendation', '')}")
        lines.append(f"Self-assessed confidence: {d.get('confidence', '?')}/100")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run_risk(
    backend: LLMBackend,
    settings: Settings,
    question: str,
    plan: dict,
    results: list[AgentResult],
) -> AgentResult:
    prompt = f"""\
# Decision under consideration
{plan.get('restated_question', question)}

Decision type: {plan.get('decision_type', 'other')}

# Specialist findings
{_findings_brief(results)}

Assess the risks of proceeding. Pay particular attention to risks that only
become visible when two domains' findings are read together.
"""
    return _call(backend, ROLE_RISK, prompt, schemas.RISK_SCHEMA, _normalise_risk)


# ---------------------------------------------------------------------------
# 8. Strategy
# ---------------------------------------------------------------------------
def run_strategy(
    backend: LLMBackend,
    settings: Settings,
    question: str,
    plan: dict,
    results: list[AgentResult],
    risk: AgentResult,
) -> AgentResult:
    risk_block = (
        json.dumps(risk.data, indent=2)[:6000]
        if risk.ok
        else f"(risk assessment unavailable: {risk.error})"
    )
    criteria = _as_list(plan.get("success_criteria"))
    sub_questions = _as_list(plan.get("sub_questions"))
    prompt = f"""\
# Original question
{question}

# The decision, as framed by the planner
{plan.get('restated_question', question)}

Sub-questions this analysis had to resolve:
{chr(10).join(f'- {q}' for q in sub_questions) or '- (none recorded)'}

A good answer must demonstrate:
{chr(10).join(f'- {c}' for c in criteria) or '- (none recorded)'}

# Specialist findings
{_findings_brief(results)}

# Risk assessment
{risk_block}

Make the call.
"""
    return _call(
        backend, ROLE_STRATEGY, prompt, schemas.STRATEGY_SCHEMA, _normalise_strategy
    )


# ---------------------------------------------------------------------------
# 9. Evaluator
# ---------------------------------------------------------------------------
def run_evaluator(
    backend: LLMBackend,
    settings: Settings,
    question: str,
    report_markdown: str,
    evidence_seen: str,
) -> AgentResult:
    prompt = f"""\
# Question the report was asked to answer
{question}

# Evidence the agents were given
This is the ground truth. A figure in the report that does not appear here is
unsupported, however plausible it sounds.

{evidence_seen}

# The report
{report_markdown}

Grade the report.
"""
    return _call(
        backend, ROLE_EVALUATOR, prompt, schemas.EVAL_SCHEMA, _normalise_eval
    )