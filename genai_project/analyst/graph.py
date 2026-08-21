"""The orchestrator — a small state machine over the eight agents.

    question
        │
    [plan]                      one call: decide what to analyse and by whom
        │
        ├──────┬──────┬──────┬──────┐
     finance sales customer market operations     ← concurrent, one call each
        └──────┴──────┴──────┴──────┘
        │
    [risk]                      reads every finding at once
        │
    [strategy]                  makes the call
        │
    [report] → [evaluate]       render, then grade with an independent judge

This is written as an explicit graph rather than with an agent framework
because the control flow is fixed and knowable: the edges never change at
runtime, so a scheduler that reasons about them buys nothing and costs a
dependency. What the graph *does* need is honest handling of partial failure,
a hard budget ceiling, and the ability to report progress to a UI — all of
which are visible in this file rather than hidden in a library.

The only branch in the graph is the planner's: it chooses which specialists
run at all. Everything else is unconditional.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import agents, reporting
from .agents import AgentResult
from .config import (
    ROLE_EVALUATOR,
    ROLE_LABELS,
    ROLE_PLANNER,
    ROLE_RISK,
    ROLE_STRATEGY,
    RUNS_DIR,
    SPECIALIST_ROLES,
    Settings,
)
from .datasets import DataRoom
from .llm import BudgetExceeded, LLMBackend, Usage, make_backend
from .retrieval import HybridRetriever, build_corpus

# Called as progress(stage, status, detail) so a UI can render a live trace.
ProgressFn = Callable[[str, str, str], None]


def _noop(stage: str, status: str, detail: str = "") -> None:  # pragma: no cover
    pass


@dataclass
class AnalysisResult:
    """Everything one run produced — enough to render, audit, or replay it."""

    question: str
    settings: Settings
    plan: AgentResult
    specialists: list[AgentResult] = field(default_factory=list)
    risk: AgentResult | None = None
    strategy: AgentResult | None = None
    evaluation: AgentResult | None = None
    report_markdown: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    aborted: str = ""  # populated when the budget ceiling stopped the run

    @property
    def ok(self) -> bool:
        return bool(self.strategy and self.strategy.ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "settings": self.settings.to_dict(),
            "plan": self.plan.as_dict(),
            "specialists": [r.as_dict() for r in self.specialists],
            "risk": self.risk.as_dict() if self.risk else None,
            "strategy": self.strategy.as_dict() if self.strategy else None,
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
            "report_markdown": self.report_markdown,
            "usage": self.usage,
            "seconds": round(self.seconds, 2),
            "aborted": self.aborted,
        }

    def save(self, directory: pathlib.Path | None = None) -> pathlib.Path:
        directory = pathlib.Path(directory or RUNS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"run-{stamp}.json"
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path


# ---------------------------------------------------------------------------
# node 1 — plan
# ---------------------------------------------------------------------------
def _plan_node(
    backend: LLMBackend,
    settings: Settings,
    question: str,
    room: DataRoom,
    progress: ProgressFn,
) -> tuple[AgentResult, list[dict]]:
    progress(ROLE_PLANNER, "running", "breaking the question into assignments")
    result = agents.run_planner(backend, settings, question, room)

    if not result.ok:
        # Without a plan there are no assignments, so fall back to running
        # every enabled specialist against the raw question. A degraded
        # analysis beats no analysis.
        progress(ROLE_PLANNER, "failed", result.error)
        return result, [
            {
                "agent": role,
                "task": f"Analyse this question from a {ROLE_LABELS[role]} perspective: {question}",
                "why": "planner unavailable — running the full default roster",
                "priority": 1,
            }
            for role in settings.enabled_specialists
        ]

    assignments = [
        a
        for a in result.data.get("assignments", [])
        if isinstance(a, dict) and a.get("agent") in settings.enabled_specialists
    ]
    if not settings.planner_selects_agents:
        # The operator has overruled the planner: run everything enabled,
        # keeping whatever tasks the planner did write.
        assigned = {a["agent"] for a in assignments}
        for role in settings.enabled_specialists:
            if role not in assigned:
                assignments.append(
                    {
                        "agent": role,
                        "task": f"Analyse this question from a {ROLE_LABELS[role]} perspective: {question}",
                        "why": "included because agent selection is disabled",
                        "priority": 3,
                    }
                )
    if not assignments:
        progress(ROLE_PLANNER, "warning", "planner assigned no agents; running all")
        assignments = [
            {
                "agent": role,
                "task": question,
                "why": "fallback",
                "priority": 2,
            }
            for role in settings.enabled_specialists
        ]

    assignments.sort(key=lambda a: a.get("priority", 3))
    names = ", ".join(ROLE_LABELS[a["agent"]] for a in assignments)
    progress(ROLE_PLANNER, "done", f"dispatching {len(assignments)} agents: {names}")
    return result, assignments


# ---------------------------------------------------------------------------
# node 2 — specialists, concurrently
# ---------------------------------------------------------------------------
def _specialist_node(
    backend: LLMBackend,
    settings: Settings,
    question: str,
    room: DataRoom,
    retriever: HybridRetriever | None,
    assignments: list[dict],
    progress: ProgressFn,
) -> tuple[list[AgentResult], dict[str, str], BudgetExceeded | None]:
    evidence_by_role: dict[str, str] = {}
    prepared = []
    for a in assignments:
        role = a["agent"]
        progress(role, "preparing", "assembling evidence")
        evidence, citations = agents.build_evidence(
            role, question, room, settings, retriever
        )
        evidence_by_role[role] = evidence
        prepared.append((a, evidence, citations))

    results: list[AgentResult] = []
    budget_hit: BudgetExceeded | None = None
    workers = max(1, min(settings.concurrency, len(prepared)))
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(
                agents.run_specialist,
                backend,
                settings,
                a["agent"],
                question,
                a.get("task", question),
                a.get("why", ""),
                evidence,
                citations,
            ): a["agent"]
            for a, evidence, citations in prepared
        }
        for role in pending.values():
            progress(role, "running", "analysing")
        for future in futures.as_completed(pending):
            try:
                result = future.result()
            except BudgetExceeded as exc:
                # Drain the rest of the fan-out rather than abandoning it, so
                # the agents that did finish still reach the report. The caller
                # aborts once every future has been collected.
                budget_hit = budget_hit or exc
                progress(pending[future], "failed", str(exc))
                continue
            results.append(result)
            if result.ok:
                headline = result.data.get("headline", "")
                progress(result.role, "done", headline)
            else:
                progress(result.role, "failed", result.error)

    # Restore the planner's priority order; completion order is arbitrary.
    order = {a["agent"]: n for n, a in enumerate(assignments)}
    results.sort(key=lambda r: order.get(r.role, 99))
    return results, evidence_by_role, budget_hit


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------
def run_analysis(
    question: str,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
    room: DataRoom | None = None,
    backend: LLMBackend | None = None,
) -> AnalysisResult:
    """Run the full pipeline for one question."""
    settings = settings or Settings()
    progress = progress or _noop
    room = room or DataRoom()
    backend = backend or make_backend(settings)
    started = time.perf_counter()

    missing = room.missing()
    if missing:
        raise FileNotFoundError(
            "Data room incomplete — run `python make_sample_data.py` first. "
            f"Missing: {', '.join(missing[:5])}"
            + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )

    retriever = None
    if settings.use_retrieval:
        progress("retrieval", "running", "indexing documents")
        corpus = build_corpus(room)
        retriever = HybridRetriever(corpus, alpha=settings.hybrid_alpha)
        mode = "BM25 + LSA dense" if retriever.dense_available else "BM25 only"
        progress("retrieval", "done", f"{len(corpus)} passages indexed ({mode})")

    result = AnalysisResult(question=question, settings=settings, plan=AgentResult(ROLE_PLANNER, False))

    try:
        plan_result, assignments = _plan_node(
            backend, settings, question, room, progress
        )
        result.plan = plan_result

        result.specialists, evidence_by_role, budget_hit = _specialist_node(
            backend, settings, question, room, retriever, assignments, progress
        )
        if budget_hit is not None:
            raise budget_hit

        if settings.fail_fast and any(not r.ok for r in result.specialists):
            failed = ", ".join(r.role for r in result.specialists if not r.ok)
            raise RuntimeError(f"fail_fast is on and these agents failed: {failed}")

        progress(ROLE_RISK, "running", "scoring risk across every domain")
        result.risk = agents.run_risk(
            backend, settings, question, plan_result.data, result.specialists
        )
        progress(
            ROLE_RISK,
            "done" if result.risk.ok else "failed",
            result.risk.data.get("top_concern", "") if result.risk.ok else result.risk.error,
        )

        progress(ROLE_STRATEGY, "running", "synthesising the recommendation")
        result.strategy = agents.run_strategy(
            backend, settings, question, plan_result.data, result.specialists, result.risk
        )
        progress(
            ROLE_STRATEGY,
            "done" if result.strategy.ok else "failed",
            result.strategy.data.get("recommendation", "")
            if result.strategy.ok
            else result.strategy.error,
        )

        result.report_markdown = reporting.render_report(result)

        if settings.run_evaluation and result.ok:
            progress(ROLE_EVALUATOR, "running", "grading the report against the evidence")
            # The judge sees exactly what the specialists saw, so "is this
            # figure in the evidence?" is a question it can actually answer.
            evidence_seen = "\n\n".join(
                f"=== Evidence shown to the {ROLE_LABELS[role]} ===\n{text}"
                for role, text in evidence_by_role.items()
            )
            result.evaluation = agents.run_evaluator(
                backend, settings, question, result.report_markdown, evidence_seen
            )
            if result.evaluation.ok:
                score = result.evaluation.data.get("overall_score", "?")
                verdict = result.evaluation.data.get("verdict", "?")
                progress(ROLE_EVALUATOR, "done", f"{score}/100 — {verdict}")
            else:
                progress(ROLE_EVALUATOR, "failed", result.evaluation.error)

    except BudgetExceeded as exc:
        result.aborted = str(exc)
        progress("budget", "failed", str(exc))
        if not result.report_markdown:
            result.report_markdown = reporting.render_report(result)

    usage = getattr(backend, "usage", None)
    result.usage = usage.as_dict() if isinstance(usage, Usage) else {}
    result.seconds = time.perf_counter() - started
    return result