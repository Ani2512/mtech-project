"""The Automatic Prompt Optimization loop.

Algorithm (Pryzant et al., 2023), per iteration:

    for each prompt in the beam:
        evaluate on a training minibatch
        take the worst-scoring cases as evidence
        ask the optimizer model for `n_gradients` criticisms   (gradient)
        turn each criticism into `n_edits` rewritten prompts    (descent step)
        paraphrase each rewrite `n_paraphrases` times           (exploration)
    rank the whole candidate pool with a UCB bandit on the dev set
    keep the top `beam_width` as the next beam

The bandit matters: naively scoring every candidate on the full dev set is
quadratic in cost, while UCB concentrates evaluations on candidates that look
promising and cheaply discards the rest.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .backends import BudgetExceeded, LLMBackend, extract_json
from .config import ROLE_OPTIMIZER, ROLE_TASK, Settings
from .data import Example, Split
from .prompts import (
    EDIT_PROMPT,
    GRADIENT_PROMPT,
    PARAPHRASE_PROMPT,
    STRING_LIST_SCHEMA,
    format_failures,
)
from .scoring import ScoreBreakdown, paired_bootstrap, score_answer

ProgressFn = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    example: Example
    prediction: str
    score: ScoreBreakdown


@dataclass
class PromptEval:
    prompt: str
    results: list[EvalResult]

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score.total for r in self.results) / len(self.results)

    def worst(self, k: int) -> list[EvalResult]:
        return sorted(self.results, key=lambda r: r.score.total)[:k]


def run_prompt_on_example(
    backend: LLMBackend, prompt: str, example: Example, settings: Settings
) -> EvalResult:
    """Execute the prompt under test on one example and score the answer."""
    completion = backend.complete(
        role=ROLE_TASK, prompt=example.render_user_message(), system=prompt
    )
    breakdown = score_answer(
        example, completion.text, settings=settings, backend=backend
    )
    return EvalResult(example=example, prediction=completion.text, score=breakdown)


def evaluate_prompt(
    backend: LLMBackend,
    prompt: str,
    examples: Iterable[Example],
    settings: Settings,
    *,
    progress: ProgressFn = _noop,
    label: str = "",
) -> PromptEval:
    examples = list(examples)
    results: list[EvalResult] = []
    done = 0

    def work(ex: Example) -> EvalResult:
        return run_prompt_on_example(backend, prompt, ex, settings)

    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
        for res in pool.map(work, examples):
            results.append(res)
            done += 1
            progress(
                {
                    "event": "eval_progress",
                    "label": label,
                    "done": done,
                    "total": len(examples),
                }
            )
    return PromptEval(prompt=prompt, results=results)


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    text: str
    origin: str = "seed"
    parent: str | None = None
    scores: list[float] = field(default_factory=list)

    @property
    def pulls(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def ucb(self, total_pulls: int, c: float) -> float:
        if not self.scores:
            return math.inf
        return self.mean + c * math.sqrt(
            math.log(max(total_pulls, 2)) / len(self.scores)
        )


def _failure_payload(results: list[EvalResult]) -> list[dict[str, Any]]:
    return [
        {
            "instruction": r.example.instruction,
            "input": r.example.input,
            "reference": r.example.output,
            "prediction": r.prediction,
            "score": r.score.total,
            "critique": r.score.critique,
            "violations": r.score.violations,
        }
        for r in results
    ]


def _ask_for_list(
    backend: LLMBackend, prompt: str, expected: int
) -> list[str]:
    comp = backend.complete(
        role=ROLE_OPTIMIZER, prompt=prompt, schema=STRING_LIST_SCHEMA
    )
    try:
        items = extract_json(comp.text).get("items", [])
    except ValueError:
        return []
    out = [str(x).strip() for x in items if str(x).strip()]
    return out[:expected]


def generate_candidates(
    backend: LLMBackend,
    parent: Candidate,
    evaluation: PromptEval,
    settings: Settings,
    *,
    progress: ProgressFn = _noop,
) -> list[Candidate]:
    """One descent step: criticise the prompt, rewrite it, paraphrase it."""
    failures = _failure_payload(evaluation.worst(settings.errors_per_gradient))
    failure_text = format_failures(failures)

    gradients = _ask_for_list(
        backend,
        GRADIENT_PROMPT.format(
            prompt=parent.text, failures=failure_text, n=settings.n_gradients
        ),
        settings.n_gradients,
    )
    progress({"event": "gradients", "count": len(gradients), "gradients": gradients})

    new: list[Candidate] = []
    for gradient in gradients:
        edits = _ask_for_list(
            backend,
            EDIT_PROMPT.format(
                prompt=parent.text,
                gradient=gradient,
                failures=failure_text,
                n=settings.n_edits,
            ),
            settings.n_edits,
        )
        for edit in edits:
            new.append(
                Candidate(text=edit, origin=f"edit: {gradient[:80]}", parent=parent.text)
            )
            if settings.n_paraphrases > 0:
                paras = _ask_for_list(
                    backend,
                    PARAPHRASE_PROMPT.format(
                        prompt=edit, n=settings.n_paraphrases
                    ),
                    settings.n_paraphrases,
                )
                new.extend(
                    Candidate(text=p, origin="paraphrase", parent=edit) for p in paras
                )

    progress({"event": "candidates", "count": len(new)})
    return new


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        key = " ".join(c.text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --------------------------------------------------------------------------
# bandit selection
# --------------------------------------------------------------------------


def ucb_select(
    backend: LLMBackend,
    candidates: list[Candidate],
    dev: list[Example],
    settings: Settings,
    *,
    rng: random.Random,
    progress: ProgressFn = _noop,
) -> list[Candidate]:
    """Rank candidates with UCB1 and return the best `beam_width`."""
    if not candidates:
        return []
    for c in candidates:
        c.scores = []

    def pull(cand: Candidate, ex: Example) -> float:
        return run_prompt_on_example(backend, cand.text, ex, settings).score.total

    # One mandatory pull per arm, run in parallel.
    first_examples = [rng.choice(dev) for _ in candidates]
    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
        for cand, score in zip(
            candidates, pool.map(pull, candidates, first_examples)
        ):
            cand.scores.append(score)
    progress(
        {"event": "ucb_init", "candidates": len(candidates), "pulls": len(candidates)}
    )

    for i in range(settings.ucb_pulls):
        total = sum(c.pulls for c in candidates)
        best = max(candidates, key=lambda c: c.ucb(total, settings.ucb_c))
        best.scores.append(pull(best, rng.choice(dev)))
        progress(
            {
                "event": "ucb_pull",
                "done": i + 1,
                "total": settings.ucb_pulls,
                "best_mean": max(c.mean for c in candidates),
            }
        )

    ranked = sorted(candidates, key=lambda c: c.mean, reverse=True)
    return ranked[: settings.beam_width]


# --------------------------------------------------------------------------
# the optimization loop
# --------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    initial_prompt: str
    best_prompt: str
    best_dev_score: float
    history: list[dict[str, Any]]
    usage: dict[str, Any]
    stopped_early: bool = False
    stop_reason: str = ""
    elapsed_seconds: float = 0.0


def optimize(
    backend: LLMBackend,
    split: Split,
    settings: Settings,
    *,
    progress: ProgressFn = _noop,
) -> OptimizationResult:
    started = time.time()
    rng = random.Random(settings.seed)

    beam = [Candidate(text=settings.initial_prompt, origin="seed")]
    history: list[dict[str, Any]] = []
    best_prompt = settings.initial_prompt
    best_score = -1.0
    since_improved = 0
    stopped_early = False
    stop_reason = ""

    try:
        for iteration in range(1, settings.iterations + 1):
            progress({"event": "iteration_start", "iteration": iteration})
            minibatch = rng.sample(
                split.train, min(settings.minibatch_size, len(split.train))
            )

            pool: list[Candidate] = list(beam)
            iteration_evals: list[dict[str, Any]] = []

            for c_i, cand in enumerate(beam):
                ev = evaluate_prompt(
                    backend,
                    cand.text,
                    minibatch,
                    settings,
                    progress=progress,
                    label=f"iter {iteration} beam {c_i + 1}/{len(beam)}",
                )
                iteration_evals.append(
                    {
                        "prompt": cand.text,
                        "train_minibatch_score": ev.mean_score,
                        "origin": cand.origin,
                    }
                )
                progress(
                    {
                        "event": "beam_evaluated",
                        "iteration": iteration,
                        "index": c_i,
                        "score": ev.mean_score,
                    }
                )
                pool.extend(
                    generate_candidates(
                        backend, cand, ev, settings, progress=progress
                    )
                )

            pool = dedupe(pool)
            progress(
                {"event": "pool_ready", "iteration": iteration, "size": len(pool)}
            )

            beam = ucb_select(
                backend, pool, split.dev, settings, rng=rng, progress=progress
            )
            if not beam:
                stop_reason = "candidate pool collapsed to empty"
                stopped_early = True
                break

            top = beam[0]
            improved = top.mean > best_score
            if improved:
                best_score, best_prompt = top.mean, top.text
                since_improved = 0
            else:
                since_improved += 1

            history.append(
                {
                    "iteration": iteration,
                    "pool_size": len(pool),
                    "beam": [
                        {
                            "prompt": c.text,
                            "dev_mean": c.mean,
                            "pulls": c.pulls,
                            "origin": c.origin,
                        }
                        for c in beam
                    ],
                    "best_dev_score": best_score,
                    "best_prompt": best_prompt,
                    "improved": improved,
                    "minibatch_evals": iteration_evals,
                    "usage": backend.usage.as_dict(),
                }
            )
            progress(
                {
                    "event": "iteration_end",
                    "iteration": iteration,
                    "best_score": best_score,
                    "improved": improved,
                }
            )

            if since_improved >= settings.patience:
                stopped_early = True
                stop_reason = (
                    f"no dev improvement for {since_improved} iteration(s) "
                    f"(patience={settings.patience})"
                )
                break

    except BudgetExceeded as exc:
        stopped_early = True
        stop_reason = str(exc)
        progress({"event": "budget_exceeded", "message": str(exc)})

    return OptimizationResult(
        initial_prompt=settings.initial_prompt,
        best_prompt=best_prompt,
        best_dev_score=best_score,
        history=history,
        usage=backend.usage.as_dict(),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        elapsed_seconds=time.time() - started,
    )


# --------------------------------------------------------------------------
# held-out comparison
# --------------------------------------------------------------------------


def compare_on_test(
    backend: LLMBackend,
    baseline_prompt: str,
    optimized_prompt: str,
    test: list[Example],
    settings: Settings,
    *,
    progress: ProgressFn = _noop,
) -> dict[str, Any]:
    """Score both prompts on the held-out test split and test significance."""
    base = evaluate_prompt(
        backend, baseline_prompt, test, settings, progress=progress, label="test/baseline"
    )
    opt = evaluate_prompt(
        backend,
        optimized_prompt,
        test,
        settings,
        progress=progress,
        label="test/optimized",
    )

    base_scores = [r.score.total for r in base.results]
    opt_scores = [r.score.total for r in opt.results]
    stats = paired_bootstrap(opt_scores, base_scores, seed=settings.seed)

    wins = sum(1 for o, b in zip(opt_scores, base_scores) if o > b + 1e-9)
    losses = sum(1 for o, b in zip(opt_scores, base_scores) if o < b - 1e-9)
    ties = len(opt_scores) - wins - losses

    def metric_means(ev: PromptEval) -> dict[str, float]:
        n = max(len(ev.results), 1)
        out = {
            "total": sum(r.score.total for r in ev.results) / n,
            "judge": sum(
                (r.score.judge or 0.0) for r in ev.results
            ) / n,
            "rouge_l": sum(r.score.rouge_l for r in ev.results) / n,
            "token_f1": sum(r.score.token_f1 for r in ev.results) / n,
        }
        # Averaged only over examples that state a constraint -- including the
        # rest as 1.0 would dilute the number towards the share of the split
        # that happens to be constrained.
        con = [r.score.constraint for r in ev.results if r.score.constraint is not None]
        if con:
            out["constraint"] = sum(con) / len(con)
            out["constraint_n"] = len(con)
            out["constraint_pass_rate"] = sum(
                1 for r in ev.results if r.score.constraint == 1.0
            ) / len(con)
        return out

    return {
        "n_test": len(test),
        "baseline_prompt": baseline_prompt,
        "optimized_prompt": optimized_prompt,
        "baseline": metric_means(base),
        "optimized": metric_means(opt),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "significance": stats,
        "per_example": [
            {
                "instruction": r_b.example.instruction,
                "input": r_b.example.input,
                "reference": r_b.example.output,
                "baseline_prediction": r_b.prediction,
                "optimized_prediction": r_o.prediction,
                "baseline_score": r_b.score.total,
                "optimized_score": r_o.score.total,
                "constraint": (
                    None
                    if r_b.score.constraint is None
                    else {
                        "baseline": r_b.score.constraint,
                        "optimized": r_o.score.constraint,
                        "baseline_violation": r_b.score.violations,
                        "optimized_violation": r_o.score.violations,
                    }
                ),
            }
            for r_b, r_o in zip(base.results, opt.results)
        ],
    }