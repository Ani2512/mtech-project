"""Scoring: lexical metrics, an LLM-as-judge, and deterministic constraints.

Alpaca has no automatic metric that correlates well with answer quality —
open-ended references mean exact match is near-useless and ROUGE only
captures surface overlap. So the primary signal is an LLM judge scoring the
candidate against the reference on a 0-100 scale, blended with cheap lexical
metrics that anchor it and make the score reproducible when the judge is off.

Runs 1-3 exposed the flaw in that arrangement: under the lenient rubric the
judge scored the *baseline* prompt 0.975 out of 1.0, so the component carrying
70% of the weight had 0.025 of room left and no effect could be resolved above
its noise. Two switches address that, both off by default so the earlier runs
stay reproducible:

  `judge_rubric="strict"`   a deduction-based rubric with a stated ceiling
                            policy, so a merely-acceptable answer scores 70
                            rather than 95;
  `constraint_weight>0`     a deterministic component that counts whether the
                            answer honours the limits its instruction stated.

The second is the more important one: it costs nothing, cannot drift, and is
the only part of the metric that does not depend on a model's opinion.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .backends import LLMBackend, extract_json
from .config import ROLE_JUDGE, Settings
from .constraints import constraint_score, describe_violations, extract_constraints
from .data import Example

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    """Bag-of-words F1 — order-insensitive content overlap."""
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 1.0 if not p and not r else 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure (longest common subsequence), order-sensitive."""
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 1.0 if not p and not r else 0.0
    lcs = _lcs_length(p, r)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "Quality of the candidate answer, 0 (useless) to 100 (matches or beats the reference).",
        },
        "critique": {
            "type": "string",
            "description": "One or two sentences naming the concrete defect, or why it is good.",
        },
    },
    "required": ["score", "critique"],
    "additionalProperties": False,
}

# The strict rubric needs its own schema. A field description is not a comment:
# the model reads it as part of the instruction, and "100 (matches or beats the
# reference)" told it to award full marks to anything reference-quality — which
# silently cancelled the deduction scale in STRICT_JUDGE_PROMPT. Measured on 30
# pilot examples the two rubrics then scored within 0.005 of each other.
STRICT_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": (
                "Quality after applying every deduction in the rubric. 90+ is "
                "reserved for answers with no nameable defect; a merely "
                "acceptable answer belongs in the 70s."
            ),
        },
        "critique": {
            "type": "string",
            "description": "The single largest deduction applied, and why.",
        },
    },
    "required": ["score", "critique"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are grading an AI assistant's answer against a reference answer.

<instruction>
{instruction}
</instruction>

<input>
{input}
</input>

<reference_answer>
{reference}
</reference_answer>

<candidate_answer>
{candidate}
</candidate_answer>

Score the candidate 0-100 on how well it fulfils the instruction.

Grading rules:
- The reference is one acceptable answer, not the only one. A different but
  equally correct answer scores highly.
- Reward: factual correctness, following the exact format/constraints the
  instruction asks for, completeness, and appropriate length.
- Penalise: ignoring an explicit constraint, wrong facts, padding, refusing a
  benign request, answering a different question, or wrapping the answer in
  preamble the instruction did not ask for.
- Judge the answer only. Do not reward or punish style you merely dislike.

In `critique`, name the single most important concrete defect (or say what
made it strong). Be specific enough that a prompt engineer could act on it."""


STRICT_JUDGE_PROMPT = """You are grading an AI assistant's answer against a reference answer.

<instruction>
{instruction}
</instruction>

<input>
{input}
</input>

<reference_answer>
{reference}
</reference_answer>

<candidate_answer>
{candidate}
</candidate_answer>

Grade the candidate 0-100. Start at 100 and subtract for every defect you can
actually name:

- -40  ignores an explicit requirement in the instruction (length limit, item
       count, output format, required audience or point of view)
- -30  states something false, or asserts a specific fact the input does not
       support
- -25  omits a main point the instruction asked for
- -15  padding: sentences that add no information the instruction requested
- -15  answers a different or broader question than the one asked
- -10  unrequested preamble, sign-off, or commentary about the answer itself
- -10  markedly longer or shorter than the task warrants

Then check the total against these bands and adjust to fit:

- 90-100  you cannot name a single defect
- 70-89   fully usable; one minor defect
- 50-69   usable, but with a defect that a careful reader would notice
- 25-49   a significant failure of the task
- 0-24    wrong, off-task, refused, or empty

Calibration matters more than generosity. An answer that is merely acceptable
scores 70, not 95; reserve 90 and above for answers with nothing to criticise.
The reference is one acceptable answer, not the only one — a different but
equally correct answer is not penalised for differing, and a reference that
itself breaks the instruction does not license the candidate to break it.

In `critique`, name the single largest deduction you applied and why. Be
specific enough that a prompt engineer could act on it."""


@dataclass
class ScoreBreakdown:
    total: float
    judge: float | None
    rouge_l: float
    token_f1: float
    exact: float
    critique: str = ""
    # Deterministic compliance with the instruction's stated limits. None when
    # the instruction states none, or when constraint_weight is 0.
    constraint: float | None = None
    violations: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "judge": None if self.judge is None else round(self.judge, 4),
            "rouge_l": round(self.rouge_l, 4),
            "token_f1": round(self.token_f1, 4),
            "exact": self.exact,
            "constraint": (
                None if self.constraint is None else round(self.constraint, 4)
            ),
            "violations": self.violations,
            "critique": self.critique,
        }


def judge_answer(
    backend: LLMBackend, example: Example, candidate: str, *, rubric: str = "lenient"
) -> tuple[float, str]:
    """Return (score in 0..1, critique)."""
    strict = rubric == "strict"
    template = STRICT_JUDGE_PROMPT if strict else JUDGE_PROMPT
    schema = STRICT_JUDGE_SCHEMA if strict else JUDGE_SCHEMA
    prompt = template.format(
        instruction=example.instruction,
        input=example.input or "(none)",
        reference=example.output,
        candidate=candidate or "(the model produced no answer)",
    )
    comp = backend.complete(role=ROLE_JUDGE, prompt=prompt, schema=schema)
    try:
        data = extract_json(comp.text)
        score = float(data["score"])
        critique = str(data.get("critique", ""))
    except (ValueError, KeyError, TypeError):
        # A judge failure should not crash a long run; fall back to lexical.
        return token_f1(candidate, example.output), "judge output unparseable"
    return max(0.0, min(100.0, score)) / 100.0, critique


def score_answer(
    example: Example,
    candidate: str,
    *,
    settings: Settings,
    backend: LLMBackend | None = None,
) -> ScoreBreakdown:
    r = rouge_l(candidate, example.output)
    f1 = token_f1(candidate, example.output)
    em = exact_match(candidate, example.output)

    judge_score: float | None = None
    critique = ""
    if settings.use_judge and backend is not None:
        judge_score, critique = judge_answer(
            backend, example, candidate, rubric=settings.judge_rubric
        )

    con_score: float | None = None
    violations = ""
    if settings.constraint_weight > 0:
        constraints = extract_constraints(example.instruction, example.input)
        if constraints:
            con_score, checks = constraint_score(constraints, candidate)
            violations = describe_violations(checks)

    # Blend whatever components are actually in play, renormalising by their
    # weights so the total always spans 0..1 and switching one off does not
    # silently rescale the others. An example with no stated constraint simply
    # drops that term rather than scoring 1.0 on it, which would otherwise
    # inflate constrained and unconstrained examples differently.
    parts: list[tuple[float, float]] = [
        (settings.rouge_weight, r),
        (settings.f1_weight, f1),
    ]
    if judge_score is not None:
        parts.append((settings.judge_weight, judge_score))
    if con_score is not None:
        parts.append((settings.constraint_weight, con_score))

    denom = sum(w for w, _ in parts)
    if denom > 0:
        total = sum(w * v for w, v in parts) / denom
    else:
        total = statistics.fmean(v for _, v in parts) if parts else 0.0

    # Surface the violation to the optimizer through the same channel the
    # judge's critique uses, so the gradient step sees it as evidence.
    if violations:
        critique = f"{critique} Constraint violated: {violations}.".strip()

    return ScoreBreakdown(
        total=total,
        judge=judge_score,
        rouge_l=r,
        token_f1=f1,
        exact=em,
        critique=critique,
        constraint=con_score,
        violations=violations,
    )


# --- significance testing ------------------------------------------------


def paired_bootstrap(
    a: list[float], b: list[float], *, n_resamples: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap over per-example score differences (a - b).

    Returns the mean difference, a 95% CI, and a two-sided p-value for the
    null hypothesis that the mean difference is zero.
    """
    import random

    if len(a) != len(b):
        raise ValueError("paired_bootstrap requires equal-length score lists")
    if not a:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    diffs = [x - y for x, y in zip(a, b)]
    observed = statistics.fmean(diffs)
    rng = random.Random(seed)
    n = len(diffs)

    means: list[float] = []
    centered = [d - observed for d in diffs]
    extreme = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(diffs[i] for i in idx) / n)
        # null distribution: resample the centered differences
        null_mean = sum(centered[i] for i in idx) / n
        if abs(null_mean) >= abs(observed):
            extreme += 1

    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return {
        "mean_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": (extreme + 1) / (n_resamples + 1),
    }