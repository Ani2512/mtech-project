"""Deterministic extraction and checking of hard output constraints.

Runs 1-3 measured prompt quality with an LLM judge blended with ROUGE-L and
token-F1. That metric turned out to be unusable for detecting improvement: the
judge scored the *baseline* prompt 0.975 out of 1.0, leaving 0.025 of headroom
on a component carrying 70% of the weight. Any real effect was smaller than the
judge's own noise, so all three runs returned p > 0.4 and none of them could
say whether the optimizer worked.

This module supplies the missing piece: a signal that is objective, free, and
has room to move. Around 900 Alpaca instructions state an explicit, countable
limit ("in less than 50 words", "in one sentence"). Whether an answer honours
it is a matter of counting, not of opinion -- no model call, no reference
answer, no judgement, no noise. A prompt that teaches the model to respect
stated limits scores higher here; one that encourages padding scores lower,
which is exactly the behaviour the Run 3 analysis suspected but could not
measure.

Only high-precision patterns are kept. A constraint extracted wrongly is worse
than one never extracted, because it penalises a correct answer and teaches the
optimizer the wrong lesson. Each pattern below was validated against Alpaca's
own reference answers and the failures inspected by hand; what survived, and
what was cut, is recorded in HOW_IT_WORKS.md section 12.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# --- number words ---------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a single": 1,
}


def _to_int(token: str) -> int | None:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


# --- patterns -------------------------------------------------------------
#
# Every pattern requires an explicit bounding preposition ("in", "to",
# "within", "using") or an explicit "or less" suffix, so a bare mention of a
# number beside the word "words" does not match.

_BOUND = (
    r"(?:less than|no more than|not more than|at most|fewer than|under|"
    r"up to|a maximum of|maximum of|max of)"
)

RE_WORDS = re.compile(
    rf"\b(?:in|to|within|using)\s+(?:{_BOUND}\s+)?(\d{{1,4}})\s+words?\b"
    rf"|\b(\d{{1,4}})\s+words?\s+or\s+(?:less|fewer)\b"
    rf"|\b{_BOUND}\s+(\d{{1,4}})\s+words?\b",
    re.I,
)

# "no less than 150 words", "at least 50 words" state a *floor*. They contain
# the substring "less than", so an unguarded max-pattern reads them backwards
# and penalises every answer that correctly writes more. Python's `re` has no
# variable-length lookbehind, so this is a separate veto pattern rather than a
# negative lookbehind inside RE_WORDS.
RE_WORDS_FLOOR = re.compile(
    r"\b(?:no|not)\s+(?:less|fewer)\s+than\s+\d{1,4}\s+words?\b"
    r"|\bat\s+least\s+\d{1,4}\s+words?\b"
    r"|\b(?:a\s+)?minimum\s+of\s+\d{1,4}\s+words?\b"
    # "3 sentences with a maximum of 40 words each" budgets per sentence, not
    # per answer; a whole-answer check reads it as an eighth of the real limit.
    r"|\b\d{1,4}\s+words?\s+(?:each|per|apiece)\b",
    re.I,
)

RE_SENTENCES_RANGE = re.compile(
    r"\b(\d{1,3})\s*(?:to|-|–|or)\s*(\d{1,3})\s+sentences?\b", re.I
)

# Deliberately excludes the bare article. "use the verb 'juggle' in a sentence"
# and "reverse the word order in a sentence" are task descriptions, not length
# limits; matching them cost 4 of the 8 sampled false positives. "a single
# sentence" survives because it is unambiguous.
_SENT_NUM = (
    r"(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|a single)"
)

RE_SENTENCES = re.compile(
    rf"\b(?:in|to|within|using|only)\s+(?:{_BOUND}\s+|exactly\s+)?"
    rf"({_SENT_NUM})\s+sentences?\b"
    rf"|\b({_SENT_NUM})\s+sentences?\s+or\s+(?:less|fewer)\b",
    re.I,
)

# "in a sentence or two" -- a real limit, but too loose to pin to a number, so
# the example is skipped rather than guessed at.
RE_SENTENCES_VAGUE = re.compile(r"\bsentences?\s+or\s+(?:two|three|so)\b", re.I)


@dataclass(frozen=True)
class Constraint:
    """One countable requirement stated by an instruction."""

    kind: str  # "max_words" | "max_sentences"
    limit: int
    source: str  # the matched instruction fragment, for reporting

    def describe(self) -> str:
        return {
            "max_words": f"at most {self.limit} words",
            "max_sentences": f"at most {self.limit} sentence(s)",
        }.get(self.kind, f"{self.kind}={self.limit}")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "limit": self.limit, "source": self.source}


def extract_constraints(instruction: str, input_text: str = "") -> list[Constraint]:
    """Find the hard, countable constraints an instruction states.

    At most one constraint per kind is returned; where an instruction states
    two limits of the same kind the first is taken, since Alpaca's phrasing
    puts the operative one first.
    """
    text = f"{instruction} {input_text}"
    found: list[Constraint] = []

    if not RE_WORDS_FLOOR.search(text):
        m = RE_WORDS.search(text)
        if m:
            limit = int(next(g for g in m.groups() if g))
            if 5 <= limit <= 1000:
                found.append(Constraint("max_words", limit, m.group(0).strip()))

    if not RE_SENTENCES_VAGUE.search(text):
        m = RE_SENTENCES_RANGE.search(text)
        if m:
            # "in 3 to 5 sentences" -- the upper bound is the binding one.
            found.append(
                Constraint("max_sentences", int(m.group(2)), m.group(0).strip())
            )
        else:
            m = RE_SENTENCES.search(text)
            if m:
                limit = _to_int(next(g for g in m.groups() if g))
                if limit and 1 <= limit <= 20:
                    found.append(
                        Constraint("max_sentences", limit, m.group(0).strip())
                    )

    return found


def has_constraint(instruction: str, input_text: str = "") -> bool:
    return bool(extract_constraints(instruction, input_text))


# --- counting -------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

_ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Mt.",
    "vs.", "etc.", "e.g.", "i.e.", "cf.", "al.", "Inc.", "Ltd.", "Co.",
    "No.", "Fig.", "Vol.", "approx.", "a.m.", "p.m.", "U.S.", "U.K.",
]

# "George H. W. Bush" is one sentence, not three: a lone capital followed by a
# period is an initial, so its dot is dropped before the split.
_INITIAL = re.compile(r"\b([A-Z])\.")

_LIST_MARKER = re.compile(
    r"^\s*(?:[-*•–●■]|\(?\d{1,2}[.)\]]|[a-z][.)])\s+", re.M
)

_SENT_END = re.compile(r"[.!?…]+(?=\s|$)")


def count_words(text: str) -> int:
    """Words, counting a hyphenated compound as one word."""
    return len(_WORD_RE.findall(text or ""))


def count_sentences(text: str) -> int:
    """Sentences, tolerating list formatting, initials and abbreviations.

    A list item carrying no terminal punctuation still counts as one sentence
    -- otherwise a bulleted answer to "in one sentence" would score as a single
    sentence no matter how far it ran.
    """
    text = (text or "").strip()
    if not text:
        return 0

    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", ""))
    protected = _INITIAL.sub(r"\1", protected)

    total = 0
    for line in protected.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        if not line:
            continue
        pieces = [p for p in _SENT_END.split(line) if p.strip()]
        total += max(len(pieces), 1)
    return total


# Item counting ("name three drinks") was implemented and then removed. Alpaca
# states an item count in ~2,800 instructions, but its reference answers name
# the requested items and then discuss them in prose, so no counter can
# separate "three items, elaborated" from "eight items". Checked against the
# gold answers it disagreed 24% of the time, against 8.5% for sentences -- and
# most of that 24% was the counter's fault, not the data's. Only what can be
# counted exactly is kept.

_COUNTERS = {
    "max_words": count_words,
    "max_sentences": count_sentences,
}


@dataclass
class ConstraintCheck:
    constraint: Constraint
    observed: int
    score: float
    satisfied: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.constraint.as_dict(),
            "observed": self.observed,
            "score": round(self.score, 4),
            "satisfied": self.satisfied,
        }

    def describe(self) -> str:
        verdict = "ok" if self.satisfied else "VIOLATED"
        return f"{self.constraint.describe()} -> observed {self.observed} [{verdict}]"


def check_constraint(constraint: Constraint, answer: str) -> ConstraintCheck:
    """Score one constraint on a graded 0..1 scale.

    Graded rather than binary so the optimizer sees how badly a prompt misses
    and not merely that it missed: overshooting a 50-word limit by five words
    is a different failure from writing 300 words, and a binary signal hides
    the difference. The score reaches 0 at twice the allowance.
    """
    observed = _COUNTERS[constraint.kind](answer)
    limit = max(constraint.limit, 1)
    overshoot = max(0, observed - constraint.limit)
    return ConstraintCheck(
        constraint=constraint,
        observed=observed,
        score=max(0.0, 1.0 - overshoot / limit),
        satisfied=overshoot == 0,
    )


def check_all(constraints: list[Constraint], answer: str) -> list[ConstraintCheck]:
    return [check_constraint(c, answer) for c in constraints]


def constraint_score(
    constraints: list[Constraint], answer: str
) -> tuple[float, list[ConstraintCheck]]:
    """Mean graded compliance across an example's constraints.

    Returns 1.0 when the instruction states no constraint, so unconstrained
    examples neither gain nor lose from this component.
    """
    if not constraints:
        return 1.0, []
    checks = check_all(constraints, answer)
    return sum(c.score for c in checks) / len(checks), checks


def describe_violations(checks: list[ConstraintCheck]) -> str:
    """One line naming what was violated, for the optimizer's error report."""
    bad = [c for c in checks if not c.satisfied]
    if not bad:
        return ""
    return "; ".join(
        f"the instruction required {c.constraint.describe()} but the answer "
        f"had {c.observed}"
        for c in bad
    )
