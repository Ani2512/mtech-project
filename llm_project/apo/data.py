"""Alpaca dataset loading and splitting.

Source: https://huggingface.co/datasets/tatsu-lab/alpaca

The HF dataset is distributed as parquet; the identical records are also
published as plain JSON in the original Stanford Alpaca repository, which is
what we download so the project has no parquet/`datasets` dependency. If the
`datasets` package happens to be installed, `load_raw` will use it instead.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import urllib.request
from dataclasses import dataclass

from .config import DATA_DIR
from .constraints import has_constraint

ALPACA_JSON_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)
LOCAL_JSON = DATA_DIR / "alpaca_data.json"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def source_text_length(text: str) -> int:
    """Characters of usable source material, ignoring bare URLs.

    Alpaca was machine-generated, and a sizeable slice of its "summarize the
    following article" items supply only a URL or a headline as the input —
    the reference answer was hallucinated from a link the generating model
    could not read either. Those items are unanswerable: the only way to score
    well is to fabricate. Measuring the input with URLs removed is how we
    detect and drop them.
    """
    return len(_URL_RE.sub("", text or "").strip())


@dataclass(frozen=True)
class Example:
    idx: int
    instruction: str
    input: str
    output: str

    def render_user_message(self) -> str:
        """The user turn shown to the task model.

        The prompt being optimized is the *system* prompt; this rendering is
        held fixed so that any change in score is attributable to the prompt.
        """
        if self.input.strip():
            return f"Instruction:\n{self.instruction}\n\nInput:\n{self.input}"
        return f"Instruction:\n{self.instruction}"


@dataclass
class Split:
    train: list[Example]
    dev: list[Example]
    test: list[Example]

    def summary(self) -> str:
        return (
            f"train={len(self.train)} dev={len(self.dev)} test={len(self.test)}"
        )


def download_if_needed(dest: pathlib.Path = LOCAL_JSON) -> pathlib.Path:
    """Download alpaca_data.json once and cache it on disk."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        ALPACA_JSON_URL, headers={"User-Agent": "apo-project/1.0"}
    )
    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def load_raw() -> list[dict]:
    """Return the full 52k-row Alpaca dataset as a list of dicts."""
    try:  # optional fast path if the user has `datasets` installed
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [
            {
                "instruction": r["instruction"],
                "input": r["input"],
                "output": r["output"],
            }
            for r in ds
        ]
    except Exception:
        pass

    path = download_if_needed()
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_split(
    *,
    seed: int = 13,
    n_train: int = 40,
    n_dev: int = 40,
    n_test: int = 60,
    require_input: bool | None = None,
    filter_keyword: str | None = None,
    max_output_chars: int = 1200,
    min_input_chars: int = 0,
    require_constraint: bool = False,
    exclude_idx: set[int] | None = None,
) -> Split:
    """Deterministically sample disjoint train/dev/test splits.

    `require_input` and `filter_keyword` narrow the dataset to a task family,
    which is what prompt optimization actually targets — a prompt tuned for
    "rewrite this text" tasks is not the same prompt as one tuned for
    open-ended Q&A.

    `min_input_chars` drops items whose input carries too little real source
    text (URLs excluded) to be answerable. On the summarisation family this
    removes ~57% of items at a threshold of 200, and those items otherwise
    dominate the error signal without being fixable by any prompt.

    `require_constraint` keeps only instructions stating a countable limit
    ("in one sentence", "in less than 50 words") -- 791 of the 52k rows. This
    is the family the constraint metric can measure, and the only one where a
    prompt's effect on instruction-following is directly observable rather
    than mediated by a judge's opinion.

    `exclude_idx` withholds rows already used by an earlier run. Enlarging a
    run's test split is not the same experiment -- the whole split is drawn as
    one `rng.sample(pool, n_train + n_dev + n_test)`, so changing `n_test`
    reshuffles train and dev as well. Replicating a finding on more data means
    drawing a fresh sample disjoint from the original, which is what this is
    for.
    """
    rows = load_raw()
    pool: list[Example] = []
    keyword = (filter_keyword or "").lower().strip()
    excluded = exclude_idx or set()

    for i, r in enumerate(rows):
        if i in excluded:
            continue
        instruction = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if not instruction or not out:
            continue
        if len(out) > max_output_chars:
            continue
        if require_input is True and not inp:
            continue
        if require_input is False and inp:
            continue
        if min_input_chars > 0 and source_text_length(inp) < min_input_chars:
            continue
        if keyword and keyword not in f"{instruction} {inp}".lower():
            continue
        if require_constraint and not has_constraint(instruction, inp):
            continue
        pool.append(Example(idx=i, instruction=instruction, input=inp, output=out))

    needed = n_train + n_dev + n_test
    if len(pool) < needed:
        raise ValueError(
            f"Filter matched only {len(pool)} examples but {needed} are needed. "
            "Loosen filter_keyword/require_input or reduce the split sizes."
        )

    rng = random.Random(seed)
    sample = rng.sample(pool, needed)
    return Split(
        train=sample[:n_train],
        dev=sample[n_train : n_train + n_dev],
        test=sample[n_train + n_dev :],
    )


def split_from_settings(settings) -> Split:
    return build_split(
        seed=settings.seed,
        n_train=settings.n_train,
        n_dev=settings.n_dev,
        n_test=settings.n_test,
        require_input=settings.require_input,
        filter_keyword=settings.filter_keyword,
        max_output_chars=settings.max_output_chars,
        min_input_chars=settings.min_input_chars,
        require_constraint=settings.require_constraint,
    )