# How the code works

A code-level guide to the Automatic Prompt Optimization system. The `README`
covers *what* the project does and how to run it; this document covers *how the
implementation works* — the modules, the data structures, the control flow, and
the places you would extend it.

**Contents**

1. [The algorithm in one page](#1-the-algorithm-in-one-page)
2. [Module map](#2-module-map)
3. [Core data structures](#3-core-data-structures)
4. [End-to-end execution flow](#4-end-to-end-execution-flow)
5. [The optimization loop in detail](#5-the-optimization-loop-in-detail)
6. [The scoring subsystem](#6-the-scoring-subsystem)
7. [The backend layer](#7-the-backend-layer)
8. [Data loading and filtering](#8-data-loading-and-filtering)
9. [Held-out evaluation and statistics](#9-held-out-evaluation-and-statistics)
10. [Configuration reference](#10-configuration-reference)
11. [Extension points](#11-extension-points)
12. [Empirical findings](#12-empirical-findings)
13. [Known limitations](#13-known-limitations)

---

## 1. The algorithm in one page

The system implements *Automatic Prompt Optimization with "Gradient Descent"
and Beam Search* (Pryzant et al., EMNLP 2023). No numerical gradients are
involved. The analogy is structural:

| Numerical gradient descent | This system |
|---|---|
| Loss on a minibatch | Mean score on a training minibatch |
| ∂loss/∂θ | An LLM-written criticism of the prompt ("textual gradient") |
| θ ← θ − α∇ | An LLM rewriting the prompt to answer that criticism |
| Random restarts | Paraphrasing the rewrite |
| Model selection | UCB1 bandit over candidates on a dev split |

One iteration, in words: run the current prompt on a training minibatch, grade
every answer, take the worst cases, ask a model *what is wrong with the prompt
that produced these*, ask it to rewrite the prompt accordingly, paraphrase the
rewrites, then rank the whole candidate pool on dev examples and keep the best
few. Repeat.

The optimized artifact is a **system prompt**. The user turn is produced by a
fixed template (`Example.render_user_message`), so any change in score is
attributable to the prompt rather than to how the task was presented.

---

## 2. Module map

```
apo/
├── config.py      Settings dataclass, provider registry, pricing, key resolution
├── data.py        Alpaca download, filtering, deterministic train/dev/test splits
├── backends.py    Provider clients, disk cache, cost budget, offline mock
├── scoring.py     ROUGE-L / token-F1 / LLM judge, paired bootstrap
├── prompts.py     The three meta-prompts and the failure formatter
├── optimizer.py   Evaluation, candidate generation, UCB selection, main loop
└── reporting.py   Budget estimation, run artifacts, markdown report
```

Top-level drivers, none of which contain algorithm logic:

```
run_optimize.py       CLI: one full search + held-out comparison
run_replicate.py      Re-score a finished run's prompt on fresh disjoint data
probe_headroom.py     ~$0.04 precondition check: does this task model leave the
                      baseline below ceiling? Run before spending on a search.
run_weak_executor.sh  Driver for the Run 6/7 arm: probe | estimate | optimize |
                      replicate
app.py                Streamlit UI over the same pipeline
list_models.py        Enumerate models the configured keys can reach
```

Dependency direction (nothing points backwards):

```
config ──┬──> data ──────┐
         ├──> backends ──┼──> optimizer ──> reporting
         └──> scoring <──┘         ^
                  prompts ─────────┘
```

`config` depends on nothing in the package. `optimizer` is the only module that
orchestrates; every other module is a library it calls. Neither `app.py` nor
`run_optimize.py` contains algorithm logic — both are thin drivers over
`optimizer.optimize()`.

---

## 3. Core data structures

All are plain dataclasses. There is no ORM, no global state, and no mutable
module-level configuration.

### `Example` — one Alpaca task (`data.py`)

```python
@dataclass(frozen=True)
class Example:
    idx: int          # row index in the original 52k dataset (for reproducibility)
    instruction: str
    input: str        # may be empty
    output: str       # the reference answer

    def render_user_message(self) -> str: ...
```

`render_user_message()` is the fixed template:

```
Instruction:
{instruction}

Input:
{input}          # omitted entirely when input is empty
```

### `Split` — the three disjoint slices

```python
@dataclass
class Split:
    train: list[Example]   # source of failures -> criticisms
    dev:   list[Example]   # candidate selection (the bandit)
    test:  list[Example]   # touched once, at the end
```

The separation is what makes the final number meaningful. Improving and
verifying on the same examples cannot distinguish a better prompt from a lucky
one.

### `Completion` and `Usage` — backend results (`backends.py`)

```python
@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_role: dict[str, int] = field(default_factory=dict)
```

### `ScoreBreakdown` — one graded answer (`scoring.py`)

```python
@dataclass
class ScoreBreakdown:
    total: float          # weighted blend, 0..1 — what the optimizer maximises
    judge: float | None   # None when the judge is disabled
    rouge_l: float
    token_f1: float
    exact: float
    critique: str = ""    # the judge's stated reason; feeds the gradient step
```

`critique` is the field that makes the method work. Without it the critic would
see only numbers; with it, the critic reads *why* each answer was marked down.

### `EvalResult` / `PromptEval` — a prompt's performance (`optimizer.py`)

```python
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
    def mean_score(self) -> float: ...      # how good this prompt is
    def worst(self, k: int) -> list[EvalResult]: ...   # what to learn from
```

### `Candidate` — a prompt competing in the search

```python
@dataclass
class Candidate:
    text: str
    origin: str = "seed"        # "seed" | "edit: <criticism>" | "paraphrase"
    parent: str | None = None   # the prompt it was derived from
    scores: list[float] = field(default_factory=list)   # one per bandit pull

    @property
    def pulls(self) -> int: ...
    @property
    def mean(self) -> float: ...
    def ucb(self, total_pulls: int, c: float) -> float: ...
```

`origin` and `parent` give every prompt in the final report a traceable
lineage back to the criticism that produced it.

### `OptimizationResult` — the run's output

```python
@dataclass
class OptimizationResult:
    initial_prompt: str
    best_prompt: str
    best_dev_score: float
    history: list[dict]      # per-iteration record, serialised to result.json
    usage: dict
    stopped_early: bool = False
    stop_reason: str = ""
    elapsed_seconds: float = 0.0
```

---

## 4. End-to-end execution flow

From `run_optimize.py main()`:

```
1.  settings_from_args(args)              -> Settings
2.  split_from_settings(settings)         -> Split          (downloads Alpaca once)
3.  estimate_budget(settings)             -> call/cost projection, printed
4.  [confirmation gate unless --yes]
5.  make_backend(settings)                -> OpenAI | Anthropic | Mock backend
6.  optimize(backend, split, settings)    -> OptimizationResult
7.  compare_on_test(...)                  -> held-out comparison + significance
8.  save_run(run_dir, ...)                -> runs/<timestamp>/ artifacts
```

`app.py` performs exactly the same sequence; the only difference is that its
`progress` callback writes to Streamlit widgets instead of stdout.

### The progress callback

Every long-running function accepts `progress: Callable[[dict], None]`. Events
are plain dicts with an `"event"` key:

```python
{"event": "iteration_start", "iteration": 2}
{"event": "eval_progress", "label": "...", "done": 3, "total": 8}
{"event": "gradients", "count": 2, "gradients": [...]}
{"event": "ucb_pull", "done": 17, "total": 120, "best_mean": 0.71}
{"event": "iteration_end", "iteration": 2, "best_score": 0.74, "improved": True}
```

This is why the CLI and the web UI can share the entire engine without the
engine importing either.

---

## 5. The optimization loop in detail

### The driver

```python
def optimize(backend, split, settings, *, progress=_noop):
    rng = random.Random(settings.seed)
    beam = [Candidate(text=settings.initial_prompt, origin="seed")]
    best_prompt, best_score, since_improved = settings.initial_prompt, -1.0, 0

    for iteration in range(1, settings.iterations + 1):
        minibatch = rng.sample(split.train, settings.minibatch_size)
        pool = list(beam)

        for cand in beam:
            ev = evaluate_prompt(backend, cand.text, minibatch, settings)
            pool.extend(generate_candidates(backend, cand, ev, settings))

        pool = dedupe(pool)
        beam = ucb_select(backend, pool, split.dev, settings, rng=rng)

        if beam[0].mean > best_score:
            best_score, best_prompt = beam[0].mean, beam[0].text
            since_improved = 0
        else:
            since_improved += 1
        if since_improved >= settings.patience:
            break
```

`beam` carries the surviving prompts between iterations. Everything else is
recomputed per iteration — deliberately, so that a candidate's score always
comes from a fresh sample rather than accumulating across iterations.

### Step 1 — evaluation

```python
def evaluate_prompt(backend, prompt, examples, settings, ...):
    def work(ex):
        return run_prompt_on_example(backend, prompt, ex, settings)
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        results = list(pool.map(work, examples))
    return PromptEval(prompt=prompt, results=results)


def run_prompt_on_example(backend, prompt, example, settings):
    completion = backend.complete(
        role=ROLE_TASK,
        prompt=example.render_user_message(),
        system=prompt,                    # <- the prompt under test
    )
    breakdown = score_answer(example, completion.text,
                             settings=settings, backend=backend)
    return EvalResult(example=example, prediction=completion.text, score=breakdown)
```

Each example costs **two** API calls when the judge is enabled: one to produce
an answer, one to grade it. `ThreadPoolExecutor` runs `settings.concurrency`
examples at a time; `Usage` accounting is mutex-protected in `_BaseBackend`.

### Step 2 — candidate generation (the descent step)

```python
def generate_candidates(backend, parent, evaluation, settings, ...):
    failures = _failure_payload(evaluation.worst(settings.errors_per_gradient))
    failure_text = format_failures(failures)

    # (a) gradient: criticise the prompt using failures as evidence
    gradients = _ask_for_list(backend,
        GRADIENT_PROMPT.format(prompt=parent.text, failures=failure_text,
                               n=settings.n_gradients),
        settings.n_gradients)

    new = []
    for gradient in gradients:
        # (b) descent: rewrite the whole prompt to answer that criticism
        edits = _ask_for_list(backend,
            EDIT_PROMPT.format(prompt=parent.text, gradient=gradient,
                               failures=failure_text, n=settings.n_edits),
            settings.n_edits)
        for edit in edits:
            new.append(Candidate(text=edit, origin=f"edit: {gradient[:80]}",
                                 parent=parent.text))
            # (c) exploration: reword the rewrite
            paras = _ask_for_list(backend,
                PARAPHRASE_PROMPT.format(prompt=edit, n=settings.n_paraphrases),
                settings.n_paraphrases)
            new.extend(Candidate(text=p, origin="paraphrase", parent=edit)
                       for p in paras)
    return new
```

The evidence bundle passed to the critic is:

```python
{"instruction": ..., "input": ..., "reference": ...,
 "prediction": ..., "score": ..., "critique": ...}
```

`format_failures()` renders it into labelled blocks and clips long fields to
600 characters, so a few verbose examples cannot crowd out the instructions in
the meta-prompt.

**Branching factor** is `n_gradients × n_edits × (1 + n_paraphrases)` children
per beam member. At defaults (2 × 1 × 2) a beam of 2 produces a pool of ~10.

### Step 3 — structured output

All three meta-prompt calls funnel through one helper:

```python
def _ask_for_list(backend, prompt, expected):
    comp = backend.complete(role=ROLE_OPTIMIZER, prompt=prompt,
                            schema=STRING_LIST_SCHEMA)
    try:
        items = extract_json(comp.text).get("items", [])
    except ValueError:
        return []                      # a malformed reply costs a candidate, not the run
    return [str(x).strip() for x in items if str(x).strip()][:expected]
```

`STRING_LIST_SCHEMA` is `{"items": [str, ...]}` with `additionalProperties:
false` and `strict: true`, so the provider constrains generation to valid JSON.
`extract_json()` is still used as a belt-and-braces parser — it tries a direct
`json.loads`, then a fenced ```` ```json ```` block, then the outermost
`{...}` span.

### Step 4 — deduplication

```python
def dedupe(candidates):
    seen, out = set(), []
    for c in candidates:
        key = " ".join(c.text.lower().split())
        if key and key not in seen:
            seen.add(key); out.append(c)
    return out
```

Paraphrasing regularly regenerates a prompt already in the pool. Deduplicating
before selection avoids paying to evaluate the same text twice.

### Step 5 — UCB selection

```python
def ucb_select(backend, candidates, dev, settings, *, rng, ...):
    for c in candidates:
        c.scores = []

    def pull(cand, ex):
        return run_prompt_on_example(backend, cand.text, ex, settings).score.total

    # one mandatory pull per arm, in parallel
    first = [rng.choice(dev) for _ in candidates]
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        for cand, score in zip(candidates, pool.map(pull, candidates, first)):
            cand.scores.append(score)

    # remaining budget spent adaptively
    for _ in range(settings.ucb_pulls):
        total = sum(c.pulls for c in candidates)
        best = max(candidates, key=lambda c: c.ucb(total, settings.ucb_c))
        best.scores.append(pull(best, rng.choice(dev)))

    return sorted(candidates, key=lambda c: c.mean, reverse=True)[:settings.beam_width]
```

with the selection rule on `Candidate`:

```python
def ucb(self, total_pulls, c):
    if not self.scores:
        return math.inf                  # untried arms are always tried first
    return self.mean + c * math.sqrt(math.log(max(total_pulls, 2)) / len(self.scores))
```

`mean` is exploitation; the square-root term is exploration and shrinks as an
arm accumulates pulls. A candidate that scores poorly twice stops receiving
budget while a promising one keeps earning it.

**One pull = one dev example.** Total dev evaluations per iteration are
therefore `len(pool) + ucb_pulls`, and mean pulls per candidate are
`1 + ucb_pulls / len(pool)`. This ratio is the single most important knob in the
system — see [Empirical findings](#12-empirical-findings).

**The adaptive phase is inherently sequential**: each result determines which
arm to pull next, so it cannot be parallelised as written. On large
`ucb_pulls` it dominates wall-clock. Batched UCB (pull the top-*k* arms
simultaneously) is the standard remedy and is not implemented here.

### Termination

Three ways a run ends:

| Condition | Mechanism |
|---|---|
| Iterations exhausted | `for` loop completes |
| No dev improvement for `patience` iterations | `since_improved` counter breaks the loop |
| Call or cost ceiling hit | `BudgetExceeded` raised in `_check_budget()`, caught in `optimize()` |

All three return a well-formed `OptimizationResult`; the budget case sets
`stopped_early=True` and `stop_reason`, so a truncated run still produces the
best prompt found and a complete report rather than losing the work.

---

## 6. The scoring subsystem

### Why a blend

Alpaca references are open-ended. Exact match is near-useless — a correct
answer phrased differently scores zero. So `score_answer()` blends three
signals:

```python
total = (judge_weight * judge + rouge_weight * rouge_l + f1_weight * token_f1)
        / (judge_weight + rouge_weight + f1_weight)
```

Defaults are 0.70 / 0.15 / 0.15. When `use_judge=False` the lexical weights are
renormalised so `total` still spans 0..1 and remains comparable across runs.

### The lexical metrics

Both are implemented directly, with no NLP dependency:

```python
def token_f1(pred, ref):      # bag-of-words F1 via Counter intersection
def rouge_l(pred, ref):       # F-measure over the longest common subsequence
def _lcs_length(a, b):        # O(len(a) x len(b)) dynamic programme, O(len(b)) memory
```

`token_f1` is order-insensitive; `rouge_l` is order-sensitive. Keeping both
means a scrambled answer is penalised by one and not the other, which is
diagnostic when the two diverge.

### The judge

```python
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},    # 0..100
        "critique": {"type": "string"},
    },
    "required": ["score", "critique"],
    "additionalProperties": False,
}
```

The judge prompt instructs the model to treat the reference as *one acceptable
answer rather than the only one*, to reward constraint-following and penalise
padding or refusal of benign requests, and to name the single most important
concrete defect in `critique`.

Failure handling is deliberate:

```python
except (ValueError, KeyError, TypeError):
    return token_f1(candidate, example.output), "judge output unparseable"
```

A malformed judge reply degrades that one example to a lexical score instead of
aborting a run that may already have cost several dollars.

---

## 7. The backend layer

Every model call in the project goes through one method:

```python
class LLMBackend(Protocol):
    def complete(self, *, role: str, prompt: str,
                 system: str | None = None,
                 schema: dict | None = None) -> Completion: ...
```

`role` is one of `ROLE_TASK`, `ROLE_JUDGE`, `ROLE_OPTIMIZER`, and it selects
the model, token limit, and reasoning effort via `Settings.model_for(role)`,
`max_tokens_for(role)`, `effort_for(role)`. This is what allows a cheap model
to answer and grade while a stronger one does the rewriting.

### `_BaseBackend` — shared machinery

Three responsibilities that every live provider inherits:

| Concern | Method | Behaviour |
|---|---|---|
| Budget | `_check_budget()` | Raises `BudgetExceeded` past `max_llm_calls` or `max_cost_usd` |
| Accounting | `_record(role, completion)` | Mutex-guarded token/cost/call tallies, per role |
| Cache | `_cache_get/_cache_put` | SHA-256 of the full request payload → `.cache/<xx>/<hash>.json` |

Cache hits are recorded as calls but cost nothing, which is why re-running an
experiment with one changed parameter is far cheaper than the first run.

### `OpenAIBackend` (default)

Written against the GPT-5.6 parameter surface:

```python
kwargs = {
    "model": model,
    "messages": messages,                     # system message first, if present
    "max_completion_tokens": max_tokens,      # not `max_tokens`
    "reasoning_effort": effort,               # low | medium | high | xhigh | max
}
if schema is not None:
    kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": schema, "strict": True},
    }
resp = self.client.chat.completions.create(**kwargs)
```

Sampling parameters (`temperature`, `top_p`) are never sent. A safety refusal
arrives as `content=None` with `refusal` populated and is converted to an empty
string rather than raising.

### `AnthropicBackend`

The same contract against Claude: `system=` as a top-level parameter,
`output_config={"effort": ..., "format": {...}}`, and text assembled from
`resp.content` blocks. Retained so the judge can run on a different model
family from the task model.

### `MockBackend`

A deterministic offline stub that keeps the whole pipeline runnable with no key
and no cost. It returns schema-valid JSON for the optimizer roles and, for the
judge role, computes `token_f1` against the reference so scoring behaves
sensibly end-to-end.

Its task model **does not attempt the instruction**, so mock scores are
structurally valid and semantically meaningless. Every report generated under
the mock backend is stamped as such.

### Dispatch

```python
def make_backend(settings):
    if settings.backend == PROVIDER_MOCK:      return MockBackend(settings)
    if settings.backend == PROVIDER_ANTHROPIC: return AnthropicBackend(settings)
    if settings.backend == PROVIDER_OPENAI:    return OpenAIBackend(settings)
    raise ValueError(...)
```

Nothing above this layer knows which provider is in use.

---

## 8. Data loading and filtering

```python
def build_split(*, seed, n_train, n_dev, n_test,
                require_input=None, filter_keyword=None,
                max_output_chars=1200, min_input_chars=0) -> Split
```

The pipeline: load 52k rows → apply filters → `random.Random(seed).sample()` of
`n_train + n_dev + n_test` → slice into three disjoint lists. Same seed and
same filters always reproduce the same split, and `split.json` records every
row index for the record.

| Filter | Purpose |
|---|---|
| `require_input` | `True` = only items with an `input`; `False` = bare instructions |
| `filter_keyword` | Narrow to a task family (`summarize`, `classify`, …) |
| `max_output_chars` | Drop pathologically long references |
| `min_input_chars` | Drop unanswerable items — see below |

### `min_input_chars` and unanswerable items

```python
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

def source_text_length(text: str) -> int:
    return len(_URL_RE.sub("", text or "").strip())
```

Alpaca is machine-generated, and a large fraction of its *"summarize the
following article"* items supply only a URL or a headline as input — the
reference answer was hallucinated from a link the generating model could not
read either. These items are unanswerable: the only way to score well is to
fabricate. Measuring input length with URLs removed is how they are detected
and dropped. Quantified in [Empirical findings](#12-empirical-findings).

---

## 9. Held-out evaluation and statistics

```python
def compare_on_test(backend, baseline_prompt, optimized_prompt, test, settings, ...):
    base = evaluate_prompt(backend, baseline_prompt, test, settings)
    opt  = evaluate_prompt(backend, optimized_prompt, test, settings)
    stats = paired_bootstrap([r.score.total for r in opt.results],
                             [r.score.total for r in base.results],
                             seed=settings.seed)
```

Both prompts are scored on the **same** examples, so the comparison is paired
and per-example difficulty cancels out.

### The significance test

```python
def paired_bootstrap(a, b, *, n_resamples=10_000, seed=0) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    observed = statistics.fmean(diffs)
    centered = [d - observed for d in diffs]      # the null distribution
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(diffs[i] for i in idx) / n)          # -> CI
        if abs(sum(centered[i] for i in idx) / n) >= abs(observed):
            extreme += 1                                       # -> p-value
    return {"mean_diff": observed, "ci_low": ..., "ci_high": ...,
            "p_value": (extreme + 1) / (n_resamples + 1)}
```

Resampling the raw differences gives the 95% confidence interval; resampling
the *centered* differences simulates the null hypothesis and gives a two-sided
p-value. The `+1` terms are the standard correction that keeps the p-value from
ever being exactly zero.

`build_report()` prints the verdict in words — "the improvement is / is NOT
statistically significant at p < 0.05" — so a null result cannot be quietly
reported as a win.

---

## 10. Configuration reference

Every knob is a field on the `Settings` dataclass, serialised into each run's
`config.json`.

| Group | Fields |
|---|---|
| **Models** | `task_model`, `judge_model`, `optimizer_model`, `*_effort`, `*_max_tokens` |
| **Data** | `seed`, `n_train`, `n_dev`, `n_test`, `require_input`, `filter_keyword`, `max_output_chars`, `min_input_chars` |
| **Search** | `iterations`, `minibatch_size`, `beam_width`, `n_gradients`, `n_edits`, `n_paraphrases`, `errors_per_gradient`, `ucb_pulls`, `ucb_c`, `patience` |
| **Scoring** | `use_judge`, `judge_weight`, `rouge_weight`, `f1_weight` |
| **Execution** | `concurrency`, `use_cache`, `backend`, `max_llm_calls`, `max_cost_usd` |
| **Prompt** | `initial_prompt` |

`reporting.estimate_budget()` projects call counts and USD from these before a
run starts, using per-role average token assumptions. It is deliberately rough
— its job is preventing an accidental $50 run, not precise accounting.

---

## 11. Extension points

### Add a provider

Subclass `_BaseBackend`, implement `complete()`, register it in
`make_backend()`, and add the model to `MODEL_PRICING` / `PROVIDER_MODELS`.
Nothing else changes — the optimizer, scoring and UI are provider-agnostic. The
OpenAI backend was added this way in about 40 lines.

### Add a metric

Write the function in `scoring.py`, add a field to `ScoreBreakdown`, and give
it a weight in `score_answer()`. Weights are renormalised, so the total stays
0..1 and remains comparable to previous runs.

### Change the search strategy

`ucb_select()` is a pure function of `(candidates, dev, settings)` returning a
ranked sublist. Successive halving, full-dev evaluation, or batched UCB can
replace it without touching anything else.

### Change the meta-prompts

`prompts.py` holds all three as module-level strings. This is the highest-
leverage file for improving results — the quality of the criticisms is what
drives the whole method.

### Testing

56 tests, all offline via `MockBackend` and stub clients:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

`tests/test_apo.py` covers metrics, statistics, splitting, budget and an
end-to-end mock run; `tests/test_openai_backend.py` verifies request shaping
against a stub client; `tests/test_app.py` executes the Streamlit app headlessly
via `AppTest`.

---

## 12. Empirical findings

Seven live experiments. Runs 1–3 use the summarisation family and all produced
**null results**; the reasons are worth recording, because they are not the same
reason each time. Run 4 changes the *measurement* rather than the data and is
the first to produce a positive effect. Run 5 is a replication of Run 4 on
disjoint data at the sample size Run 4's own power analysis called for, and it
comes back null.

Runs 1–5 all run the task role on `gpt-5.6-luna`. Their combined verdict — that
a frontier model on Alpaca has no headroom for a prompt to win in — is a
falsifiable claim, so Runs 6–7 test it by moving the task role onto
`gpt-3.5-turbo` and changing nothing else. That arm produces the project's only
effect that survives replication.

### Run 1 — underpowered (12/12/12, $0.21)

| | |
|---|---|
| Dev | 0.7599 → 0.8312 |
| Test | 0.7246 → 0.7075 (−0.0171, p = 0.59) |

Diagnosis, computed from the run's own artifacts:

- Winning candidates were ranked on **4–8 dev examples each**
  (`ucb_pulls=24` spread over a 10-candidate pool).
- Example difficulty has standard deviation **0.176**, while a real prompt
  effect is **0.02–0.05** — noise roughly 5× the signal.
- At n=12 test examples the minimum detectable difference was **0.056**, above
  the entire plausible effect range. The experiment could not have returned a
  significant positive result regardless of prompt quality.

### Run 2 — properly powered (40/40/60, `ucb_pulls=120`, $0.70)

| | |
|---|---|
| Dev | 0.7064 → 0.7415 → 0.7439 |
| Test | 0.7051 → 0.6958 (−0.0093, p = 0.487) |
| Win / loss / tie | 28 / 26 / 6 |
| 95% CI | [−0.039, +0.013] |

A genuine null: the CI excludes any gain larger than +0.013 and win/loss is a
coin flip. The dev gain did not transfer.

### The cause: irreducible dataset noise

All three iterations produced criticisms about URL handling. Inspecting the
worst-scoring examples showed why:

```
INSTRUCTION: Summarize the following news article about the current pandemic.
INPUT      : https://www.nytimes.com/2020/08/22/world/coronavirus-news.html
REFERENCE  : The current pandemic is continuing to impact countries worldwide...
MODEL SAID : I can't access the New York Times article directly from the link.
             Please paste the article's text and I'll summarize it.
SCORE      : 0.035
```

The model behaved correctly and scored 0.035, because the reference was
hallucinated by the dataset's generating model. Measured across the test split:

| | n | Mean baseline score |
|---|---|---|
| Input contains a URL | 12 | 0.516 |
| Input has real source text | 48 | 0.752 |

A **0.236** gap — an order of magnitude larger than any plausible prompt
effect. Across the whole family, **57%** of `summarize` items with an input
carry under 200 characters of real source text.

The optimizer therefore spent its entire budget attacking a defect no prompt
can fix, and the resulting prompt drifted toward instructing the model to
answer from "reliable prior knowledge" when a source is unavailable — i.e.
toward confabulation, because that is what the metric rewarded.

`min_input_chars` was added in response; at a threshold of 200 it leaves 256
usable examples in the family. Run 3 is the experiment that tests whether that
fix was sufficient.

### Run 3 — clean data (40/40/60, `min_input_chars=200`, $0.69)

Identical to Run 2 in every parameter — `iterations=3`, `beam=2`,
`ucb_pulls=120`, `n_gradients=1`, `seed=13` — so the filter is the only changed
variable. It removes every unanswerable item:

| | examples | contain a URL | < 200 chars of real source text |
|---|---|---|---|
| Run 2 (`min_input_chars=0`) | 140 | 22 | **87 (62%)** |
| Run 3 (`min_input_chars=200`) | 140 | **0** | **0** |

| | |
|---|---|
| Dev | 0.8174 → 0.8184 (improved every iteration) |
| Test | 0.8153 → 0.8145 (−0.0008, p = 0.844) |
| Win / loss / tie | 37 / 22 / 1 |
| 95% CI | [−0.0095, +0.0061] |

**What the fix did achieve.** The baseline rose from 0.7051 to **0.8153** —
+0.11 in absolute score, purely from deleting unanswerable items. And the
criticisms finally attacked prompting rather than URL handling: iteration 1
targeted format/length under-specification, iterations 2–3 argued about
grounding strictness. The optimizer was, at last, working on something a prompt
can change.

**What it did not achieve.** The test delta stayed null. Removing the dataset
noise did not make the method work.

**But the shape of the null changed, and that is the finding.** Run 2 was a coin
flip. Run 3 is not:

| | Run 2 (noisy) | Run 3 (clean) |
|---|---|---|
| Win / loss | 28 / 26 | **37 / 22** |
| Median delta | — | **+0.0040** |
| Mean delta | −0.0093 | −0.0008 |
| Mean win size | — | +0.0139 |
| Mean loss size | — | **−0.0255** |
| Sign test | — | p = 0.067 |
| Paired bootstrap | p = 0.487 | p = 0.844 |

The optimized prompt wins on 63% of decided examples and its *median* effect is
positive, but it loses roughly twice as hard as it wins. A broad base of small
gains is cancelled by a thin tail of large losses. The sign test and the
magnitude test disagree — the honest signature of *helps usually, hurts badly
sometimes*.

The mechanism is identifiable. The optimized prompt instructs the model to "add
reasonable context, significance, consequences, benefits, applications, or
broader implications". The worst losses are all tightly length-constrained:

```
-0.187  Summarize the text "The Cat in the Hat" in less than 100 words
-0.058  Summarize the text to 50 words. Output should contain only one sentence
```

The elaboration clause pads exactly where the instruction demands brevity. The
edit that raised the average lowered the floor, and beam search had already
surfaced the trade-off — iteration 3's two beam members produced directly
contradictory criticisms, one demanding stricter grounding and the other
objecting that grounding was already too strict.

**A caveat that must be stated rather than buried:** the single −0.1875 example
carries the entire negative sign. Excluding it the mean flips to +0.0024. That
is not a licence to drop it — discarding inconvenient observations is how false
results are manufactured — but the fact that one point in 60 determines the
direction is itself a measurement of how underpowered a 60-example split is
against a ~0.005 effect. `--test 300` would be the honest next experiment.

### Run 4 — fixing the measurement, not the data (constraint family, strict rubric, $0.94)

Runs 1–3 all ended at the same wall, and §12's own closing line named the way
past it: a task family with more headroom, or a scoring function that can see
constraint violations. Run 4 does both. The decisive number from Run 3 is the
decomposition of its −0.0008:

```
judge     0.9750 → 0.9702   ×0.70  =  −0.0034
rouge_l   0.3938 → 0.3954   ×0.15  =  +0.0003
token_f1  0.4916 → 0.5072   ×0.15  =  +0.0023
                                      ───────
                                       −0.0008
```

The whole sign came from the judge — and the judge scored the *baseline* 0.975
out of 1.0. A component with 0.025 of headroom carried 70% of the weight, so the
metric could not have resolved a real effect in either direction. Runs 1–3 were
not measuring prompt quality; they were measuring judge noise.

**Two changes**, both off by default so Runs 1–3 stay reproducible:

1. `constraints.py` — deterministic checking of countable limits stated in the
   instruction ("in less than 50 words", "in one sentence"). No model call, no
   reference, no opinion. `--require-constraint` selects that family, 791 rows.
2. `--judge-rubric strict` — a deduction-based rubric with an explicit ceiling
   policy ("a merely acceptable answer scores 70, not 95").

**A pilot on 30 examples was run before spending anything, and it refuted half
of this design.** Measured on the baseline prompt:

| Family (n=30) | judge lenient | judge strict | constraint | fully met |
|---|---|---|---|---|
| any stated constraint | 0.8730 | 0.8817 | **0.9896** | 27/30 (90%) |
| *tight* only (≤30 words / ≤2 sentences) | 0.9350 | 0.9530 | **1.0000** | **30/30 (100%)** |

Tightening the constraints made compliance *better*, reaching a perfect 1.0.
`gpt-5.6-luna` does not violate stated length limits; the three failures in the
looser family were overruns of one word (151 vs 150), one word (6 vs 5), and 21
words. **The constraint metric has less headroom than the judge it was built to
replace**, so it went into the run at weight 0.10 as a tripwire against padding
rather than as the primary signal. Building it was still worth it — a metric
that cannot move is a finding about the task, and it is the only component in
the system that cannot drift.

The pilot also exposed a bug in the strict rubric. The first pass scored strict
≈ lenient (0.8783 vs 0.8730), which made no sense against a rubric that deducts
40 points for ignoring a constraint. The cause was `JUDGE_SCHEMA`, whose `score`
field carried the description *"0 (useless) to 100 (matches or beats the
reference)"*. **A JSON-schema field description is not a comment — the model
reads it as instruction**, and it silently cancelled the deduction scale in the
prompt body. Giving the strict rubric its own `STRICT_JUDGE_SCHEMA` fixed it:
p25 fell 0.960 → 0.850 and the minimum 0.000 → 0.300.

Run 4 config: constraint family, strict rubric, weights judge 0.70 / rouge 0.10
/ f1 0.10 / constraint 0.10, otherwise identical to Runs 2–3 (40/40/60, 3
iterations, beam 2, minibatch 8, gradients 1, `ucb_pulls=120`, seed 13).

| Component | Baseline | Optimized | Delta |
|---|---|---|---|
| **total** | **0.7889** | **0.8071** | **+0.0181** |
| judge (strict) | 0.8943 | 0.9208 | +0.0265 |
| rouge_l | 0.2777 | 0.2729 | −0.0048 |
| token_f1 | 0.3647 | 0.3595 | −0.0052 |
| constraint | 0.9867 | 0.9925 | +0.0058 |

Win / loss / tie **28 / 26 / 6**, p = 0.2064. The baseline fell from 0.8153 to
0.7889 — the rubric change worked, giving the metric **0.118 of headroom against
Run 3's 0.025**, roughly 5×. First positive point estimate in four runs, still
not significant.

**Did it game the metric?** The optimized prompt contains the tell:

> *"When the input is incomplete or underspecified, make reasonable inferences
> from the wording, context, titles, and ordinary knowledge instead of refusing
> or merely explaining what is missing."*

That is the Run 2 confabulation drift again — Alpaca punishes a model for
refusing to describe a portrait it was never shown, because the gold answer was
itself invented. So the gain was decomposed by whether the baseline had
deflected:

| Group | n | mean Δ | share of total gain |
|---|---|---|---|
| baseline deflected | 3 | +0.0294 | **8.1%** |
| baseline answered | 57 | +0.0175 | 91.9% |

Excluding the three deflection items: +0.0175, p = 0.2249 — unchanged. **The
exploit is present in the prompt's wording but contributes almost none of the
effect.** The predicted failure mode did not materialise at scale.

What blocked a verdict was no longer measurement but sample size: sd of
per-example differences 0.1123, so n ≈ 148 was needed to resolve +0.018 at
p < 0.05.

### Run 5 — replication at adequate power ($1.25)

Re-running Run 4 with `--test 150` would not have answered the question. A run
draws its whole split with one `rng.sample(pool, n_train + n_dev + n_test)`, so
enlarging the test set reshuffles train and dev too; the optimizer would train
on different data and produce a different prompt. That is a second independent
experiment, not a confirmation.

`run_replicate.py` does the correct thing instead: load the finished run's
`config.json` so scoring is byte-identical, read its `split.json` and withhold
every one of the 140 rows it saw, draw **250 fresh examples** from the remaining
651, and re-run the paired comparison on the *same* optimized prompt. No search,
no optimizer calls — 1000 calls total.

| Component | Baseline | Optimized | Delta |
|---|---|---|---|
| **total** | **0.8149** | **0.8115** | **−0.0034** |
| judge (strict) | 0.9257 | 0.9225 | −0.0032 |
| rouge_l | 0.2946 | 0.2901 | −0.0045 |
| token_f1 | 0.3828 | 0.3778 | −0.0050 |
| constraint | 0.9919 | 0.9899 | −0.0020 |

Win / loss / tie **109 / 113 / 28**. Bootstrap p = 0.5098, sign test p = 0.8405,
95% CI **[−0.0137, +0.0067]**.

**Run 4's +0.0181 did not replicate.** The confidence interval excludes it: the
upper bound is +0.0067, well below Run 4's point estimate. Every component moved
slightly negative, and wins and losses are dead even. The sd of per-example
differences also fell from 0.1123 to 0.0814, which is what a fluctuating
60-example sample looks like from the far side.

| | n | mean | sd | MDE at 95% |
|---|---|---|---|---|
| Run 4 | 60 | +0.0181 | 0.1123 | ±0.0284 |
| Run 5 | 250 | −0.0034 | 0.0814 | ±0.0101 |
| **pooled** | **310** | **+0.0008** | 0.0885 | **±0.0098** |

### Run 6 — a weak executor (`gpt-3.5-turbo` task model, $0.61)

Runs 1–5 all left the same escape hatch open. Every one of them ran the task
role on a frontier model, and the closing diagnosis of Run 5 was that such a
model has no room to improve. That is a hypothesis about the *benchmark*, and it
predicts something specific: move the task role onto a model that genuinely
fails, and the effect should appear.

Run 6 does exactly that and nothing else. The task role runs on
`gpt-3.5-turbo`; judge and optimizer stay on `gpt-5.6-luna`. Same strict rubric,
same 40/40/60 split at seed 13, same three iterations at beam width 2, same
weights. Only the executor changes.

**Precondition first.** `probe_headroom.py` exists so this is checked before
money is spent: it runs the baseline prompt on 20 examples for roughly $0.04 and
reports where the executor lands relative to the frontier model's 0.789.

| | Baseline score |
|---|---|
| `gpt-5.6-luna` (Runs 4–5 reference) | 0.789 |
| `gpt-3.5-turbo` (probe, n=20) | **0.6948** |

Roughly nine points of headroom, and the verdict is *usable*. Only then does the
run proceed.

| Component | Baseline | Optimized | Delta |
|---|---|---|---|
| **total** | **0.7016** | **0.7542** | **+0.0526** |
| judge (strict) | 0.8335 | 0.8902 | +0.0567 |
| rouge_l | 0.3660 | 0.4097 | +0.0437 |
| token_f1 | 0.4216 | 0.4642 | +0.0426 |

Win / loss / tie **39 / 17 / 4**. Bootstrap p = **0.0006**, 95% CI
**[+0.0258, +0.0811]**. Dev score reached 0.8758 and improved at every one of
the three iterations — no early stop. 299 calls, $0.355 for the search, ~$0.61
including the held-out comparison.

Every component moves in the same direction, which is the signature Run 2
lacked: the judge, ROUGE-L and token-F1 agree on both sign and rough magnitude,
so this is not one metric being gamed while the others sag.

But Run 4 also looked like this, and Run 4 did not replicate.

### Run 7 — replication of the weak-executor result ($1.08)

Same protocol as Run 5. `run_replicate.py` loads Run 6's `config.json` so scoring
is byte-identical, withholds all 140 rows Run 6 saw, draws **250 fresh examples**
at seed 101, and re-runs the paired comparison on Run 6's optimized prompt. No
search, no optimizer calls. 1000 calls, $1.08.

| Component | Baseline | Optimized | Delta |
|---|---|---|---|
| **total** | **0.6875** | **0.7131** | **+0.0256** |
| judge (strict) | 0.8343 | 0.8570 | +0.0227 |
| rouge_l | 0.3063 | 0.3403 | +0.0340 |
| token_f1 | 0.3835 | 0.4141 | +0.0306 |

Win / loss / tie **125 / 109 / 16**. Bootstrap p = **0.0057**, 95% CI
**[+0.0080, +0.0435]**.

**The effect survived.** This is the first result in the project that does. The
interval excludes zero on 250 disjoint examples, and every component again moves
positive.

It is also about **half** the discovery estimate. Run 6's +0.0526 sits outside
Run 7's interval, exactly as Run 4's +0.0181 sat outside Run 5's — the winner's
curse is present here too. The honest point estimate is Run 7's, not Run 6's.

| | n | mean | sd | MDE at 95% |
|---|---|---|---|---|
| Run 6 | 60 | +0.0526 | 0.1104 | ±0.0279 |
| Run 7 | 250 | +0.0256 | 0.1421 | ±0.0176 |
| **pooled** | **310** | **+0.0308** | 0.1368 | **±0.0152** |

#### What kind of effect it is

The mean is real but it does not describe a typical example. The distribution of
per-example differences is concentrated, and saying so matters more than the
p-value:

| Statistic | Value |
|---|---|
| mean | +0.0256 |
| **median** | **0.0000** |
| sign test (125 up / 109 down / 16 tied) | **p = 0.3268** |
| 10 largest \|delta\| contribute | **+0.0149** of the +0.0256 |
| examples with delta > +0.3 | 9 |
| examples with delta < −0.3 | 1 |
| the 17 examples scoring < 0.4 at baseline | **+0.1982** mean gain |

The median change is exactly zero and the sign test is null: for most examples
the optimized prompt changes nothing measurable. Roughly 58% of the mean comes
from ten examples out of 250.

Trimming confirms the shape without contradicting the result:

| Trim | Mean |
|---|---|
| none | +0.0256 |
| 5% | +0.0195 |
| 10% | +0.0159 |
| 20% | +0.0131 |

The estimate shrinks as the tails are removed but **never flips sign**, so this
is not a handful of outliers manufacturing an effect out of nothing — there is a
positive drift underneath. The accurate one-line description is: *the optimized
prompt rescues catastrophic failures; it does not broadly lift typical answers.*
That is consistent with what the prompt actually says — most of its text is
instructions to commit to a conclusion, follow the requested format, and stop,
which are precisely the failure modes that produce a near-zero score on a weak
executor.

#### A confound worth recording

An earlier pass at this analysis split the 250 examples into quintiles by
*baseline score* and found a clean monotonic gradient: Q1 +0.048 falling
smoothly to Q5 −0.024. It reads as decisive evidence that the method helps
exactly where the baseline struggles.

It is not admissible. Conditioning on a noisy measured score guarantees
regression to the mean at both tails — examples that scored low partly by bad
luck will score higher on re-measurement whatever the prompt does, and the
mirror holds at the top. The gradient is what that artifact looks like, and it
would appear even if the prompt were unchanged.

Re-splitting by features fixed *before* measurement — whether the example has an
`input` field, reference length, instruction verb — every subgroup interval
contains zero. The `baseline < 0.4` figure in the table above carries the same
caveat and is reported as description, not as evidence of moderation.

### What the seven experiments establish together

**The project's main result: automatic prompt optimization produces no
measurable improvement on Alpaca when the task model is already good at the
task, and a small but replicable one when it is not. The binding constraint is
headroom in the benchmark, not the optimizer.**

Runs 1–5 establish the first half; Runs 6–7 establish the second by
falsification. Neither half is interesting alone — together they identify the
variable that decides the outcome.

- **Run 1** could not have detected an effect: noise 5× the signal, minimum
  detectable difference above the entire plausible effect range.
- **Run 2** showed the failure mode when the benchmark contains unanswerable
  items: the optimizer optimizes toward noise, competently, and drifts the
  prompt toward confabulation because that is what the metric rewarded.
- **Run 3** removed the noise; the failure mode changed rather than disappeared,
  and the decomposition revealed why — the judge was at 0.975 and had no room to
  move.
- **Run 4** rebuilt the measurement. The strict rubric delivered 5× the
  headroom; the deterministic constraint metric, contrary to its design premise,
  delivered none, because the model already complies 99% of the time. The point
  estimate turned positive for the first time (+0.0181) and was not driven by
  the metric-gaming clause the optimizer had inserted.
- **Run 5** replicated Run 4's prompt on 250 disjoint examples and returned
  −0.0034, excluding the Run 4 estimate. Pooled over 310 held-out examples on a
  frontier executor the effect is **+0.0008 ± 0.0098** — a genuine null, not an
  inconclusive one.
- **Run 6** changed one variable, the task model, to `gpt-3.5-turbo`, after a
  $0.04 probe confirmed the baseline sat nine points below the frontier model's.
  The effect appeared: **+0.0526**, p = 0.0006, all components agreeing.
- **Run 7** replicated it on 250 disjoint examples: **+0.0256**, 95% CI
  **[+0.0080, +0.0435]**, p = 0.0057. Halved, as the winner's curse predicts,
  but it survived — the first effect in the project that did.

The two arms are the same experiment run either side of a threshold. Same
optimizer, same rubric, same split sizes, same seed, same statistics; the only
difference that matters is whether the baseline had anywhere to go. On a
frontier executor the 95% interval is [−0.0090, +0.0106] and the answer is no.
On a weak executor it is [+0.0080, +0.0435] and the answer is yes.

This is what makes the Run 1–5 null interpretable rather than merely
disappointing. A null result alone is compatible with a broken implementation;
the same code producing a replicated positive effect the moment headroom exists
rules that out. **The optimizer works. Runs 1–5 were measuring a task with no
room left in it.**

Two qualifications belong next to the positive result, and neither is small.

*The effect is concentrated, not broad.* Median change 0.0000, sign test null,
58% of the mean from ten examples out of 250. It rescues catastrophic failures
rather than lifting typical answers. A practitioner reading "+2.6 points" and
expecting every response to improve slightly would be misled; the realistic
expectation is that a small number of bad answers stop being bad.

*The effect is smaller than the cost of finding it.* Run 6 plus Run 7 cost
$1.69 to establish +0.0256 on one prompt for one model on one dataset. Nothing
here says that trade is worthwhile in practice — only that the improvement is
real.

Neither arm contradicts Pryzant et al. Their gains come from binary
classification — jailbreak detection, sarcasm — scored by hard F1 against a
discrete correct answer, with baselines far below ceiling. The mechanism this
project reproduces is the same one: **the method needs failures to learn from.**
Supply them and it works; withhold them and there is nothing for a textual
gradient to point at.

The remaining open question is no longer whether the optimizer functions but
whether the concentrated, tail-driven improvement it produces generalises beyond
a weak executor on Alpaca — to domain-specific instruction sets, to adversarial
inputs, or to tasks where the reference answer is not a single sentence.
---

## 13. Known limitations

| Limitation | Detail |
|---|---|
| **Judge/task model coupling** | Both default to `gpt-5.6-luna`, so the judge shares the blind spots of the model it grades. Set `--judge-model`, or run the judge on the Anthropic backend. |
| **Sequential bandit** | The adaptive UCB phase cannot be parallelised as written; it dominates wall-clock at high `ucb_pulls`. Batched UCB would fix it. |
| **Prompt bloat** | Each iteration appends constraints; prompts grow monotonically. `EDIT_PROMPT` caps length at 200 words, but nothing prunes. |
| **Single task family per run** | Criticisms from different task types contradict, producing hedged prompts. Use `--filter`. |
| **Dev overfitting** | The bandit selects on dev; with small `ucb_pulls`, it selects noise. The held-out split detects this but cannot prevent it. |
| **Cost scales multiplicatively** | `iterations × beam × candidates × minibatch`. Raise one at a time. |
| **No resume** | A killed run loses progress, though the LLM cache makes a re-run substantially cheaper. |

---

## References

1. Pryzant, R. et al. (2023). *Automatic Prompt Optimization with "Gradient
   Descent" and Beam Search.* EMNLP. [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
2. Zhou, Y. et al. (2023). *Large Language Models Are Human-Level Prompt
   Engineers.* ICLR. [arXiv:2211.01910](https://arxiv.org/abs/2211.01910)
3. Taori, R. et al. (2023). *Stanford Alpaca.* https://github.com/tatsu-lab/stanford_alpaca
4. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot
   Arena.* NeurIPS. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
5. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of
   the Multiarmed Bandit Problem.* Machine Learning, 47, 235–256.