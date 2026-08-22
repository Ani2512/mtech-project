# Automatic Prompt Optimization

A system that **rewrites a prompt for you, using feedback from its own
failures.** You give it a starting prompt like *"You are a helpful assistant.
Answer the user's instruction."*; it runs that prompt on real tasks, grades the
answers, works out what the prompt is doing wrong, rewrites it, and keeps the
version that actually scores better on data it has never optimized against.

**Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
— 52,002 `instruction` / `input` / `output` triples.

**Model:** `gpt-5.6-luna` (OpenAI) by default. The backend is swappable —
Anthropic Claude and an offline mock are also supported.

---

## 1. Why this is a hard problem

Prompt engineering is normally a human guessing loop: change some wording, eyeball
a few outputs, keep whatever felt better. That fails for three reasons — you test
on a handful of examples, "felt better" is not measurable, and the space of
possible prompts is infinite.

This project replaces each of those with something mechanical:

| Human loop | This system |
|---|---|
| Try a few examples | Score on a sampled minibatch, select on a held-out dev split |
| "That looks better" | A blended numeric score (LLM judge + ROUGE-L + token-F1) |
| Guess the next edit | Derive the edit from the actual failures ("textual gradient") |
| Hope it generalises | Confirm on a **held-out test split** with a significance test |

The core idea is from Pryzant et al. (2023), *Automatic Prompt Optimization with
"Gradient Descent" and Beam Search*. There is no backpropagation involved — the
"gradient" is a natural-language criticism of the prompt, and the "descent step"
is an LLM rewriting the prompt to address that criticism.

---

## 2. The optimization loop

```
                    ┌──────────────────────────────────────────┐
                    │  BEAM: the current best prompts          │
                    └───────────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────▼─────────────────────────┐
              │ 1. EVALUATE on a training minibatch               │
              │    run the prompt → grade each answer 0..1        │
              └─────────────────────────┬─────────────────────────┘
                                        │  worst-scoring cases
              ┌─────────────────────────▼─────────────────────────┐
              │ 2. GRADIENT — criticise the prompt                │
              │    "it never tells the model to obey an explicit  │
              │     output format, so it pads short answers"      │
              └─────────────────────────┬─────────────────────────┘
                                        │
              ┌─────────────────────────▼─────────────────────────┐
              │ 3. DESCENT — rewrite the prompt to fix that       │
              │ 4. PARAPHRASE — reword the rewrite (exploration)  │
              └─────────────────────────┬─────────────────────────┘
                                        │  candidate pool
              ┌─────────────────────────▼─────────────────────────┐
              │ 5. SELECT — UCB bandit on the dev split           │
              │    spend evaluations on promising candidates,     │
              │    discard the rest cheaply → new beam            │
              └─────────────────────────┬─────────────────────────┘
                                        │
                         repeat, or stop on `patience`
```

**Why a bandit?** Scoring every candidate on the whole dev split is quadratic in
cost. UCB1 gives each candidate one evaluation, then concentrates the remaining
budget on candidates whose score *or* uncertainty is high — so a bad prompt is
dropped after a couple of evaluations while a promising one earns more.

**Why beam search?** Pure hill-climbing commits to the first improvement it finds.
Keeping the top *k* prompts lets the search recover when the most obvious fix
turns out to be a local optimum. Set `--beam 1` to compare against greedy search.

---

## 3. How answers are scored

Alpaca references are open-ended, so exact match is useless — a correct answer
worded differently scores zero. The score blends three signals (weights are
configurable in `apo/config.py`):

| Component | Weight | What it catches | Weakness |
|---|---|---|---|
| **LLM judge** (0–100) | 0.70 | Correctness, constraint-following, appropriate length | Costs an API call; some variance |
| **ROUGE-L** | 0.15 | Content overlap in the right order | Blind to meaning |
| **Token-F1** | 0.15 | Content overlap, order-insensitive | Blind to meaning |

The judge is prompted to treat the reference as *one* acceptable answer rather
than the only one, and to return a structured `{score, critique}` object. That
critique is not just for scoring — it is fed back into the gradient step as
evidence, which is what makes the criticisms concrete instead of generic.

With `--no-judge` the run uses lexical metrics only. That halves the cost and is
useful for a quick sanity check, but the signal is weak on open-ended answers.

---

## 4. Project layout

```
llm_project/
├── apo/
│   ├── config.py      Settings dataclass, model pricing, API-key resolution
│   ├── data.py        Alpaca download, filtering, deterministic train/dev/test
│   ├── backends.py    OpenAI + Anthropic clients, disk cache, budget, mock
│   ├── constraints.py Deterministic checking of stated length limits
│   ├── scoring.py     ROUGE-L / token-F1 / LLM judge + paired bootstrap
│   ├── prompts.py     The three meta-prompts (gradient, edit, paraphrase)
│   ├── optimizer.py   Evaluation, candidate generation, UCB, the main loop
│   └── reporting.py   Budget estimation, run artifacts, markdown report
├── app.py             Streamlit UI
├── run_optimize.py    Command-line runner
├── run_replicate.py   Re-tests a finished run's prompt on fresh, disjoint data
└── list_models.py     Prints the model IDs your key can actually reach
```

Every model call goes through `backends.py`, which is what makes the cost
ceiling, the response cache, the offline mode, and provider-swapping possible
at all.

> **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** is the code-level companion to this
> file — data structures, control flow, every function in the optimization
> loop, extension points, and the empirical findings from the live runs.

### Providers

`backends.py` exposes one interface (`complete(role, prompt, system, schema)`)
with three implementations, chosen with `--backend`:

| Backend | Model | Notes |
|---|---|---|
| `openai` *(default)* | `gpt-5.6-luna` | Chat Completions, `reasoning_effort`, strict JSON-schema outputs |
| `anthropic` | `claude-opus-5` etc. | Kept working; needs `ANTHROPIC_API_KEY` |
| `mock` | — | Offline stub, no key, no cost |

Nothing above the backend layer knows which provider is in use — the optimizer,
scoring, UI and tests are provider-agnostic.

---

## 5. Setup

```powershell
cd "C:\Users\Anirudh\OneDrive\Desktop\MTech Project\llm_project"
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then add your API key — copy `.env.example` to `.env` and paste the key in:

```
OPENAI_API_KEY=sk-proj-...
```

`.env` is gitignored. Check which models the key can reach:

```powershell
.\venv\Scripts\python.exe list_models.py --filter gpt-5.6
```

**Note on key hygiene:** a key sitting in a plaintext file inside a synced
OneDrive folder is one sharing mistake away from being public — keep keys in
`.env` only, and rotate any key that has been stored elsewhere.

Verify everything works without spending anything:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

---

## 6. Usage

### Streamlit UI

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Five tabs: **Optimize** (configure and watch the run live), **Results** (score
trajectory, baseline-vs-optimized metrics, per-example diff), **Dataset**
(browse Alpaca), **Runs** (reload past runs), **Playground** (try any prompt on
any instruction).

### Command line

```powershell
# What would this cost? Nothing is called.
.\venv\Scripts\python.exe run_optimize.py --dry-run

# Full offline smoke test — no key, no cost
.\venv\Scripts\python.exe run_optimize.py --backend mock --iterations 2

# A real run, restricted to summarisation-style tasks
.\venv\Scripts\python.exe run_optimize.py --filter summarize --with-input --iterations 3

# Cheaper: skip the judge, smaller splits
.\venv\Scripts\python.exe run_optimize.py --no-judge --train 20 --dev 20 --test 30
```

`--help` lists every flag. The most useful ones:

| Flag | Effect |
|---|---|
| `--filter WORD` / `--with-input` | Narrow to one task family — this matters (see §9) |
| `--iterations N` `--beam K` | Search depth and width |
| `--gradients` `--edits` `--paraphrases` | Candidates generated per step |
| `--backend` | `openai` (default), `anthropic`, or `mock` |
| `--model` | Set all three roles at once (e.g. `gpt-5.6-luna`) |
| `--no-judge` | Lexical scoring only — much cheaper, much noisier |
| `--max-cost` | Hard spend ceiling; the run stops cleanly when hit |
| `--dry-run` | Print the budget estimate and exit |

---

## 7. Cost control

An optimization run makes hundreds of API calls, so cost is a first-class concern:

- **Budget estimate up front.** `--dry-run` prints projected call counts and USD.
- **Hard ceilings.** `max_cost_usd` and `max_llm_calls` stop the run cleanly and
  still write a report with whatever was found.
- **Disk cache.** Identical calls are served from `.cache/`, so re-running an
  experiment with one changed parameter costs a fraction of the first run.
- **Per-role models.** The judge and task model can be cheaper than the optimizer,
  which is the only role where reasoning quality really pays.
- **Per-role effort.** Task and judge calls run at `low` `reasoning_effort`, the
  optimizer at `high`.

Defaults (3 iterations, beam 2, 40/40/60 split, judge on) come to roughly 540
calls. On `gpt-5.6-luna` at $1 / $6 per 1M tokens that is about **$1.30**.
Luna is a cost-tier model, which is what makes a search this call-heavy
practical — the same run on a frontier-tier model costs several times more.

---

## 8. Output

Each run writes `runs/<timestamp>/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable summary — trajectory, both prompts, test results |
| `best_prompt.txt` | The optimized prompt, ready to use |
| `result.json` | Full search history, every candidate and score, usage |
| `comparison.json` | Per-example baseline-vs-optimized predictions and scores |
| `config.json`, `split.json` | Everything needed to reproduce the run |

### Reading the results honestly

The report ends with a **paired bootstrap test** over per-example score
differences, reporting a 95% confidence interval and a p-value. This is the part
that separates a real result from noise. A prompt that scores +0.03 on 60 test
examples with p = 0.4 has **not** been shown to be better — the report says so
rather than declaring victory. Increase `--test` for more statistical power.

### What the runs actually found

Seven live experiments. Runs 1–5 run the task role on `gpt-5.6-luna` and **none
found a significant improvement**. Runs 6–7 change one variable — the task model
— and the effect appears and replicates. Reporting both halves is the point.

| Run | Config | Baseline → Optimized | p | Win/loss |
|---|---|---|---|---|
| 1 | 12/12/12, summarize | 0.7246 → 0.7075 | 0.59 | — |
| 2 | 40/40/60, summarize | 0.7051 → 0.6958 | 0.49 | 28 / 26 |
| 3 | + `--min-input-chars 200` | 0.8153 → 0.8145 | 0.84 | **37 / 22** |
| 4 | constraint family, `--judge-rubric strict` | 0.7889 → **0.8071** | 0.21 | 28 / 26 |
| 5 | replication of run 4, **n=250** disjoint | 0.8149 → 0.8115 | 0.51 | 109 / 113 |
| 6 | task model → **`gpt-3.5-turbo`**, strict rubric | 0.7016 → **0.7542** | **0.0006** | **39 / 17** |
| 7 | replication of run 6, **n=250** disjoint | 0.6875 → **0.7131** | **0.0057** | 125 / 109 |

Run 2's failure had a diagnosable cause: **62% of the evaluation set was
unanswerable**. Alpaca's `summarize` items frequently supply a bare URL as the
input with a hallucinated reference answer, so a model that correctly says "I
can't open that link" scores 0.035. The optimizer spent its whole budget
attacking a defect no prompt can fix, and drifted the prompt toward
confabulation because that is what the metric rewarded.

`--min-input-chars` was built to fix this, and Run 3 tested it. Filtering
removed every unanswerable item (87/140 → 0/140) and lifted the baseline by
**+0.11**. The criticisms finally attacked prompting instead of URL handling.
But the test delta stayed null — the fix was necessary and not sufficient.

What changed was the *shape*: Run 3's prompt wins **37 of 59** decided examples
with a **positive median** (+0.004), yet loses nearly twice as hard as it wins,
so the mean stays flat. Sign test p = 0.067, magnitude test p = 0.84 — the
signature of *helps usually, hurts badly sometimes*. The regressions concentrate
on length-constrained items, where the optimized prompt's instruction to add
"broader implications" pads exactly where brevity was demanded.

Runs 4 and 5 attacked the *measurement* instead. Decomposing Run 3's delta showed
the whole sign came from the judge — which had scored the **baseline** 0.975 out
of 1.0. A component with 0.025 of headroom carried 70% of the weight, so the
first three runs were measuring judge noise, not prompt quality. Run 4 added a
strict deduction-based rubric (5× the headroom) and `constraints.py`, a
deterministic check of stated length limits. Its point estimate turned positive
for the first time: **+0.0181, p = 0.21**.

Run 5 tested whether that survived. Re-running with a bigger `--test` would have
reshuffled train and dev as well, so `run_replicate.py` instead re-scored the
*same* optimized prompt on **250 fresh examples disjoint from everything Run 4
saw**. It returned **−0.0034**, with a 95% CI of [−0.0137, +0.0067] that excludes
Run 4's estimate, and wins and losses dead even at 109/113. Run 4's gain was
sampling noise.

Pooled over all 310 held-out examples on a frontier executor the effect is
**+0.0008 ± 0.0098** — a genuine null, not an inconclusive one. The cause is
visible in the baseline: `"You are a helpful assistant."` already scores
0.79–0.82 under a rubric reserving 90+ for flawless work, and obeys explicit
length limits 99% of the time. There is no room for a prompt to win in.

That explanation is falsifiable, so Runs 6–7 tested it. Moving the task role to
`gpt-3.5-turbo` — judge and optimizer unchanged, same rubric, same split sizes,
same seed — dropped the baseline to 0.70 and the effect appeared: **+0.0526,
p = 0.0006**. Run 7 re-scored that prompt on 250 disjoint examples and it
**survived**: **+0.0256, 95% CI [+0.0080, +0.0435], p = 0.0057**. Halved, as the
winner's curse predicts, but the first effect in the project that replicated.

**The result: automatic prompt optimization produces no measurable improvement
on Alpaca when the task model is already good at the task, and a small but
replicable one when it is not. The binding constraint is headroom in the
benchmark, not the optimizer.** The null arm is what makes the positive arm
interpretable, and vice versa — a null alone is compatible with broken code,
whereas the same code producing a replicated effect the moment headroom exists
rules that out.

Two qualifications belong on the positive result. It is **concentrated, not
broad**: the median change is 0.0000, the sign test is null (p = 0.33), and 58%
of the mean comes from ten examples out of 250. It rescues catastrophic failures
rather than lifting typical answers. And it **cost more to find than it is
worth** — $1.69 to establish +0.0256 on one prompt, one model, one dataset.

Reproduce the weak-executor arm with `./run_weak_executor.sh probe`, then
`optimize`, then `replicate <run_dir>`. See
[HOW_IT_WORKS.md §12](HOW_IT_WORKS.md) for the full analysis, including a
JSON-schema bug that silently cancelled the strict rubric on its first attempt
and a regression-to-the-mean artifact that nearly became a finding.

---

## 9. Limitations

- **A prompt is optimized for a task family, not for everything.** Optimizing
  across all of Alpaca produces bland, generic prompts, because the criticisms
  from a translation failure and a maths failure pull in opposite directions.
  Use `--filter` / `--with-input` to define a coherent slice; that is where the
  method actually shows gains.
- **The judge is a model, not ground truth.** It shares blind spots with the
  model being evaluated. Judge and task models being the same family is a known
  bias — by default both are `gpt-5.6-luna`, which is the strongest form of this
  problem. Setting `--judge-model` to a different model, or `--backend anthropic`
  for the judge role, is the mitigation the config supports, and is worth doing
  before quoting any number in a report. Runs 6–7 avoid it by construction: the
  task model is `gpt-3.5-turbo` and the judge is `gpt-5.6-luna`, so the positive
  result there is not a model grading its own family.
- **Small splits overfit.** With 40 dev examples, the bandit can select a prompt
  that happens to suit those 40. The held-out test split exists precisely to
  detect this — a large dev gain with a flat test result means overfitting.
- **Cost scales multiplicatively.** Iterations × beam × candidates × minibatch.
  Raise one at a time.
- **The mock backend proves the pipeline runs, not that prompts improve.** Its
  task model ignores the prompt entirely, so mock scores are structurally valid
  and semantically meaningless. Every mock report is stamped as such.

---

## 10. References

1. Pryzant, R., Iter, D., Li, J., Lee, Y. T., Zhu, C., & Zeng, M. (2023).
   *Automatic Prompt Optimization with "Gradient Descent" and Beam Search.*
   EMNLP 2023. [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
2. Zhou, Y. et al. (2023). *Large Language Models Are Human-Level Prompt
   Engineers.* ICLR 2023. [arXiv:2211.01910](https://arxiv.org/abs/2211.01910)
3. Taori, R. et al. (2023). *Stanford Alpaca: An Instruction-following LLaMA
   Model.* https://github.com/tatsu-lab/stanford_alpaca
4. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot
   Arena.* NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
5. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of
   the Multiarmed Bandit Problem.* Machine Learning, 47, 235–256.