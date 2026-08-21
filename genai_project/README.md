# Multi-Agent Business Analyst

A team of eight specialist AI agents that read one company's internal data room
and answer a **decision** question — not *"summarise this document"* but
*"should we do this?"*

The problem is not finding data. A business analyst already has the financials,
the CRM export, the support tickets, the competitor tracker. The problem is
**synthesis**: no single document contains the answer, and the answer usually
lives in the interaction between documents — a margin decline in the income
statement that is really a vendor problem in the operations file, showing up as
a complaint theme in the support tickets.

So instead of one model reading one file, this system runs a planner that
breaks the question into assignments, five domain specialists that work
concurrently on their own evidence, a risk agent that reads all of their
findings at once, and a strategy agent that has to commit to an answer. A ninth
agent then grades the result against the evidence it was built from.

---

## 1. What it produces

Ask **"Why did profit decline?"** and you get a board-ready report. This is
verbatim from a real run (the full output is in [`sample_report.md`](sample_report.md)):

> **CONDITIONAL GO** — Launch within the next 30 days a CFO-led
> operating-margin recovery program covering logistics, manufacturing, Sales &
> Marketing discounting, and Pulse/Wave working capital, while holding further
> volume-led discount expansion until a reconciled product-region-channel
> contribution bridge is complete.
>
> Finance's TTM P&L shows revenue up 19.1% to $56.41M, but gross profit rose
> only 15.7% to $22.98M, gross margin fell 1.2 points to 40.7%, and net income
> fell 46.2% to $1.12M. […] Logistics is the most acute specific escalation,
> reaching $3.07M for the TTM and 7.0% of June revenue, consistent with
> Operations evidence of freight cost per unit rising to $1.19 and 63.1% of
> inbound shipments arriving more than three days late. […] Working capital is
> a separate cash risk: free cash flow moved from positive $1.45M to negative
> $1.02M, with DSO at 70 days and inventory days at 106.
>
> **Confidence 68%** — *the Sales TTM revenue of $59.20M does not reconcile to
> Finance's period, and product margins and realized prices are unavailable.*

Three things in that output are the whole point of the design: the numbers are
real and traceable, the conclusion crosses four domains that no single file
contains, and the confidence is argued *down* from certainty with named gaps
rather than asserted.

Other questions the data room can answer:

| | |
|---|---|
| Should we expand into Southeast Asia next year? | Which products should we discontinue? |
| What is hurting customer satisfaction most? | Which market has the highest growth potential? |
| How do we compare with our competitors right now? | What are our biggest risks over four quarters? |

---

## 2. The agent team

| # | Agent | Reads | Produces |
|---|---|---|---|
| 1 | **Planner** | The question + a data-room inventory | Restated decision, 3-6 sub-questions, one assignment per specialist |
| 2 | **Finance** | Income statement, balance sheet, cash flow, budget | Margin trends, cash conversion, the line item driving the change |
| 3 | **Sales** | Orders, CRM pipeline | Regional and product mix shifts, pipeline coverage, ASP vs volume |
| 4 | **Customer Intelligence** | Reviews, support tickets, surveys | Complaint clusters weighted by frequency *and* severity, verbatim quotes |
| 5 | **Market Intelligence** | News feed, competitor tracker, industry research | Competitor moves, regulation, growth rates — fact separated from forecast |
| 6 | **Operations** | Inventory, vendors, shipments, production | Vendor concentration, on-time delivery, stockouts and overstock |
| 7 | **Risk** | *Every specialist's findings* | Risks scored likelihood × impact, mitigations, early-warning metrics |
| 8 | **Strategy** | Everything, plus the risk assessment | The recommendation, steelmanned counter-arguments, action plan, confidence |
| 9 | **Evaluator** | The finished report + the evidence behind it | An independent grade, and every claim the evidence does not support |

Agents 2-6 run **concurrently** — they share no state, so there is nothing to
serialise. Agents 7 and 8 are strictly sequential: risk needs every finding,
and strategy needs the risk assessment.

---

## 3. Architecture

```
                          "Why did profit decline?"
                                     │
                          ┌──────────▼──────────┐
                          │   PLANNER AGENT     │  restates the decision,
                          │                     │  assigns work, and drops
                          └──────────┬──────────┘  agents it judges irrelevant
                                     │
        ┌──────────┬─────────────────┼─────────────────┬──────────┐
        │          │                 │                 │          │
   ┌────▼────┐┌────▼────┐      ┌─────▼─────┐     ┌─────▼────┐┌────▼─────┐
   │ FINANCE ││  SALES  │      │ CUSTOMER  │     │  MARKET  ││   OPS    │
   └────┬────┘└────┬────┘      └─────┬─────┘     └─────┬────┘└────┬─────┘
        │          │                 │                 │          │
        └──────────┴─────────────────┼─────────────────┴──────────┘
                                     │  typed findings, never prose
                          ┌──────────▼──────────┐
                          │     RISK AGENT      │  likelihood × impact,
                          │                     │  compound cross-domain risks
                          └──────────┬──────────┘
                          ┌──────────▼──────────┐
                          │   STRATEGY AGENT    │  the recommendation
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   EXECUTIVE REPORT  │
                          └──────────┬──────────┘
                          ┌──────────▼──────────┐
                          │  EVALUATOR (judge)  │  grades it against the
                          └─────────────────────┘  evidence it was built from
```

The graph is written out explicitly in [`analyst/graph.py`](analyst/graph.py)
rather than delegated to an agent framework, because the control flow is fixed
and knowable — the edges never change at runtime. What the orchestrator
actually needs is honest partial-failure handling, a hard budget ceiling, and
live progress reporting, all of which are visible in that one file instead of
hidden behind a library.

**Agents hand each other typed data, never prose.** Every agent returns JSON
validated against a schema in [`analyst/schemas.py`](analyst/schemas.py), so
the Risk Agent receives structured findings rather than paragraphs it would
have to re-parse, and the report renderer never has to guess.

---

## 4. Where the numbers come from

This is the design decision the project turns on.

**Language models cannot be trusted to do arithmetic over a CSV, so they are
never asked to.** Every figure an agent sees has already been computed in
Python by [`analyst/analytics.py`](analyst/analytics.py) — growth rates, margin
bridges, concentration ratios, days of supply, complaint frequencies, linear
forecasts. The agent's job is interpretation, not calculation.

The unstructured half of the data room is handled differently. The news feed,
competitor tracker and industry research are prose, far too long to paste into
every prompt, and mostly irrelevant to any one question. So they are retrieved
against the question being asked:

```
                       ┌──────────────────────────────┐
   markdown files ───▶ │  chunk on headings/blocks    │ ──▶ 23 passages
                       └──────────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
      ┌───────────────┐                            ┌────────────────┐
      │     BM25      │  exact terms:              │   LSA / SVD    │  latent
      │   (lexical)   │  "tariff", "Vietnam"       │    (dense)     │  meaning
      └───────┬───────┘                            └────────┬───────┘
              └──────────────────┬──────────────────────────┘
                                 ▼
                   reciprocal rank fusion ──▶ top-k passages, cited
```

Neither retriever is sufficient alone: BM25 misses paraphrase, and dense
retrieval misses rare proper nouns. On the query *"competitor pricing
pressure"* the dense retriever contributes two passages BM25 ranks nowhere,
because they discuss cell price increases and a competitor launch price without
using the word "pressure".

The two are combined with **reciprocal rank fusion** rather than a weighted
score sum, because BM25 scores are unbounded while cosine similarities live in
[-1, 1] — ranks are the only scale the two share.

---

## 5. Grading the output

A report that reads well and cites a number that does not exist is the
characteristic failure of this kind of system, and it is invisible to anyone
who has not read the source data.

So a ninth agent grades the finished report against **the exact evidence the
specialists were shown**, scoring groundedness, relevance, completeness,
actionability and internal consistency, and listing every claim the evidence
does not support.

It works, and it is not merely pedantic. On the real run above it rejected the
report's central framing by citing evidence the report had quietly stepped
around, and caught a subtler error — treating a variance against budget as if
it were a deterioration over time:

> **[high]** *"The profit decline is primarily an operating-cost and
> margin-compression problem, not a TTM revenue-volume collapse."* — Revenue
> growth and operating deterioration are established, but the evidence does not
> prove operating costs were the primary cause rather than price/mix effects or
> unmeasured COGS changes. **The latest six months also show a 5.7% revenue
> decline, so the dismissal of volume weakness is too broad.**
>
> **[high]** *"manufacturing the largest individual budget variance and
> logistics nearly as large"* — The evidence shows these lines were over
> budget, but not their period-over-period movement. **A variance to budget may
> reflect an inaccurate budget rather than deterioration.**
>
> **Verdict: 82/100 — REVISE**

Note what it does *not* flag: figures the report derived by arithmetic from the
evidence. An earlier version of this prompt marked every computed growth rate
as "unsupported" because the number did not appear verbatim, which made the
judge useless — it cried wolf on every calculation. The current version checks
the arithmetic and reserves its objections for causation. That distinction is
the difference between a judge worth reading and a noise generator.

The judge is deliberately not bound by the same house rules as the analysis
agents: it is checking whether those rules were followed, so it must not share
their framing.

---

## 6. Tech choices, and what I deliberately did not use

| Component | Used | Instead of | Why |
|---|---|---|---|
| Orchestration | ~330 lines of explicit graph | LangGraph, CrewAI, AutoGen | The control flow is fixed. A framework that reasons about edges buys nothing here and costs a dependency plus a debugging layer. |
| Retrieval | BM25 + numpy SVD | FAISS, Chroma, Pinecone | The corpus is ~25 passages. The index rebuilds in milliseconds; a vector database would be more infrastructure than corpus. |
| Numbers | stdlib `csv` + `statistics` | pandas | Every figure the agents see is computed by code you can read in one file, with no dependency that can silently change a rounding rule. |
| Output validation | Provider-native strict JSON schemas | Pydantic parsing + retries | Validation happens at the API boundary, so malformed output is impossible rather than merely caught. |
| Evaluation | LLM-as-judge over the real evidence | RAGAS, DeepEval | Those score retrieval pipelines against reference answers. There is no reference answer for "should we expand?" — the meaningful test is whether the claims trace to the evidence. |

The honest caveat: **dense retrieval here is LSA/SVD fitted on this corpus, not
a pretrained neural encoder.** That bridges vocabulary gaps *within* the data
room, which is the failure mode BM25 actually has here, but it is not a
substitute for a real embedding model on open-domain text. `LsaIndex` in
[`analyst/retrieval.py`](analyst/retrieval.py) is the swap point — anything
exposing `.scores(query) -> list[float]` drops in, and nothing else changes,
because fusion consumes ranks rather than raw scores.

---

## 7. Reliability

| Concern | How it is handled |
|---|---|
| Malformed agent output | Strict JSON schemas at the API boundary; ranges clamped in Python afterwards, since strict mode cannot express `minimum`/`maximum` |
| One agent failing | Returns `ok=False` instead of raising. The report renders with that section marked failed, and downstream agents are told which agent is missing |
| Runaway cost | Hard ceilings on both call count and USD, checked before every request; the run aborts cleanly and still renders what it has |
| Repeat runs | Every response is cached to disk by prompt hash, so re-asking a question is free and instant |
| No API key | A mock backend walks the schemas and fills them with deterministic filler, exercising the real orchestration, validation and rendering paths at zero cost |
| Model refusals | Detected via `stop_reason` before the response body is read |

### Tests

```bash
python -m unittest discover -s tests -t .      # 47 tests, ~4s, no API key, no network
```

The suite is built around defects that actually occurred rather than around
coverage:

| Test | The bug it guards |
|---|---|
| `test_ceiling_holds_under_concurrency` | The call ceiling was read-then-decide, so five specialists starting together all saw the same under-budget count and a ceiling of two admitted six calls. Ten threads race for three slots; the old code admits all ten. |
| `test_budget_exceeded_propagates_out_of_an_agent` | `BudgetExceeded` was swallowed by the agent-level catch-all, so the graph kept dispatching against a ceiling it had already hit and never recorded the abort. |
| `test_budget_abort_is_recorded_and_partial_work_survives` | Aborting must not discard the specialists already paid for. |
| `test_model_defaults_follow_the_backend` | Switching backend without naming a model sent an Anthropic model id to OpenAI — a 404 on every agent. |
| `test_committed_data_matches_its_generator` | The committed CSVs had drifted from `make_sample_data.py`, so the documented regeneration command would have silently replaced the data the sample report describes. |
| `test_request_matches_the_installed_sdk_shape` | Covers the Anthropic request path without spending credit, and asserts the keys we send are a subset of what the installed SDK's `OutputConfigParam` declares. |

Everything runs on the mock backend, so the suite is free and offline.

---

## 8. Running it

```bash
cd genai_project
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**You may not need a key at all.** Keys are resolved from `genai_project/.env`,
then from a `.env` one directory up in the shared `MTech Project` folder, then
from `llm_project/.env` — so a key already set up for a sibling project is
found automatically, without copying the secret into a second file. Otherwise
copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. **Or skip keys
entirely** and use the mock backend.

The default backend is OpenAI (`gpt-5.6-luna`); Anthropic (`claude-opus-5`) is
one dropdown or `--backend anthropic` away, and the model defaults to the right
one for whichever backend you pick.

**Web UI:**

```bash
streamlit run app.py
```

**Command line:**

```bash
python run_analysis.py "Why did profit decline?"
python run_analysis.py "Should we expand into Southeast Asia?" --backend mock
python run_analysis.py "Which products should we discontinue?" \
    --agents finance,sales --effort high --save
```

| Flag | Effect |
|---|---|
| `--backend mock` | Run offline with no key and no cost |
| `--agents finance,sales` | Restrict which specialists the planner may use |
| `--effort low\|medium\|high\|xhigh\|max` | Reasoning depth for every agent |
| `--all-agents` | Overrule the planner and run every specialist |
| `--no-retrieval` / `--no-eval` | Ablate the retriever or the judge |
| `--max-cost 2.00` | Hard spend ceiling |
| `--save` / `--out report.md` | Write the structured run / the markdown |

To regenerate the synthetic company from scratch: `python make_sample_data.py`.

---

## 9. The data room

Seventeen files describing **Nimbus Audio**, a synthetic consumer-audio company
with a deliberately diagnosable problem — revenue growing while margin and cash
quietly deteriorate.

```
data/
├── company_profile.md
├── finance/      income_statement · balance_sheet · cash_flow · budget_vs_actual
├── sales/        orders · crm_pipeline
├── customer/     reviews · support_tickets · surveys
├── market/       news_feed · competitor_tracker · industry_research
└── operations/   inventory · vendors · shipments · production
```

It is synthetic on purpose. Real company data cannot be published with an
academic submission, and a generated data room lets the *answer* be known in
advance — the margin decline is planted, so whether the agents find it is a
test rather than an anecdote. `make_sample_data.py` builds all of it
deterministically from a seed.

---

## 10. What one run actually costs

Measured on the "Why did profit decline?" run quoted above, GPT-5.6-Luna, all
five specialists, retrieval and evaluation on:

| | |
|---|---|
| Model calls | 9 (1 planner + 5 specialists + risk + strategy + judge) |
| Tokens | 51,864 in / 35,046 out |
| Cost | **$0.26** |
| Wall clock | 203 s — the five specialists run concurrently in ~25 s total |
| Re-run of the same question | free, served from the disk cache |

The specialists are the cheap part. Risk and strategy dominate both cost and
latency, because they reason over every finding at once — which is why they are
the two agents given a higher effort setting by default.

---

## 11. Limitations

- **Synthetic data.** The pipeline is real; the company is not. Nothing here
  validates the agents against messy real-world data.
- **LSA is not a neural encoder** (§6). On a larger or more open-domain corpus
  this is the first component that should be replaced.
- **The judge is a model too.** It catches unsupported figures reliably; it is
  weaker at catching an argument that is well-grounded but wrong. It reduces
  the failure rate rather than eliminating it, and its own score should not be
  reported as ground truth.
- **No multi-turn follow-up.** Each question is a fresh run. Follow-ups
  ("now break that down by region") would need conversation state the graph
  does not currently carry.
- **The planner can drop an agent it should have kept.** `--all-agents`
  overrules it, at roughly double the cost.

---

## 12. Layout

```
genai_project/
├── app.py                  Streamlit UI — live agent trace, six audit tabs
├── run_analysis.py         CLI
├── make_sample_data.py     regenerates the synthetic data room
├── sample_report.md        a real report, so the output is readable without a key
├── tests/                  47 offline tests, built around defects that occurred
└── analyst/
    ├── config.py           one Settings object describing an entire run
    ├── datasets.py         data-room loading, stdlib only
    ├── analytics.py        every figure the agents see, computed in Python
    ├── retrieval.py        BM25 + LSA hybrid retrieval with rank fusion
    ├── schemas.py          the JSON contract each agent must satisfy
    ├── prompts.py          nine system prompts, and the rules they share
    ├── llm.py              backends, disk cache, budget ceiling, usage accounting
    ├── agents.py           the agents; evidence assembly and clamping
    ├── graph.py            the orchestrator
    └── reporting.py        markdown rendering, risk stars, evaluation tables
```