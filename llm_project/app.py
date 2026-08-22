"""Streamlit UI for the Automatic Prompt Optimization system.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import pathlib
import traceback

import altair as alt
import pandas as pd
import streamlit as st

from apo.backends import BudgetExceeded, make_backend
from apo.config import (
    DEFAULT_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_KEY_ENV,
    PROVIDER_MOCK,
    PROVIDER_MODELS,
    PROVIDER_OPENAI,
    RUNS_DIR,
    Settings,
    load_api_key,
)
from apo.data import split_from_settings
from apo.optimizer import compare_on_test, optimize
from apo.reporting import build_report, estimate_budget, new_run_dir, save_run

# Categorical slots 1 and 2 of the reference palette, in fixed order.
# Baseline always takes slot 1, optimized always slot 2, so a chart never
# repaints when one of them is missing.
SERIES_1 = "#2a78d6"  # blue   - baseline
SERIES_2 = "#eb6834"  # orange - optimized
MUTED = "#898781"
GRID = "#e1e0d9"

st.set_page_config(
    page_title="Automatic Prompt Optimization", page_icon="🧭", layout="wide"
)


def explain_api_error(exc: Exception) -> str | None:
    """Turn common provider errors into an actionable message."""
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return (
            "The OpenAI account has no remaining quota. Add credits at "
            "platform.openai.com → Billing, or switch the backend to `mock` "
            "to exercise the pipeline offline."
        )
    if "credit balance is too low" in msg:
        return (
            "The Anthropic account has no credits. Add credits at "
            "console.anthropic.com → Plans & Billing, or switch the backend "
            "to `mock` to exercise the pipeline offline."
        )
    if "model_not_found" in msg or "does not exist" in msg:
        return (
            "The API rejected the model ID. Check it against your provider's "
            "model list — `python list_models.py` prints the models your key "
            "can actually reach."
        )
    if "invalid_api_key" in msg or "authentication" in msg or "401" in msg:
        return (
            "The API key was rejected. Check the key in `llm_project/.env` "
            "matches the selected backend."
        )
    return None


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def trajectory_chart(history: list[dict]) -> alt.Chart:
    """Best dev score per iteration — one series, so no legend is needed."""
    df = pd.DataFrame(
        [
            {"Iteration": h["iteration"], "Best dev score": h["best_dev_score"]}
            for h in history
        ]
    )
    base = alt.Chart(df).encode(
        x=alt.X(
            "Iteration:O",
            axis=alt.Axis(
                labelColor=MUTED, titleColor=MUTED, domainColor=GRID, tickColor=GRID
            ),
        ),
        y=alt.Y(
            "Best dev score:Q",
            scale=alt.Scale(zero=False, nice=True),
            axis=alt.Axis(
                labelColor=MUTED,
                titleColor=MUTED,
                gridColor=GRID,
                domain=False,
                ticks=False,
                format=".3f",
            ),
        ),
        tooltip=[
            alt.Tooltip("Iteration:O"),
            alt.Tooltip("Best dev score:Q", format=".4f"),
        ],
    )
    line = base.mark_line(color=SERIES_2, strokeWidth=2)
    dots = base.mark_point(
        color=SERIES_2, filled=True, size=90, stroke="white", strokeWidth=2
    )
    return (line + dots).properties(height=280)


def metric_comparison_chart(comparison: dict) -> alt.Chart:
    """Grouped bars: baseline vs optimized across every metric."""
    rows = []
    for metric in ("total", "judge", "rouge_l", "token_f1"):
        rows.append(
            {
                "Metric": metric,
                "Prompt": "Baseline",
                "Score": comparison["baseline"][metric],
            }
        )
        rows.append(
            {
                "Metric": metric,
                "Prompt": "Optimized",
                "Score": comparison["optimized"][metric],
            }
        )
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X(
                "Metric:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelColor=MUTED,
                    titleColor=MUTED,
                    domainColor=GRID,
                    ticks=False,
                ),
            ),
            xOffset=alt.XOffset("Prompt:N", sort=["Baseline", "Optimized"]),
            y=alt.Y(
                "Score:Q",
                axis=alt.Axis(
                    labelColor=MUTED,
                    titleColor=MUTED,
                    gridColor=GRID,
                    domain=False,
                    ticks=False,
                    format=".2f",
                ),
            ),
            color=alt.Color(
                "Prompt:N",
                scale=alt.Scale(
                    domain=["Baseline", "Optimized"], range=[SERIES_1, SERIES_2]
                ),
                legend=alt.Legend(title=None, labelColor=MUTED),
            ),
            tooltip=[
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Prompt:N"),
                alt.Tooltip("Score:Q", format=".4f"),
            ],
        )
        .properties(height=300)
    )


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------


def sidebar() -> tuple[Settings, bool, bool]:
    st.sidebar.title("Configuration")

    backend = st.sidebar.radio(
        "Backend",
        [PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_MOCK],
        index=0,
        help="`mock` runs the full pipeline offline with a deterministic stub. "
        "It costs nothing and needs no key, but its scores are not meaningful.",
    )

    api_key = load_api_key(backend) if backend != PROVIDER_MOCK else None
    if backend == PROVIDER_MOCK:
        st.sidebar.info("Offline mock backend — no key needed, no cost.")
    elif api_key:
        st.sidebar.success(f"{backend} key found (…{api_key[-6:]})")
    else:
        var = PROVIDER_KEY_ENV.get(backend, "API key")
        st.sidebar.warning(
            f"No {var} found. Add it to `llm_project/.env`, or switch the "
            "backend to `mock`."
        )

    st.sidebar.subheader("Models")
    model_options = PROVIDER_MODELS.get(backend, [DEFAULT_MODEL])
    task_model = st.sidebar.selectbox("Task model", model_options, index=0)
    judge_model = st.sidebar.selectbox("Judge model", model_options, index=0)
    optimizer_model = st.sidebar.selectbox("Optimizer model", model_options, index=0)
    use_judge = st.sidebar.checkbox(
        "Use LLM judge",
        value=True,
        help="Off = lexical metrics only. Halves the cost, but the signal is "
        "much weaker on open-ended Alpaca answers.",
    )

    st.sidebar.subheader("Search")
    iterations = st.sidebar.slider("Iterations", 1, 10, 3)
    beam_width = st.sidebar.slider("Beam width", 1, 5, 2)
    minibatch_size = st.sidebar.slider("Minibatch size", 2, 32, 8)
    n_gradients = st.sidebar.slider("Criticisms per prompt", 1, 5, 2)
    n_edits = st.sidebar.slider("Rewrites per criticism", 1, 4, 1)
    n_paraphrases = st.sidebar.slider("Paraphrases per rewrite", 0, 3, 1)
    ucb_pulls = st.sidebar.slider("Bandit pulls", 4, 100, 24)

    st.sidebar.subheader("Data")
    seed = st.sidebar.number_input("Seed", value=13, step=1)
    n_train = st.sidebar.number_input("Train size", 8, 500, 40, step=4)
    n_dev = st.sidebar.number_input("Dev size", 8, 500, 40, step=4)
    n_test = st.sidebar.number_input("Test size", 4, 500, 60, step=4)
    input_mode = st.sidebar.selectbox(
        "Example type",
        ["any", "with input field", "instruction only"],
        help="Narrows the dataset to one task family. A prompt tuned for "
        "'rewrite this text' tasks is not the same prompt as one tuned for "
        "open-ended Q&A.",
    )
    keyword = st.sidebar.text_input(
        "Keyword filter", value="", placeholder="e.g. summarize"
    )
    min_input_chars = st.sidebar.number_input(
        "Min source text (chars)",
        0,
        5000,
        0,
        step=50,
        help="Drops examples whose `input` has too little real source text "
        "(URLs don't count). Alpaca contains many unanswerable "
        "'summarise this link' items whose reference answers were "
        "hallucinated — they can't be fixed by any prompt and they dominate "
        "the error signal. Try 200 for summarisation.",
    )

    st.sidebar.subheader("Safety limits")
    max_cost = st.sidebar.number_input("Max spend (USD)", 0.5, 500.0, 25.0)
    concurrency = st.sidebar.slider("Concurrency", 1, 12, 4)
    use_cache = st.sidebar.checkbox("Cache LLM calls", value=True)

    settings = Settings(
        backend=backend,
        task_model=task_model,
        judge_model=judge_model,
        optimizer_model=optimizer_model,
        use_judge=use_judge,
        iterations=iterations,
        beam_width=beam_width,
        minibatch_size=minibatch_size,
        n_gradients=n_gradients,
        n_edits=n_edits,
        n_paraphrases=n_paraphrases,
        ucb_pulls=ucb_pulls,
        seed=int(seed),
        n_train=int(n_train),
        n_dev=int(n_dev),
        n_test=int(n_test),
        require_input={"any": None, "with input field": True, "instruction only": False}[
            input_mode
        ],
        filter_keyword=keyword.strip() or None,
        min_input_chars=int(min_input_chars),
        max_cost_usd=float(max_cost),
        concurrency=concurrency,
        use_cache=use_cache,
    )

    budget = estimate_budget(settings)
    st.sidebar.subheader("Budget estimate")
    st.sidebar.caption(
        f"~{budget['total_calls']} API calls"
        + (
            " · $0.00 (mock)"
            if backend == PROVIDER_MOCK
            else f" · ~${budget['estimated_cost_usd']:.2f}"
        )
    )

    run_test = st.sidebar.checkbox("Run held-out comparison", value=True)
    go = st.sidebar.button("Run optimization", type="primary", width="stretch")
    return settings, go, run_test


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------


def render_optimize_tab(settings: Settings, go: bool, run_test: bool) -> None:
    st.subheader("Optimization run")
    st.caption(
        "Each iteration: run the prompt on a training minibatch, grade the "
        "answers, turn the worst cases into written criticism, rewrite the "
        "prompt against that criticism, then keep the best candidates using a "
        "UCB bandit on the dev split."
    )

    initial = st.text_area(
        "Starting prompt",
        value=st.session_state.get(
            "initial_prompt", Settings().initial_prompt
        ),
        height=110,
    )
    st.session_state["initial_prompt"] = initial
    settings.initial_prompt = initial.strip() or Settings().initial_prompt

    if not go:
        st.info("Set the configuration in the sidebar, then press **Run optimization**.")
        return

    log_box = st.empty()
    status_line = st.empty()
    log: list[str] = []

    def progress(event: dict) -> None:
        kind = event.get("event")
        if kind == "iteration_start":
            log.append(f"**Iteration {event['iteration']}**")
        elif kind == "eval_progress":
            status_line.caption(
                f"Evaluating [{event['label']}] {event['done']}/{event['total']}"
            )
            return
        elif kind == "beam_evaluated":
            log.append(
                f"- beam member {event['index'] + 1} minibatch score "
                f"`{event['score']:.4f}`"
            )
        elif kind == "gradients":
            log.append(f"- {event['count']} criticism(s) generated")
            for g in event.get("gradients", []):
                log.append(f"    - _{g}_")
        elif kind == "pool_ready":
            log.append(f"- candidate pool: {event['size']}")
        elif kind == "ucb_pull":
            status_line.caption(
                f"Bandit selection {event['done']}/{event['total']} "
                f"(best mean {event['best_mean']:.4f})"
            )
            return
        elif kind == "iteration_end":
            flag = "improved" if event["improved"] else "no improvement"
            log.append(f"- best dev score `{event['best_score']:.4f}` ({flag})")
        elif kind == "budget_exceeded":
            log.append(f"- **stopped:** {event['message']}")
        else:
            return
        log_box.markdown("\n".join(log[-60:]))

    try:
        with st.spinner("Loading Alpaca dataset…"):
            split = split_from_settings(settings)
        st.caption(f"Splits — {split.summary()}")

        backend = make_backend(settings)
        with st.status("Optimizing…", expanded=True) as status:
            result = optimize(backend, split, settings, progress=progress)
            comparison = None
            budget_note = None
            if run_test:
                status.update(label="Evaluating on held-out test split…")
                try:
                    comparison = compare_on_test(
                        backend,
                        result.initial_prompt,
                        result.best_prompt,
                        split.test,
                        settings,
                        progress=progress,
                    )
                except BudgetExceeded as exc:
                    # The optimization is already paid for; keep it rather than
                    # losing the whole run to the ceiling.
                    budget_note = str(exc)
            status.update(label="Done", state="complete")

        run_dir = new_run_dir()
        save_run(
            run_dir,
            settings=settings,
            split=split,
            result=result,
            comparison=comparison,
        )

        st.session_state["result"] = result
        st.session_state["comparison"] = comparison
        st.session_state["settings"] = settings
        st.session_state["run_dir"] = str(run_dir)
        st.success(f"Finished. Artifacts saved to `{run_dir}`")
        if budget_note:
            st.warning(
                f"Budget limit reached during held-out evaluation: {budget_note} "
                "The optimized prompt is saved; the comparison is not."
            )
        st.info("Open the **Results** tab for the optimized prompt and scores.")

    except BudgetExceeded as exc:
        st.error(f"Budget limit reached: {exc}")
    except Exception as exc:  # surface API errors legibly rather than a stack dump
        friendly = explain_api_error(exc)
        if friendly:
            st.error(friendly)
        else:
            st.error(f"Run failed: {exc}")
            st.code(traceback.format_exc(), language="text")


def render_results_tab() -> None:
    result = st.session_state.get("result")
    if result is None:
        st.info("No results yet. Run an optimization, or load one from the **Runs** tab.")
        return

    comparison = st.session_state.get("comparison")
    settings: Settings = st.session_state.get("settings", Settings())

    if settings.backend == "mock":
        st.warning(
            "These numbers come from the **mock** backend. They confirm the "
            "pipeline runs; they are not meaningful results."
        )

    if comparison:
        b = comparison["baseline"]["total"]
        o = comparison["optimized"]["total"]
        sig = comparison["significance"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline (test)", f"{b:.4f}")
        c2.metric("Optimized (test)", f"{o:.4f}", delta=f"{o - b:+.4f}")
        c3.metric(
            "Win / loss / tie",
            f"{comparison['wins']} / {comparison['losses']} / {comparison['ties']}",
        )
        c4.metric(
            "p-value",
            f"{sig['p_value']:.4f}",
            help="Paired bootstrap over per-example score differences.",
        )
        if sig["p_value"] < 0.05:
            st.success(
                f"Improvement is statistically significant "
                f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}])."
            )
        else:
            st.warning(
                "The difference is not statistically significant at p < 0.05. "
                "A larger test split or more iterations would be needed to "
                "make a claim."
            )

    left, right = st.columns(2)
    with left:
        st.markdown("**Starting prompt**")
        st.code(result.initial_prompt, language="text")
    with right:
        st.markdown("**Optimized prompt**")
        st.code(result.best_prompt, language="text")

    if result.history:
        st.markdown("#### Best dev score by iteration")
        st.altair_chart(trajectory_chart(result.history), width="stretch")

    if comparison:
        st.markdown("#### Test-split scores by metric")
        st.altair_chart(
            metric_comparison_chart(comparison), width="stretch"
        )

        st.markdown("#### Per-example comparison")
        df = pd.DataFrame(comparison["per_example"])
        df["delta"] = df["optimized_score"] - df["baseline_score"]
        st.dataframe(
            df[
                [
                    "instruction",
                    "baseline_score",
                    "optimized_score",
                    "delta",
                    "baseline_prediction",
                    "optimized_prediction",
                    "reference",
                ]
            ].sort_values("delta"),
            width="stretch",
            height=340,
        )

    with st.expander("Search trajectory detail"):
        for h in result.history:
            st.markdown(
                f"**Iteration {h['iteration']}** — pool {h['pool_size']}, "
                f"best dev `{h['best_dev_score']:.4f}`"
            )
            for c in h["beam"]:
                st.caption(
                    f"dev `{c['dev_mean']:.4f}` ({c['pulls']} pulls) · "
                    f"{c['origin']}"
                )
                st.code(c["prompt"], language="text")

    u = result.usage
    st.caption(
        f"{u.get('calls', 0)} API calls ({u.get('cache_hits', 0)} served from "
        f"cache) · estimated spend ${u.get('cost_usd', 0):.3f} · "
        f"{result.elapsed_seconds:.1f}s"
    )
    if st.session_state.get("run_dir"):
        report = build_report(settings, result, comparison)
        st.download_button(
            "Download report (markdown)",
            report,
            file_name="apo_report.md",
            mime="text/markdown",
        )


def render_dataset_tab() -> None:
    st.subheader("Alpaca dataset")
    st.caption(
        "Source: https://huggingface.co/datasets/tatsu-lab/alpaca — 52,002 "
        "instruction/input/output triples."
    )
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("Rows to show", 5, 200, 25, step=5)
    mode = c2.selectbox("Example type", ["any", "with input field", "instruction only"])
    kw = c3.text_input("Keyword filter", value="")

    if st.button("Load sample"):
        try:
            split = split_from_settings(
                Settings(
                    n_train=int(n),
                    n_dev=1,
                    n_test=1,
                    require_input={
                        "any": None,
                        "with input field": True,
                        "instruction only": False,
                    }[mode],
                    filter_keyword=kw.strip() or None,
                )
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "instruction": e.instruction,
                            "input": e.input,
                            "output": e.output,
                        }
                        for e in split.train
                    ]
                ),
                width="stretch",
                height=460,
            )
        except ValueError as exc:
            st.error(str(exc))


def render_runs_tab() -> None:
    st.subheader("Saved runs")
    if not RUNS_DIR.exists():
        st.info("No runs saved yet.")
        return
    dirs = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True
    )
    if not dirs:
        st.info("No runs saved yet.")
        return

    choice = st.selectbox("Run", [p.name for p in dirs])
    path = RUNS_DIR / choice
    result_file = path / "result.json"
    if not result_file.exists():
        st.error("This run directory has no result.json.")
        return

    data = json.loads(result_file.read_text(encoding="utf-8"))
    st.markdown("**Optimized prompt**")
    st.code(data["best_prompt"], language="text")
    st.caption(
        f"best dev score {data['best_dev_score']:.4f} · "
        f"{data['usage'].get('calls', 0)} calls · "
        f"${data['usage'].get('cost_usd', 0):.3f}"
    )
    report = path / "report.md"
    if report.exists():
        with st.expander("Full report"):
            st.markdown(report.read_text(encoding="utf-8"))


def render_playground_tab() -> None:
    st.subheader("Try a prompt")
    st.caption(
        "Run any instruction through a prompt to see the difference directly. "
        "Uses the backend selected in the sidebar."
    )
    result = st.session_state.get("result")
    default_prompt = result.best_prompt if result else Settings().initial_prompt

    prompt = st.text_area("System prompt", value=default_prompt, height=140)
    instruction = st.text_area(
        "Instruction", value="Explain why the sky is blue to a 10-year-old.", height=90
    )
    extra = st.text_area("Input (optional)", value="", height=70)

    if st.button("Run", type="primary"):
        settings: Settings = st.session_state.get("settings", Settings())
        try:
            backend = make_backend(settings)
            user = f"Instruction:\n{instruction}"
            if extra.strip():
                user += f"\n\nInput:\n{extra.strip()}"
            completion = backend.complete(role="task", prompt=user, system=prompt)
            st.markdown("**Answer**")
            st.write(completion.text or "_(empty response)_")
            st.caption(
                f"{completion.input_tokens} in / {completion.output_tokens} out tokens"
            )
        except Exception as exc:
            st.error(explain_api_error(exc) or f"Failed: {exc}")


# --------------------------------------------------------------------------


def main() -> None:
    st.title("Automatic Prompt Optimization")
    st.caption(
        "Refines a prompt from execution feedback on the Alpaca dataset — "
        "textual gradients, beam search, and a UCB bandit."
    )

    settings, go, run_test = sidebar()
    tabs = st.tabs(["Optimize", "Results", "Dataset", "Runs", "Playground"])
    with tabs[0]:
        render_optimize_tab(settings, go, run_test)
    with tabs[1]:
        render_results_tab()
    with tabs[2]:
        render_dataset_tab()
    with tabs[3]:
        render_runs_tab()
    with tabs[4]:
        render_playground_tab()


if __name__ == "__main__":
    main()