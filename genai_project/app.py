"""Streamlit front end for the Multi-Agent Business Analyst.

    streamlit run app.py

The UI exists to make the *process* visible, not just the answer: you watch
the planner choose a roster, the specialists run concurrently, and the judge
grade the result. The tabs then let you audit any step — which passages an
agent retrieved, what JSON it returned, what the whole run cost.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from analyst import (
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    SPECIALIST_ROLES,
    DataRoom,
    Settings,
    load_api_key,
    render_evaluation,
    run_analysis,
)
from analyst.config import (
    EFFORT_LEVELS,
    PROVIDER_MODELS,
    ROLE_LABELS,
    ROLE_RISK,
    ROLE_STRATEGY,
)
from analyst.reporting import stars
from analyst.retrieval import HybridRetriever, build_corpus
from run_analysis import EXAMPLE_QUESTIONS

st.set_page_config(
    page_title="Multi-Agent Business Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_ICONS = {
    "preparing": "⏳",
    "running": "⚙️",
    "done": "✅",
    "failed": "❌",
    "warning": "⚠️",
}


# ---------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_room() -> DataRoom:
    return DataRoom()


@st.cache_resource(show_spinner=False)
def get_retriever(alpha: float) -> HybridRetriever:
    return HybridRetriever(build_corpus(get_room()), alpha=alpha)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def sidebar() -> Settings:
    st.sidebar.title("⚙️ Configuration")

    has_anthropic = bool(load_api_key(PROVIDER_ANTHROPIC))
    has_openai = bool(load_api_key(PROVIDER_OPENAI))

    backends = [PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_MOCK]
    default_backend = (
        PROVIDER_OPENAI if has_openai else PROVIDER_ANTHROPIC if has_anthropic else PROVIDER_MOCK
    )
    backend = st.sidebar.selectbox(
        "Backend",
        backends,
        index=backends.index(default_backend),
        format_func=lambda b: {
            PROVIDER_ANTHROPIC: "Anthropic (Claude)",
            PROVIDER_OPENAI: "OpenAI",
            PROVIDER_MOCK: "Mock — offline, no key, no cost",
        }[b],
    )

    if backend == PROVIDER_MOCK:
        st.sidebar.info(
            "The mock backend exercises the real orchestration, schema "
            "validation and reporting paths, but its findings are filler — "
            "not analysis."
        )
    elif backend == PROVIDER_ANTHROPIC and not has_anthropic:
        st.sidebar.error("No `ANTHROPIC_API_KEY` found. Add one to `.env`.")
    elif backend == PROVIDER_OPENAI and not has_openai:
        st.sidebar.error("No `OPENAI_API_KEY` found. Add one to `.env`.")

    models = PROVIDER_MODELS.get(backend, [])
    # Empty string means "resolve from the backend" — see Settings.__post_init__.
    model = st.sidebar.selectbox("Model", models) if models else ""

    st.sidebar.divider()
    st.sidebar.subheader("Agents")
    enabled = st.sidebar.multiselect(
        "Specialists available to the planner",
        SPECIALIST_ROLES,
        default=SPECIALIST_ROLES,
        format_func=lambda r: ROLE_LABELS[r],
    )
    planner_selects = st.sidebar.checkbox(
        "Let the planner drop irrelevant agents",
        value=True,
        help="Off means every selected specialist runs, whatever the planner thinks. "
        "On is cheaper and usually sharper.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Reasoning")
    effort = st.sidebar.select_slider(
        "Effort for the specialists",
        options=EFFORT_LEVELS,
        value="medium",
        help="Risk and Strategy always run one level up — they reconcile "
        "conflicting evidence, which is the expensive part.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Retrieval")
    use_retrieval = st.sidebar.checkbox("Hybrid document retrieval", value=True)
    alpha = st.sidebar.slider(
        "Dense weight in the blend",
        0.0, 1.0, 0.5, 0.1,
        disabled=not use_retrieval,
        help="0 = pure BM25 lexical, 1 = pure LSA dense.",
    )
    top_k = st.sidebar.slider("Passages per agent", 3, 15, 8, disabled=not use_retrieval)

    st.sidebar.divider()
    st.sidebar.subheader("Evaluation")
    run_eval = st.sidebar.checkbox(
        "Grade the report with an LLM judge",
        value=True,
        help="An independent pass that checks every figure against the evidence.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Budget")
    max_cost = st.sidebar.number_input("Max spend (USD)", 0.25, 50.0, 5.0, 0.25)
    max_calls = st.sidebar.number_input("Max model calls", 5, 200, 60, 5)
    use_cache = st.sidebar.checkbox(
        "Reuse cached responses",
        value=True,
        help="Identical prompts are served from disk, so re-running a question is free.",
    )

    settings = Settings(
        backend=backend,
        model=model,
        enabled_specialists=enabled or list(SPECIALIST_ROLES),
        planner_selects_agents=planner_selects,
        use_retrieval=use_retrieval,
        hybrid_alpha=alpha,
        retrieval_top_k=top_k,
        run_evaluation=run_eval,
        use_cache=use_cache,
        max_cost_usd=float(max_cost),
        max_llm_calls=int(max_calls),
    )
    # The two synthesis agents get one notch more effort than the specialists.
    idx = EFFORT_LEVELS.index(effort)
    deeper = EFFORT_LEVELS[min(idx + 1, len(EFFORT_LEVELS) - 1)]
    settings.role_effort = {
        **{r: effort for r in SPECIALIST_ROLES},
        "planner": effort,
        ROLE_RISK: deeper,
        ROLE_STRATEGY: deeper,
        "evaluator": deeper,
    }
    return settings


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
def render_overview(result) -> None:
    strategy = result.strategy.data if (result.strategy and result.strategy.ok) else {}
    risk = result.risk.data if (result.risk and result.risk.ok) else {}

    # A quarter-width metric truncates anything longer than ~11 characters,
    # so "CONDITIONAL GO" renders as "CONDITIONA…". Abbreviate deliberately
    # rather than letting the browser cut it mid-word.
    short = {"go": "GO", "conditional_go": "COND. GO", "hold": "HOLD", "no_go": "NO GO"}
    decision = strategy.get("decision", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", short.get(decision, str(decision).replace("_", " ").upper() or "—"))
    c2.metric("Confidence", f"{strategy.get('confidence', '—')}%")
    c3.metric("Overall risk", stars(risk.get("overall_risk", 0)) if risk else "—")
    if result.evaluation and result.evaluation.ok:
        c4.metric(
            "Judge score",
            f"{result.evaluation.data.get('overall_score', '—')}/100",
            result.evaluation.data.get("verdict", ""),
        )
    else:
        c4.metric("Judge score", "—")

    if strategy.get("recommendation"):
        st.success(f"**{strategy['recommendation']}**")
    if risk.get("top_concern"):
        st.warning(f"**Top concern:** {risk['top_concern']}")
    st.markdown(result.report_markdown)


def render_agents(result) -> None:
    everything = list(result.specialists)
    for extra in (result.risk, result.strategy, result.evaluation):
        if extra is not None:
            everything.append(extra)

    for r in everything:
        icon = "✅" if r.ok else "❌"
        with st.expander(
            f"{icon} {r.label} — {r.seconds:.1f}s"
            + ("  ·  cached" if r.cached else "")
            + (f"  ·  {r.evidence_chars:,} chars of evidence" if r.evidence_chars else "")
        ):
            if not r.ok:
                st.error(r.error)
                continue
            if r.citations:
                st.caption("Retrieved: " + ", ".join(dict.fromkeys(r.citations)))
            st.json(r.data, expanded=False)


def render_evidence(result, settings: Settings) -> None:
    st.subheader("What the agents were given")
    st.caption(
        "Numbers come from `analytics.py`, which computes them from the CSVs in "
        "code. Prose comes from hybrid retrieval over the markdown files. No "
        "agent ever sees a raw file it has to do arithmetic on."
    )

    if not settings.use_retrieval:
        st.info("Retrieval was disabled for this run.")
        return

    retriever = get_retriever(settings.hybrid_alpha)
    st.write(
        f"**{len(retriever.corpus)} passages indexed** — "
        f"dense retriever {'available' if retriever.dense_available else 'unavailable (BM25 only)'}"
    )

    query = st.text_input("Try the retriever directly", value=result.question)
    if query:
        hits = retriever.search(query, top_k=settings.retrieval_top_k)
        if not hits:
            st.info("No passage matched.")
        for n, hit in enumerate(hits, 1):
            st.markdown(
                f"**P{n} · `{hit.passage.citation}`**  \n"
                f"<small>fused {hit.score:.4f} — {hit.why()}</small>",
                unsafe_allow_html=True,
            )
            st.code(hit.passage.text, language=None)


def render_data_room() -> None:
    room = get_room()
    st.subheader("The data room")
    st.caption("Everything the team can see. Nothing else is available to it.")
    st.code(room.inventory_report(), language=None)
    st.markdown(room.text("company_profile.md"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📊 Multi-Agent Business Analyst")
    st.caption(
        "Eight specialist agents read one company's data room and answer a "
        "decision question — not *summarise this document*, but *should we do this?*"
    )

    settings = sidebar()

    if "question" not in st.session_state:
        st.session_state.question = EXAMPLE_QUESTIONS[0]

    st.markdown("**Try one of these**")
    cols = st.columns(4)
    for n, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[n % 4].button(example, key=f"ex{n}", use_container_width=True):
            st.session_state.question = example

    question = st.text_area(
        "Executive question",
        key="question",
        height=80,
        placeholder="Should we expand into Southeast Asia next year?",
    )
    run = st.button("Run the analysis", type="primary", disabled=not question.strip())

    if run:
        room = get_room()
        missing = room.missing()
        if missing:
            st.error(
                "The data room is incomplete. Run `python make_sample_data.py` "
                f"to regenerate it. Missing: {', '.join(missing[:5])}"
            )
            st.stop()

        log: list[str] = []
        started = time.perf_counter()
        with st.status("Running the agent team…", expanded=True) as status_box:
            trace = st.empty()

            def progress(stage: str, kind: str, detail: str = "") -> None:
                label = ROLE_LABELS.get(stage, stage.title())
                icon = STATUS_ICONS.get(kind, "•")
                elapsed = time.perf_counter() - started
                log.append(f"{icon} `{elapsed:5.1f}s` **{label}** — {detail}")
                trace.markdown("\n\n".join(log[-40:]))

            try:
                result = run_analysis(question, settings, progress=progress)
            except FileNotFoundError as exc:
                status_box.update(label="Data room incomplete", state="error")
                st.error(str(exc))
                st.stop()
            except RuntimeError as exc:
                status_box.update(label="Could not start", state="error")
                st.error(str(exc))
                st.info("Switch the backend to **Mock** in the sidebar to run without a key.")
                st.stop()

            state = "complete" if result.ok else "error"
            status_box.update(
                label=f"Finished in {result.seconds:.1f}s — "
                f"{result.usage.get('calls', 0)} calls, "
                f"${result.usage.get('cost_usd', 0):.4f}",
                state=state,
            )
        st.session_state.result = result
        st.session_state.result_settings = settings

    result = st.session_state.get("result")
    if result is None:
        st.info("Ask a question above to start. No API key? Choose the **Mock** backend.")
        render_data_room()
        return

    if result.aborted:
        st.warning(f"The run stopped early: {result.aborted}")

    # Deliberately not st.tabs: any widget inside a tab triggers a rerun, and
    # st.tabs always reopens on the first tab afterwards. Typing a query into
    # the Evidence retriever would bounce the user back to Report every time.
    # A keyed selector keeps its value across reruns, so the view survives.
    views = ["📄 Report", "🔎 Evaluation", "🤖 Agents", "📚 Evidence", "🗄️ Data room", "⤓ Export"]
    view = st.segmented_control(
        "View", views, default=views[0], key="view", label_visibility="collapsed"
    ) or views[0]
    st.divider()

    if view == views[0]:
        render_overview(result)
    elif view == views[1]:
        st.markdown(render_evaluation(result.evaluation))
    elif view == views[2]:
        render_agents(result)
    elif view == views[3]:
        render_evidence(result, st.session_state.get("result_settings", settings))
    elif view == views[4]:
        render_data_room()
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        full = result.report_markdown
        if result.evaluation is not None:
            full += "\n\n" + render_evaluation(result.evaluation)
        st.download_button(
            "Download report (Markdown)",
            full,
            file_name=f"analysis-{stamp}.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download full run (JSON)",
            json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
            file_name=f"run-{stamp}.json",
            mime="application/json",
        )
        st.json(result.usage)


if __name__ == "__main__":
    main()