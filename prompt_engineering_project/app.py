"""Streamlit front-end for the Microsoft AI Show semantic search.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from semantic_search import (
    TOP_K,
    IndexNotBuilt,
    SearchEngine,
    build_vectors,
    index_exists,
    load_segments,
)

EXAMPLE_QUERIES = [
    "What are Jupyter Notebooks?",
    "Can you use RStudio with Azure ML?",
    "How do I detect objects in images?",
    "Responsible AI and model fairness",
]

st.set_page_config(page_title="AI Show Semantic Search")

st.markdown(
    """
    <style>
      .result-card {
          border: 1px solid rgba(128,128,128,.25);
          border-radius: .6rem;
          padding: 1rem 1.15rem;
          margin-bottom: .6rem;
          background: rgba(128,128,128,.06);
      }
      .result-title { font-size: 1.05rem; font-weight: 600; margin-bottom: .15rem; }
      .result-meta  { font-size: .82rem; opacity: .75; margin-bottom: .5rem; }
      .result-summary { font-size: .92rem; line-height: 1.5; }
      .badge {
          display: inline-block; padding: .1rem .5rem; border-radius: 1rem;
          font-size: .75rem; font-weight: 600; margin-right: .4rem;
          border: 1px solid rgba(128,128,128,.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> SearchEngine:
    """Load the index, building it on first launch if it does not exist yet.

    Deliberately draws nothing: Streamlit replays elements written inside a
    cached function on every later rerun, which would leave a stale progress
    box on screen. The caller owns the messaging instead.
    """
    try:
        return SearchEngine.load()
    except IndexNotBuilt:
        segments = load_segments()
        return SearchEngine(segments, build_vectors(segments))


def main() -> None:
    st.title("Microsoft AI Show — Semantic Video Search")
    st.caption(
        "Ask a question in plain English and jump straight to the moment in the "
        "video where it is explained."
    )

    message = (
        "Loading search index…"
        if index_exists()
        else "First run: downloading the embedding model and embedding every "
        "segment. This takes about a minute, and happens only once."
    )
    try:
        with st.spinner(message):
            engine = load_engine()
    except (FileNotFoundError, IndexNotBuilt) as exc:
        st.error(str(exc))
        st.stop()

    if "query" not in st.session_state:
        st.session_state.query = ""

    st.write("**Try one of these:**")
    for column, example in zip(st.columns(len(EXAMPLE_QUERIES)), EXAMPLE_QUERIES):
        if column.button(example, use_container_width=True):
            st.session_state.query = example

    query = st.text_input(
        "Your question", key="query", placeholder="e.g. What are Jupyter Notebooks?"
    )
    if not query.strip():
        st.stop()

    with st.spinner("Searching…"):
        results = engine.search(query, top_k=TOP_K)

    if not results:
        st.warning("No segments matched. Try rephrasing your question.")
        st.stop()

    st.subheader(f"Top {len(results)} results")
    for result in results:
        segment = result.segment
        speaker = f" ·{segment.speaker}" if segment.speaker else ""
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-title">{result.rank}. {segment.title}</div>
              <div class="result-meta">
                <span class="badge">similarity {result.score:.4f}</span>
                <span class="badge">{segment.start}</span>{speaker}
              </div>
              <div class="result-summary">{segment.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(f""
                       f""
                       f""
                       f""
                       f""
                       f""
                       f""
                       f"Watch at {segment.timestamp}", segment.youtube_url)


if __name__ == "__main__":
    main()
