# Semantic Search over Microsoft AI Show Videos

A semantic search application that lets students ask a question in plain English
— *"What are Jupyter Notebooks?"*, *"Can you use RStudio with Azure ML?"* — and
jump straight to the timestamped moment in a Microsoft AI Show video where it is
explained.

Unlike keyword search, matching is done on **meaning**: both the query and every
transcript segment are represented as embedding vectors, and relevance is their
**cosine similarity**. Everything runs locally and free — no API keys, and no
data leaves your machine.

> **Why semantic search matters, in one example.** For the query *"Can you use
> RStudio with Azure ML?"*, a keyword baseline (TF-IDF) ranks a generic
> *Introduction to Azure ML Services* segment first at 0.127, because it only
> matches "Azure" and "ML". Semantic search returns **"Using R with Azure
> Machine Learning" at 0.788** — even though the word "RStudio" never appears in
> that summary. The model understands that R ≈ RStudio.

---

## 1. The dataset

`embedding_index_3m.json` (~49 MB) — **1,409 transcript segments across 279
videos**, each a 3-minute chunk:

| Field | Meaning |
|---|---|
| `title` | Video title |
| `speaker` | Speaker(s) in the episode |
| `summary` | LLM-generated summary of the 3-minute chunk |
| `videoId` | YouTube video id |
| `start` / `seconds` | Chunk start time (`00:06:07` / `367`) |
| `ada_v2` | 1536-dim OpenAI embedding shipped with the dataset |

## 2. How it works

```
        query text                        1,409 summaries
             │                                   │
             ▼                                   ▼
   ┌──────────────────────┐        ┌─────────────────────────────┐
   │ embed query with     │        │ embed segments with the     │
   │ bge-small-en-v1.5    │        │ SAME model, once, on the    │
   │ (local, ~8 ms)       │        │ app's first launch          │
   └──────────┬───────────┘        └──────────────┬──────────────┘
              │ q (unit vector)                   │ V (unit rows)
              └────────────► scores = V · q ◄─────┘
                                   │
                      argpartition → Top-5 → results
```

**Why the segments are re-embedded.** A query can only be compared against
vectors produced by the *same* model — cosine similarity across two different
models is meaningless. The `ada_v2` vectors in the dataset come from OpenAI's
paid API, so querying them would need a key. Embedding the summaries locally
with `BAAI/bge-small-en-v1.5` (384-dim) puts the query and the segments in one
space, at zero cost. This is genuine semantic search; it just uses a smaller
model than ada-002.

**Notes on the implementation**

* Every row of `V` and the query `q` are scaled to unit length, so cosine
  similarity is a single matrix–vector product — the full index ranks in ~8 ms.
* `np.argpartition` finds the Top-K without fully sorting all 1,409 scores.
* Parsing 49 MB of JSON takes ~20 s, so the segments are cached on first run.
  The cache self-invalidates if the dataset changes.
* The model runs on ONNX Runtime (`fastembed`), not PyTorch — ~15× smaller, and
  it works on locked-down Windows machines where Application Control blocks
  torch's DLLs.

## 3. Setup and running

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's it — `app.py` is the only file you need to run. On its very first launch
it downloads the embedding model and embeds all 1,409 segments, showing progress
in the browser. That takes about a minute and happens only once; the results go
into a `.cache/` folder next to `app.py`, and every later launch is instant.

To do that step up front in a terminal instead — so the first user of the app
never waits — run it once beforehand:

```bash
python build_local_index.py     # optional
```

Open <http://localhost:8501>, then type a question or click an example query.
Every search returns the **Top 5** most relevant segments, each showing the
**video title, summary, similarity score, timestamp, and a clickable YouTube
link** that starts playback at that exact moment.

## 4. Verified results

| Query | Top result | Score |
|---|---|---|
| What are Jupyter Notebooks? | Jupyter Notebooks in Visual Studio Code @ 6:07 | 0.805 |
| Can you use RStudio with Azure ML? | Using R with Azure Machine Learning @ 0:00 | 0.788 |
| How do I detect objects in images? | Computer Vision Made Easy! @ 24:31 | 0.759 |
| Deploying a model to production | Learn How to Deploy Machine Learning Models! @ 15:10 | 0.787 |

## 5. Project layout

```
prompt_engineering_project/
├── app.py                     # Streamlit UI -- the only file you run
├── semantic_search.py         # dataset loading, embedding, cosine ranking
├── build_local_index.py       # optional: embed the segments from a terminal
├── embedding_index_3m.json    # provided dataset
├── requirements.txt
└── .cache/                    # model + vectors, generated on first run
```
