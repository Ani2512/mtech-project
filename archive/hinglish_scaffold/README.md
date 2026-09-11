# Project Dhwani — phase 1: baseline failure study

Hindi/Hinglish audio LLM. Phase 1 asks one question: **how do existing audio
LLMs fail on Hinglish code-switched speech, and in what proportions?** The
answer is the empirical basis for the morphology-aware DPO method in phase 2
(the rejected responses are built from these failure modes).

This folder is a standalone Python package (`dhwani`) plus a runner. It has no
notebook state; the Colab notebook only clones this repo and runs the CLI.

## Failure taxonomy (v0.1, to be refined against native-speaker annotation)

| Code | Mode | Example (reference → typical hypothesis) |
|---|---|---|
| `FUSION` | Morphological fusion: English stem + Hindi inflection inside one word | *filmein* (film + -ein) → *films* / *film me* |
| `LIGHT_VERB` | English content word + Hindi light verb (*karna/hona/dena/lena*) | *adjust karna* → *samayojit karna* / *adjust* dropped |
| `SCRIPT` | Right word, unexpected script (English word in Devanagari or Hindi word in Roman) | *phone* → *फ़ोन* |
| `TRANSLATE` | Translation instead of transcription (Hindi span rendered in English, or vice versa) | *mujhe kal jaana hai* → *I have to go tomorrow* |
| `OTHER` | Any remaining substitution/deletion/insertion | — |

`SCRIPT` is not counted as an error under the script-agnostic metric; it is
counted separately because it is the mode that Sarvam Audio's transcription
modes explicitly control, and it must be separated from the true lexical errors.

## Layout

```
dhwani/
  taxonomy.py    failure modes + rule-based attribution of each error token to a mode
  normalize.py   Devanagari/Roman handling, script-agnostic token normalisation
  metrics.py     WER / CER, script-agnostic WER, per-mode counts
  data.py        loaders: HiACC (Zenodo), MUCS 2021 Hindi-English (OpenSLR 104), IndicVoices (HF)
  models.py      backends: Qwen2-Audio-7B-Instruct, Shuka-1, mock
  run_baseline.py  CLI: run a model over a dataset, write JSONL + summary
configs/         one YAML per (model, dataset) run
tests/           unit tests; run locally with no GPU or model
docs/            findings that gate the project (competitors, data licensing)
```

## Run

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # harness sanity, no model needed

# Smoke test the whole pipeline with the mock backend (no GPU):
python -m dhwani.run_baseline --model mock --dataset synthetic --n 50 --out runs/mock

# Real run on a T4/A100 (Colab):
python -m dhwani.run_baseline --model qwen2-audio --dataset hiacc --split adult --n 200 --out runs/qwen2_hiacc
python -m dhwani.run_baseline --model shuka   --dataset hiacc --split adult --n 200 --out runs/shuka_hiacc
```

Each run writes `predictions.jsonl` (one row per utterance: reference,
hypothesis, aligned errors, attributed modes) and `summary.json` (WER, CER,
script-agnostic WER, per-mode counts and rates).

## Datasets and licences (verified 2026-09-11, see docs/data_licensing.md)

| Dataset | Use | Licence | Notes |
|---|---|---|---|
| HiACC | Primary eval; taxonomy attribution | CC BY-NC 4.0 | 5.24 h, 44 speakers, **word-level hi/en labels** |
| MUCS 2021 Hindi-English | Phase 2 preference data | CC BY-SA 4.0 | 89.86 h train, technical lectures |
| IndicVoices (Hindi) | Monolingual Hindi control | CC BY 4.0 | spontaneous speech |

## Baselines

`Qwen2-Audio-7B-Instruct` (the model phase 2 fine-tunes) and `sarvamai/shuka-1`
(the existing open Hindi audio LLM; see docs/competitors.md). Neither has
been trained for Hinglish code-switching.
