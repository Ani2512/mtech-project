# Data access and licensing — verified 2026-09-11

| Dataset | Source | Licence | Size (Hindi/Hinglish) | Labels | Role | Constraint |
|---|---|---|---|---|---|---|
| **HiACC** | Zenodo record 15551669 | CC BY-NC 4.0 | 5.24 h (3.22 h adult, 2.04 h children), 44 speakers, 16 kHz mono | Devanagari for Hindi, Latin for English, **token-level hi/en labels in JSON** | Primary evaluation set; taxonomy attribution | Non-commercial only — fine for a thesis. Children's subset recorded in classrooms with noise. Manual download. |
| **MUCS 2021 Hindi-English** | OpenSLR 104 (`Hindi-English_train.tar.gz` 7.3 GB, `_test.tar.gz` 443 MB) | CC BY-SA 4.0 (code-switching subsets only; the monolingual MUCS subsets are under an MSR licence instead) | 89.86 h train, 5.18 h test | Mixed-script transcripts, no token language labels | Phase 2 preference-pair construction; secondary eval | Share-alike: any released derived transcripts/preference data must be CC BY-SA 4.0. Domain is technical spoken tutorials, so code-switching is mostly technical vocabulary. |
| **IndicVoices (Hindi)** | `ai4bharat/IndicVoices` on Hugging Face | CC BY 4.0 | 12,000 h across 22 languages; Hindi subset large; 76 % extempore, 15 % conversational | Native script | Monolingual Hindi control (mode rates should be near zero) | None material. |
| `agarwalayushi/hinglish` compilation | Hugging Face | Compilation says CC BY 4.0 | 2,264 h / 815 k clips | `<hi-en>` tags on code-switched passages | **Do not rely on it.** | Six of 14 upstream sources are "licence varies"; several are TTS-synthesised, not natural speech. Usable only after tracing each source. |
| `sarvamai/audiollm-evals`, `sarvamai/contextual_asr_benchmark` | Hugging Face | Check card | eval-only | — | Extra eval sets for the Shuka comparison | Check licence on card before use. |

## Decisions

- Phase 1 evaluation runs on **HiACC** (labelled, so the taxonomy can be
  attributed automatically) and **MUCS test** (larger, unlabelled, WER only).
- Phase 2 preference data is built from **MUCS train**; released artefacts
  inherit CC BY-SA 4.0.
- Native-speaker annotation for the taxonomy audit is done on a 300-utterance
  HiACC adult sample first (clean audio), children's subset second.

## Still to confirm by hand

1. HiACC archive layout after extraction (`dhwani/data.py::load_hiacc` assumes
   `transcripts.csv` + `lang_labels.json`; adjust to the real filenames).
2. MUCS tarball internal layout (`transcripts/text`, `wav.scp`); the loader
   searches recursively so minor differences should not matter.
3. IndicVoices config name for Hindi on HF (`"hindi"` assumed).
