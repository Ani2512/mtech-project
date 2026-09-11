# Competitor check — 2026-09-11

Question: does an instruction-following Hindi/Hinglish audio LLM already exist,
and is AI4Bharat building one?

## Verdict

**The proposal's "first instruction-following audio LLM for Hindi" claim is
false as written and must be narrowed.** Sarvam's Shuka v1 has existed since
August 2024. The defensible claim is: *first audio LLM trained and evaluated
specifically for Hindi–English code-switching, with a morphology-aware
preference method.* Nothing found addresses intra-word code-switching in an
audio LLM. Shuka v1 becomes a **baseline**, not a competitor.

## Findings

| Effort | What it is | Open? | Hinglish / code-switching | Impact |
|---|---|---|---|---|
| **Shuka v1** (Sarvam, Aug 2024) — `sarvamai/shuka-1` | Saaras v1 encoder + 60M projector + Llama-3-8B-Instruct, projector-only training on <100 h, Hindi+English, audio-text-to-text | Yes (Llama 3 licence) | Not mentioned anywhere in card or blog | Baseline #2. Also invalidates "first Hindi audio LLM". |
| **Sarvam Audio** (Feb 2026) | Audio extension of Sarvam 3B; ASR with five transcription modes incl. "Normalised Code-Mixed" (native script + English in Roman) and "Romanised"; diarisation; contextual ASR for voice bots | **No** — API only ("available soon on the Sarvam Dashboard"); no weights, no paper | Yes at the *script-control* level (which script to render English words in) | Not a baseline (closed). Confirms `SCRIPT` mode is a product concern; our contribution must be on the lexical modes (`FUSION`, `LIGHT_VERB`, `TRANSLATE`). Their `sarvamai/contextual_asr_benchmark` and `sarvamai/audiollm-evals` datasets are usable for eval. |
| **Bodhan AI + AI4Bharat, four models** (Sep 2026) | IndicOCR, Indic-Translate, Indic-Transcribe (ASR, "Flex" mode keeps English in Latin script), Indic-Speak (TTS) | Yes | Script-level only (ASR) | Plain ASR, not an audio LLM. Indic-Transcribe Flex is a useful ASR reference point. |
| **IndicContextEval** (Khapra group / AI4Bharat, Interspeech 2026, arXiv 2606.19157) | Benchmark: do audio LLMs use provided context across 8 Indic languages; evaluates 5 existing models; trains nothing | Data CC BY 4.0 | No | **Signal, not competitor**: AI4Bharat is now *evaluating* audio LLMs on Indic speech, the usual precursor to building one. Re-check their arXiv/HF every month. |
| TTS-STT Flywheel (arXiv 2605.03073) | LoRA on Whisper for entity-dense Indic ASR, reports code-mixed Hindi as a weak point | CC BY 4.0 | Yes (entity-level) | Plain ASR. Corroborates that code-mixed Hindi is an open failure area. |
| HiACC corpus paper (2025) | Hinglish speech corpus with token-level language labels; ASR baselines only | CC BY-NC 4.0 | Yes | Our primary eval set. No LLM work in the paper. |

Not found: any Hindi/Hinglish fine-tune of Qwen2-Audio or Audio Flamingo; any
Hugging Face model tagged `audio-text-to-text` matching "hindi" or "indic"
besides `sarvamai/shuka-1`; any DPO / preference-based method for
code-switching in an Indic audio LLM.

## Consequences for the proposal

1. Replace "first instruction-following audio LLM for Hindi/Hinglish" with
   "first Hinglish code-switching-aware audio LLM; first preference-based
   method for intra-word code-switching". Cite Shuka v1 as prior work.
2. Add Shuka v1 as a baseline in every phase 1 table. Its projector-only,
   <100 h training makes it plausible that it is *worse* than Qwen2-Audio on
   Hinglish, which would itself be a reportable finding.
3. Keep the `SCRIPT` mode in the taxonomy but do not build the contribution
   on it: Sarvam Audio already sells script control.
4. Risk register: AI4Bharat (IndicContextEval authors) and Sarvam (Shuka v2
   rumoured, nothing published) are the two groups who could scoop this.
   Mitigation is speed to a first result on HiACC, not idea secrecy.

## Sources

- Shuka v1 model card: https://huggingface.co/sarvamai/shuka_v1
- Meta blog on Shuka: https://ai.meta.com/blog/sarvam-india-audio-language-model-llama/
- Sarvam Audio blog: https://www.sarvam.ai/blogs/sarvam-audio
- Bodhan AI release (Sep 2026): https://www.analyticsvidhya.com/blog/2026/09/bodhan-ai-indic-models/
- IndicContextEval: https://arxiv.org/abs/2606.19157
- TTS-STT Flywheel: https://arxiv.org/abs/2605.03073
- HiACC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12329218/

## Addendum 2026-09-11 (later): adjacent method papers

| Paper | What it is | Overlap with Dhwani | Use |
|---|---|---|---|
| **RLVR for code-switched ASR** (Ye & Vickers, arXiv 2607.02757, Jul 2026) | GRPO on Qwen2-Audio, decoder-only updates, reward = −CER + 0.05·[all chars in allowed scripts]; 10 X–English pairs, trained on 128 h of TTS (CS-FLEURS), zero-shot to SwitchLingua; 20 % data beats 100 % LoRA SFT (CER 0.147 vs 0.159) | Closest method paper so far. **Hindi–English excluded on purpose** ("pairs where the base model has non-trivial recognition capability"). Script reward is character-set membership only; no morphology, no intra-word switching, no preference pairs. No code released. | Cite as the baseline method to beat; their script reward is the trivial version of our `SCRIPT` mode. Their exclusion of Hindi is a quotable statement of the gap. Add GRPO-with-script-reward as a phase 2 comparison arm if compute allows. |
| **LaRA** (Shaik et al., Findings of EMNLP 2024) | "Large Rank Adaptation": speech–text cross-modal learning needs LoRA ranks comparable to the pretrained weight size; HuBERT units + LLM, LibriSpeech/DailyTalk, English only; code on GitHub | Orthogonal: it is about adapter capacity, not code-switching. | Informs the LoRA rank sweep in phase 2 (try r=64 as in 2607.02757 and a high-rank arm). |
| AuRA (arXiv 2606.11033) | Distils an ASR encoder into a LoRA adapter so the LLM "hears" without an external encoder | Orthogonal | Not needed. |
| GELU adapters for code-switching (arXiv 2506.00291) | Whisper + encoder adapter, SEAME/ASCEND | **Withdrawn by the author for incorrect results.** | Do not cite. |
