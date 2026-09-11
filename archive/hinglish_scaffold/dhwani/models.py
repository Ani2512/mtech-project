"""Model backends with one interface: transcribe(audio_path, prompt) -> str.

qwen2-audio  Qwen/Qwen2-Audio-7B-Instruct  (phase 2 fine-tunes this)
shuka        sarvamai/shuka-1               (existing open Hindi audio LLM, Llama3-8B decoder)
mock         no audio; corrupts the reference with taxonomy-shaped errors so the
             harness can be exercised end-to-end on a CPU.
"""
from __future__ import annotations

import random
import re
from typing import Protocol

from .taxonomy import LIGHT_VERB_STEMS, TRANSLATION_SEED, is_fusion
from .normalize import phonetic_key, script_of, tokenize

DEFAULT_PROMPT = (
    "Transcribe this Hindi-English (Hinglish) speech exactly as spoken. "
    "Do not translate. Keep English words in Roman script and Hindi words in Devanagari."
)


class Backend(Protocol):
    name: str
    def transcribe(self, audio_path: str | None, prompt: str = DEFAULT_PROMPT, reference: str | None = None) -> str: ...


class MockBackend:
    """Injects FUSION / LIGHT_VERB / SCRIPT / TRANSLATE errors at fixed rates."""
    name = "mock"

    def __init__(self, seed: int = 0, p_fusion=0.7, p_light=0.5, p_script=0.3, p_translate=0.2, p_sentence=0.05):
        self.rng = random.Random(seed)
        self.p = dict(fusion=p_fusion, light=p_light, script=p_script, translate=p_translate, sentence=p_sentence)
        self._en2hi = {en: sorted(his)[0] for en, his in TRANSLATION_SEED.items()}

    def transcribe(self, audio_path, prompt=DEFAULT_PROMPT, reference=None):
        assert reference is not None, "mock backend needs the reference"
        toks = tokenize(reference)
        if self.rng.random() < self.p["sentence"]:
            return "I need to do this tomorrow"        # whole-sentence translation
        out = []
        for i, t in enumerate(toks):
            r = self.rng.random()
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if is_fusion(t) and r < self.p["fusion"]:
                out.append(re.sub(r"(ein|on|ों|ें)$", "", t) + "s")          # filmein -> films
            elif any(nxt.startswith(s) for s in LIGHT_VERB_STEMS) and script_of(t) == "latin" and r < self.p["light"]:
                continue                                                     # drop the English content word
            elif phonetic_key(t) in self._en2hi and r < self.p["translate"]:
                out.append({"kal": "कल", "aaj": "आज", "paani": "पानी", "thik": "ठीक"}.get(self._en2hi[phonetic_key(t)], self._en2hi[phonetic_key(t)]))
            elif script_of(t) == "latin" and r < self.p["script"]:
                out.append({"phone": "फ़ोन", "school": "स्कूल", "office": "ऑफिस", "doctor": "डॉक्टर", "time": "टाइम", "bus": "बस", "rate": "रेट", "problem": "प्रॉब्लम"}.get(t, t))
            else:
                out.append(t)
        return " ".join(out)


class Qwen2AudioBackend:
    name = "qwen2-audio"

    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct", device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype or torch.float16, device_map=device
        ).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def transcribe(self, audio_path, prompt=DEFAULT_PROMPT, reference=None):
        import librosa

        audio, _ = librosa.load(audio_path, sr=self.sr)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_path}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ids = ids[:, inputs.input_ids.size(1):]
        return self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class ShukaBackend:
    name = "shuka"

    def __init__(self, model_id: str = "sarvamai/shuka-1", device: str = "cuda"):
        from transformers import pipeline

        self.pipe = pipeline(model=model_id, trust_remote_code=True, device=0 if device == "cuda" else -1)

    def transcribe(self, audio_path, prompt=DEFAULT_PROMPT, reference=None):
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000)
        turns = [
            {"role": "system", "content": "Respond only with the verbatim transcript."},
            {"role": "user", "content": f"<|audio|>\n{prompt}"},
        ]
        out = self.pipe({"audio": audio, "turns": turns, "sampling_rate": sr}, max_new_tokens=256)
        return (out if isinstance(out, str) else str(out)).strip()


def get_backend(name: str, **kw) -> Backend:
    return {"mock": MockBackend, "qwen2-audio": Qwen2AudioBackend, "shuka": ShukaBackend}[name](**kw)
