"""Model backends. Interface: ground(audio_path, query_text) -> raw text.

mock:oracle            returns the ground truth (harness sanity)
mock:ignore_condition  returns every occurrence of X, ignoring the condition
mock:first_only        returns the first occurrence of X only (TAG-Bench's dominant failure)
mock:jitter            oracle with +-0.4 s boundary noise
qwen2.5-omni           Qwen/Qwen2.5-Omni-7B thinker (text out)
qwen2-audio            Qwen/Qwen2-Audio-7B-Instruct
gemini                 Gemini via google-genai (GEMINI_API_KEY)
"""
from __future__ import annotations

import json
import os
import random

SYSTEM = (
    "You are an audio analysis assistant. You will hear one recording and receive a query "
    "describing which sound events to locate. Reply with ONLY a JSON list of [start, end] pairs "
    "in seconds, e.g. [[1.2, 2.0], [7.5, 8.1]]. If nothing in the recording satisfies the query, reply []."
)


def prompt_for(query_text: str, duration: float | None = None) -> str:
    d = f" The recording is {duration:.1f} seconds long." if duration else ""
    return f"Locate: {query_text}.{d} Reply with only the JSON list."


class MockBackend:
    def __init__(self, mode: str = "oracle", seed: int = 0):
        self.mode, self.rng = mode, random.Random(seed)
        self.name = f"mock:{mode}"

    def ground(self, audio_path, query_text, query=None, duration=None):
        assert query is not None, "mock backends need the Query object"
        if self.mode == "oracle":
            iv = query.answer
        elif self.mode == "ignore_condition":
            iv = query.plain_intervals
        elif self.mode == "first_only":
            iv = query.plain_intervals[:1]
        elif self.mode == "jitter":
            iv = [(max(0, a + self.rng.uniform(-0.4, 0.4)), b + self.rng.uniform(-0.4, 0.4)) for a, b in query.answer]
        else:
            raise KeyError(self.mode)
        return json.dumps([[round(a, 2), round(b, 2)] for a, b in iv])


class Qwen25OmniBackend:
    name = "qwen2.5-omni"

    def __init__(self, model_id="Qwen/Qwen2.5-Omni-7B", device="auto"):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, enable_audio_output=False
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def ground(self, audio_path, query_text, query=None, duration=None):
        from qwen_omni_utils import process_mm_info

        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": prompt_for(query_text, duration)}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, return_audio=False, max_new_tokens=200, do_sample=False)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class Qwen2AudioBackend:
    name = "qwen2-audio"

    def __init__(self, model_id="Qwen/Qwen2-Audio-7B-Instruct", device="auto"):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16, device_map=device).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def ground(self, audio_path, query_text, query=None, duration=None):
        import librosa

        audio, _ = librosa.load(audio_path, sr=self.sr)
        conv = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "audio", "audio_url": audio_path}, {"type": "text", "text": prompt_for(query_text, duration)}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        ids = ids[:, inputs.input_ids.size(1):]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model_id="gemini-2.5-pro"):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_id = model_id

    def ground(self, audio_path, query_text, query=None, duration=None):
        from google.genai import types

        data = open(audio_path, "rb").read()
        r = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Part.from_bytes(data=data, mime_type="audio/wav"), SYSTEM + "\n" + prompt_for(query_text, duration)],
        )
        return (r.text or "").strip()


def get_backend(name: str):
    if name.startswith("mock:"):
        return MockBackend(name.split(":", 1)[1])
    return {"qwen2.5-omni": Qwen25OmniBackend, "qwen2-audio": Qwen2AudioBackend, "gemini": GeminiBackend}[name]()
