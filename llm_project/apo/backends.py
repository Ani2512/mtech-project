"""LLM backends: live Anthropic calls, disk caching, and an offline mock.

Every model call in the project goes through `LLMBackend.complete`, which
gives one place to enforce the cost budget, cache repeated calls across runs,
and swap in a deterministic stub when no API credits are available.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import (
    CACHE_DIR,
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    ROLE_JUDGE,
    ROLE_OPTIMIZER,
    ROLE_TASK,
    Settings,
    estimate_cost,
    load_api_key,
)


class BudgetExceeded(RuntimeError):
    """Raised when a run hits its call or cost ceiling."""


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_role: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "by_role": dict(self.by_role),
        }


class LLMBackend(Protocol):
    def complete(
        self,
        *,
        role: str,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Completion: ...


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _BaseBackend:
    """Shared caching, budget enforcement and usage accounting."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usage = Usage()
        self._lock = threading.Lock()

    # -- budget ---------------------------------------------------------
    def _check_budget(self) -> None:
        s = self.settings
        if self.usage.calls >= s.max_llm_calls:
            raise BudgetExceeded(
                f"Reached max_llm_calls={s.max_llm_calls}. Raise the limit or "
                "shrink the run (fewer iterations / smaller splits)."
            )
        if self.usage.cost_usd >= s.max_cost_usd:
            raise BudgetExceeded(
                f"Reached max_cost_usd=${s.max_cost_usd:.2f} "
                f"(spent ${self.usage.cost_usd:.2f})."
            )

    def _record(self, role: str, c: Completion) -> None:
        with self._lock:
            self.usage.calls += 1
            self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
            if c.cached:
                self.usage.cache_hits += 1
                return
            self.usage.input_tokens += c.input_tokens
            self.usage.output_tokens += c.output_tokens
            self.usage.cost_usd += estimate_cost(
                c.model, c.input_tokens, c.output_tokens
            )

    # -- cache ----------------------------------------------------------
    def _cache_path(self, key: str):
        return CACHE_DIR / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Completion | None:
        if not self.settings.use_cache:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Completion(
            text=d["text"],
            model=d.get("model", "?"),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cached=True,
        )

    def _cache_put(self, key: str, c: Completion) -> None:
        if not self.settings.use_cache:
            return
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "text": c.text,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(p)


class OpenAIBackend(_BaseBackend):
    """Live calls against the OpenAI Chat Completions API.

    Written against the GPT-5.6 family parameter surface: `reasoning_effort`
    instead of a thinking budget, `max_completion_tokens` instead of
    `max_tokens`, and strict JSON-schema structured outputs.
    """

    def __init__(self, settings: Settings, api_key: str | None = None):
        super().__init__(settings)
        import openai

        key = api_key or load_api_key(PROVIDER_OPENAI)
        if not key:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY, add it to "
                "llm_project/.env, or run with backend='mock'."
            )
        self._openai = openai
        self.client = openai.OpenAI(api_key=key, max_retries=5)

    def complete(
        self,
        *,
        role: str,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Completion:
        s = self.settings
        model = s.model_for(role)
        max_tokens = s.max_tokens_for(role)
        effort = s.effort_for(role)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response_format: dict[str, Any] | None = None
        if schema is not None:
            # Strict mode requires additionalProperties:false and every
            # property listed in `required` — our schemas satisfy both.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": schema,
                    "strict": True,
                },
            }

        key = _cache_key(
            {
                "provider": PROVIDER_OPENAI,
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "reasoning_effort": effort,
                "response_format": response_format,
            }
        )
        hit = self._cache_get(key)
        if hit is not None:
            self._record(role, hit)
            return hit

        self._check_budget()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": effort,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self.client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        # A safety refusal comes back as a populated `refusal` field with
        # `content` set to None; treat it as an empty answer rather than
        # crashing a long run.
        text = choice.message.content or ""

        usage = getattr(resp, "usage", None)
        c = Completion(
            text=text,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        self._cache_put(key, c)
        self._record(role, c)
        return c


class AnthropicBackend(_BaseBackend):
    """Live calls against the Claude API."""

    def __init__(self, settings: Settings, api_key: str | None = None):
        super().__init__(settings)
        import anthropic

        key = api_key or load_api_key(PROVIDER_ANTHROPIC)
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY, add it to "
                "llm_project/.env, or run with backend='mock'."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=5)

    def complete(
        self,
        *,
        role: str,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Completion:
        s = self.settings
        model = s.model_for(role)
        max_tokens = s.max_tokens_for(role)
        output_config: dict[str, Any] = {"effort": s.effort_for(role)}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        key = _cache_key(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "output_config": output_config,
            }
        )
        hit = self._cache_get(key)
        if hit is not None:
            self._record(role, hit)
            return hit

        self._check_budget()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)

        if resp.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")

        c = Completion(
            text=text,
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self._cache_put(key, c)
        self._record(role, c)
        return c


class MockBackend(_BaseBackend):
    """Deterministic offline stand-in.

    This exists so the full pipeline can be exercised (and unit-tested)
    without API credits. It produces *structurally* valid outputs but the
    scores it yields are NOT meaningful research results — the mock task model
    does not actually attempt the instruction.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def complete(
        self,
        *,
        role: str,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Completion:
        seed = int(
            hashlib.sha256(
                f"{role}|{system}|{prompt}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        if schema is not None:
            text = self._mock_structured(role, prompt, seed)
        else:
            text = self._mock_text(role, prompt, seed)
        c = Completion(text=text, model=f"mock-{role}", input_tokens=0, output_tokens=0)
        self._record(role, c)
        return c

    # The mock judge scores by lexical overlap with the reference, so the
    # scoring pipeline behaves sensibly end-to-end.
    def _mock_structured(self, role: str, prompt: str, seed: int) -> str:
        if role == ROLE_JUDGE:
            ref = _between(prompt, "<reference_answer>", "</reference_answer>")
            cand = _between(prompt, "<candidate_answer>", "</candidate_answer>")
            from .scoring import token_f1

            score = int(round(100 * token_f1(cand, ref)))
            return json.dumps(
                {"score": score, "critique": "mock judge: lexical overlap only"}
            )
        # optimizer roles return a list of strings
        n = 1 + (seed % 2)
        items = [
            f"[mock-{role} variant {i + 1}] "
            "Answer the instruction directly, follow any format the user asks "
            "for, and keep the response focused."
            for i in range(n)
        ]
        return json.dumps({"items": items})

    def _mock_text(self, role: str, prompt: str, seed: int) -> str:
        if role == ROLE_TASK:
            body = _between(prompt, "Instruction:\n", "\n\nInput:") or prompt
            words = body.strip().split()[:40]
            return "(mock answer) " + " ".join(words)
        return f"(mock {role} output #{seed % 1000})"


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i == -1:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i:j] if j != -1 else text[i:]


def extract_json(text: str) -> Any:
    """Parse a JSON object from model output, tolerating stray prose/fences."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"could not parse JSON from model output: {text[:200]!r}")


def make_backend(settings: Settings) -> LLMBackend:
    """Build the backend named by `settings.backend`."""
    if settings.backend == PROVIDER_MOCK:
        return MockBackend(settings)
    if settings.backend == PROVIDER_ANTHROPIC:
        return AnthropicBackend(settings)
    if settings.backend == PROVIDER_OPENAI:
        return OpenAIBackend(settings)
    raise ValueError(
        f"Unknown backend {settings.backend!r}. "
        f"Expected one of: {PROVIDER_OPENAI}, {PROVIDER_ANTHROPIC}, {PROVIDER_MOCK}."
    )