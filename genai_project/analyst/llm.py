"""The single choke point for every model call in the system.

Putting all traffic through `LLMBackend.complete` buys four things at once:
a disk cache that makes repeat runs free, a hard budget stop, per-agent usage
accounting for the report footer, and a deterministic offline backend so the
whole pipeline can be demonstrated without an API key.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import (
    CACHE_DIR,
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    Settings,
    estimate_cost,
    load_api_key,
)


class BudgetExceeded(RuntimeError):
    """Raised when a run hits its call or cost ceiling."""


class AgentCallError(RuntimeError):
    """A single agent's call failed (bad JSON, refusal, transport error)."""


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
        # Slots claimed by in-flight requests. Tracked separately from
        # usage.calls, which only rises once a call has finished.
        self._reserved = 0

    # -- budget ---------------------------------------------------------
    def _reserve_call(self) -> None:
        """Claim one call against the ceiling, atomically.

        Reading the count and then deciding is not enough: the five
        specialists start together, all observe the same under-budget total
        before any of them has finished, and all proceed — so a ceiling of two
        lets five calls through. Claiming the slot inside the same lock that
        reads it is what turns the ceiling into a real limit.

        The cost ceiling is necessarily softer: a request's cost is unknown
        until it returns, so concurrent calls can overshoot by at most one
        round of in-flight requests.
        """
        s = self.settings
        with self._lock:
            if self._reserved >= s.max_llm_calls:
                raise BudgetExceeded(
                    f"Reached max_llm_calls={s.max_llm_calls}. Raise the ceiling or "
                    "disable some specialists."
                )
            if self.usage.cost_usd >= s.max_cost_usd:
                raise BudgetExceeded(
                    f"Reached max_cost_usd=${s.max_cost_usd:.2f} "
                    f"(spent ${self.usage.cost_usd:.2f})."
                )
            self._reserved += 1

    def _record(self, role: str, c: Completion) -> None:
        with self._lock:
            self.usage.calls += 1
            self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
            if c.cached:
                self.usage.cache_hits += 1
                return
            self.usage.input_tokens += c.input_tokens
            self.usage.output_tokens += c.output_tokens
            self.usage.cost_usd += estimate_cost(c.model, c.input_tokens, c.output_tokens)

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
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
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

    Written for the GPT-5.6 parameter surface: `reasoning_effort` rather than a
    thinking budget, `max_completion_tokens` rather than `max_tokens`, and
    strict JSON-schema structured outputs.
    """

    def __init__(self, settings: Settings, api_key: str | None = None):
        super().__init__(settings)
        import openai

        key = api_key or load_api_key(PROVIDER_OPENAI)
        if not key:
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY, add it to "
                "llm_project/.env, or run with --backend mock."
            )
        self.client = openai.OpenAI(api_key=key, max_retries=4)

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
            # Strict mode requires additionalProperties:false and every key
            # listed in `required` — schemas.py satisfies both.
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
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

        self._reserve_call()

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
        # A safety refusal arrives as a populated `refusal` with content=None.
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
                "llm_project/.env, or run with --backend mock."
            )
        self.client = anthropic.Anthropic(api_key=key, max_retries=4)

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
                "provider": PROVIDER_ANTHROPIC,
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

        self._reserve_call()

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)
        text = "" if resp.stop_reason == "refusal" else "".join(
            b.text for b in resp.content if b.type == "text"
        )
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

    It walks the requested JSON schema and fills every field with plausible
    filler seeded from the prompt, so the orchestration, validation and
    reporting paths can all be exercised without credits. The *wiring* it
    exercises is real; the *judgements* it returns are not analysis.
    """

    def complete(
        self,
        *,
        role: str,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Completion:
        # Claim a slot like the live backends do. The mock costs nothing, but
        # skipping this would make budget ceilings unenforceable offline — and
        # untestable without spending real money on the very path most likely
        # to be wrong.
        self._reserve_call()

        seed = int(
            hashlib.sha256(f"{role}|{system}|{prompt}".encode("utf-8")).hexdigest()[:12],
            16,
        )
        rng = random.Random(seed)
        if schema is None:
            text = f"(mock {role} output #{seed % 1000})"
        else:
            text = json.dumps(_mock_from_schema(schema, rng, role), ensure_ascii=False)
        c = Completion(text=text, model=f"mock-{role}", input_tokens=0, output_tokens=0)
        self._record(role, c)
        return c


_MOCK_SENTENCES = [
    "Mock finding: the trailing twelve months show a clear directional move.",
    "Mock finding: the most recent quarter diverges from the two before it.",
    "Mock finding: one segment carries a disproportionate share of the change.",
    "Mock finding: the underlying driver is cost, not volume.",
    "Mock finding: evidence is consistent across two independent sources.",
]


# Strict structured outputs reject `minimum`/`maxItems`, so the schemas carry
# their ranges in prose. The mock recovers a sensible range from the field name.
_ONE_TO_FIVE = {"likelihood", "impact", "score", "severity", "overall_risk", "priority"}
_PERCENT = {"confidence", "probability", "probability_pct"}


def _mock_from_schema(
    schema: dict, rng: random.Random, role: str, name: str = ""
) -> Any:
    """Build a value that satisfies `schema`. Handles the subset we emit."""
    if "enum" in schema:
        return schema["enum"][rng.randrange(len(schema["enum"]))]

    kind = schema.get("type", "string")
    if kind == "object":
        props = schema.get("properties", {})
        return {k: _mock_from_schema(v, rng, role, k) for k, v in props.items()}
    if kind == "array":
        return [
            _mock_from_schema(schema["items"], rng, role, name)
            for _ in range(2 + rng.randrange(2))
        ]
    if kind in ("integer", "number"):
        if name in _ONE_TO_FIVE:
            lo, hi = 1, 5
        elif name in _PERCENT:
            lo, hi = 55, 92
        else:
            lo, hi = 0, 100
        return rng.randint(lo, hi) if kind == "integer" else round(rng.uniform(lo, hi), 2)
    if kind == "boolean":
        return rng.random() < 0.5
    return f"[mock {role}] {rng.choice(_MOCK_SENTENCES)}"


def extract_json(text: str) -> Any:
    """Parse a JSON object from model output, tolerating fences and prose."""
    text = (text or "").strip()
    if not text:
        raise AgentCallError("empty model output")
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
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AgentCallError(f"could not parse JSON: {exc}") from exc
    raise AgentCallError(f"could not parse JSON from model output: {text[:200]!r}")


def make_backend(settings: Settings) -> LLMBackend:
    """Build the backend named by `settings.backend`."""
    if settings.backend == PROVIDER_MOCK:
        return MockBackend(settings)
    if settings.backend == PROVIDER_ANTHROPIC:
        return AnthropicBackend(settings)
    if settings.backend == PROVIDER_OPENAI:
        return OpenAIBackend(settings)
    raise ValueError(
        f"Unknown backend {settings.backend!r}. Expected one of: "
        f"{PROVIDER_OPENAI}, {PROVIDER_ANTHROPIC}, {PROVIDER_MOCK}."
    )