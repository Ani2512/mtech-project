"""Configuration for the Automatic Prompt Optimization system."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"
RUNS_DIR = PROJECT_ROOT / "runs"

# Providers the system can talk to. `mock` is the offline stub.
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_MOCK = "mock"

# Which environment variable holds each provider's key.
PROVIDER_KEY_ENV = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
}

# USD per 1M tokens: (input, output). Used only for cost reporting.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.6-luna": (1.00, 6.00),
    # Cheaper executors, for running the task role on a model that is weak
    # enough to leave the baseline prompt below ceiling. Verified against
    # developers.openai.com/api/docs/pricing on 2026-08-21.
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic (kept so existing runs still price correctly)
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Models offered in the UI, per provider.
PROVIDER_MODELS = {
    # The weaker executors are offered so the weak-executor experiment can be
    # reproduced from the UI, not only from the command line. Pick one as the
    # task model and leave judge/optimizer on the default.
    PROVIDER_OPENAI: [
        "gpt-5.6-luna",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ],
    PROVIDER_ANTHROPIC: ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
}

DEFAULT_MODEL = "gpt-5.6-luna"

# The gpt-4.x / 3.5 families reject `reasoning_effort` outright with a 400.
# That only matters once the task role runs on a cheap executor, which is the
# point of the weak-executor experiment.
NON_REASONING_PREFIXES = ("gpt-4", "gpt-3.5", "chatgpt", "babbage", "davinci")


def supports_reasoning_effort(model: str) -> bool:
    """Whether `model` accepts the `reasoning_effort` request parameter."""
    return not model.startswith(NON_REASONING_PREFIXES)

# The three roles the system plays. Each can use a different model.
ROLE_TASK = "task"  # answers Alpaca instructions using the prompt under test
ROLE_JUDGE = "judge"  # scores an answer against the reference answer
ROLE_OPTIMIZER = "optimizer"  # writes criticisms and rewrites prompts


@dataclass
class Settings:
    """All knobs for a single optimization run."""

    # --- models -------------------------------------------------------
    task_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    optimizer_model: str = DEFAULT_MODEL
    # Reasoning effort per role. Task/judge are shallow calls; the optimizer
    # is where reasoning quality actually matters. Valid on both providers:
    # low | medium | high | xhigh | max (OpenAI also accepts none/minimal).
    task_effort: str = "low"
    judge_effort: str = "low"
    optimizer_effort: str = "high"
    task_max_tokens: int = 1024
    judge_max_tokens: int = 1024
    optimizer_max_tokens: int = 4096

    # --- data ---------------------------------------------------------
    seed: int = 13
    n_train: int = 40
    n_dev: int = 40
    n_test: int = 60
    # None = keep everything; True = only examples that carry an `input`
    # field; False = only bare instructions.
    require_input: bool | None = None
    filter_keyword: str | None = None
    max_output_chars: int = 1200
    # Minimum characters of real source text (URLs excluded) an example's
    # input must carry. Guards against Alpaca's unanswerable "summarise this
    # link" items, whose references were hallucinated. 0 disables the check.
    min_input_chars: int = 0
    # Keep only examples whose instruction states a hard, countable limit
    # ("in less than 50 words", "in one sentence"). These are the examples on
    # which `constraint_weight` has anything to measure.
    require_constraint: bool = False

    # --- optimization -------------------------------------------------
    iterations: int = 3
    minibatch_size: int = 8
    beam_width: int = 2
    n_gradients: int = 2  # criticisms drawn per prompt per iteration
    n_edits: int = 1  # rewrites produced per criticism
    n_paraphrases: int = 1  # extra rewordings per edited prompt
    errors_per_gradient: int = 3  # worst-scoring examples shown to the critic
    ucb_pulls: int = 24  # dev evaluations used to rank the candidate pool
    ucb_c: float = 1.0  # exploration constant
    patience: int = 2  # iterations without improvement before stopping

    # --- scoring ------------------------------------------------------
    # Weights need not sum to 1; the total is renormalised by whatever is
    # actually in play, so switching a component off does not rescale the rest.
    use_judge: bool = True
    judge_weight: float = 0.7  # remainder goes to the lexical metrics
    rouge_weight: float = 0.15
    f1_weight: float = 0.15
    # Deterministic compliance with the instruction's stated limits. 0 keeps
    # the Run 1-3 metric exactly as it was; raising it trades the judge's
    # saturated opinion for a signal that can actually move. See constraints.py.
    constraint_weight: float = 0.0
    # "lenient" is the Run 1-3 rubric, which scored the baseline prompt 0.975
    # and so could not resolve any real effect. "strict" deducts from a stated
    # ceiling for enumerated defects and reserves the top band for flawless
    # answers.
    judge_rubric: str = "lenient"

    # --- execution ----------------------------------------------------
    concurrency: int = 4
    use_cache: bool = True
    backend: str = PROVIDER_OPENAI  # "openai" | "anthropic" | "mock"
    max_llm_calls: int = 4000  # hard stop, protects against runaway loops
    max_cost_usd: float = 25.0  # hard stop on estimated spend

    initial_prompt: str = (
        "You are a helpful assistant. Answer the user's instruction."
    )

    def model_for(self, role: str) -> str:
        return {
            ROLE_TASK: self.task_model,
            ROLE_JUDGE: self.judge_model,
            ROLE_OPTIMIZER: self.optimizer_model,
        }[role]

    def effort_for(self, role: str) -> str:
        return {
            ROLE_TASK: self.task_effort,
            ROLE_JUDGE: self.judge_effort,
            ROLE_OPTIMIZER: self.optimizer_effort,
        }[role]

    def max_tokens_for(self, role: str) -> int:
        return {
            ROLE_TASK: self.task_max_tokens,
            ROLE_JUDGE: self.judge_max_tokens,
            ROLE_OPTIMIZER: self.optimizer_max_tokens,
        }[role]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_api_key(provider: str = PROVIDER_OPENAI) -> str | None:
    """Resolve the API key for a provider.

    Order: the provider's env var (OPENAI_API_KEY / ANTHROPIC_API_KEY), then
    the same name in the project's .env file, then a file path given by
    <VAR>_FILE. Returns None if the key cannot be found, so callers can fall
    back to the offline backend.
    """
    var = PROVIDER_KEY_ENV.get(provider)
    if var is None:
        return None

    key = os.environ.get(var)
    if key:
        return key.strip()

    env = _parse_env_file(PROJECT_ROOT / ".env")
    if env.get(var):
        return env[var]

    key_file = os.environ.get(f"{var}_FILE") or env.get(f"{var}_FILE")
    if key_file:
        p = pathlib.Path(key_file).expanduser()
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (5.00, 25.00))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000