"""Configuration for the Multi-Agent Business Analyst.

One `Settings` object describes an entire analysis run: which provider and
model each agent uses, how hard each agent is allowed to think, and the hard
ceilings that stop a runaway loop from burning the API budget.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

# `analyst/` is the package; the project root is one level up, so `data/`,
# `.cache/` and `runs/` all sit beside `app.py` where you would expect them.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"
RUNS_DIR = PROJECT_ROOT / "runs"

# A key can live in this project's own .env, one level up in the shared MTech
# Project folder, or in the sibling llm_project that already carries one — so
# a key you have already set up once is found without copying the secret into
# a second file. First match wins, in that order.
ENV_SEARCH_PATHS = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT.parent / ".env",
    PROJECT_ROOT.parent / "llm_project" / ".env",
]

# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_MOCK = "mock"  # offline stub — no API key, no network, no cost

PROVIDER_KEY_ENV = {
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
}

# USD per 1M tokens: (input, output). Reporting only.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (1.00, 6.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

PROVIDER_MODELS = {
    PROVIDER_OPENAI: ["gpt-5.6-luna"],
    PROVIDER_ANTHROPIC: ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
}

DEFAULT_MODEL = "claude-opus-5"


def default_model_for(backend: str) -> str:
    """The model a backend should use when the caller did not name one.

    Without this, switching backend but not model sends an Anthropic model id
    to OpenAI, and every agent fails with a 404.
    """
    models = PROVIDER_MODELS.get(backend)
    return models[0] if models else DEFAULT_MODEL

# Effort levels accepted by the Claude `output_config.effort` knob, cheapest
# first. Higher effort buys deeper reasoning at more tokens.
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]

# ---------------------------------------------------------------------------
# agent roles
# ---------------------------------------------------------------------------
ROLE_PLANNER = "planner"
ROLE_FINANCE = "finance"
ROLE_SALES = "sales"
ROLE_CUSTOMER = "customer"
ROLE_MARKET = "market"
ROLE_OPERATIONS = "operations"
ROLE_RISK = "risk"
ROLE_STRATEGY = "strategy"
# Not part of the analysis chain — it grades the finished report afterwards.
ROLE_EVALUATOR = "evaluator"

# The five domain specialists that fan out in parallel after planning.
SPECIALIST_ROLES = [
    ROLE_FINANCE,
    ROLE_SALES,
    ROLE_CUSTOMER,
    ROLE_MARKET,
    ROLE_OPERATIONS,
]
ALL_ROLES = [ROLE_PLANNER] + SPECIALIST_ROLES + [ROLE_RISK, ROLE_STRATEGY]

ROLE_LABELS = {
    ROLE_PLANNER: "Planner Agent",
    ROLE_FINANCE: "Finance Agent",
    ROLE_SALES: "Sales Agent",
    ROLE_CUSTOMER: "Customer Intelligence Agent",
    ROLE_MARKET: "Market Intelligence Agent",
    ROLE_OPERATIONS: "Operations Agent",
    ROLE_RISK: "Risk Agent",
    ROLE_STRATEGY: "Strategy Agent",
    ROLE_EVALUATOR: "Evaluator (LLM-as-judge)",
}

# Reasoning effort per role. The specialists summarise a pre-computed digest,
# which is shallow work; risk and strategy have to reconcile conflicting
# evidence, so they get the expensive setting.
DEFAULT_EFFORT = {
    ROLE_PLANNER: "medium",
    ROLE_FINANCE: "medium",
    ROLE_SALES: "medium",
    ROLE_CUSTOMER: "medium",
    ROLE_MARKET: "medium",
    ROLE_OPERATIONS: "medium",
    ROLE_RISK: "high",
    ROLE_STRATEGY: "high",
    ROLE_EVALUATOR: "high",
}

# These ceilings cover thinking *and* the visible answer. Claude Opus 5 and
# Sonnet 5 think by default, so a budget sized around the JSON alone would
# truncate mid-object; every value here leaves room for both.
DEFAULT_MAX_TOKENS = {
    ROLE_PLANNER: 8000,
    ROLE_FINANCE: 12000,
    ROLE_SALES: 12000,
    ROLE_CUSTOMER: 12000,
    ROLE_MARKET: 12000,
    ROLE_OPERATIONS: 12000,
    ROLE_RISK: 16000,
    ROLE_STRATEGY: 20000,
    ROLE_EVALUATOR: 12000,
}


@dataclass
class Settings:
    """Every knob for a single analysis run."""

    # --- provider / models --------------------------------------------
    backend: str = PROVIDER_OPENAI  # "openai" | "anthropic" | "mock"
    # Left empty, the model is resolved from the backend in __post_init__, so
    # `Settings(backend="openai")` does the obvious thing.
    model: str = ""
    # Per-role overrides; anything missing falls back to `model`.
    role_models: dict[str, str] = field(default_factory=dict)
    role_effort: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EFFORT))
    role_max_tokens: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_TOKENS))

    # --- orchestration -------------------------------------------------
    # Specialists the planner is allowed to dispatch. Trimming this list is
    # the cheapest way to shorten a run.
    enabled_specialists: list[str] = field(default_factory=lambda: list(SPECIALIST_ROLES))
    # Let the planner drop specialists it judges irrelevant to the question.
    planner_selects_agents: bool = True
    concurrency: int = 5  # specialists run in a thread pool

    # --- evidence sizing -----------------------------------------------
    # How much raw text each agent sees. Digests are pre-computed, so these
    # ceilings are about trimming long tails, not truncating the analysis.
    max_digest_chars: int = 9000
    max_market_chars: int = 14000
    max_quotes: int = 12  # verbatim review/ticket snippets shown to the customer agent

    # --- retrieval (RAG over the unstructured documents) -----------------
    # The CSVs are summarised by `analytics.py`; the markdown files are too
    # long to paste whole, so passages are retrieved against the question.
    use_retrieval: bool = True
    retrieval_top_k: int = 8
    # Weight of the dense (embedding) score in the hybrid blend. The rest of
    # the weight goes to BM25. 0.0 = pure lexical, which is the automatic
    # fallback whenever no embedding backend is reachable.
    hybrid_alpha: float = 0.5

    # --- evaluation ------------------------------------------------------
    # After the report is built, grade it with an independent LLM judge.
    run_evaluation: bool = True

    # --- execution ------------------------------------------------------
    use_cache: bool = True
    max_llm_calls: int = 60
    max_cost_usd: float = 5.0
    fail_fast: bool = False  # if False, one dead specialist does not sink the run

    def __post_init__(self) -> None:
        if not self.model:
            self.model = default_model_for(self.backend)

    def model_for(self, role: str) -> str:
        return self.role_models.get(role, self.model)

    def effort_for(self, role: str) -> str:
        return self.role_effort.get(role, DEFAULT_EFFORT.get(role, "medium"))

    def max_tokens_for(self, role: str) -> int:
        return self.role_max_tokens.get(role, DEFAULT_MAX_TOKENS.get(role, 3072))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        known = set(cls.__dataclass_fields__)
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
    """Resolve a provider's key: environment first, then any .env we can see.

    Returns None rather than raising so callers can fall back to the mock
    backend and still demonstrate the pipeline.
    """
    var = PROVIDER_KEY_ENV.get(provider)
    if var is None:
        return None

    key = os.environ.get(var)
    if key:
        return key.strip()

    for path in ENV_SEARCH_PATHS:
        env = _parse_env_file(path)
        if env.get(var):
            return env[var]
        key_file = env.get(f"{var}_FILE")
        if key_file:
            p = pathlib.Path(key_file).expanduser()
            if p.exists():
                return p.read_text(encoding="utf-8").strip()

    key_file = os.environ.get(f"{var}_FILE")
    if key_file:
        p = pathlib.Path(key_file).expanduser()
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (1.00, 6.00))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000