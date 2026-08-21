"""Automatic Prompt Optimization on the Alpaca dataset.

Refines a system prompt from execution feedback: run it, grade the answers,
turn the failures into natural-language criticism ("textual gradients"),
rewrite the prompt against that criticism, and keep the winners via beam
search with a UCB bandit.
"""

from .config import Settings, load_api_key
from .data import Example, Split, build_split, split_from_settings
from .backends import (
    make_backend,
    OpenAIBackend,
    AnthropicBackend,
    MockBackend,
    BudgetExceeded,
)
from .optimizer import (
    OptimizationResult,
    compare_on_test,
    evaluate_prompt,
    optimize,
)
from .reporting import build_report, estimate_budget, new_run_dir, save_run

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "load_api_key",
    "Example",
    "Split",
    "build_split",
    "split_from_settings",
    "make_backend",
    "OpenAIBackend",
    "AnthropicBackend",
    "MockBackend",
    "BudgetExceeded",
    "OptimizationResult",
    "optimize",
    "evaluate_prompt",
    "compare_on_test",
    "estimate_budget",
    "new_run_dir",
    "save_run",
    "build_report",
]