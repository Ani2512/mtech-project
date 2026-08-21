"""Multi-Agent Business Analyst.

A team of specialist agents that read a company's own data room and answer a
decision question with an executive-quality report.

    from analyst import Settings, run_analysis

    result = run_analysis("Should we expand into Southeast Asia?", Settings())
    print(result.report_markdown)
"""

from .config import (
    ALL_ROLES,
    DEFAULT_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    ROLE_LABELS,
    SPECIALIST_ROLES,
    Settings,
    load_api_key,
)
from .datasets import DataRoom
from .graph import AnalysisResult, run_analysis
from .llm import BudgetExceeded, make_backend
from .reporting import render_evaluation, render_report

__all__ = [
    "ALL_ROLES",
    "AnalysisResult",
    "BudgetExceeded",
    "DEFAULT_MODEL",
    "DataRoom",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_MOCK",
    "PROVIDER_OPENAI",
    "ROLE_LABELS",
    "SPECIALIST_ROLES",
    "Settings",
    "load_api_key",
    "make_backend",
    "render_evaluation",
    "render_report",
    "run_analysis",
]

__version__ = "1.0.0"