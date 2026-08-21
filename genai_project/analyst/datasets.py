"""Loading the company data room — standard library only.

`pandas` is deliberately not used: on this machine its compiled extension is
blocked by Windows Application Control, and the whole analytics layer fits
comfortably in `csv` + `statistics` anyway. Fewer moving parts, and every
number the agents see is computed by code you can read.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable

from .config import DATA_DIR

# Every source file the system knows about, grouped by the agent that owns it.
SOURCES: dict[str, list[str]] = {
    "profile": ["company_profile.md"],
    "finance": [
        "finance/income_statement.csv",
        "finance/balance_sheet.csv",
        "finance/cash_flow.csv",
        "finance/budget_vs_actual.csv",
    ],
    "sales": ["sales/orders.csv", "sales/crm_pipeline.csv"],
    "customer": [
        "customer/reviews.csv",
        "customer/support_tickets.csv",
        "customer/surveys.csv",
    ],
    "market": [
        "market/news_feed.md",
        "market/competitor_tracker.md",
        "market/industry_research.md",
    ],
    "operations": [
        "operations/inventory.csv",
        "operations/vendors.csv",
        "operations/shipments.csv",
        "operations/production.csv",
    ],
}


def _coerce(value: str) -> Any:
    """Turn CSV strings into numbers where that is unambiguous."""
    v = value.strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class DataRoom:
    """Handle on the data directory, with light caching per file."""

    root: pathlib.Path = DATA_DIR

    def __post_init__(self) -> None:
        self.root = pathlib.Path(self.root)
        self._cache: dict[str, Any] = {}

    # -- access ---------------------------------------------------------
    def path(self, rel: str) -> pathlib.Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def rows(self, rel: str) -> list[dict[str, Any]]:
        """Read a CSV into a list of dicts with numeric coercion."""
        if rel in self._cache:
            return self._cache[rel]
        p = self.path(rel)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing data file {rel}. Run `python make_sample_data.py` first."
            )
        with p.open(newline="", encoding="utf-8") as fh:
            out = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        self._cache[rel] = out
        return out

    def text(self, rel: str) -> str:
        if rel in self._cache:
            return self._cache[rel]
        p = self.path(rel)
        if not p.exists():
            raise FileNotFoundError(f"Missing data file {rel}.")
        out = p.read_text(encoding="utf-8")
        self._cache[rel] = out
        return out

    # -- introspection --------------------------------------------------
    def available(self, domain: str) -> list[str]:
        return [rel for rel in SOURCES.get(domain, []) if self.exists(rel)]

    def missing(self) -> list[str]:
        return [
            rel for group in SOURCES.values() for rel in group if not self.exists(rel)
        ]

    def inventory_report(self) -> str:
        """Human-readable listing of what the data room holds."""
        lines = []
        for domain, files in SOURCES.items():
            lines.append(f"{domain}:")
            for rel in files:
                p = self.path(rel)
                if not p.exists():
                    lines.append(f"  - {rel}  (MISSING)")
                elif rel.endswith(".csv"):
                    lines.append(f"  - {rel}  ({len(self.rows(rel))} rows)")
                else:
                    lines.append(f"  - {rel}  ({len(self.text(rel)):,} chars)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# small aggregation helpers used across the analytics module
# ---------------------------------------------------------------------------
def group_sum(
    rows: Iterable[dict], key: str, value: str, where=None
) -> dict[Any, float]:
    out: dict[Any, float] = {}
    for r in rows:
        if where and not where(r):
            continue
        out[r[key]] = out.get(r[key], 0.0) + (r.get(value) or 0.0)
    return out


def group_count(rows: Iterable[dict], key: str, where=None) -> dict[Any, int]:
    out: dict[Any, int] = {}
    for r in rows:
        if where and not where(r):
            continue
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def group_mean(
    rows: Iterable[dict], key: str, value: str, where=None
) -> dict[Any, float]:
    sums: dict[Any, float] = {}
    counts: dict[Any, int] = {}
    for r in rows:
        if where and not where(r):
            continue
        v = r.get(value)
        if v is None:
            continue
        sums[r[key]] = sums.get(r[key], 0.0) + v
        counts[r[key]] = counts.get(r[key], 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def month_of(date_str: str) -> str:
    """'2026-04-17' -> '2026-04'."""
    return str(date_str)[:7]