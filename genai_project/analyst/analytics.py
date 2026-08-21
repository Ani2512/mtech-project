"""Deterministic analytics — the layer that keeps the agents honest.

Every number an agent quotes is computed here, in Python, from the CSVs. The
model's job is interpretation: *why* the operating margin moved, *what it
means* for the decision. It is never asked to do the arithmetic, because an
LLM asked to average 4,000 rows in its head will invent a plausible number.

Each `*_digest()` returns a compact markdown brief that goes straight into the
agent's prompt.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .datasets import DataRoom, group_count, group_mean, group_sum, month_of

# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    a = abs(x)
    if a >= 1_000_000_000:
        return f"${x / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}%"


def signed_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:+.{digits}f}%"


def pts(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:+.{digits}f} pts"


def growth(new: float, old: float) -> float | None:
    """Percentage change, or None when the base is zero."""
    return None if not old else 100.0 * (new - old) / abs(old)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, rule] + body)


def top_n(d: dict[Any, float], n: int = 5, reverse: bool = True) -> list[tuple[Any, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)[:n]


def linear_forecast(series: list[float], periods: int = 3) -> list[float]:
    """Ordinary least-squares trend extrapolation. Intentionally simple —
    it is a sanity check on direction, not a forecasting model."""
    n = len(series)
    if n < 3:
        return [series[-1]] * periods if series else []
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(series) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, series)) / denom
    intercept = my - slope * mx
    return [intercept + slope * (n + k) for k in range(periods)]


def _split_periods(months: list[str], size: int = 12) -> tuple[list[str], list[str]]:
    """Trailing window and the window before it."""
    recent = months[-size:]
    prior = months[-2 * size : -size]
    return recent, prior


# ---------------------------------------------------------------------------
# 1. Finance
# ---------------------------------------------------------------------------
def finance_digest(room: DataRoom) -> str:
    pnl = sorted(room.rows("finance/income_statement.csv"), key=lambda r: r["month"])
    bs = sorted(room.rows("finance/balance_sheet.csv"), key=lambda r: r["quarter"])
    cf = sorted(room.rows("finance/cash_flow.csv"), key=lambda r: r["quarter"])
    budget = room.rows("finance/budget_vs_actual.csv")

    months = [r["month"] for r in pnl]
    recent, prior = _split_periods(months)
    r_rows = [r for r in pnl if r["month"] in set(recent)]
    p_rows = [r for r in pnl if r["month"] in set(prior)]

    def agg(rows: list[dict], field: str) -> float:
        return sum(r[field] for r in rows)

    rev_r, rev_p = agg(r_rows, "revenue"), agg(p_rows, "revenue")
    gp_r, gp_p = agg(r_rows, "gross_profit"), agg(p_rows, "gross_profit")
    op_r, op_p = agg(r_rows, "operating_income"), agg(p_rows, "operating_income")
    ni_r, ni_p = agg(r_rows, "net_income"), agg(p_rows, "net_income")
    log_r, log_p = agg(r_rows, "logistics_cost"), agg(p_rows, "logistics_cost")

    gm_r, gm_p = 100 * gp_r / rev_r, 100 * gp_p / rev_p
    om_r, om_p = 100 * op_r / rev_r, 100 * op_p / rev_p
    lg_r, lg_p = 100 * log_r / rev_r, 100 * log_p / rev_p

    # Half-year view exposes the recent inflection the annual view smooths out.
    h1 = pnl[-6:]
    h0 = pnl[-12:-6]
    om_h1 = 100 * agg(h1, "operating_income") / agg(h1, "revenue")
    om_h0 = 100 * agg(h0, "operating_income") / agg(h0, "revenue")
    lg_h1 = 100 * agg(h1, "logistics_cost") / agg(h1, "revenue")
    lg_h0 = 100 * agg(h0, "logistics_cost") / agg(h0, "revenue")

    opex_r = agg(r_rows, "opex_rnd") + agg(r_rows, "opex_sales_marketing") + agg(r_rows, "opex_ga")
    opex_p = agg(p_rows, "opex_rnd") + agg(p_rows, "opex_sales_marketing") + agg(p_rows, "opex_ga")

    lines = [
        f"## Finance evidence ({months[0]} to {months[-1]}, {len(months)} months)",
        "",
        f"**Trailing 12 months ({recent[0]}..{recent[-1]}) vs prior 12 ({prior[0]}..{prior[-1]})**",
        "",
        table(
            ["Metric", "TTM", "Prior TTM", "Change"],
            [
                ["Revenue", money(rev_r), money(rev_p), signed_pct(growth(rev_r, rev_p))],
                ["Gross profit", money(gp_r), money(gp_p), signed_pct(growth(gp_r, gp_p))],
                ["Gross margin", pct(gm_r), pct(gm_p), pts(gm_r - gm_p)],
                ["Operating income", money(op_r), money(op_p), signed_pct(growth(op_r, op_p))],
                ["Operating margin", pct(om_r), pct(om_p), pts(om_r - om_p)],
                ["Net income", money(ni_r), money(ni_p), signed_pct(growth(ni_r, ni_p))],
                ["Logistics cost", money(log_r), money(log_p), signed_pct(growth(log_r, log_p))],
                ["Logistics % of revenue", pct(lg_r), pct(lg_p), pts(lg_r - lg_p)],
                ["Opex (R&D+S&M+G&A)", money(opex_r), money(opex_p), signed_pct(growth(opex_r, opex_p))],
                ["Opex % of revenue", pct(100 * opex_r / rev_r), pct(100 * opex_p / rev_p),
                 pts(100 * opex_r / rev_r - 100 * opex_p / rev_p)],
            ],
        ),
        "",
        "**Recent inflection — last 6 months vs the 6 before**",
        "",
        table(
            ["Metric", f"{h1[0]['month']}..{h1[-1]['month']}", f"{h0[0]['month']}..{h0[-1]['month']}", "Change"],
            [
                ["Revenue", money(agg(h1, "revenue")), money(agg(h0, "revenue")),
                 signed_pct(growth(agg(h1, "revenue"), agg(h0, "revenue")))],
                ["Operating margin", pct(om_h1), pct(om_h0), pts(om_h1 - om_h0)],
                ["Logistics % of revenue", pct(lg_h1), pct(lg_h0), pts(lg_h1 - lg_h0)],
            ],
        ),
        "",
        "**Monthly P&L, last 12 months**",
        "",
        table(
            ["Month", "Revenue", "Gross margin", "Logistics %", "Op income", "Op margin", "Net income"],
            [
                [
                    r["month"], money(r["revenue"]),
                    pct(100 * r["gross_profit"] / r["revenue"]),
                    pct(100 * r["logistics_cost"] / r["revenue"]),
                    money(r["operating_income"]),
                    pct(100 * r["operating_income"] / r["revenue"]),
                    money(r["net_income"]),
                ]
                for r in pnl[-12:]
            ],
        ),
    ]

    # Balance sheet and cash flow
    last_bs, first_bs = bs[-1], bs[-5] if len(bs) >= 5 else bs[0]
    q_rev = {}
    for r in pnl:
        y, m = int(r["month"][:4]), int(r["month"][5:])
        q = f"{y}-Q{(m - 1) // 3 + 1}"
        q_rev[q] = q_rev.get(q, 0.0) + r["revenue"]
    dso = 90 * last_bs["accounts_receivable"] / q_rev[last_bs["quarter"]]
    dso_prior = 90 * first_bs["accounts_receivable"] / q_rev[first_bs["quarter"]]
    dio = 90 * last_bs["inventory"] / (q_rev[last_bs["quarter"]] * 0.6)
    dio_prior = 90 * first_bs["inventory"] / (q_rev[first_bs["quarter"]] * 0.6)

    fcf_recent = sum(r["free_cash_flow"] for r in cf[-4:])
    fcf_prior = sum(r["free_cash_flow"] for r in cf[-8:-4]) if len(cf) >= 8 else None
    burn = [r["free_cash_flow"] for r in cf[-4:]]
    avg_q_fcf = sum(burn) / len(burn)
    runway = (
        f"{last_bs['cash'] / abs(avg_q_fcf):.1f} quarters at the current burn"
        if avg_q_fcf < 0
        else "not applicable — free cash flow is positive"
    )

    lines += [
        "",
        f"**Balance sheet ({last_bs['quarter']}) and cash flow**",
        "",
        table(
            ["Metric", "Latest", f"4 quarters earlier ({first_bs['quarter']})", "Change"],
            [
                ["Cash", money(last_bs["cash"]), money(first_bs["cash"]),
                 signed_pct(growth(last_bs["cash"], first_bs["cash"]))],
                ["Accounts receivable", money(last_bs["accounts_receivable"]),
                 money(first_bs["accounts_receivable"]),
                 signed_pct(growth(last_bs["accounts_receivable"], first_bs["accounts_receivable"]))],
                ["Inventory", money(last_bs["inventory"]), money(first_bs["inventory"]),
                 signed_pct(growth(last_bs["inventory"], first_bs["inventory"]))],
                ["DSO (days)", f"{dso:.0f}", f"{dso_prior:.0f}", f"{dso - dso_prior:+.0f}"],
                ["Inventory days", f"{dio:.0f}", f"{dio_prior:.0f}", f"{dio - dio_prior:+.0f}"],
                ["Total debt", money(last_bs["short_term_debt"] + last_bs["long_term_debt"]),
                 money(first_bs["short_term_debt"] + first_bs["long_term_debt"]), ""],
                ["Equity", money(last_bs["shareholders_equity"]),
                 money(first_bs["shareholders_equity"]), ""],
            ],
        ),
        "",
        f"- Free cash flow, last 4 quarters: {money(fcf_recent)}"
        + (f" vs {money(fcf_prior)} in the 4 before" if fcf_prior is not None else ""),
        f"- Latest quarter operating cash flow: {money(cf[-1]['operating_cash_flow'])}, "
        f"capex {money(cf[-1]['capex'])}, free cash flow {money(cf[-1]['free_cash_flow'])}",
        f"- Cash runway: {runway}",
    ]

    # Budget variance
    var_by_dept: dict[str, list[float]] = {}
    for r in budget:
        var_by_dept.setdefault(r["department"], [0.0, 0.0])
        var_by_dept[r["department"]][0] += r["budget_usd"]
        var_by_dept[r["department"]][1] += r["actual_usd"]
    lines += [
        "",
        "**Budget vs actual, trailing 12 months**",
        "",
        table(
            ["Department", "Budget", "Actual", "Variance", "Variance %"],
            [
                [dept, money(b), money(a), money(a - b), signed_pct(growth(a, b))]
                for dept, (b, a) in sorted(
                    var_by_dept.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True
                )
            ],
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Sales
# ---------------------------------------------------------------------------
def sales_digest(room: DataRoom) -> str:
    orders = room.rows("sales/orders.csv")
    pipeline = room.rows("sales/crm_pipeline.csv")

    months = sorted({month_of(r["order_date"]) for r in orders})
    recent, prior = _split_periods(months)
    r_set, p_set = set(recent), set(prior)

    def in_recent(r: dict) -> bool:
        return month_of(r["order_date"]) in r_set

    def in_prior(r: dict) -> bool:
        return month_of(r["order_date"]) in p_set

    total_r = sum(r["net_revenue_usd"] for r in orders if in_recent(r))
    total_p = sum(r["net_revenue_usd"] for r in orders if in_prior(r))

    def compare(key: str) -> list[list[Any]]:
        rec = group_sum(orders, key, "net_revenue_usd", in_recent)
        pri = group_sum(orders, key, "net_revenue_usd", in_prior)
        rows = []
        for name in sorted(rec, key=lambda k: rec[k], reverse=True):
            g = growth(rec[name], pri.get(name, 0.0))
            rows.append(
                [
                    name, money(rec[name]), money(pri.get(name, 0.0)),
                    signed_pct(g), pct(100 * rec[name] / total_r),
                ]
            )
        return rows

    units_r = group_sum(orders, "product", "units", in_recent)
    units_p = group_sum(orders, "product", "units", in_prior)
    disc_r = group_mean(orders, "region", "discount_pct", in_recent)
    disc_p = group_mean(orders, "region", "discount_pct", in_prior)

    monthly = {m: 0.0 for m in months}
    for r in orders:
        monthly[month_of(r["order_date"])] += r["net_revenue_usd"]
    series = [monthly[m] for m in months]
    fc = linear_forecast(series[-12:], 3)

    lines = [
        f"## Sales evidence ({months[0]} to {months[-1]})",
        "",
        f"Booked revenue TTM {money(total_r)} vs prior TTM {money(total_p)} "
        f"({signed_pct(growth(total_r, total_p))}). {len(orders):,} orders in the file.",
        "",
        "**By region — TTM vs prior TTM**",
        "",
        table(["Region", "TTM", "Prior TTM", "Growth", "Share of TTM"], compare("region")),
        "",
        "**By product — TTM vs prior TTM**",
        "",
        table(["Product", "TTM", "Prior TTM", "Growth", "Share of TTM"], compare("product")),
        "",
        "**Units by product**",
        "",
        table(
            ["Product", "TTM units", "Prior TTM units", "Growth"],
            [
                [p, f"{int(units_r[p]):,}", f"{int(units_p.get(p, 0)):,}",
                 signed_pct(growth(units_r[p], units_p.get(p, 0)))]
                for p in sorted(units_r, key=lambda k: units_r[k], reverse=True)
            ],
        ),
        "",
        "**By channel — TTM vs prior TTM**",
        "",
        table(["Channel", "TTM", "Prior TTM", "Growth", "Share of TTM"], compare("channel")),
        "",
        "**By customer segment — TTM vs prior TTM**",
        "",
        table(["Segment", "TTM", "Prior TTM", "Growth", "Share of TTM"], compare("customer_segment")),
        "",
        "**Average discount by region (percentage points off list)**",
        "",
        table(
            ["Region", "TTM", "Prior TTM", "Change"],
            [
                [reg, pct(disc_r[reg]), pct(disc_p.get(reg, 0.0)),
                 pts(disc_r[reg] - disc_p.get(reg, 0.0))]
                for reg in sorted(disc_r, key=lambda k: disc_r[k], reverse=True)
            ],
        ),
        "",
        "**Monthly booked revenue, last 12 months, with a 3-month linear trend projection**",
        "",
        table(
            ["Month", "Revenue"],
            [[m, money(monthly[m])] for m in months[-12:]]
            + [[f"{i + 1}-month projection", money(v)] for i, v in enumerate(fc)],
        ),
        "",
        f"- Top countries TTM: "
        + ", ".join(
            f"{c} {money(v)}"
            for c, v in top_n(group_sum(orders, "country", "net_revenue_usd", in_recent), 5)
        ),
    ]

    # Pipeline
    open_stages = {"Qualification", "Proposal", "Negotiation", "Contracting"}
    open_opps = [o for o in pipeline if o["stage"] in open_stages]
    weighted_by_region: dict[str, float] = {}
    weighted_by_product: dict[str, float] = {}
    for o in open_opps:
        w = o["value_usd"] * o["probability_pct"] / 100
        weighted_by_region[o["region"]] = weighted_by_region.get(o["region"], 0.0) + w
        weighted_by_product[o["product"]] = weighted_by_product.get(o["product"], 0.0) + w

    won = [o for o in pipeline if o["stage"] == "Closed Won"]
    lost = [o for o in pipeline if o["stage"] == "Closed Lost"]
    win_rate = 100 * len(won) / max(1, len(won) + len(lost))

    lines += [
        "",
        f"**CRM pipeline** — {len(open_opps)} open opportunities, "
        f"{money(sum(o['value_usd'] for o in open_opps))} gross, "
        f"{money(sum(weighted_by_region.values()))} probability-weighted. "
        f"Historic win rate in this file: {pct(win_rate)}.",
        "",
        table(
            ["Region", "Weighted pipeline"],
            [[k, money(v)] for k, v in top_n(weighted_by_region, 10)],
        ),
        "",
        table(
            ["Product", "Weighted pipeline"],
            [[k, money(v)] for k, v in top_n(weighted_by_product, 10)],
        ),
        "",
        "Open pipeline by stage: "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(group_count(open_opps, "stage").items())
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Customer intelligence
# ---------------------------------------------------------------------------
# A small lexicon does the topic clustering. It is transparent and auditable,
# which matters more here than the marginal accuracy of an embedding model —
# and every hit can be traced back to the review that produced it.
TOPIC_LEXICON: dict[str, list[str]] = {
    "battery life": ["battery", "charge", "charging", "drain", "hours", "juice", "power"],
    "pricing": ["price", "pricing", "expensive", "cost", "cheap", "value", "hike", "money"],
    "delivery speed": ["delivery", "shipping", "shipped", "arrived", "dispatch", "packaging"],
    "sound quality": ["sound", "bass", "audio", "clarity", "call", "noise", "stage"],
    "connectivity": ["bluetooth", "pairing", "pair", "connect", "drops", "multipoint", "link"],
    "comfort and fit": ["comfort", "fit", "ear", "fall out", "fatigue", "tips"],
    "support and warranty": ["support", "warranty", "reply", "service", "rma", "response"],
    "app and firmware": ["app", "firmware", "update", "software"],
    "build quality": ["build", "solid", "scratch", "durable", "broke", "cracked"],
}


def _topic_hits(texts: Iterable[str]) -> dict[str, int]:
    counts = {t: 0 for t in TOPIC_LEXICON}
    for text in texts:
        low = str(text).lower()
        for topic, words in TOPIC_LEXICON.items():
            if any(re.search(rf"\b{re.escape(w)}", low) for w in words):
                counts[topic] += 1
    return {k: v for k, v in counts.items() if v}


def customer_digest(room: DataRoom, max_quotes: int = 12) -> str:
    reviews = room.rows("customer/reviews.csv")
    tickets = room.rows("customer/support_tickets.csv")
    surveys = room.rows("customer/surveys.csv")

    months = sorted({month_of(r["date"]) for r in reviews})
    recent6 = set(months[-6:])
    prior6 = set(months[-12:-6])

    def rev_recent(r: dict) -> bool:
        return month_of(r["date"]) in recent6

    def rev_prior(r: dict) -> bool:
        return month_of(r["date"]) in prior6

    r_recent = [r for r in reviews if rev_recent(r)]
    r_prior = [r for r in reviews if rev_prior(r)]
    avg = lambda rows: sum(r["rating"] for r in rows) / max(1, len(rows))  # noqa: E731

    by_prod_r = group_mean(reviews, "product", "rating", rev_recent)
    by_prod_p = group_mean(reviews, "product", "rating", rev_prior)
    by_region_r = group_mean(reviews, "region", "rating", rev_recent)

    negative = [r for r in reviews if r["rating"] <= 2]
    positive = [r for r in reviews if r["rating"] >= 4]
    neg_recent = [r for r in r_recent if r["rating"] <= 2]
    pos_recent = [r for r in r_recent if r["rating"] >= 4]

    neg_topics = _topic_hits(r["review_text"] for r in negative)
    pos_topics = _topic_hits(r["review_text"] for r in positive)
    neg_topics_recent = _topic_hits(r["review_text"] for r in neg_recent)

    # NPS = %promoters (9-10) - %detractors (0-6)
    def nps(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        promoters = sum(1 for r in rows if r["nps_score"] >= 9)
        detractors = sum(1 for r in rows if r["nps_score"] <= 6)
        return 100.0 * (promoters - detractors) / len(rows)

    s_months = sorted({month_of(r["date"]) for r in surveys})
    s_recent = [r for r in surveys if month_of(r["date"]) in set(s_months[-6:])]
    s_prior = [r for r in surveys if month_of(r["date"]) in set(s_months[-12:-6])]
    nps_by_region = {
        reg: nps([r for r in s_recent if r["region"] == reg])
        for reg in sorted({r["region"] for r in s_recent})
    }

    t_months = sorted({month_of(r["date"]) for r in tickets})
    t_recent = [r for r in tickets if month_of(r["date"]) in set(t_months[-6:])]
    t_prior = [r for r in tickets if month_of(r["date"]) in set(t_months[-12:-6])]
    cat_r = group_count(t_recent, "category")
    cat_p = group_count(t_prior, "category")
    res_r = sum(r["resolution_hours"] for r in t_recent) / max(1, len(t_recent))
    res_p = sum(r["resolution_hours"] for r in t_prior) / max(1, len(t_prior))

    lines = [
        f"## Customer evidence ({months[0]} to {months[-1]})",
        "",
        f"{len(reviews):,} reviews, {len(tickets):,} support tickets, {len(surveys):,} survey responses.",
        "",
        "**Ratings**",
        "",
        table(
            ["Window", "Reviews", "Avg rating", "% 1-2 star", "% 4-5 star"],
            [
                ["Last 6 months", len(r_recent), f"{avg(r_recent):.2f}",
                 pct(100 * len(neg_recent) / max(1, len(r_recent))),
                 pct(100 * len(pos_recent) / max(1, len(r_recent)))],
                ["Previous 6 months", len(r_prior), f"{avg(r_prior):.2f}",
                 pct(100 * sum(1 for r in r_prior if r["rating"] <= 2) / max(1, len(r_prior))),
                 pct(100 * sum(1 for r in r_prior if r["rating"] >= 4) / max(1, len(r_prior)))],
            ],
        ),
        "",
        "**Average rating by product (last 6 months vs previous 6)**",
        "",
        table(
            ["Product", "Last 6m", "Prev 6m", "Change"],
            [
                [p, f"{by_prod_r[p]:.2f}", f"{by_prod_p.get(p, 0):.2f}",
                 f"{by_prod_r[p] - by_prod_p.get(p, 0):+.2f}"]
                for p in sorted(by_prod_r, key=lambda k: by_prod_r[k])
            ],
        ),
        "",
        "**Average rating by region (last 6 months)**: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(by_region_r.items(), key=lambda kv: kv[1])),
        "",
        "**Complaint themes** (keyword clustering over 1-2 star reviews; a review "
        "can hit more than one theme)",
        "",
        table(
            ["Theme", "All-time mentions", "Last 6 months", "% of negative reviews (all-time)"],
            [
                [t, c, neg_topics_recent.get(t, 0), pct(100 * c / max(1, len(negative)))]
                for t, c in top_n(neg_topics, 8)
            ],
        ),
        "",
        "**Praise themes** (keyword clustering over 4-5 star reviews)",
        "",
        table(
            ["Theme", "Mentions", "% of positive reviews"],
            [[t, c, pct(100 * c / max(1, len(positive)))] for t, c in top_n(pos_topics, 8)],
        ),
        "",
        "**Survey / NPS**",
        "",
        f"- NPS last 6 months: {nps(s_recent):+.0f} (previous 6 months: {nps(s_prior):+.0f})",
        f"- Average CSAT last 6 months: "
        f"{sum(r['csat_score'] for r in s_recent) / max(1, len(s_recent)):.2f} / 5",
        "- NPS by region (last 6 months): "
        + ", ".join(f"{k} {v:+.0f}" for k, v in sorted(nps_by_region.items(), key=lambda kv: -kv[1])),
        "",
        "**Support tickets**",
        "",
        table(
            ["Category", "Last 6m", "Prev 6m", "Change"],
            [
                [c, n, cat_p.get(c, 0), signed_pct(growth(n, cat_p.get(c, 0)))]
                for c, n in top_n({k: float(v) for k, v in cat_r.items()}, 10)
            ],
        ),
        "",
        f"- Mean resolution time: {res_r:.1f}h last 6 months vs {res_p:.1f}h in the previous 6 "
        f"({signed_pct(growth(res_r, res_p))})",
        f"- P1 share of tickets last 6 months: "
        f"{pct(100 * sum(1 for r in t_recent if r['priority'] == 'P1') / max(1, len(t_recent)))}",
        f"- Open or escalated tickets in file: "
        f"{sum(1 for r in tickets if r['status'] in ('open', 'escalated'))}",
    ]

    # Verbatims, so the agent can quote a customer rather than paraphrase a table.
    quotes_neg = [r for r in neg_recent][: max_quotes // 2]
    quotes_pos = [r for r in pos_recent][: max_quotes - len(quotes_neg)]
    lines += [
        "",
        "**Verbatim samples (recent)**",
        "",
    ]
    for r in quotes_neg + quotes_pos:
        lines.append(
            f'- [{r["rating"]}★ {r["product"]}, {r["region"]}, {r["date"]}] "{r["review_text"]}"'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Market intelligence
# ---------------------------------------------------------------------------
def market_digest(room: DataRoom, max_chars: int = 14000) -> str:
    parts = ["## Market evidence (external sources)", ""]
    budget = max_chars
    for rel in room.available("market"):
        text = room.text(rel)
        if len(text) > budget:
            text = text[:budget] + "\n[...truncated...]"
        budget -= len(text)
        parts += [f"### Source: {rel}", "", text, ""]
        if budget <= 0:
            break
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 5. Operations
# ---------------------------------------------------------------------------
def operations_digest(room: DataRoom) -> str:
    inv = room.rows("operations/inventory.csv")
    vendors = room.rows("operations/vendors.csv")
    ships = room.rows("operations/shipments.csv")
    prod = room.rows("operations/production.csv")

    inv_value = sum(r["on_hand_units"] * r["unit_cost_usd"] for r in inv)
    by_product_value: dict[str, float] = {}
    for r in inv:
        by_product_value[r["product"]] = (
            by_product_value.get(r["product"], 0.0) + r["on_hand_units"] * r["unit_cost_usd"]
        )
    overstock = sorted(
        [r for r in inv if r["days_of_supply"] > 90], key=lambda r: -r["days_of_supply"]
    )
    understock = sorted([r for r in inv if r["days_of_supply"] < 35], key=lambda r: r["days_of_supply"])

    months = sorted({r["ship_date"][:7] for r in ships})
    recent6, prior6 = set(months[-6:]), set(months[-12:-6])
    s_recent = [r for r in ships if r["ship_date"][:7] in recent6]
    s_prior = [r for r in ships if r["ship_date"][:7] in prior6]

    def delay_stats(rows: list[dict]) -> tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0
        avg_delay = sum(r["delay_days"] for r in rows) / len(rows)
        late_pct = 100 * sum(1 for r in rows if r["delay_days"] > 3) / len(rows)
        cost_per_unit = sum(r["freight_cost_usd"] for r in rows) / max(
            1, sum(r["units"] for r in rows)
        )
        return avg_delay, late_pct, cost_per_unit

    d_r, l_r, c_r = delay_stats(s_recent)
    d_p, l_p, c_p = delay_stats(s_prior)

    delay_by_dest_r = group_mean(s_recent, "destination_region", "delay_days")
    delay_by_dest_p = group_mean(s_prior, "destination_region", "delay_days")

    p_months = sorted({r["month"] for r in prod})
    p_recent, p_prior = set(p_months[-6:]), set(p_months[-12:-6])
    yield_r = group_mean([r for r in prod if r["month"] in p_recent], "line", "yield_pct")
    yield_p = group_mean([r for r in prod if r["month"] in p_prior], "line", "yield_pct")
    down_r = group_sum([r for r in prod if r["month"] in p_recent], "line", "downtime_hours")
    down_p = group_sum([r for r in prod if r["month"] in p_prior], "line", "downtime_hours")
    attain_r = {
        line: 100 * group_sum([r for r in prod if r["month"] in p_recent], "line", "produced_units")[line]
        / group_sum([r for r in prod if r["month"] in p_recent], "line", "planned_units")[line]
        for line in yield_r
    }

    single_source = [v for v in vendors if v["single_source"] == "yes"]

    lines = [
        "## Operations evidence",
        "",
        f"**Inventory** — {sum(r['on_hand_units'] for r in inv):,} units on hand across "
        f"{len({r['warehouse'] for r in inv})} warehouses, {money(inv_value)} at cost.",
        "",
        table(
            ["Product", "Inventory at cost", "Share"],
            [
                [p, money(v), pct(100 * v / inv_value)]
                for p, v in top_n(by_product_value, 10)
            ],
        ),
        "",
        f"Overstocked SKUs (>90 days of supply): {len(overstock)} of {len(inv)}.",
        "",
        table(
            ["SKU", "Product", "Warehouse", "On hand", "Monthly demand", "Days of supply", "Value at cost"],
            [
                [r["sku"], r["product"], r["warehouse"], f"{r['on_hand_units']:,}",
                 f"{r['avg_monthly_demand_units']:,}", f"{r['days_of_supply']:.0f}",
                 money(r["on_hand_units"] * r["unit_cost_usd"])]
                for r in overstock[:8]
            ],
        ),
        "",
        f"Thin cover (<35 days of supply): {len(understock)} SKUs.",
        "",
        table(
            ["SKU", "Product", "Warehouse", "On hand", "Reorder point", "Days of supply"],
            [
                [r["sku"], r["product"], r["warehouse"], f"{r['on_hand_units']:,}",
                 f"{r['reorder_point_units']:,}", f"{r['days_of_supply']:.0f}"]
                for r in understock[:8]
            ],
        ),
        "",
        "**Inbound logistics**",
        "",
        table(
            ["Metric", "Last 6 months", "Previous 6 months", "Change"],
            [
                ["Average delay vs plan (days)", f"{d_r:.1f}", f"{d_p:.1f}", f"{d_r - d_p:+.1f}"],
                ["Shipments >3 days late", pct(l_r), pct(l_p), pts(l_r - l_p)],
                ["Freight cost per unit", f"${c_r:.2f}", f"${c_p:.2f}", signed_pct(growth(c_r, c_p))],
            ],
        ),
        "",
        table(
            ["Destination", "Avg delay last 6m (days)", "Previous 6m", "Change"],
            [
                [k, f"{v:.1f}", f"{delay_by_dest_p.get(k, 0):.1f}",
                 f"{v - delay_by_dest_p.get(k, 0):+.1f}"]
                for k, v in sorted(delay_by_dest_r.items(), key=lambda kv: -kv[1])
            ],
        ),
        "",
        "**Vendors**",
        "",
        table(
            ["Vendor", "Component", "Country", "On-time %", "Defect %", "Lead time (d)",
             "Contract", "Single source"],
            [
                [v["vendor"], v["component"], v["country"], pct(v["on_time_delivery_pct"]),
                 pct(v["defect_rate_pct"]), v["lead_time_days"],
                 money(v["annual_contract_value_usd"]), v["single_source"]]
                for v in sorted(vendors, key=lambda v: v["on_time_delivery_pct"])
            ],
        ),
        "",
        f"- Single-sourced components: {len(single_source)} "
        f"({', '.join(v['component'] + ' / ' + v['vendor'] for v in single_source)}), "
        f"{money(sum(v['annual_contract_value_usd'] for v in single_source))} of annual contract value.",
        "",
        "**Manufacturing**",
        "",
        table(
            ["Line", "Yield last 6m", "Yield prev 6m", "Change", "Downtime last 6m (h)",
             "Downtime prev 6m (h)", "Plan attainment last 6m"],
            [
                [line, pct(yield_r[line]), pct(yield_p.get(line, 0)),
                 pts(yield_r[line] - yield_p.get(line, 0)),
                 f"{down_r[line]:.0f}", f"{down_p.get(line, 0):.0f}", pct(attain_r[line])]
                for line in sorted(yield_r)
            ],
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def company_profile(room: DataRoom) -> str:
    try:
        return room.text("company_profile.md")
    except FileNotFoundError:
        return "(no company profile on file)"


DIGEST_BUILDERS = {
    "finance": finance_digest,
    "sales": sales_digest,
    "customer": customer_digest,
    "market": market_digest,
    "operations": operations_digest,
}


def build_digest(role: str, room: DataRoom, settings=None) -> str:
    """Build the evidence brief for one specialist, trimmed to its budget."""
    builder = DIGEST_BUILDERS[role]
    if role == "customer" and settings is not None:
        text = builder(room, max_quotes=settings.max_quotes)
    elif role == "market" and settings is not None:
        text = builder(room, max_chars=settings.max_market_chars)
    else:
        text = builder(room)

    limit = getattr(settings, "max_digest_chars", 9000) if settings else 9000
    if role == "market":
        limit = getattr(settings, "max_market_chars", 14000) if settings else 14000
    if len(text) > limit:
        text = text[:limit] + "\n[...evidence truncated to fit the context budget...]"
    return text