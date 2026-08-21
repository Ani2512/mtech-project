"""Generate the synthetic company dataset the agents analyse.

The fictional company is **Nimbus Audio** — a consumer-audio brand selling
wireless earbuds, headphones and speakers through retail partners,
distributors and its own e-commerce store.

Everything here is seeded, so re-running the script reproduces byte-identical
files. The generator deliberately bakes a *coherent story* into the numbers so
the agents have something real to find:

  * revenue is growing (~18% YoY) but operating margin is compressing (~5pts),
    driven by a logistics-cost spike in the last two quarters;
  * India is the fastest-growing region from a small base, Europe is stalling;
  * the Wave headphone line is declining while Pulse earbuds grow;
  * customers praise delivery speed and sound, and complain about battery life
    (Pulse) and pricing;
  * operations carry overstocked Wave inventory, thin Pulse cover, and a
    single-sourced battery vendor whose defect rate is climbing.

The launch candidate — **Nimbus Pulse Pro** — has no sales history, which is
what makes "should we launch Product X?" a genuine decision question.

Usage:
    python make_sample_data.py            # writes ./data
    python make_sample_data.py --force    # overwrite existing files
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
from datetime import date, timedelta

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

SEED = 7

# 24 complete months: 2024-07 .. 2026-06.
START_YEAR, START_MONTH = 2024, 7
N_MONTHS = 24

REGIONS = ["North America", "Europe", "India", "Southeast Asia", "Middle East"]
COUNTRY_BY_REGION = {
    "North America": ["United States", "Canada"],
    "Europe": ["Germany", "United Kingdom", "France", "Netherlands"],
    "India": ["India"],
    "Southeast Asia": ["Singapore", "Indonesia", "Vietnam", "Thailand"],
    "Middle East": ["United Arab Emirates", "Saudi Arabia"],
}

# Region revenue mix drifts across the window: India roughly doubles its share
# while Europe gives ground.
REGION_MIX_START = {
    "North America": 0.42,
    "Europe": 0.28,
    "India": 0.07,
    "Southeast Asia": 0.13,
    "Middle East": 0.10,
}
REGION_MIX_END = {
    "North America": 0.38,
    "Europe": 0.21,
    "India": 0.17,
    "Southeast Asia": 0.14,
    "Middle East": 0.10,
}

PRODUCTS = ["Nimbus Pulse", "Nimbus Wave", "Nimbus Echo", "Nimbus Clip"]
PRODUCT_MIX_START = {
    "Nimbus Pulse": 0.45,
    "Nimbus Wave": 0.30,
    "Nimbus Echo": 0.18,
    "Nimbus Clip": 0.07,
}
PRODUCT_MIX_END = {
    "Nimbus Pulse": 0.52,
    "Nimbus Wave": 0.20,
    "Nimbus Echo": 0.19,
    "Nimbus Clip": 0.09,
}
UNIT_PRICE = {
    "Nimbus Pulse": 79.0,
    "Nimbus Wave": 149.0,
    "Nimbus Echo": 119.0,
    "Nimbus Clip": 49.0,
}

CHANNELS = ["retail_partner", "distributor", "ecommerce", "direct_b2b"]
CHANNEL_WEIGHTS = [0.38, 0.27, 0.25, 0.10]
SEGMENTS = ["consumer_retail", "telecom_bundle", "enterprise"]
SEGMENT_WEIGHTS = [0.68, 0.20, 0.12]

# November/December lift, February/March dip.
SEASONALITY = {
    1: 0.92, 2: 0.88, 3: 0.95, 4: 0.98, 5: 1.00, 6: 1.02,
    7: 0.99, 8: 1.01, 9: 1.04, 10: 1.08, 11: 1.22, 12: 1.26,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def month_list() -> list[tuple[int, int]]:
    out = []
    y, m = START_YEAR, START_MONTH
    for _ in range(N_MONTHS):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_label(y: int, m: int) -> str:
    return f"{y}-{m:02d}"


def quarter_label(y: int, m: int) -> str:
    return f"{y}-Q{(m - 1) // 3 + 1}"


def ramp(start: float, end: float, i: int, n: int = N_MONTHS) -> float:
    """Linear interpolation from `start` to `end` across the window."""
    if n <= 1:
        return end
    return start + (end - start) * (i / (n - 1))


def blend(a: dict[str, float], b: dict[str, float], i: int) -> dict[str, float]:
    mixed = {k: ramp(a[k], b[k], i) for k in a}
    total = sum(mixed.values())
    return {k: v / total for k, v in mixed.items()}


def pick(rng: random.Random, items: list[str], weights: list[float]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def write_csv(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.relative_to(DATA_DIR.parent)}  ({len(rows)} rows)")


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(DATA_DIR.parent)}")


# --------------------------------------------------------------------------
# finance
# --------------------------------------------------------------------------
def build_finance(rng: random.Random) -> list[dict]:
    """Monthly P&L. Returns the rows so sales generation can reconcile to them."""
    months = month_list()
    rows: list[dict] = []
    base = 3_600_000.0

    for i, (y, m) in enumerate(months):
        trend = (1.0136 ** i)  # ~17-18% annualised
        revenue = base * trend * SEASONALITY[m] * rng.uniform(0.975, 1.025)

        # Margin compression: COGS creeps up, logistics spikes in the final
        # two quarters when the Red Sea re-routing hits.
        cogs_ratio = ramp(0.575, 0.598, i)
        logistics_ratio = ramp(0.040, 0.052, i)
        if i >= N_MONTHS - 6:
            logistics_ratio += 0.018 * ((i - (N_MONTHS - 7)) / 6)

        rnd_ratio = ramp(0.088, 0.095, i)
        sm_ratio = ramp(0.140, 0.152, i)
        ga_ratio = ramp(0.072, 0.070, i)

        cogs = revenue * cogs_ratio
        logistics = revenue * logistics_ratio
        rnd = revenue * rnd_ratio
        sm = revenue * sm_ratio
        ga = revenue * ga_ratio

        gross_profit = revenue - cogs
        operating_income = gross_profit - logistics - rnd - sm - ga
        interest = 46_000 + 900 * i
        pretax = operating_income - interest
        tax = max(0.0, pretax * 0.26)
        net_income = pretax - tax

        rows.append(
            {
                "month": month_label(y, m),
                "revenue": round(revenue, 2),
                "cogs": round(cogs, 2),
                "gross_profit": round(gross_profit, 2),
                "logistics_cost": round(logistics, 2),
                "opex_rnd": round(rnd, 2),
                "opex_sales_marketing": round(sm, 2),
                "opex_ga": round(ga, 2),
                "operating_income": round(operating_income, 2),
                "interest_expense": round(interest, 2),
                "tax": round(tax, 2),
                "net_income": round(net_income, 2),
            }
        )

    write_csv(
        DATA_DIR / "finance" / "income_statement.csv",
        list(rows[0].keys()),
        [list(r.values()) for r in rows],
    )
    return rows


def build_balance_and_cash(rng: random.Random, pnl: list[dict]) -> None:
    quarters: list[str] = []
    for y, m in month_list():
        q = quarter_label(y, m)
        if q not in quarters:
            quarters.append(q)

    bs_rows, cf_rows = [], []
    cash = 14_200_000.0
    inventory = 9_800_000.0
    ar = 7_400_000.0
    lt_debt = 12_000_000.0

    for qi, q in enumerate(quarters):
        q_pnl = [r for r in pnl if quarter_label(int(r["month"][:4]), int(r["month"][5:])) == q]
        q_rev = sum(r["revenue"] for r in q_pnl)
        q_op = sum(r["operating_income"] for r in q_pnl)
        q_net = sum(r["net_income"] for r in q_pnl)

        # Working capital deteriorates: receivables and inventory both build.
        ar = q_rev * ramp(0.62, 0.78, qi, len(quarters))
        inventory = q_rev * ramp(0.55, 0.71, qi, len(quarters))
        other_ca = 2_100_000 + 40_000 * qi
        ppe = 18_500_000 + 380_000 * qi
        intangibles = 6_200_000 - 120_000 * qi

        ap = q_rev * 0.41
        st_debt = 4_500_000 + (900_000 if qi >= len(quarters) - 2 else 0)
        lt_debt = 12_000_000 - 250_000 * qi

        capex = q_rev * rng.uniform(0.035, 0.052)
        working_capital_change = -(q_rev * ramp(0.02, 0.055, qi, len(quarters)))
        depreciation = 620_000 + 18_000 * qi
        operating_cf = q_net + depreciation + working_capital_change
        investing_cf = -capex - rng.uniform(150_000, 400_000)
        financing_cf = (-250_000) + (900_000 if qi >= len(quarters) - 2 else 0)
        cash = cash + operating_cf + investing_cf + financing_cf

        total_assets = cash + ar + inventory + other_ca + ppe + intangibles
        total_liabilities = ap + st_debt + lt_debt + 3_100_000
        equity = total_assets - total_liabilities

        bs_rows.append(
            [
                q,
                round(cash, 2), round(ar, 2), round(inventory, 2), round(other_ca, 2),
                round(ppe, 2), round(intangibles, 2), round(total_assets, 2),
                round(ap, 2), round(st_debt, 2), round(lt_debt, 2),
                round(total_liabilities, 2), round(equity, 2),
            ]
        )
        cf_rows.append(
            [
                q,
                round(q_net, 2), round(depreciation, 2), round(working_capital_change, 2),
                round(operating_cf, 2), round(capex, 2), round(investing_cf, 2),
                round(financing_cf, 2), round(operating_cf - capex, 2), round(cash, 2),
            ]
        )
        _ = q_op

    write_csv(
        DATA_DIR / "finance" / "balance_sheet.csv",
        [
            "quarter", "cash", "accounts_receivable", "inventory", "other_current_assets",
            "ppe_net", "intangibles", "total_assets", "accounts_payable", "short_term_debt",
            "long_term_debt", "total_liabilities", "shareholders_equity",
        ],
        bs_rows,
    )
    write_csv(
        DATA_DIR / "finance" / "cash_flow.csv",
        [
            "quarter", "net_income", "depreciation", "working_capital_change",
            "operating_cash_flow", "capex", "investing_cash_flow", "financing_cash_flow",
            "free_cash_flow", "closing_cash",
        ],
        cf_rows,
    )


def build_budget(rng: random.Random, pnl: list[dict]) -> None:
    """Departmental budget vs actual for the trailing 12 months."""
    departments = {
        "R&D": ("opex_rnd", 1.00),
        "Sales & Marketing": ("opex_sales_marketing", 0.96),
        "G&A": ("opex_ga", 1.02),
        "Logistics": ("logistics_cost", 0.86),   # chronically under-budgeted
        "Manufacturing": ("cogs", 0.99),
    }
    rows = []
    for r in pnl[-12:]:
        for dept, (field, plan_factor) in departments.items():
            actual = r[field]
            budget = actual * plan_factor * rng.uniform(0.985, 1.015)
            rows.append(
                [
                    r["month"], dept, round(budget, 2), round(actual, 2),
                    round(actual - budget, 2), round(100 * (actual - budget) / budget, 2),
                ]
            )
    write_csv(
        DATA_DIR / "finance" / "budget_vs_actual.csv",
        ["month", "department", "budget_usd", "actual_usd", "variance_usd", "variance_pct"],
        rows,
    )


# --------------------------------------------------------------------------
# sales
# --------------------------------------------------------------------------
def build_sales(rng: random.Random, pnl: list[dict]) -> None:
    months = month_list()
    rows = []
    order_no = 100_000

    for i, (y, m) in enumerate(months):
        target = pnl[i]["revenue"]
        region_mix = blend(REGION_MIX_START, REGION_MIX_END, i)
        product_mix = blend(PRODUCT_MIX_START, PRODUCT_MIX_END, i)
        days = (date(y + (m == 12), (m % 12) + 1, 1) - date(y, m, 1)).days

        for region, r_share in region_mix.items():
            # Europe softens further than the mix alone implies.
            region_target = target * r_share
            for product, p_share in product_mix.items():
                # Wave sells disproportionately in Europe/NA, Clip in India/SEA.
                skew = 1.0
                if product == "Nimbus Wave" and region in ("Europe", "North America"):
                    skew = 1.18
                if product == "Nimbus Clip" and region in ("India", "Southeast Asia"):
                    skew = 1.35
                if product == "Nimbus Pulse" and region == "India":
                    skew = 1.22
                bucket = region_target * p_share * skew
                remaining = bucket
                while remaining > 1_000:
                    order_no += 1
                    price = UNIT_PRICE[product]
                    # Discounting deepens over time, worst in Europe.
                    disc = ramp(0.04, 0.11, i) + (0.035 if region == "Europe" else 0.0)
                    disc = max(0.0, min(0.35, rng.gauss(disc, 0.02)))
                    net_price = price * (1 - disc)
                    units = max(20, int(rng.lognormvariate(5.6, 0.7)))
                    line = min(remaining, units * net_price)
                    units = max(10, int(round(line / net_price)))
                    remaining -= units * net_price
                    order_date = date(y, m, 1) + timedelta(days=rng.randrange(days))
                    rows.append(
                        [
                            f"SO-{order_no}",
                            order_date.isoformat(),
                            region,
                            rng.choice(COUNTRY_BY_REGION[region]),
                            pick(rng, CHANNELS, CHANNEL_WEIGHTS),
                            pick(rng, SEGMENTS, SEGMENT_WEIGHTS),
                            product,
                            units,
                            round(price, 2),
                            round(100 * disc, 2),
                            round(units * net_price, 2),
                        ]
                    )

    rows.sort(key=lambda r: r[1])
    write_csv(
        DATA_DIR / "sales" / "orders.csv",
        [
            "order_id", "order_date", "region", "country", "channel", "customer_segment",
            "product", "units", "list_price_usd", "discount_pct", "net_revenue_usd",
        ],
        rows,
    )

    # CRM pipeline — open opportunities, weighted toward India for the Pulse Pro.
    stages = ["Qualification", "Proposal", "Negotiation", "Contracting", "Closed Won", "Closed Lost"]
    prob = {"Qualification": 20, "Proposal": 40, "Negotiation": 65, "Contracting": 85,
            "Closed Won": 100, "Closed Lost": 0}
    accounts = [
        "Reliance Digital", "Croma", "Flipkart", "Amazon IN", "Vijay Sales",
        "Best Buy", "Target", "Walmart", "Amazon US", "Costco",
        "MediaMarkt", "Fnac Darty", "Currys", "Coolblue",
        "Shopee SG", "Lazada ID", "FPT Retail", "Central Thailand",
        "Sharaf DG", "Jarir Bookstore", "Noon.com",
        "Airtel Bundles", "Jio Devices", "Vodafone EU", "T-Mobile US",
    ]
    owners = ["A. Menon", "L. Fischer", "S. Rahman", "J. Alvarez", "P. Kaur", "T. Okafor"]
    pipe = []
    for n in range(140):
        account = accounts[n % len(accounts)]
        region = (
            "India" if account in ("Reliance Digital", "Croma", "Flipkart", "Amazon IN",
                                   "Vijay Sales", "Airtel Bundles", "Jio Devices")
            else "North America" if account in ("Best Buy", "Target", "Walmart", "Amazon US",
                                                "Costco", "T-Mobile US")
            else "Europe" if account in ("MediaMarkt", "Fnac Darty", "Currys", "Coolblue",
                                         "Vodafone EU")
            else "Southeast Asia" if account in ("Shopee SG", "Lazada ID", "FPT Retail",
                                                 "Central Thailand")
            else "Middle East"
        )
        stage = pick(rng, stages, [0.26, 0.22, 0.16, 0.10, 0.16, 0.10])
        value = rng.lognormvariate(12.4, 0.75)
        product = pick(rng, PRODUCTS + ["Nimbus Pulse Pro"], [0.34, 0.12, 0.14, 0.10, 0.30])
        close = date(2026, 7, 1) + timedelta(days=rng.randrange(210))
        pipe.append(
            [
                f"OPP-{5000 + n}", account, region, product, stage,
                round(value, 2), prob[stage], close.isoformat(), rng.choice(owners),
            ]
        )
    write_csv(
        DATA_DIR / "sales" / "crm_pipeline.csv",
        ["opportunity_id", "account", "region", "product", "stage", "value_usd",
         "probability_pct", "expected_close_date", "owner"],
        pipe,
    )


# --------------------------------------------------------------------------
# customer intelligence
# --------------------------------------------------------------------------
POSITIVE_TEMPLATES = [
    ("delivery speed", "Ordered on Monday, wearing them Wednesday — {product} shipping was quicker than promised."),
    ("delivery speed", "Delivery was unbelievably fast and the packaging on the {product} was spotless."),
    ("sound quality", "The sound stage on the {product} is rich and the bass is clean at any volume."),
    ("sound quality", "Call clarity on the {product} beats my old set by a mile, even on a windy street."),
    ("build quality", "The {product} feels solid — two months of gym use and not a scratch."),
    ("comfort", "I can wear the {product} for a full workday without any ear fatigue."),
    ("app experience", "Pairing the {product} through the Nimbus app took ten seconds flat."),
    ("value", "For the price the {product} is honestly hard to beat right now."),
]
NEGATIVE_TEMPLATES = [
    ("battery life", "The {product} claims eight hours but I get closer to four and a half before the case is dead."),
    ("battery life", "Battery on my {product} degraded noticeably after three months — it barely survives a commute now."),
    ("battery life", "Charging case for the {product} drains overnight even when I don't use it."),
    ("pricing", "At this price the {product} should not be missing basic features — the competition is cheaper."),
    ("pricing", "Another price hike on the {product} and still no charger in the box."),
    ("connectivity", "The {product} drops the Bluetooth link whenever my phone is in my back pocket."),
    ("connectivity", "Multipoint pairing on the {product} keeps switching to the wrong device."),
    ("shipping", "The {product} shipped nine days late with no explanation from support."),
    ("support", "Two weeks to get a warranty reply about my {product}. Not acceptable."),
    ("fit", "The {product} tips fall out of my ears during any kind of run."),
]
NEUTRAL_TEMPLATES = [
    ("mixed", "The {product} sounds great but the battery is average — fine for the money."),
    ("mixed", "Good {product} overall; the app is clunky and updates are slow to arrive."),
    ("mixed", "Solid {product}, though I expected better noise cancellation at this price."),
]


def build_customer(rng: random.Random) -> None:
    months = month_list()
    reviews, tickets, surveys = [], [], []

    for i, (y, m) in enumerate(months):
        days = (date(y + (m == 12), (m % 12) + 1, 1) - date(y, m, 1)).days
        n_reviews = 24 + rng.randrange(8)
        for _ in range(n_reviews):
            product = pick(
                rng, PRODUCTS,
                list(blend(PRODUCT_MIX_START, PRODUCT_MIX_END, i).values()),
            )
            region = pick(rng, REGIONS, list(blend(REGION_MIX_START, REGION_MIX_END, i).values()))
            # Two deliberate effects, so the sentiment signal is real rather than
            # sampling noise: Pulse sours through the window as the battery
            # problem spreads, and Europe runs unhappy (price pressure, freight
            # delays) while India runs happy — matching the NPS-by-region data.
            neg_rate = 0.20
            if product == "Nimbus Pulse":
                neg_rate += 0.26 * (i / (N_MONTHS - 1))
            neg_rate += {
                "Europe": 0.12, "North America": 0.02, "Middle East": 0.0,
                "Southeast Asia": -0.03, "India": -0.07,
            }[region]
            if i >= N_MONTHS - 6:      # freight delays reach the customer
                neg_rate += 0.04
            roll = rng.random()
            if roll < neg_rate:
                # Pick a complaint that fits the situation: Pulse owners talk
                # about battery, Europe talks about price, late shipments show
                # up as shipping complaints once the delays start.
                theme, text = rng.choice(NEGATIVE_TEMPLATES)
                for _try in range(3):
                    if theme == "battery life" and product != "Nimbus Pulse" and rng.random() < 0.7:
                        theme, text = rng.choice(NEGATIVE_TEMPLATES)
                    elif theme == "pricing" and region != "Europe" and rng.random() < 0.4:
                        theme, text = rng.choice(NEGATIVE_TEMPLATES)
                    elif theme == "shipping" and i < N_MONTHS - 6 and rng.random() < 0.7:
                        theme, text = rng.choice(NEGATIVE_TEMPLATES)
                    else:
                        break
                rating = rng.choice([1, 1, 2, 2, 3])
            elif roll < neg_rate + 0.12:
                theme, text = rng.choice(NEUTRAL_TEMPLATES)
                rating = 3
            else:
                theme, text = rng.choice(POSITIVE_TEMPLATES)
                rating = rng.choice([4, 5, 5, 5])
            d = date(y, m, 1) + timedelta(days=rng.randrange(days))
            reviews.append(
                [
                    f"RV-{len(reviews) + 1:05d}", d.isoformat(), product, region, rating,
                    theme, text.format(product=product), rng.choice(["yes", "yes", "yes", "no"]),
                ]
            )

        n_tickets = 18 + rng.randrange(8)
        for _ in range(n_tickets):
            product = pick(rng, PRODUCTS, [0.5, 0.22, 0.19, 0.09])
            region = pick(rng, REGIONS, list(blend(REGION_MIX_START, REGION_MIX_END, i).values()))
            battery_weight = 0.14 + 0.20 * (i / (N_MONTHS - 1))
            ship_weight = 0.10 + (0.16 if i >= N_MONTHS - 6 else 0.0)
            cats = ["battery", "connectivity", "shipping_delay", "charging_case",
                    "audio_quality", "billing", "warranty_claim"]
            weights = [battery_weight, 0.17, ship_weight, 0.11, 0.12, 0.08, 0.12]
            category = pick(rng, cats, weights)
            priority = pick(rng, ["P1", "P2", "P3"], [0.16, 0.44, 0.40])
            base_hours = {"P1": 9, "P2": 26, "P3": 58}[priority]
            hours = max(1.0, rng.gauss(base_hours * (1.0 + 0.35 * (i / (N_MONTHS - 1))), base_hours * 0.3))
            d = date(y, m, 1) + timedelta(days=rng.randrange(days))
            tickets.append(
                [
                    f"TK-{len(tickets) + 1:05d}", d.isoformat(), product, region, category,
                    priority, pick(rng, ["resolved", "resolved", "resolved", "open", "escalated"],
                                   [0.4, 0.25, 0.2, 0.1, 0.05]),
                    round(hours, 1),
                    f"{category.replace('_', ' ').title()} reported on {product}",
                ]
            )

        n_surveys = 9 + rng.randrange(4)
        for _ in range(n_surveys):
            region = pick(rng, REGIONS, list(blend(REGION_MIX_START, REGION_MIX_END, i).values()))
            centre = {"India": 8.6, "Southeast Asia": 8.1, "Middle East": 7.8,
                      "North America": 7.6, "Europe": 6.6}[region]
            nps = int(max(0, min(10, round(rng.gauss(centre, 1.9)))))
            csat = int(max(1, min(5, round(rng.gauss(1 + nps / 2.6, 0.7)))))
            d = date(y, m, 1) + timedelta(days=rng.randrange(days))
            surveys.append(
                [
                    f"SV-{len(surveys) + 1:05d}", d.isoformat(), region, nps, csat,
                    "yes" if nps >= 8 else "no",
                    rng.choice(
                        [
                            "Would buy again — delivery was fast.",
                            "Price keeps creeping up.",
                            "Battery does not last a full day.",
                            "Great sound for the money.",
                            "Support took too long to respond.",
                            "Love the app, hate the case.",
                        ]
                    ),
                ]
            )

    write_csv(
        DATA_DIR / "customer" / "reviews.csv",
        ["review_id", "date", "product", "region", "rating", "topic", "review_text",
         "verified_purchase"],
        reviews,
    )
    write_csv(
        DATA_DIR / "customer" / "support_tickets.csv",
        ["ticket_id", "date", "product", "region", "category", "priority", "status",
         "resolution_hours", "summary"],
        tickets,
    )
    write_csv(
        DATA_DIR / "customer" / "surveys.csv",
        ["response_id", "date", "region", "nps_score", "csat_score", "would_recommend",
         "comment"],
        surveys,
    )


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------
def build_operations(rng: random.Random) -> None:
    warehouses = [
        ("WH-NA1", "Reno, NV", "North America"),
        ("WH-NA2", "Columbus, OH", "North America"),
        ("WH-EU1", "Rotterdam", "Europe"),
        ("WH-IN1", "Bhiwandi", "India"),
        ("WH-SEA1", "Singapore", "Southeast Asia"),
        ("WH-ME1", "Jebel Ali", "Middle East"),
    ]
    inv = []
    for code, city, region in warehouses:
        for product in PRODUCTS:
            sku = f"{product.split()[1][:2].upper()}-{code[-3:]}"
            demand = {
                "Nimbus Pulse": 9_800, "Nimbus Wave": 3_100,
                "Nimbus Echo": 4_200, "Nimbus Clip": 2_600,
            }[product] * {"North America": 1.0, "Europe": 0.62, "India": 0.55,
                          "Southeast Asia": 0.44, "Middle East": 0.3}[region]
            demand = max(200, int(demand * rng.uniform(0.85, 1.15) / 2))
            if product == "Nimbus Wave":         # overstocked, demand fading
                on_hand = int(demand * rng.uniform(3.6, 5.4))
            elif product == "Nimbus Pulse":      # running thin
                on_hand = int(demand * rng.uniform(0.7, 1.3))
            else:
                on_hand = int(demand * rng.uniform(1.6, 2.6))
            reserved = int(on_hand * rng.uniform(0.05, 0.18))
            unit_cost = round(UNIT_PRICE[product] * rng.uniform(0.52, 0.6), 2)
            inv.append(
                [
                    sku, product, code, city, region, on_hand, reserved,
                    int(demand * 0.9), demand, unit_cost,
                    round(30 * (on_hand - reserved) / demand, 1),
                ]
            )
    write_csv(
        DATA_DIR / "operations" / "inventory.csv",
        ["sku", "product", "warehouse", "location", "region", "on_hand_units",
         "reserved_units", "reorder_point_units", "avg_monthly_demand_units",
         "unit_cost_usd", "days_of_supply"],
        inv,
    )

    vendors = [
        ("VN-01", "Anshan Cell Works", "Battery cell", "China", 0.71, 4.9, 42, 8_900_000, "yes"),
        ("VN-02", "Haiphong Precision", "Enclosure", "Vietnam", 0.86, 1.8, 28, 4_200_000, "no"),
        ("VN-03", "Penang Acoustics", "Driver unit", "Malaysia", 0.93, 0.9, 24, 6_400_000, "no"),
        ("VN-04", "Shenzhen Linkwave", "BT SoC", "China", 0.88, 1.4, 35, 7_100_000, "yes"),
        ("VN-05", "Chennai Moulding", "Charging case", "India", 0.90, 2.2, 18, 1_900_000, "no"),
        ("VN-06", "Suzhou FlexPCB", "PCB assembly", "China", 0.82, 2.6, 31, 3_600_000, "no"),
        ("VN-07", "Bangkok Foam", "Ear tips", "Thailand", 0.96, 0.6, 14, 620_000, "no"),
        ("VN-08", "Guadalajara Pack", "Retail packaging", "Mexico", 0.94, 0.8, 12, 880_000, "no"),
        ("VN-09", "Taipei Magnetics", "Magnet array", "Taiwan", 0.89, 1.1, 27, 2_400_000, "yes"),
        ("VN-10", "Pune Cable Co", "Cabling", "India", 0.92, 1.6, 16, 540_000, "no"),
    ]
    write_csv(
        DATA_DIR / "operations" / "vendors.csv",
        ["vendor_id", "vendor", "component", "country", "on_time_delivery_pct",
         "defect_rate_pct", "lead_time_days", "annual_contract_value_usd", "single_source"],
        [
            [v[0], v[1], v[2], v[3], round(100 * v[4], 1), v[5], v[6], v[7], v[8]]
            for v in vendors
        ],
    )

    lanes = [
        ("Shenzhen", "North America", "sea", 28),
        ("Shenzhen", "Europe", "sea", 34),
        ("Shenzhen", "India", "sea", 21),
        ("Haiphong", "Southeast Asia", "sea", 9),
        ("Haiphong", "Europe", "sea", 33),
        ("Shenzhen", "Middle East", "sea", 19),
        ("Shenzhen", "North America", "air", 5),
        ("Chennai", "India", "road", 4),
    ]
    ship_rows = []
    for i, (y, m) in enumerate(month_list()):
        days = (date(y + (m == 12), (m % 12) + 1, 1) - date(y, m, 1)).days
        for _ in range(9 + rng.randrange(4)):
            origin, dest, mode, planned = rng.choice(lanes)
            # Delays escalate in the final two quarters, worst on Europe sea lanes.
            slip = rng.gauss(1.2, 1.6)
            if i >= N_MONTHS - 6:
                slip += rng.gauss(5.5 if dest == "Europe" else 3.2, 2.0)
            actual = max(2, int(round(planned + slip)))
            units = 2_000 + rng.randrange(14_000)
            base_rate = {"sea": 0.42, "air": 3.10, "road": 0.28}[mode]
            rate = base_rate * (1.0 + (0.45 if i >= N_MONTHS - 6 else 0.10))
            d = date(y, m, 1) + timedelta(days=rng.randrange(days))
            ship_rows.append(
                [
                    f"SH-{len(ship_rows) + 1:05d}", d.isoformat(), origin, dest, mode,
                    planned, actual, actual - planned, units,
                    round(units * rate, 2),
                    "delivered" if actual - planned <= 3 else "delivered_late",
                ]
            )
    write_csv(
        DATA_DIR / "operations" / "shipments.csv",
        ["shipment_id", "ship_date", "origin", "destination_region", "mode",
         "planned_transit_days", "actual_transit_days", "delay_days", "units",
         "freight_cost_usd", "status"],
        ship_rows,
    )

    prod_rows = []
    for i, (y, m) in enumerate(month_list()):
        for line in ["Line A (earbuds)", "Line B (headphones)", "Line C (speakers)"]:
            planned = {"Line A (earbuds)": 210_000, "Line B (headphones)": 74_000,
                       "Line C (speakers)": 96_000}[line]
            planned = int(planned * (1.0 + 0.012 * i) * rng.uniform(0.95, 1.05))
            yield_pct = rng.gauss(96.4 - (1.9 if line.startswith("Line A") and i >= 18 else 0), 0.8)
            produced = int(planned * min(1.0, rng.gauss(0.97, 0.03)))
            scrap = int(produced * (100 - yield_pct) / 100)
            downtime = round(max(0.0, rng.gauss(22 + (10 if i >= 18 else 0), 7)), 1)
            prod_rows.append(
                [month_label(y, m), line, planned, produced, scrap, downtime,
                 round(yield_pct, 2)]
            )
    write_csv(
        DATA_DIR / "operations" / "production.csv",
        ["month", "line", "planned_units", "produced_units", "scrap_units",
         "downtime_hours", "yield_pct"],
        prod_rows,
    )


# --------------------------------------------------------------------------
# market intelligence (unstructured text)
# --------------------------------------------------------------------------
NEWS = """
# Industry News Feed — Consumer Audio (rolling 9 months)

**2026-06-18 — Reuters.** India's true-wireless earbud shipments grew 22% year on year
in Q1 FY27, the fifth consecutive quarter of double-digit growth. Analysts attribute
the expansion to sub-$60 price bands and telecom bundling deals rather than premium demand.

**2026-06-02 — Nikkei Asia.** Ocean freight rates on Asia–Europe lanes rose a further
14% after continued Red Sea re-routing. Carriers expect elevated rates through at least
Q3 2026. Consumer electronics brands with thin logistics buffers are reported to be
absorbing the cost rather than repricing.

**2026-05-27 — The Verge.** SonicaLabs launched the Sonica Air 3 at $89 with a claimed
11-hour battery, undercutting incumbents in the mid-tier earbud segment. Early reviews
praise battery endurance and criticise call quality.

**2026-05-11 — Economic Times.** Indian government extended PLI (production-linked
incentive) benefits for wearables assembled domestically, effective FY27. Brands with
local assembly gain roughly 4-6% landed-cost advantage over imports.

**2026-04-30 — Bloomberg.** EU regulators finalised the common-charger and
battery-replaceability rules; portable audio devices sold in the EU after 2027 must
offer user-replaceable batteries. Compliance retooling estimated at $2-4M per SKU family.

**2026-04-09 — TechCrunch.** AuraSound raised $140M Series D to expand its India
distribution and open a Pune assembly line, explicitly targeting the $50-100 earbud band.

**2026-03-22 — WSJ.** US tariff review placed consumer audio imports from China under a
proposed additional 7.5% duty, with a decision expected in Q4 2026.

**2026-02-14 — CNBC.** Holiday-quarter results across the category showed volume growth
with average selling prices down 6% year on year — the third straight year of ASP erosion
in earbuds.

**2026-01-19 — Mint.** Indian e-commerce platforms reported record Republic Day audio
sales, with 68% of units sold below Rs 5,000 (~$60).

**2025-12-05 — FT.** Battery cell suppliers signalled 2026 price increases of 8-12% on
small-format lithium cells, citing raw-material contracts and capacity reallocation to EVs.
"""

COMPETITORS = """
# Competitor Tracker — Q2 2026

## SonicaLabs (global, listed)
- Revenue run-rate ~$1.9B, earbuds ~62% of mix.
- Launched Sonica Air 3 ($89) in May 2026: 11h battery, IPX5, no ANC.
- Aggressive channel rebates in Europe; estimated 4-6 pts of share gain there since 2025.
- Weakness: call quality complaints, thin service network in India.

## AuraSound (private, India-focused challenger)
- Estimated FY26 revenue $310M, growing >45% YoY.
- $140M raised April 2026; Pune assembly line targeted for Q1 2027 (PLI-eligible).
- Price band $35-75. Heavy telecom bundling with two national carriers.
- Weakness: brand perception at premium tiers, 3.9-star average retail rating.

## Kestrel Audio (premium, EU/NA)
- Premium ANC headphones $249-399. Not a direct earbud competitor.
- Announced replaceable-battery redesign ahead of EU 2027 rules — a compliance head start.

## Boru (China domestic + SEA export)
- Ultra-value earbuds $19-39. Expanding in Indonesia and Vietnam.
- Quality variance high; 12-month failure rates reported around 9%.

## Share movement (mid-tier earbuds, $50-120, estimated)
| Region        | Nimbus 2025 | Nimbus Q2 2026 | Leader           |
|---------------|-------------|----------------|------------------|
| North America | 11.4%       | 11.1%          | SonicaLabs 19.8% |
| Europe        | 9.2%        | 7.6%           | SonicaLabs 22.4% |
| India         | 3.1%        | 4.8%           | AuraSound 16.2%  |
| Southeast Asia| 6.0%        | 6.3%           | Boru 14.9%       |
| Middle East   | 8.1%        | 8.0%           | SonicaLabs 17.1% |
"""

RESEARCH = """
# Industry Research Digest — Wearable Audio, 2026

## Market size and growth (third-party estimates)
| Market         | 2025 TAM | 2028 TAM (est.) | CAGR  | Notes                              |
|----------------|----------|-----------------|-------|------------------------------------|
| Global TWS     | $42.1B   | $58.9B          | 11.9% | Volume-led; ASPs eroding ~5%/yr    |
| India TWS      | $2.4B    | $4.6B           | 24.2% | Fastest large market; price-driven |
| Europe TWS     | $9.8B    | $11.2B          | 4.5%  | Saturating; replacement-cycle led  |
| North America  | $12.6B   | $15.1B          | 6.2%  | Premium-tier resilient             |
| SE Asia TWS    | $3.3B    | $5.4B           | 17.8% | Value tier dominant                |

## Demand drivers
- Replacement cycles have shortened to ~22 months, largely because of battery degradation.
- Battery endurance is now the single most cited purchase driver in mid-tier earbuds
  (37% of surveyed buyers), ahead of sound quality (29%) and ANC (18%).
- Bundling with telecom plans accounts for an estimated 21% of India unit volume.

## Cost and regulation
- Small-format lithium cell prices expected +8-12% in 2026.
- EU battery-replaceability mandate applies to devices placed on the EU market from 2027.
- India PLI for wearables favours local assembly; imported units face ~18% effective duty.
- Proposed US tariff on China-origin consumer audio: +7.5%, decision expected Q4 2026.

## Analyst commentary
- Mid-tier ($50-120) is where volume is growing but margin is thinnest; two of the last
  three entrants at that price point exited within 18 months.
- Brands winning in India did so with local assembly, carrier bundles and a service
  network, not with feature parity.
"""


def build_market() -> None:
    write_text(DATA_DIR / "market" / "news_feed.md", NEWS)
    write_text(DATA_DIR / "market" / "competitor_tracker.md", COMPETITORS)
    write_text(DATA_DIR / "market" / "industry_research.md", RESEARCH)


COMPANY_PROFILE = """
# Nimbus Audio — Company Profile

**Business.** Consumer audio hardware. Wireless earbuds, headphones and portable
speakers sold through retail partners, distributors, telecom bundles and a direct
e-commerce store.

**Fiscal window in this dataset.** 2024-07 through 2026-06 (24 complete months).

**Product lines**
| Product           | Category            | List price | Status                          |
|-------------------|---------------------|-----------|----------------------------------|
| Nimbus Pulse      | Mid-tier earbuds    | $79       | Core volume driver               |
| Nimbus Wave       | Over-ear headphones | $149      | Mature, declining                |
| Nimbus Echo       | Portable speaker    | $119      | Stable                           |
| Nimbus Clip       | Sport earbuds       | $49       | Small, growing in India/SEA      |
| **Nimbus Pulse Pro** | Premium earbuds  | $129 (planned) | **Unlaunched — decision pending** |

**Regions.** North America, Europe, India, Southeast Asia, Middle East.

**Open decision.** Whether to launch the Nimbus Pulse Pro, and if so in which
region and on what timeline.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Nimbus Audio sample dataset.")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data/ directory")
    args = ap.parse_args()

    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.csv")) and not args.force:
        print(f"{DATA_DIR} already contains data. Re-run with --force to regenerate.")
        return

    rng = random.Random(SEED)
    print(f"Generating Nimbus Audio dataset in {DATA_DIR} ...")
    pnl = build_finance(rng)
    build_balance_and_cash(rng, pnl)
    build_budget(rng, pnl)
    build_sales(rng, pnl)
    build_customer(rng)
    build_operations(rng)
    build_market()
    write_text(DATA_DIR / "company_profile.md", COMPANY_PROFILE)
    print("Done.")


if __name__ == "__main__":
    main()