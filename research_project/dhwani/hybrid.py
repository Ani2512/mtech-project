"""Per-type arm selection: direct prompting or decompose-and-combine.

docs/phase2_decomposition.md shows the two arms fail in different places.
Decomposition wins on the relational types (AFTER, NEXT_AFTER, WHILE,
NOT_FOLLOWED) and loses on ORDINAL and BEFORE, which the model already handles
and where grounding a second event only adds error.

Choosing the arm per condition type is therefore worth a few points, but only
if the choice is made on validation clips and reported on test clips. Selecting
on the test set and reporting the same numbers would be choosing the maximum of
two noisy estimates and calling it a method.

    python -m dhwani.hybrid --direct runs/esc50/qwen25_omni --agent runs/esc50/agent_omni \
           --out runs/esc50/hybrid
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import summarize
from .split import assign


def load(run_dir: Path) -> dict[str, dict]:
    rows = {}
    for line in open(run_dir / "predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["qid"]] = r
    return rows


def clip_of(qid: str) -> str:
    return qid.rsplit("_q", 1)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True, help="run dir for direct prompting")
    ap.add_argument("--agent", required=True, help="run dir for decompose-and-combine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--frac-val", type=float, default=0.15)
    ap.add_argument("--metric", default="f1@0.5")
    a = ap.parse_args(argv)

    direct, agent = load(Path(a.direct)), load(Path(a.agent))
    shared = sorted(set(direct) & set(agent))
    if not shared:
        raise SystemExit("the two runs share no query ids; score them on the same benchmark")
    print(f"{len(shared)} queries scored by both arms")

    split_of = {q: assign(clip_of(q), a.seed, a.frac_train, a.frac_val) for q in shared}
    n_split = defaultdict(int)
    for s in split_of.values():
        n_split[s] += 1
    print("queries per split:", dict(n_split))

    # choose on val only
    val_scores: dict[str, dict[str, list]] = defaultdict(lambda: {"direct": [], "agent": []})
    for q in shared:
        if split_of[q] != "val":
            continue
        t = direct[q]["qtype"]
        val_scores[t]["direct"].append(direct[q][a.metric])
        val_scores[t]["agent"].append(agent[q][a.metric])

    choice = {}
    print(f"\narm chosen per type, on VAL ({a.metric})")
    print(f"{'type':<14}{'direct':>10}{'agent':>10}{'n':>6}  chosen")
    for t in sorted(val_scores):
        d = val_scores[t]["direct"]; g = val_scores[t]["agent"]
        md = sum(d) / len(d) if d else 0.0
        mg = sum(g) / len(g) if g else 0.0
        choice[t] = "agent" if mg > md else "direct"
        print(f"{t:<14}{md:>10.3f}{mg:>10.3f}{len(d):>6}  {choice[t]}")

    # report on test only
    rows_hybrid, rows_direct, rows_agent = [], [], []
    for q in shared:
        if split_of[q] != "test":
            continue
        t = direct[q]["qtype"]
        src = agent if choice.get(t, "direct") == "agent" else direct
        rows_hybrid.append(src[q])
        rows_direct.append(direct[q])
        rows_agent.append(agent[q])

    if not rows_hybrid:
        raise SystemExit("no test-split queries in these runs; score more of the benchmark")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summaries = {"hybrid": summarize(rows_hybrid), "direct": summarize(rows_direct),
                 "agent": summarize(rows_agent)}
    (out / "summary.json").write_text(json.dumps(
        {"model": "hybrid", "choice": choice, "n_test": len(rows_hybrid),
         "by_type": summaries["hybrid"], "arms": {k: v for k, v in summaries.items() if k != "hybrid"}},
        indent=2), encoding="utf-8")

    TY = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
    print(f"\nTEST split, {len(rows_hybrid)} queries ({a.metric})")
    print(f"{'arm':<10}" + "".join(f"{t:>13}" for t in TY))
    print("-" * (10 + 13 * len(TY)))
    for name in ("direct", "agent", "hybrid"):
        by = summaries[name]
        print(f"{name:<10}" + "".join(
            (f"{by[t][a.metric]:>13.3f}" if isinstance(by.get(t, {}).get(a.metric), (int, float))
             else f"{'-':>13}") for t in TY))


if __name__ == "__main__":
    main()
