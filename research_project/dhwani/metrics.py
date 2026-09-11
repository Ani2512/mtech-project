"""Interval parsing and metrics.

union_iou      TAG-Bench style: IoU in seconds between merged interval sets
matched_f1     AEGBench style: one-to-one Hungarian matching on IoU, F1 at a threshold
count_acc      |pred| == |gt|
rejection      queries whose ground truth is empty: did the model return []?
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

Interval = tuple[float, float]

_PAIR = re.compile(r"\[?\s*(\d+(?:\.\d+)?)\s*(?:,|-|–|to)\s*(\d+(?:\.\d+)?)\s*\]?")
_EMPTY = re.compile(r"\[\s*\]|\bnone\b|\bno such\b|does not occur|not present|no occurrence", re.I)


def parse_intervals(text: str) -> list[Interval] | None:
    """Return a list of (start, end); [] for an explicit empty answer; None if unparseable."""
    if text is None:
        return None
    text = text.strip()
    # 1) JSON anywhere in the text
    for m in re.finditer(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            if not obj:
                return []
            pairs = []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    pairs.append((float(it[0]), float(it[1])))
                elif isinstance(it, dict) and {"start", "end"} <= set(it):
                    pairs.append((float(it["start"]), float(it["end"])))
            if pairs:
                return _clean(pairs)
    # 2) explicit empty
    if _EMPTY.search(text):
        return []
    # 3) loose "12.3-15.6" / "12.3 to 15.6" pairs
    pairs = [(float(a), float(b)) for a, b in _PAIR.findall(text)]
    return _clean(pairs) if pairs else None


def _clean(pairs):
    out = []
    for a, b in pairs:
        if b < a:
            a, b = b, a
        if b > a:
            out.append((a, b))
    return out


def _merge(iv: list[Interval]) -> list[Interval]:
    iv = sorted(iv)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _length(iv):
    return sum(b - a for a, b in iv)


def _inter(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: Interval, b: Interval) -> float:
    i = _inter(a, b)
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def union_iou(pred: list[Interval], gt: list[Interval]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    p, g = _merge(pred), _merge(gt)
    inter = sum(_inter(a, b) for a in p for b in g)
    union = _length(_merge(p + g))
    return inter / union if union > 0 else 0.0


def matched_f1(pred: list[Interval], gt: list[Interval], thr: float = 0.5) -> tuple[float, float, float]:
    """(precision, recall, f1) with one-to-one Hungarian matching at IoU >= thr."""
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    from scipy.optimize import linear_sum_assignment

    M = np.array([[iou(p, g) for g in gt] for p in pred])
    r, c = linear_sum_assignment(-M)
    tp = int(sum(M[i, j] >= thr for i, j in zip(r, c)))
    prec, rec = tp / len(pred), tp / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score_query(pred: list[Interval] | None, gt: list[Interval], expects_empty: bool) -> dict:
    parse_fail = pred is None
    p = pred or []
    p5 = matched_f1(p, gt, 0.5)
    p7 = matched_f1(p, gt, 0.7)
    return {
        "parse_fail": parse_fail,
        "union_iou": 0.0 if parse_fail else union_iou(p, gt),
        "f1@0.5": 0.0 if parse_fail else p5[2],
        "f1@0.7": 0.0 if parse_fail else p7[2],
        "recall@0.5": 0.0 if parse_fail else p5[1],
        "n_pred": len(p),
        "n_gt": len(gt),
        "count_acc": (not parse_fail) and len(p) == len(gt),
        "expects_empty": expects_empty,
        "pred_empty": (not parse_fail) and len(p) == 0,
    }


def summarize(rows: list[dict], key: str = "qtype") -> dict:
    """Aggregate per condition type and overall."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
        groups["ALL"].append(r)
    out = {}
    for g, rs in groups.items():
        n = len(rs)
        emp = [r for r in rs if r["expects_empty"]]
        nonemp = [r for r in rs if not r["expects_empty"]]
        pred_emp = [r for r in rs if r["pred_empty"]]
        correct_emp = [r for r in emp if r["pred_empty"]]
        # With no rejection queries of this type there is nothing to reject, so the
        # rejection metrics are undefined. Reporting 0.0 would read as a failure at
        # a task that was never posed; over-rejection is covered by
        # false_rejection_rate instead.
        has_rej = bool(emp)
        rej_p = (len(correct_emp) / len(pred_emp) if pred_emp else 0.0) if has_rej else None
        rej_r = (len(correct_emp) / len(emp)) if has_rej else None
        out[g] = {
            "n": n,
            "n_rejection_queries": len(emp),
            "parse_fail_rate": sum(r["parse_fail"] for r in rs) / n,
            "union_iou": float(np.mean([r["union_iou"] for r in nonemp])) if nonemp else None,
            "f1@0.5": float(np.mean([r["f1@0.5"] for r in nonemp])) if nonemp else None,
            "f1@0.7": float(np.mean([r["f1@0.7"] for r in nonemp])) if nonemp else None,
            "count_acc": sum(r["count_acc"] for r in rs) / n,
            "under_report_rate": (sum(r["n_pred"] < r["n_gt"] for r in nonemp) / len(nonemp)) if nonemp else None,
            "rejection_precision": rej_p,
            "rejection_recall": rej_r,
            "rejection_f1": (2 * rej_p * rej_r / (rej_p + rej_r)
                             if has_rej and (rej_p + rej_r) else (None if not has_rej else 0.0)),
            "false_rejection_rate": (sum(r["pred_empty"] for r in nonemp) / len(nonemp)) if nonemp else None,
        }
    return out
