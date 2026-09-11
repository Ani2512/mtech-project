"""Interval parsing and metrics.

union_iou      TAG-Bench style: IoU in seconds between merged interval sets
matched_f1     AEGBench style: one-to-one Hungarian matching on IoU, F1 at a threshold
count_acc      |pred| == |gt|
rejection      queries whose ground truth is empty: did the model return []?
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

import numpy as np

Interval = tuple[float, float]

_PAIR = re.compile(r"\[?\s*(\d+(?:\.\d+)?)\s*(?:,|-|–|to)\s*(\d+(?:\.\d+)?)\s*\]?")
_EMPTY = re.compile(r"\[\s*\]|\bnone\b|\bno such\b|does not occur|not present|no occurrence", re.I)


def _pairs_from_obj(obj) -> list[Interval] | None:
    """Pull (start, end) pairs out of a decoded list.

    Models do not stick to [[s, e], ...]. Observed from Qwen2-Audio:
        [{'sneeze_start': '0.63', 'sneeze_end': '1.09'}, ...]
        [{'start': '16.39', 'end': '16.94'}, ...]
        [{'sneeze': '1.96-2.34'}, ...]
    Returning None for these would score the parser, not the model.
    """
    if not isinstance(obj, list):
        return None
    if not obj:
        return []
    pairs: list[Interval] = []
    for it in obj:
        if isinstance(it, (list, tuple)) and len(it) == 2 and all(_isnum(v) for v in it):
            pairs.append((float(it[0]), float(it[1])))
        elif isinstance(it, dict):
            got = _pair_from_dict(it)
            if got:
                pairs.append(got)
    # a flat [s, e] pair, e.g. "[19.43, 20.00]"
    if not pairs and len(obj) == 2 and all(_isnum(v) for v in obj):
        pairs.append((float(obj[0]), float(obj[1])))
    return _clean(pairs) if pairs else None


def _isnum(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.strip()); return True
        except ValueError:
            return False
    return False


def _pair_from_dict(d: dict) -> Interval | None:
    # keys ending in start/end, with any prefix: 'start', 'sneeze_start', 'onset'
    start = end = None
    for k, v in d.items():
        kl = str(k).lower()
        if not _isnum(v):
            continue
        if kl.endswith("start") or kl in ("onset", "from", "begin", "s"):
            start = float(str(v).strip())
        elif kl.endswith("end") or kl in ("offset", "to", "stop", "e"):
            end = float(str(v).strip())
    if start is not None and end is not None:
        return (start, end)
    # a single value holding a range: {'sneeze': '1.96-2.34'} or {'x': [1.9, 2.3]}
    for v in d.values():
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(_isnum(x) for x in v):
            return (float(v[0]), float(v[1]))
        if isinstance(v, str):
            m = _PAIR.search(v)
            if m:
                return (float(m.group(1)), float(m.group(2)))
    # exactly two numeric values and nothing else to go on
    nums = [float(str(v).strip()) for v in d.values() if _isnum(v)]
    if len(nums) == 2:
        return (nums[0], nums[1])
    return None


def parse_intervals(text: str) -> list[Interval] | None:
    """Return a list of (start, end); [] for an explicit empty answer; None if unparseable."""
    if text is None:
        return None
    text = text.strip()
    # 1) a bracketed literal anywhere in the text, as JSON then as a Python literal
    #    (models frequently emit single-quoted dicts, which json.loads rejects)
    for m in re.finditer(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text):
        blob = m.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(blob)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
            got = _pairs_from_obj(obj)
            if got is not None:
                return got
            break
    # 2) an explicit empty answer
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
