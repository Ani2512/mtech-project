"""Re-test a finished run's optimized prompt on fresh, disjoint examples.

Why this exists
---------------
Run 4 measured its optimized prompt at +0.0181 with p = 0.21 on 60 held-out
examples: the right sign, too little power. Per-example differences had a
standard deviation of 0.11, so resolving an effect that size needs about 148
examples, not 60.

The obvious move -- re-run with `--test 150` -- would not answer the question.
A run draws its whole split with one `rng.sample(pool, n_train + n_dev +
n_test)`, so enlarging the test set reshuffles train and dev too; the optimizer
would see different training data, produce a different prompt, and the result
would be a second independent experiment rather than a confirmation of the
first.

What actually confirms a finding is evaluating the *same* prompt on data it has
never touched. This script loads a run's `config.json` so the scoring is
identical, reads its `split.json` so every example the run saw is withheld,
draws a fresh sample from what remains, and re-runs the paired comparison at
whatever size the power calculation asked for. It runs no search and calls no
optimizer, so it costs two task calls and two judge calls per example.

Usage
-----
    python run_replicate.py runs/20260815-122019 --n 250
    python run_replicate.py runs/20260815-122019 --n 250 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

from apo.backends import make_backend
from apo.config import Settings, estimate_cost, load_api_key
from apo.data import build_split
from apo.optimizer import compare_on_test
from apo.reporting import new_run_dir


def load_run(run_dir: pathlib.Path) -> tuple[Settings, str, str, set[int]]:
    """Recover a finished run's settings, prompts, and the rows it consumed."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    settings = Settings.from_dict(cfg)

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    baseline = result["initial_prompt"]
    optimized = result["best_prompt"]

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    used = set(split["train_idx"]) | set(split["dev_idx"]) | set(split["test_idx"])
    return settings, baseline, optimized, used


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    """Two-sided exact binomial test on the signs, ignoring ties.

    Reported alongside the bootstrap because the two answer different
    questions: the bootstrap asks whether the mean moved, the sign test asks
    whether more examples improved than worsened. Run 3 had them disagree
    sharply (p = 0.84 against p = 0.07), which was the most informative thing
    in it.
    """
    pos = sum(1 for d in diffs if d > 1e-9)
    neg = sum(1 for d in diffs if d < -1e-9)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=pathlib.Path, help="A finished run directory.")
    p.add_argument("--n", type=int, default=250, help="Fresh examples to draw.")
    p.add_argument("--seed", type=int, default=101, help="Sampling seed; must differ from the original run's.")
    p.add_argument("--max-cost", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not (run_dir / "config.json").exists():
        print(f"Not a run directory: {run_dir}", file=sys.stderr)
        return 2

    settings, baseline, optimized, used = load_run(run_dir)
    if args.seed == settings.seed:
        print(
            f"--seed {args.seed} matches the original run's seed; use a "
            "different one so the fresh sample is genuinely independent.",
            file=sys.stderr,
        )
        return 2
    settings.max_cost_usd = args.max_cost

    print(f"Replicating: {run_dir}")
    print(f"  scoring    : judge={settings.judge_rubric} w={settings.judge_weight}, "
          f"rouge={settings.rouge_weight}, f1={settings.f1_weight}, "
          f"constraint={settings.constraint_weight}")
    print(f"  withholding: {len(used)} examples the original run saw")

    split = build_split(
        seed=args.seed,
        n_train=0,
        n_dev=0,
        n_test=args.n,
        require_input=settings.require_input,
        filter_keyword=settings.filter_keyword,
        max_output_chars=settings.max_output_chars,
        min_input_chars=settings.min_input_chars,
        require_constraint=settings.require_constraint,
        exclude_idx=used,
    )
    fresh = split.test
    overlap = used & {e.idx for e in fresh}
    assert not overlap, f"fresh sample overlaps the original run: {overlap}"
    print(f"  fresh test : {len(fresh)} examples, 0 overlap with the original")

    calls = 2 * len(fresh) * (2 if settings.use_judge else 1)
    cost = len(fresh) * 2 * estimate_cost(settings.task_model, 700, 350)
    if settings.use_judge:
        cost += len(fresh) * 2 * estimate_cost(settings.judge_model, 950, 120)
    print(f"  budget     : ~{calls} calls, ~${cost:.2f}")

    if args.dry_run:
        return 0
    if settings.backend != "mock":
        if not load_api_key(settings.backend):
            print(f"No API key for '{settings.backend}'.", file=sys.stderr)
            return 2
        if not args.yes:
            if input("\nProceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return 1

    backend = make_backend(settings)
    done = {"n": 0}

    def progress(event: dict) -> None:
        if event.get("event") == "eval_progress":
            print(
                f"  evaluating [{event['label']}] {event['done']}/{event['total']}".ljust(60),
                end="\r",
                flush=True,
            )

    comparison = compare_on_test(
        backend, baseline, optimized, fresh, settings, progress=progress
    )

    diffs = [
        r["optimized_score"] - r["baseline_score"] for r in comparison["per_example"]
    ]
    pos, neg, sp = sign_test(diffs)
    sig = comparison["significance"]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    out = new_run_dir()
    payload = {
        "replicates": str(run_dir),
        "n": len(fresh),
        "seed": args.seed,
        "comparison": comparison,
        "sign_test": {"positive": pos, "negative": neg, "p_value": sp},
        "sd_of_differences": sd,
        "usage": backend.usage.as_dict(),
    }
    (out / "replication.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    b = comparison["baseline"]["total"]
    o = comparison["optimized"]["total"]
    print("\n" + "=" * 68)
    print(f"REPLICATION of {run_dir.name} on {len(fresh)} fresh examples")
    print("=" * 68)
    for k in ("total", "judge", "rouge_l", "token_f1", "constraint"):
        if k in comparison["baseline"]:
            bb = comparison["baseline"][k]
            oo = comparison["optimized"][k]
            print(f"  {k:12s} {bb:.4f} -> {oo:.4f}  ({oo - bb:+.4f})")
    print(
        f"\n  mean diff {sig['mean_diff']:+.4f}  95% CI "
        f"[{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}]  "
        f"bootstrap p = {sig['p_value']:.4f}"
    )
    print(f"  win/loss/tie {comparison['wins']}/{comparison['losses']}/{comparison['ties']}"
          f"   sign test p = {sp:.4f}")
    print(f"  sd of per-example differences = {sd:.4f}")
    verdict = (
        "SIGNIFICANT at p < 0.05"
        if sig["p_value"] < 0.05
        else "not significant at p < 0.05"
    )
    print(f"  verdict: {verdict}")
    u = backend.usage.as_dict()
    print(f"\nCalls: {u['calls']} ({u['cache_hits']} cached) | Cost: ${u['cost_usd']:.3f}")
    print(f"Artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
