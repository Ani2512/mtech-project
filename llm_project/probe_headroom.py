"""Measure how much headroom a task model leaves the baseline prompt.

The weak-executor experiment only makes sense if the executor is weak enough
that the default prompt scores well below ceiling. Running the full
optimization to find that out costs ~$0.80; this costs a few cents.

    python probe_headroom.py --task-model gpt-3.5-turbo
    python probe_headroom.py --task-model gpt-4o-mini --n 30

Compare the printed baseline against the gpt-5.6-luna reference of 0.789
(run 20260815-122019, strict rubric). Lower is better here: it is the room an
optimizer would have to win in.
"""

from __future__ import annotations

import argparse
import sys

from apo.backends import make_backend
from apo.config import DEFAULT_MODEL, Settings
from apo.data import split_from_settings
from apo.optimizer import evaluate_prompt

# The score the same prompt/rubric reached with gpt-5.6-luna as executor.
REFERENCE_BASELINE = 0.789
REFERENCE_MODEL = DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-model", required=True, help="Executor to probe.")
    p.add_argument("--n", type=int, default=20, help="Test examples to score.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--judge-rubric", choices=["lenient", "strict"], default="strict")
    args = p.parse_args(argv)

    settings = Settings(
        backend="openai",
        task_model=args.task_model,
        judge_rubric=args.judge_rubric,
        seed=args.seed,
        n_train=1,
        n_dev=1,
        n_test=args.n,
    )
    split = split_from_settings(settings)
    backend = make_backend(settings)

    print(f"Probing {args.task_model} on {args.n} examples ...", flush=True)
    ev = evaluate_prompt(backend, settings.initial_prompt, split.test, settings)

    total = ev.mean_score
    room = 1.0 - total
    print()
    print(f"  baseline prompt : {settings.initial_prompt!r}")
    print(f"  executor        : {args.task_model}")
    print(f"  baseline score  : {total:.4f}   (headroom {room:.4f})")
    print(f"  reference       : {REFERENCE_BASELINE:.4f} on {REFERENCE_MODEL}")
    print(f"  cost            : ${backend.usage.cost_usd:.4f}"
          f"  ({backend.usage.calls} calls)")
    print()
    if total < 0.65:
        print("  VERDICT: usable — the baseline is off the ceiling, so an")
        print("           improvement has somewhere to show up.")
    elif total < REFERENCE_BASELINE - 0.03:
        print("  VERDICT: marginal — lower than the reference, but the margin")
        print("           is small next to the 0.08 per-example spread.")
    else:
        print("  VERDICT: not usable — this executor is no worse than the")
        print("           reference, so the ceiling problem is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
