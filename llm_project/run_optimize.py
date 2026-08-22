"""Command-line entry point for the Automatic Prompt Optimization system.

Examples
--------
Estimate cost without spending anything:
    python run_optimize.py --dry-run

Smoke-test the whole pipeline offline (no API key, no cost):
    python run_optimize.py --backend mock --iterations 2

A real run on summarisation-style instructions:
    python run_optimize.py --filter summarize --with-input --iterations 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from apo.backends import BudgetExceeded, make_backend
from apo.config import Settings, load_api_key
from apo.data import split_from_settings
from apo.optimizer import compare_on_test, optimize
from apo.reporting import estimate_budget, new_run_dir, save_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatically refine a prompt using execution feedback "
        "on the Alpaca dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = Settings()

    g = p.add_argument_group("data")
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--train", type=int, default=d.n_train, dest="n_train")
    g.add_argument("--dev", type=int, default=d.n_dev, dest="n_dev")
    g.add_argument("--test", type=int, default=d.n_test, dest="n_test")
    g.add_argument(
        "--filter",
        default=None,
        dest="filter_keyword",
        help="Keep only examples whose instruction/input contains this text, "
        "e.g. 'summarize' or 'classify'.",
    )
    g.add_argument(
        "--with-input",
        action="store_true",
        help="Keep only examples that have a non-empty `input` field.",
    )
    g.add_argument(
        "--no-input",
        action="store_true",
        help="Keep only bare instructions (empty `input`).",
    )
    g.add_argument(
        "--min-input-chars",
        type=int,
        default=d.min_input_chars,
        help="Drop examples with less than this much real source text in "
        "`input` (URLs don't count). Alpaca's 'summarise this link' items are "
        "unanswerable and otherwise dominate the error signal; 200 is a good "
        "value for summarisation.",
    )
    g.add_argument(
        "--require-constraint",
        action="store_true",
        help="Keep only instructions that state a countable limit ('in one "
        "sentence', 'in less than 50 words'). 791 of Alpaca's 52k rows qualify. "
        "Pair with --constraint-weight to measure compliance directly.",
    )

    g = p.add_argument_group("search")
    g.add_argument("--iterations", type=int, default=d.iterations)
    g.add_argument("--beam", type=int, default=d.beam_width, dest="beam_width")
    g.add_argument(
        "--minibatch", type=int, default=d.minibatch_size, dest="minibatch_size"
    )
    g.add_argument("--gradients", type=int, default=d.n_gradients, dest="n_gradients")
    g.add_argument("--edits", type=int, default=d.n_edits, dest="n_edits")
    g.add_argument(
        "--paraphrases", type=int, default=d.n_paraphrases, dest="n_paraphrases"
    )
    g.add_argument("--ucb-pulls", type=int, default=d.ucb_pulls, dest="ucb_pulls")
    g.add_argument("--patience", type=int, default=d.patience)

    g = p.add_argument_group("models")
    g.add_argument(
        "--model",
        default=None,
        help="Set task, judge and optimizer models at once.",
    )
    g.add_argument("--task-model", default=d.task_model)
    g.add_argument("--judge-model", default=d.judge_model)
    g.add_argument("--optimizer-model", default=d.optimizer_model)
    g.add_argument(
        "--no-judge",
        action="store_true",
        help="Score with lexical metrics only (much cheaper, much noisier).",
    )

    g = p.add_argument_group("scoring")
    g.add_argument(
        "--judge-rubric",
        choices=["lenient", "strict"],
        default=d.judge_rubric,
        help="'lenient' is the original rubric, which scored the baseline "
        "prompt 0.975 and left no headroom to detect an improvement. 'strict' "
        "deducts from a ceiling for named defects and reserves 90+ for "
        "flawless answers.",
    )
    g.add_argument(
        "--constraint-weight",
        type=float,
        default=d.constraint_weight,
        help="Weight of the deterministic constraint-compliance component. 0 "
        "reproduces the original metric.",
    )
    g.add_argument("--judge-weight", type=float, default=d.judge_weight)
    g.add_argument("--rouge-weight", type=float, default=d.rouge_weight)
    g.add_argument("--f1-weight", type=float, default=d.f1_weight)

    g = p.add_argument_group("prompt")
    g.add_argument("--initial-prompt", default=d.initial_prompt)
    g.add_argument(
        "--initial-prompt-file",
        type=pathlib.Path,
        default=None,
        help="Read the starting prompt from a file (overrides --initial-prompt).",
    )

    g = p.add_argument_group("execution")
    g.add_argument(
        "--backend", choices=["openai", "anthropic", "mock"], default=d.backend
    )
    g.add_argument("--concurrency", type=int, default=d.concurrency)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--max-cost", type=float, default=d.max_cost_usd)
    g.add_argument("--max-calls", type=int, default=d.max_llm_calls)
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the budget estimate and exit without calling the API.",
    )
    g.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the held-out baseline-vs-optimized comparison.",
    )
    g.add_argument("-y", "--yes", action="store_true", help="Skip confirmation.")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    if args.with_input and args.no_input:
        raise SystemExit("--with-input and --no-input are mutually exclusive")
    require_input = True if args.with_input else (False if args.no_input else None)

    initial = args.initial_prompt
    if args.initial_prompt_file:
        initial = args.initial_prompt_file.read_text(encoding="utf-8").strip()

    task = judge = optimizer = None
    if args.model:
        task = judge = optimizer = args.model

    return Settings(
        seed=args.seed,
        n_train=args.n_train,
        n_dev=args.n_dev,
        n_test=args.n_test,
        require_input=require_input,
        filter_keyword=args.filter_keyword,
        min_input_chars=args.min_input_chars,
        require_constraint=args.require_constraint,
        iterations=args.iterations,
        beam_width=args.beam_width,
        minibatch_size=args.minibatch_size,
        n_gradients=args.n_gradients,
        n_edits=args.n_edits,
        n_paraphrases=args.n_paraphrases,
        ucb_pulls=args.ucb_pulls,
        patience=args.patience,
        task_model=task or args.task_model,
        judge_model=judge or args.judge_model,
        optimizer_model=optimizer or args.optimizer_model,
        use_judge=not args.no_judge,
        judge_rubric=args.judge_rubric,
        judge_weight=args.judge_weight,
        rouge_weight=args.rouge_weight,
        f1_weight=args.f1_weight,
        constraint_weight=args.constraint_weight,
        initial_prompt=initial,
        backend=args.backend,
        concurrency=args.concurrency,
        use_cache=not args.no_cache,
        max_cost_usd=args.max_cost,
        max_llm_calls=args.max_calls,
    )


def make_printer():
    state = {"last": ""}

    def progress(event: dict) -> None:
        kind = event.get("event")
        if kind == "iteration_start":
            print(f"\n=== Iteration {event['iteration']} ===", flush=True)
        elif kind == "eval_progress":
            msg = f"  evaluating [{event['label']}] {event['done']}/{event['total']}"
            print(msg.ljust(70), end="\r", flush=True)
            state["last"] = msg
        elif kind == "beam_evaluated":
            print(
                f"  beam member {event['index'] + 1} minibatch score: "
                f"{event['score']:.4f}".ljust(70),
                flush=True,
            )
        elif kind == "gradients":
            print(f"  gradients generated: {event['count']}", flush=True)
            for g in event.get("gradients", []):
                print(f"    - {g[:150]}", flush=True)
        elif kind == "candidates":
            print(f"  new candidate prompts: {event['count']}", flush=True)
        elif kind == "pool_ready":
            print(f"  candidate pool (deduped): {event['size']}", flush=True)
        elif kind == "ucb_pull":
            print(
                f"  bandit pulls {event['done']}/{event['total']} "
                f"best mean {event['best_mean']:.4f}".ljust(70),
                end="\r",
                flush=True,
            )
        elif kind == "iteration_end":
            flag = "improved" if event["improved"] else "no improvement"
            print(
                f"\n  best dev score: {event['best_score']:.4f} ({flag})",
                flush=True,
            )
        elif kind == "budget_exceeded":
            print(f"\n!! {event['message']}", flush=True)

    return progress


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)

    print("Loading Alpaca dataset ...", flush=True)
    split = split_from_settings(settings)
    print(f"Splits: {split.summary()}", flush=True)

    budget = estimate_budget(settings)
    print("\nBudget estimate (rough):")
    print(f"  task calls      : {budget['task_calls']}")
    print(f"  judge calls     : {budget['judge_calls']}")
    print(f"  optimizer calls : {budget['optimizer_calls']}")
    print(f"  total calls     : {budget['total_calls']}")
    is_live = settings.backend != "mock"
    if is_live:
        print(f"  estimated cost  : ~${budget['estimated_cost_usd']:.2f}")
    else:
        print("  estimated cost  : $0.00 (mock backend)")

    if args.dry_run:
        return 0

    if is_live:
        if not load_api_key(settings.backend):
            var = "OPENAI_API_KEY" if settings.backend == "openai" else "ANTHROPIC_API_KEY"
            print(
                f"\nNo API key found for '{settings.backend}'. Set {var} or add "
                "it to llm_project/.env, or run with --backend mock.",
                file=sys.stderr,
            )
            return 2
        if not args.yes:
            reply = input("\nProceed? [y/N] ").strip().lower()
            if reply not in {"y", "yes"}:
                print("Aborted.")
                return 1

    backend = make_backend(settings)
    progress = make_printer()

    result = optimize(backend, split, settings, progress=progress)

    comparison = None
    if not args.skip_test:
        print("\n=== Held-out evaluation ===", flush=True)
        try:
            comparison = compare_on_test(
                backend,
                result.initial_prompt,
                result.best_prompt,
                split.test,
                settings,
                progress=progress,
            )
        except BudgetExceeded as exc:
            # The optimization itself is already paid for; losing it here would
            # throw away the whole run. Save what we have and say so.
            print(f"\nBudget limit reached during held-out evaluation: {exc}")
            print("The optimized prompt is still saved; the comparison is not.")

    run_dir = new_run_dir()
    save_run(
        run_dir,
        settings=settings,
        split=split,
        result=result,
        comparison=comparison,
    )

    print("\n" + "=" * 68)
    print("OPTIMIZED PROMPT\n")
    print(result.best_prompt)
    print("=" * 68)
    if comparison:
        b = comparison["baseline"]["total"]
        o = comparison["optimized"]["total"]
        sig = comparison["significance"]
        print(
            f"Test score: baseline {b:.4f} -> optimized {o:.4f} "
            f"({o - b:+.4f}, p={sig['p_value']:.4f})"
        )
        print(
            f"Win/loss/tie: {comparison['wins']}/{comparison['losses']}/"
            f"{comparison['ties']}"
        )
    u = backend.usage.as_dict()
    print(
        f"Calls: {u['calls']} ({u['cache_hits']} cached) | "
        f"Estimated cost: ${u['cost_usd']:.3f}"
    )
    print(f"Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())