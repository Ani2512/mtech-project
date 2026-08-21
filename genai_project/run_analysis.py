"""Command-line entry point.

    python run_analysis.py "Why did profit decline?"
    python run_analysis.py "Should we expand into Southeast Asia?" --backend mock
    python run_analysis.py "Which products should we discontinue?" --agents finance,sales

Writes the report to stdout and, with --save, the full structured run to
runs/run-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import sys

from analyst import (
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENAI,
    SPECIALIST_ROLES,
    Settings,
    render_evaluation,
    run_analysis,
)
from analyst.config import EFFORT_LEVELS, PROVIDER_MODELS, ROLE_LABELS

# Questions worth asking of this data room, for --list-examples and the UI.
EXAMPLE_QUESTIONS = [
    "Should we expand into Southeast Asia next year?",
    "Why did profit decline?",
    "Which products should we discontinue?",
    "What is hurting customer satisfaction most?",
    "Which market has the highest growth potential for us?",
    "How do we compare with our competitors right now?",
    "What are our biggest risks over the next four quarters?",
    "Should we increase the marketing budget?",
]


def _force_utf8() -> None:
    """Windows consoles default to cp1252, which cannot encode ★ or █.

    The report is full of both, so reconfigure the streams rather than
    stripping the characters that make the output readable.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # already detached or redirected oddly
                pass


def _progress(stage: str, status: str, detail: str = "") -> None:
    marks = {"running": "..", "preparing": "..", "done": "OK", "failed": "!!", "warning": "??"}
    label = ROLE_LABELS.get(stage, stage.title())
    print(f"  [{marks.get(status, '  ')}] {label:<28} {detail[:80]}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-agent business analyst.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example questions:\n"
        + "\n".join(f"  {q}" for q in EXAMPLE_QUESTIONS),
    )
    p.add_argument("question", nargs="?", help="The decision question to analyse.")
    p.add_argument(
        "--backend",
        default=PROVIDER_OPENAI,
        choices=[PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_MOCK],
        help="mock runs the whole pipeline offline with no API key (default: openai).",
    )
    p.add_argument("--model", default=None, help="Override the model for every agent.")
    p.add_argument(
        "--agents",
        default=None,
        help=f"Comma-separated subset of: {','.join(SPECIALIST_ROLES)}",
    )
    p.add_argument(
        "--effort",
        default=None,
        choices=EFFORT_LEVELS,
        help="Override reasoning effort for every agent.",
    )
    p.add_argument("--no-retrieval", action="store_true", help="Skip hybrid document retrieval.")
    p.add_argument("--no-eval", action="store_true", help="Skip the LLM-as-judge pass.")
    p.add_argument("--no-cache", action="store_true", help="Ignore the on-disk response cache.")
    p.add_argument("--all-agents", action="store_true", help="Run every specialist; ignore the planner's selection.")
    p.add_argument("--max-cost", type=float, default=5.0, help="Hard USD ceiling (default: 5.00).")
    p.add_argument("--max-calls", type=int, default=60, help="Hard model-call ceiling (default: 60).")
    p.add_argument("--save", action="store_true", help="Write the full run to runs/.")
    p.add_argument("--out", default=None, help="Write the report markdown to this file.")
    p.add_argument("--list-examples", action="store_true", help="Print example questions and exit.")
    p.add_argument("--list-models", action="store_true", help="Print available models and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)

    if args.list_examples:
        for q in EXAMPLE_QUESTIONS:
            print(q)
        return 0
    if args.list_models:
        for provider, models in PROVIDER_MODELS.items():
            print(f"{provider}: {', '.join(models)}")
        return 0
    if not args.question:
        build_parser().print_help()
        return 2

    settings = Settings(
        backend=args.backend,
        use_cache=not args.no_cache,
        use_retrieval=not args.no_retrieval,
        run_evaluation=not args.no_eval,
        planner_selects_agents=not args.all_agents,
        max_cost_usd=args.max_cost,
        max_llm_calls=args.max_calls,
    )
    if args.model:
        settings.model = args.model
    if args.agents:
        chosen = [a.strip() for a in args.agents.split(",") if a.strip()]
        unknown = [a for a in chosen if a not in SPECIALIST_ROLES]
        if unknown:
            print(f"Unknown agent(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Valid: {', '.join(SPECIALIST_ROLES)}", file=sys.stderr)
            return 2
        settings.enabled_specialists = chosen
    if args.effort:
        settings.role_effort = {role: args.effort for role in settings.role_effort}

    print(f"Question: {args.question}", file=sys.stderr)
    print(f"Backend:  {settings.backend} / {settings.model}\n", file=sys.stderr)

    try:
        result = run_analysis(args.question, settings, progress=_progress)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:  # missing API key, unknown backend
        print(f"\n{exc}", file=sys.stderr)
        return 1

    report = result.report_markdown
    if result.evaluation is not None:
        report += "\n\n" + render_evaluation(result.evaluation)

    print(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"\nReport written to {args.out}", file=sys.stderr)
    if args.save:
        path = result.save()
        print(f"Full run written to {path}", file=sys.stderr)

    u = result.usage
    print(
        f"\n{u.get('calls', 0)} calls, "
        f"{u.get('input_tokens', 0):,} in / {u.get('output_tokens', 0):,} out, "
        f"${u.get('cost_usd', 0):.4f}, {result.seconds:.1f}s",
        file=sys.stderr,
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())