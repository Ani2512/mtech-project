"""Print the model IDs your API key can actually reach.

Useful when a run fails with `model_not_found` — this shows the exact ID
strings the provider accepts, rather than guessing.

    python list_models.py                # OpenAI (default)
    python list_models.py --provider anthropic
    python list_models.py --filter gpt-5.6
"""

from __future__ import annotations

import argparse
import sys

from apo.config import PROVIDER_ANTHROPIC, PROVIDER_KEY_ENV, PROVIDER_OPENAI, load_api_key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--provider",
        choices=[PROVIDER_OPENAI, PROVIDER_ANTHROPIC],
        default=PROVIDER_OPENAI,
    )
    p.add_argument("--filter", default="", help="Only show IDs containing this text.")
    args = p.parse_args(argv)

    key = load_api_key(args.provider)
    if not key:
        var = PROVIDER_KEY_ENV[args.provider]
        print(
            f"No {var} found. Add it to llm_project/.env first.", file=sys.stderr
        )
        return 2

    try:
        if args.provider == PROVIDER_OPENAI:
            import openai

            client = openai.OpenAI(api_key=key)
            ids = [m.id for m in client.models.list()]
        else:
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            ids = [m.id for m in client.models.list()]
    except Exception as exc:
        print(f"Could not list models: {exc}", file=sys.stderr)
        return 1

    needle = args.filter.lower()
    shown = sorted(i for i in ids if needle in i.lower())
    if not shown:
        print(f"No models matched {args.filter!r} (of {len(ids)} available).")
        return 0

    print(f"{len(shown)} model(s) available to this key:\n")
    for i in shown:
        print(f"  {i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())