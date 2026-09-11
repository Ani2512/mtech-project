"""Phase 1 runner: model x dataset -> predictions.jsonl + summary.json.

    python -m dhwani.run_baseline --model mock --dataset synthetic --n 50 --out runs/mock
    python -m dhwani.run_baseline --model qwen2-audio --dataset hiacc --split adult --n 200 --out runs/q2a_hiacc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import LOADERS
from .metrics import score_utterance, summarize
from .models import DEFAULT_PROMPT, get_backend


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mock", "qwen2-audio", "shuka"])
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=None, help="max utterances")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = get_backend(args.model, **({"seed": args.seed} if args.model == "mock" else {}))

    loader = LOADERS[args.dataset]
    kw = {}
    if args.dataset == "synthetic":
        kw = {"n": args.n or 50, "seed": args.seed}
    elif args.split:
        kw = {"split": args.split}
    utts = loader(**kw)

    rows, t0 = [], time.time()
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for k, u in enumerate(utts):
            if args.n is not None and k >= args.n:
                break
            hyp = backend.transcribe(u.audio_path, args.prompt, reference=u.reference)
            s = score_utterance(u.ref_for_scoring(), hyp)
            row = {"id": u.id, "audio": u.audio_path, "reference": u.reference, "hypothesis": hyp, **s, **u.meta}
            rows.append(s)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 25 == 0:
                print(f"{k + 1} done, running WER {summarize(rows)['wer']:.3f}, {time.time() - t0:.0f}s")

    summary = {"model": backend.name, "dataset": args.dataset, "split": args.split, "prompt": args.prompt, **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
