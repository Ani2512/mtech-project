"""Run artifacts: budget estimation, persistence, and a markdown report."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any

from .config import RUNS_DIR, Settings, estimate_cost
from .data import Split
from .optimizer import OptimizationResult

# Rough average token counts per call, used only for the pre-run estimate.
_AVG_TOKENS = {
    "task": (700, 350),
    "judge": (950, 120),
    "optimizer": (1800, 600),
}


def estimate_budget(settings: Settings) -> dict[str, Any]:
    """Approximate the number of API calls and USD cost before running."""
    per_eval = 2 if settings.use_judge else 1
    beam = settings.beam_width
    n_new = settings.n_gradients * settings.n_edits * (1 + settings.n_paraphrases)
    pool = beam + beam * n_new

    task_calls = 0
    judge_calls = 0
    opt_calls = 0

    for i in range(settings.iterations):
        active = 1 if i == 0 else beam
        # minibatch evaluation of each beam member
        task_calls += active * settings.minibatch_size
        judge_calls += active * settings.minibatch_size * (per_eval - 1)
        # gradient + edit + paraphrase calls
        opt_calls += active * (
            1
            + settings.n_gradients
            + settings.n_gradients * settings.n_edits
            * (1 if settings.n_paraphrases else 0)
        )
        # bandit pulls over the candidate pool
        pulls = (1 + n_new) * active + settings.ucb_pulls
        task_calls += pulls
        judge_calls += pulls * (per_eval - 1)

    # held-out comparison: two prompts over the test split
    task_calls += 2 * settings.n_test
    judge_calls += 2 * settings.n_test * (per_eval - 1)

    cost = 0.0
    for role, calls, model in (
        ("task", task_calls, settings.task_model),
        ("judge", judge_calls, settings.judge_model),
        ("optimizer", opt_calls, settings.optimizer_model),
    ):
        tin, tout = _AVG_TOKENS[role]
        cost += calls * estimate_cost(model, tin, tout)

    total = task_calls + judge_calls + opt_calls
    return {
        "task_calls": task_calls,
        "judge_calls": judge_calls,
        "optimizer_calls": opt_calls,
        "total_calls": total,
        "estimated_cost_usd": round(cost, 2),
        "candidate_pool_per_iteration": pool,
    }


def new_run_dir(root: pathlib.Path = RUNS_DIR) -> pathlib.Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: pathlib.Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_run(
    run_dir: pathlib.Path,
    *,
    settings: Settings,
    split: Split,
    result: OptimizationResult,
    comparison: dict[str, Any] | None = None,
) -> pathlib.Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "config.json", settings.to_dict())
    _write(
        run_dir / "split.json",
        {
            "sizes": {
                "train": len(split.train),
                "dev": len(split.dev),
                "test": len(split.test),
            },
            "train_idx": [e.idx for e in split.train],
            "dev_idx": [e.idx for e in split.dev],
            "test_idx": [e.idx for e in split.test],
        },
    )
    _write(
        run_dir / "result.json",
        {
            "initial_prompt": result.initial_prompt,
            "best_prompt": result.best_prompt,
            "best_dev_score": result.best_dev_score,
            "stopped_early": result.stopped_early,
            "stop_reason": result.stop_reason,
            "elapsed_seconds": result.elapsed_seconds,
            "usage": result.usage,
            "history": result.history,
        },
    )
    (run_dir / "best_prompt.txt").write_text(result.best_prompt, encoding="utf-8")
    if comparison is not None:
        _write(run_dir / "comparison.json", comparison)
    (run_dir / "report.md").write_text(
        build_report(settings, result, comparison), encoding="utf-8"
    )
    return run_dir


def build_report(
    settings: Settings,
    result: OptimizationResult,
    comparison: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Automatic Prompt Optimization — run report\n")
    a(f"- Generated: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    a(f"- Backend: `{settings.backend}`")
    a(
        f"- Models: task=`{settings.task_model}`, judge=`{settings.judge_model}`, "
        f"optimizer=`{settings.optimizer_model}`"
    )
    a(
        f"- Search: {settings.iterations} iterations, beam={settings.beam_width}, "
        f"minibatch={settings.minibatch_size}, gradients={settings.n_gradients}, "
        f"edits={settings.n_edits}, paraphrases={settings.n_paraphrases}"
    )
    a(f"- Elapsed: {result.elapsed_seconds:.1f}s")
    u = result.usage
    a(
        f"- Usage: {u.get('calls', 0)} calls "
        f"({u.get('cache_hits', 0)} cache hits), "
        f"${u.get('cost_usd', 0):.3f} estimated"
    )
    if result.stopped_early:
        a(f"- **Stopped early:** {result.stop_reason}")
    if settings.backend == "mock":
        a(
            "\n> **Mock backend.** These numbers exercise the pipeline only and "
            "are not meaningful results."
        )

    a("\n## Search trajectory\n")
    a("| Iteration | Pool size | Best dev score | Improved |")
    a("|---|---|---|---|")
    for h in result.history:
        a(
            f"| {h['iteration']} | {h['pool_size']} | "
            f"{h['best_dev_score']:.4f} | {'yes' if h['improved'] else 'no'} |"
        )

    a("\n## Initial prompt\n")
    a("```\n" + result.initial_prompt.strip() + "\n```")
    a("\n## Optimized prompt\n")
    a("```\n" + result.best_prompt.strip() + "\n```")

    if comparison:
        a("\n## Held-out test results\n")
        b, o = comparison["baseline"], comparison["optimized"]
        a(f"Test examples: {comparison['n_test']}\n")
        a("| Metric | Baseline | Optimized | Delta |")
        a("|---|---|---|---|")
        for k in ("total", "judge", "rouge_l", "token_f1", "constraint"):
            if k not in b or k not in o:
                continue
            a(
                f"| {k} | {b[k]:.4f} | {o[k]:.4f} | "
                f"{o[k] - b[k]:+.4f} |"
            )
        if "constraint_pass_rate" in b:
            a(
                f"| constraint (fully met) | {b['constraint_pass_rate']:.1%} | "
                f"{o['constraint_pass_rate']:.1%} | "
                f"{o['constraint_pass_rate'] - b['constraint_pass_rate']:+.1%} |"
            )
            a(
                f"\n{b['constraint_n']} of {comparison['n_test']} test examples "
                f"state a countable constraint."
            )
        sig = comparison["significance"]
        a(
            f"\n- Win / loss / tie: **{comparison['wins']} / "
            f"{comparison['losses']} / {comparison['ties']}**"
        )
        a(
            f"- Mean score difference: **{sig['mean_diff']:+.4f}** "
            f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], "
            f"paired bootstrap p = {sig['p_value']:.4f})"
        )
        verdict = (
            "statistically significant at p < 0.05"
            if sig["p_value"] < 0.05
            else "NOT statistically significant at p < 0.05"
        )
        a(f"- Verdict: the improvement is {verdict}.")

    return "\n".join(lines) + "\n"