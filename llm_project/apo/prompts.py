"""Meta-prompts driving the optimizer.

This implements the "textual gradient" scheme of Pryzant et al. (2023),
*Automatic Prompt Optimization with Gradient Descent and Beam Search*:

  1. run the current prompt on a minibatch and collect failures
  2. ask an LLM to criticise the prompt given those failures  -> "gradient"
  3. ask an LLM to edit the prompt in the direction that fixes the criticism
     -> "descent step"
  4. paraphrase the winners to explore the local neighbourhood -> Monte Carlo

All three meta-prompts use structured outputs so parsing never depends on the
model's formatting whims.
"""

from __future__ import annotations

from typing import Any

STRING_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The requested list of results.",
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


GRADIENT_PROMPT = """You are diagnosing why a system prompt is underperforming.

The system prompt below is used to make an AI assistant answer instructions
from the Alpaca dataset. Each answer is graded 0-100 against a reference.

<current_prompt>
{prompt}
</current_prompt>

Here are its worst-scoring cases on a sample of the training set:

{failures}

Write {n} DISTINCT criticisms of the prompt itself.

Rules for each criticism:
- Attack the prompt, not the individual examples. The prompt must generalise
  to unseen instructions, so never propose hard-coding anything specific to
  these cases.
- Point at a concrete failure mode visible in the evidence above, and say
  which property of the prompt causes it (something it fails to say, says too
  vaguely, or says in a way that backfires).
- Each criticism must be independent — do not restate one criticism {n} ways.
- One or two sentences each. No preamble, no numbering."""


EDIT_PROMPT = """You are improving a system prompt for an AI assistant that answers
instructions from the Alpaca dataset.

<current_prompt>
{prompt}
</current_prompt>

A reviewer raised this specific criticism:

<criticism>
{gradient}
</criticism>

Evidence the criticism is based on:

{failures}

Write {n} improved version(s) of the ENTIRE system prompt that fix this
criticism.

Rules:
- Output the complete replacement prompt each time, not a diff or a comment.
- Fix the stated criticism while keeping whatever already works.
- Stay general. The prompt is applied to thousands of unseen instructions, so
  it must not mention these examples or their subject matter.
- Do not include placeholders like {{instruction}} — the user's instruction is
  supplied separately in the conversation.
- Keep it under 200 words. A prompt that is long and hedged usually scores
  worse than one that is direct.
- No markdown fences, no explanation of your changes."""


PARAPHRASE_PROMPT = """Rewrite the system prompt below in {n} different way(s).

<prompt>
{prompt}
</prompt>

Rules:
- Preserve the meaning and every instruction it contains exactly.
- Vary the wording, ordering, and phrasing — this is a search for a better
  surface form of the same idea, so genuinely different rewrites are useful
  and near-copies are not.
- Output the complete prompt each time. No fences, no commentary."""


def format_failures(failures: list[dict[str, Any]]) -> str:
    """Render failing cases as evidence for the critic/editor."""
    if not failures:
        return "(no failing cases were recorded)"
    blocks: list[str] = []
    for i, f in enumerate(failures, 1):
        block = [
            f"### Failing case {i} (score {f['score']:.2f})",
            f"Instruction: {f['instruction']}",
        ]
        if f.get("input"):
            block.append(f"Input: {f['input']}")
        block.append(f"Reference answer: {_clip(f['reference'], 600)}")
        block.append(f"Model's answer: {_clip(f['prediction'], 600)}")
        if f.get("violations"):
            # Stated separately from the critique because this one is counted,
            # not judged — the critic should treat it as fact, not opinion.
            block.append(f"Measured constraint violation: {f['violations']}")
        if f.get("critique"):
            block.append(f"Grader's critique: {f['critique']}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"