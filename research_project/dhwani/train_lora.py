"""QLoRA fine-tuning of Qwen2.5-Omni's thinker for temporal grounding.

Only the thinker is trained: it is the audio-plus-text-to-text path, and the
talker (speech synthesis) is irrelevant here and is not loaded.

Mix rationale lives in docs/phase2_decomposition.md. Briefly: with perfect
events the conditions are trivial (agent scores 1.000), while at the model's
real grounding quality flawless condition logic still reaches only 0.263. So
grounding is the binding constraint and `dhwani.sft_data --plain-ratio` should
stay high rather than drilling conditional phrasing.

    python -m dhwani.train_lora --data data/esc50/sft_train.jsonl \
           --val data/esc50/sft_val.jsonl --out runs/lora_omni --epochs 2

Then evaluate with the adapter:
    python -m dhwani.run_zeroshot --model qwen2.5-omni --adapter runs/lora_omni \
           --bench data/esc50/benchmark_test.jsonl --out runs/esc50/omni_lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class GroundingCollator:
    """Builds a batch and masks the prompt so loss is taken on the answer only.

    Training on the prompt tokens teaches the model to reproduce our own
    instruction text, which wastes capacity and is not what is being measured.
    """

    def __init__(self, processor, sr: int = 16000, max_audio_s: float = 30.0):
        self.p = processor
        self.sr = sr
        self.max_audio_s = max_audio_s

    def __call__(self, batch: list[dict]):
        import librosa
        import torch

        texts, audios, prompt_lens = [], [], []
        for ex in batch:
            conv = []
            for m in ex["messages"]:
                if m["role"] == "user":
                    content = [{"type": "audio", "audio": ex["audio"]},
                               {"type": "text", "text": m["content"]}]
                else:
                    content = [{"type": "text", "text": m["content"]}]
                conv.append({"role": m["role"], "content": content})
            prompt = self.p.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            texts.append(prompt + ex["target"] + self.p.tokenizer.eos_token)
            prompt_lens.append(len(self.p.tokenizer(prompt, add_special_tokens=False).input_ids))
            a, _ = librosa.load(ex["audio"], sr=self.sr)
            audios.append(a[: int(self.max_audio_s * self.sr)])

        enc = self.p(text=texts, audio=audios, sampling_rate=self.sr,
                     return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[labels == self.p.tokenizer.pad_token_id] = -100
        for i, n in enumerate(prompt_lens):
            labels[i, :n] = -100          # loss on the answer only
        enc["labels"] = labels
        return enc


def build_model(model_id: str, precision: str | None, lora_r: int, lora_alpha: int,
                lora_dropout: float, time_tokens: bool = False, max_seconds: float = 30.0,
                resolution: float = 0.1):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    from .models import _fit_plan

    label, kw = _fit_plan(8.4, precision)
    print(f"[train] loading thinker in {label}")
    # Load the thinker alone: the talker is speech synthesis and is dead weight here.
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **kw)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    vocab = None
    if time_tokens:
        from .timetokens import TimeVocab

        vocab = TimeVocab(max_seconds, resolution)
        added = processor.tokenizer.add_tokens(vocab.tokens, special_tokens=False)
        model.resize_token_embeddings(len(processor.tokenizer))
        # TEMPO (arXiv:2608.29999): initialise each new embedding as the mean of
        # the BPE pieces of the number it stands for, so the tokens start where
        # the model already represents those digits rather than at random.
        n = vocab.init_embeddings(processor.tokenizer, model.get_input_embeddings().weight)
        print(f"[train] added {added} timestamp tokens, initialised {n} embeddings")

    if "4bit" in label or "8bit" in label:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        # Language side only. The audio encoder is frozen: with ~200 training clips
        # there is not enough signal to retrain perception, and unfreezing it is the
        # fastest way to overfit the composed-audio distribution.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # New timestamp embeddings are not low-rank updates to existing weights;
        # they are new rows and must be trained in full, or they stay at their
        # initialisation and the whole scheme is inert.
        modules_to_save=(["embed_tokens", "lm_head"] if time_tokens else None),
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model, processor, vocab


def time_loss_terms(logits, labels, time_ids, Q, ignore_index: int = -100):
    """TEMPO's distance-aware auxiliary loss, restricted to timestamp positions.

    Plain cross-entropy treats a prediction 0.1 s off as exactly as wrong as one
    10 s off, which throws away the ordinal structure that makes a timestamp
    vocabulary worth having. Instead score those positions against a Gaussian
    over neighbouring times:

        q_k  proportional to  exp(-(t_k - t*)^2 / (2 sigma^2))
        L_time = - sum_k q_k log p_k

    `Q[i]` is the precomputed soft target for the i-th time token.
    Returns (loss, n_positions); loss is 0 when the batch has no timestamps.
    """
    import torch

    # causal shift: position j predicts token j+1
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    lut = torch.full((int(logits.shape[-1]),), -1, dtype=torch.long, device=labels.device)
    lut[time_ids] = torch.arange(len(time_ids), device=labels.device)
    safe = labels.clamp_min(0)
    row = torch.where(labels == ignore_index, torch.full_like(labels, -1), lut[safe])
    mask = row >= 0
    n = int(mask.sum())
    if n == 0:
        return logits.new_zeros(()), 0

    sel = logits[mask][:, time_ids]                      # [n, T] timestamp columns only
    logp = torch.log_softmax(sel.float(), dim=-1)
    q = Q[row[mask]]                                     # [n, T]
    return -(q * logp).sum(dim=-1).mean(), n


def _make_time_trainer():
    """Built lazily so importing this module does not require transformers."""
    from transformers import Trainer

    class TimeAwareTrainer(Trainer):
        def configure_time_loss(self, tokenizer, vocab, sigma: float, lam: float):
            import torch

            from .timetokens import EMPTY_TOKEN

            ids = [tokenizer.convert_tokens_to_ids(t) for t in vocab.tokens
                   if t != EMPTY_TOKEN]
            if any(i is None or i < 0 for i in ids):
                raise ValueError("timestamp tokens are missing from the tokenizer; "
                                 "add_tokens must run before the trainer is built")
            self._time_ids = torch.tensor(ids, dtype=torch.long)
            self._Q = torch.tensor([vocab.soft_labels(t, sigma)[:-1] for t in vocab.times],
                                   dtype=torch.float)
            self._lam = float(lam)
            self._time_seen = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss
            if labels is not None and getattr(self, "_lam", 0.0) > 0:
                dev = outputs.logits.device
                if self._time_ids.device != dev:
                    self._time_ids = self._time_ids.to(dev)
                    self._Q = self._Q.to(dev)
                l_time, n = time_loss_terms(outputs.logits, labels, self._time_ids, self._Q)
                self._time_seen += n
                loss = loss + self._lam * l_time
            return (loss, outputs) if return_outputs else loss

    return TimeAwareTrainer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--precision", default=None, choices=["fp16", "8bit", "4bit"])
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--time-tokens", action="store_true",
                    help="atomic timestamp tokens plus the distance-aware Gaussian loss")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--time-sigma", type=float, default=0.3,
                    help="TEMPO uses 0.3 s")
    ap.add_argument("--time-lambda", type=float, default=0.5,
                    help="TEMPO uses 0.5")
    a = ap.parse_args(argv)

    from transformers import Trainer, TrainingArguments

    train = load_examples(Path(a.data))
    val = load_examples(Path(a.val)) if a.val else None
    print(f"[train] {len(train)} examples" + (f", {len(val)} val" if val else ""))

    model, processor, vocab = build_model(a.model_id, a.precision, a.lora_r, a.lora_alpha,
                                          a.lora_dropout, a.time_tokens, a.max_seconds,
                                          a.resolution)
    collate = GroundingCollator(processor)

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val else "no",
        bf16=False, fp16=True,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    if a.time_tokens:
        trainer = _make_time_trainer()(model=model, args=args, train_dataset=train,
                                       eval_dataset=val, data_collator=collate)
        trainer.configure_time_loss(processor.tokenizer, vocab, a.time_sigma, a.time_lambda)
    else:
        trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                          data_collator=collate)
    trainer.train()
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    print(f"[train] adapter saved to {a.out}")


if __name__ == "__main__":
    main()
