"""Run all of phase 2 unattended, then print the comparison table.

    %run /kaggle/working/bootstrap.py      # once, to set up and smoke test
    %run /kaggle/working/phase2.py         # this

Every step is skipped if its output already exists, so re-running after a crash
resumes rather than restarting. Roughly 8 hours on a T4, inside Kaggle's
12-hour session.

One epoch, not two: there are only 200 training clips, so a second pass buys
little and doubles the most expensive item.
"""
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working/mtech-project/research_project"
TEST = "data/esc50/benchmark_test.jsonl"
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")
os.chdir(WORK)

EPOCHS = os.environ.get("DHWANI_EPOCHS", "1")


def run(label, cmd, produces):
    if produces and os.path.exists(produces):
        print(f"\n=== {label}: already done ===", flush=True)
        return True
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-m"] + cmd).returncode
    print(f"--- {label}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time()-t0)/60:.0f} min",
          flush=True)
    return rc == 0


ok = True

# --- arm C: QLoRA, plain-text timestamps ------------------------------------
ok &= run("train arm C (text timestamps)",
          ["dhwani.train_lora", "--data", "data/esc50/sft_train.jsonl",
           "--val", "data/esc50/sft_val.jsonl", "--out", "/kaggle/temp/lora_text",
           "--epochs", EPOCHS],
          "/kaggle/temp/lora_text/adapter_model.safetensors")
ok &= run("eval arm C",
          ["dhwani.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", "/kaggle/temp/lora_text",
           "--bench", TEST, "--out", "runs/esc50/test_lora_text"],
          "runs/esc50/test_lora_text/summary.json")

# --- arm E: QLoRA, atomic timestamp tokens ----------------------------------
ok &= run("train arm E (timestamp tokens)",
          ["dhwani.train_lora", "--data", "data/esc50/sft_train_tt.jsonl",
           "--val", "data/esc50/sft_val_tt.jsonl", "--out", "/kaggle/temp/lora_tt",
           "--epochs", EPOCHS, "--time-tokens", "--time-sigma", "0.3", "--time-lambda", "0.5"],
          "/kaggle/temp/lora_tt/adapter_model.safetensors")
ok &= run("eval arm E",
          ["dhwani.run_zeroshot", "--model", "qwen2.5-omni", "--adapter", "/kaggle/temp/lora_tt",
           "--bench", TEST, "--out", "runs/esc50/test_lora_tt"],
          "runs/esc50/test_lora_tt/summary.json")

# --- untrained baselines on the same test split -----------------------------
ok &= run("arm A (direct prompting)",
          ["dhwani.run_zeroshot", "--model", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_direct"],
          "runs/esc50/test_direct/summary.json")
ok &= run("arm B (decompose and combine)",
          ["dhwani.run_agent", "--grounder", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_agent"],
          "runs/esc50/test_agent/summary.json")
ok &= run("recall-biased decoding (k=5, 2 votes)",
          ["dhwani.run_zeroshot", "--model", "qwen2.5-omni", "--bench", TEST,
           "--out", "runs/esc50/test_union", "--samples", "5", "--min-votes", "2",
           "--temperature", "0.7"],
          "runs/esc50/test_union/summary.json")

# --- arm D: per-type selection, chosen on val, reported on test --------------
run("arm D (hybrid)",
    ["dhwani.hybrid", "--direct", "runs/esc50/test_direct",
     "--agent", "runs/esc50/test_agent", "--out", "runs/esc50/test_hybrid"],
    "runs/esc50/test_hybrid/summary.json")

# --- results ----------------------------------------------------------------
import glob

TYPES = ["PLAIN", "ORDINAL", "AFTER", "BEFORE", "NEXT_AFTER", "WHILE", "NOT_FOLLOWED", "ALL"]
runs = {}
for p in sorted(glob.glob("runs/esc50/test_*/summary.json")):
    name = os.path.basename(os.path.dirname(p)).replace("test_", "")
    runs[name] = json.load(open(p))["by_type"]

for metric in ("f1@0.5", "f_beta", "count_acc"):
    print(f"\n{metric}" + ("   (recall weighted 5.6x, the measured asymmetry)"
                           if metric == "f_beta" else ""))
    print("-" * (14 + 13 * len(TYPES)))
    print(f"{'arm':<14}" + "".join(f"{t:>13}" for t in TYPES))
    for m in sorted(runs):
        print(f"{m:<14}" + "".join(
            (f"{runs[m][t][metric]:>13.3f}"
             if isinstance(runs[m].get(t, {}).get(metric), (int, float)) else f"{'-':>13}")
            for t in TYPES))

# persist: /kaggle/working is kept as the notebook output
import shutil

out = "/kaggle/working/results"
os.makedirs(out, exist_ok=True)
for src in glob.glob("runs/esc50/*"):
    dst = os.path.join(out, os.path.basename(src))
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
for doc in glob.glob("docs/*.md"):
    shutil.copy(doc, out)
print(f"\nresults copied to {out} (download from the Output tab)")
print("ALL STEPS OK" if ok else "SOME STEPS FAILED - check the log above")
