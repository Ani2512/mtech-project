"""One self-contained cell. Paste into Kaggle and run; safe to re-run any time.

Updates the code without deleting anything, rebuilds only what is missing, and
finishes with the trainer smoke test. It has no ordering dependency on other
cells, which is what kept going wrong.

    !curl -sL https://raw.githubusercontent.com/Ani2512/mtech-project/compositional-temporal-grounding/research_project/kaggle_bootstrap.py -o /kaggle/working/bootstrap.py
    %run /kaggle/working/bootstrap.py
"""
import os
import subprocess
import sys

REPO = "/kaggle/working/mtech-project"
WORK = f"{REPO}/research_project"
BRANCH = "compositional-temporal-grounding"
CACHE = "/kaggle/temp/hf"

os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.makedirs(CACHE, exist_ok=True)


def run(cmd, **kw):
    print(f"$ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}", flush=True)
    return subprocess.run(cmd, **kw)


def step(label):
    print(f"\n=== {label} ===", flush=True)


# ---------------------------------------------------------------- code
step("code")
os.chdir("/kaggle/working")
if os.path.isdir(f"{REPO}/.git"):
    # data/ and runs/ are gitignored, so a hard reset refreshes tracked code and
    # leaves the dataset, the benchmark and finished runs untouched.
    run(["git", "-C", REPO, "fetch", "-q", "origin"], check=True)
    run(["git", "-C", REPO, "reset", "-q", "--hard", f"origin/{BRANCH}"], check=True)
    print("updated in place; data preserved")
else:
    run(["git", "clone", "-q", "-b", BRANCH,
         "https://github.com/Ani2512/mtech-project.git", REPO], check=True)
    print("cloned fresh")
os.chdir(WORK)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())

# ---------------------------------------------------------------- deps
step("dependencies")
run([sys.executable, "-m", "pip", "install", "-q", "scipy", "soundfile", "librosa",
     "pyyaml", "pytest", "transformers>=4.52", "qwen-omni-utils", "accelerate",
     "bitsandbytes", "peft"], check=False)

step("self-test")
if run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:warnings"]).returncode:
    raise SystemExit("tests failed; stopping before anything expensive")

# ---------------------------------------------------------------- data
step("benchmark")
if os.path.exists("data/esc50/benchmark.jsonl"):
    print("already built")
else:
    run([sys.executable, "-m", "dhwani.build_benchmark", "--source", "esc50",
         "--n-clips", "300", "--p-overlap", "0.45", "--out", "data/esc50",
         "--esc50-root", "data/esc50_raw"], check=True)

step("splits")
if os.path.exists("data/esc50/benchmark_train.jsonl"):
    print("already split")
else:
    run([sys.executable, "-m", "dhwani.split", "--bench", "data/esc50/benchmark.jsonl",
         "--out", "data/esc50"], check=True)

step("training data")
for split in ("train", "val"):
    for tt, suffix in ((False, ""), (True, "_tt")):
        out = f"data/esc50/sft_{split}{suffix}.jsonl"
        if os.path.exists(out):
            continue
        cmd = [sys.executable, "-m", "dhwani.sft_data",
               "--bench", f"data/esc50/benchmark_{split}.jsonl",
               "--timelines", "data/esc50/timelines.jsonl",
               "--out", out, "--plain-ratio", "0.6"]
        if tt:
            cmd.append("--time-tokens")
        run(cmd, check=True)
for f in sorted(os.listdir("data/esc50")):
    if f.startswith("sft_"):
        print(" ", f, sum(1 for _ in open(f"data/esc50/{f}")), "examples")

# ---------------------------------------------------------------- smoke test
step("trainer smoke test (20 steps)")
rc = run([sys.executable, "-m", "dhwani.train_lora",
          "--data", "data/esc50/sft_train.jsonl", "--out", "/kaggle/temp/smoke",
          "--max-steps", "20", "--grad-accum", "1"]).returncode

print("\n" + "=" * 70)
if rc == 0:
    print("SMOKE TEST PASSED.")
    print("Check the log above: the loss should DECREASE and grad_norm must not be nan.")
    print("If it printed a WARNING about not training, tell me and we drop the lr.")
else:
    print(f"SMOKE TEST FAILED (exit {rc}). Send me the traceback above.")
print("=" * 70)
