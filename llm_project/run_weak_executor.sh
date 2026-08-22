#!/usr/bin/env bash
# The weak-executor experiment.
#
# Every run so far has returned no effect because the baseline prompt already
# scores 0.79-0.82 — there is no room for a better prompt to win. HOW_IT_WORKS
# section 12 names the fix: run the task role on a model weak enough that the
# baseline actually fails, and see whether optimization then helps.
#
# Everything except the executor is held at run 20260815-122019's settings
# (seed 13, strict rubric, 40/40/60), so the result is directly comparable.
#
#   ./run_weak_executor.sh probe        # a few cents  — is the executor weak enough?
#   ./run_weak_executor.sh optimize     # ~$0.80       — the experiment
#   ./run_weak_executor.sh replicate <run_dir>   # ~$1.30 — does the gain survive?
#
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
EXECUTOR="${EXECUTOR:-gpt-3.5-turbo}"   # override: EXECUTOR=gpt-4o-mini ./run_weak_executor.sh ...

# Python 3.14 from python.org ships without root certificates; without this the
# Alpaca download dies with CERTIFICATE_VERIFY_FAILED.
export SSL_CERT_FILE="$($PY -c 'import certifi; print(certifi.where())')"

# Judge and optimizer stay on the strong model. Only the executor is weakened,
# so any change is attributable to the executor and not to the grading.
COMMON=(
  --backend openai
  --task-model "$EXECUTOR"
  --judge-rubric strict
  --seed 13
  --train 40 --dev 40 --test 60
  --iterations 3 --beam 2
)

case "${1:-}" in
  probe)
    $PY probe_headroom.py --task-model "$EXECUTOR" --n "${2:-20}"
    ;;
  estimate)
    $PY run_optimize.py "${COMMON[@]}" --dry-run
    ;;
  optimize)
    $PY run_optimize.py "${COMMON[@]}" --max-cost 3.0 -y
    ;;
  replicate)
    if [ -z "${2:-}" ]; then
      echo "usage: $0 replicate <run_dir>   (e.g. runs/20260822-101500)" >&2
      exit 2
    fi
    # seed 101 != 13, so the replication draws disjoint examples.
    $PY run_replicate.py "$2" --n 250 --seed 101 --max-cost 3.0 -y
    ;;
  *)
    sed -n '2,14p' "$0"
    exit 2
    ;;
esac
