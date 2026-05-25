"""
benchmark/run_baseline.py

Run a frontier model baseline (direct prompt, no GBNF, no self-repair) on all
benchmark operations and save results to results_baseline_<model_slug>.json.

Usage (Colab):
    # Baseline 1 — DeepSeek V4
    BASELINE_MODEL=accounts/fireworks/models/deepseek-v4-0 \
        python benchmark/run_baseline.py

    # Baseline 2 — Kimi K2
    BASELINE_MODEL=accounts/fireworks/models/kimi-k2p5 \
        python benchmark/run_baseline.py

    # Or in Colab without env vars:
    %run benchmark/run_baseline.py
    # (uses default model below)
"""
import sys, os, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.operations import BENCHMARK_OPS
from src.baselines.baseline_runner import run_baseline

# ── Config ────────────────────────────────────────────────────────────────────
BASELINE_MODEL = os.environ.get(
    "BASELINE_MODEL",
    "accounts/fireworks/models/deepseek-v4-0",
)

# Derive a clean slug for the output filename
model_slug   = BASELINE_MODEL.split("/")[-1]
OUTPUT_FILE  = os.environ.get("RESULTS_FILE", f"results_baseline_{model_slug}.json")

print(f"{'=' * 60}")
print(f"BASELINE BENCHMARK")
print(f"  Model      : {BASELINE_MODEL}")
print(f"  No grammar, no self-repair (single shot)")
print(f"  Operations : {len(BENCHMARK_OPS)}")
print(f"  Output     : {OUTPUT_FILE}")
print(f"{'=' * 60}\n")

results = []

for op in BENCHMARK_OPS:
    print(f"[{op['id']}]  {op['pytorch_code']}")
    t0 = time.time()

    result = run_baseline(
        pytorch_code = op["pytorch_code"],
        input_shapes = op["input_shapes"],
        pytorch_fn   = op["pytorch_fn"],
        test_inputs  = op["test_inputs"],
        model        = BASELINE_MODEL,
        verbose      = False,
    )

    elapsed = time.time() - t0
    status  = "✅" if result["success"] else "❌"
    print(f"  {status}  success={result['success']}  "
          f"compilation={result['compilation_pass']}  "
          f"correctness={result['correctness_pass']}  "
          f"max_diff={result.get('max_diff', 'N/A')}  "
          f"time={elapsed:.1f}s")

    results.append({
        "op_id":            op["id"],
        "category":         op["category"],
        "pytorch_code":     op["pytorch_code"],
        "success":          result["success"],
        "attempts":         1,                      # baselines always 1 attempt
        "compilation_pass": result["compilation_pass"],
        "correctness_pass": result["correctness_pass"],
        "max_diff":         result.get("max_diff"),
        "elapsed_s":        round(elapsed, 2),
        "errors":           result.get("errors", []),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
total     = len(results)
n_success = sum(r["success"] for r in results)
n_compile = sum(r["compilation_pass"] for r in results)

print(f"\n{'=' * 60}")
print(f"SUMMARY — {model_slug}")
print(f"  Correctness rate : {n_success}/{total}  ({100*n_success/total:.1f}%)")
print(f"  Compilation rate : {n_compile}/{total}  ({100*n_compile/total:.1f}%)")
print(f"  Attempts         : always 1 (no repair)")
print(f"{'=' * 60}")

output = {
    "method":       "baseline_direct_prompt",
    "model":        BASELINE_MODEL,
    "model_slug":   model_slug,
    "grammar":      False,
    "self_repair":  False,
    "summary": {
        "total":            total,
        "correctness_rate": round(n_success / total, 4),
        "compilation_rate": round(n_compile / total, 4),
        "mean_attempts":    1.0,
    },
    "results": results,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
